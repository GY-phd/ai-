"""Fixed label budget, missing-class and order checks; no Batch 2 access."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter, process_time

from recipe import StructuredModel, anchor_update, np, ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "A_research_round08"))
from A_research_round08.models import TRAIN_SHA, ex
from A_research_round08.study import shift, fit_baselines, measurement, cost, process_peak
sys.path.insert(0, str(ROOT / "TaskAEngine_v1"))
from task_a_engine.data import fingerprints


def partitions(x, y, fold):
    assignment, _ = ex.source_folds(x, y)
    target = assignment == fold
    old_test = np.zeros(len(y), dtype=bool)
    pool, gap = [], []
    for label in range(6):
        old_test[np.flatnonzero(~target & (y == label))[4::5]] = True
        rows = np.flatnonzero(target & (y == label))
        pool.append(rows[:4])
        gap.append(int(rows[4]))
    pool = np.array(pool)
    evaluation = target.copy()
    evaluation[pool.ravel()] = False
    evaluation[gap] = False
    fit = ~target & ~old_test
    parts = [np.flatnonzero(fit), np.flatnonzero(old_test), pool.ravel(), np.array(gap), np.flatnonzero(evaluation)]
    assert len(set(np.concatenate(parts))) == sum(map(len, parts)) == len(x)
    keys = [set(fingerprints(x[part])) for part in parts]
    assert all(not keys[i] & keys[j] for i in range(len(keys)) for j in range(i))
    audit = {name: rows.tolist() for name, rows in zip(("fit", "old_test", "calibration_pool", "gap", "evaluation"), parts)}
    audit["supports"] = {name: np.bincount(y[rows], minlength=6).tolist() for name, rows in zip(("fit", "old_test", "pool", "gap", "evaluation"), parts)}
    return fit, old_test, pool, evaluation, audit


def scores(model, x, y, old_test, evaluation):
    return {"old_source": measurement(y[old_test], model.predict(x[old_test])),
            "clean": measurement(y[evaluation], model.predict(x[evaluation])),
            "composite": measurement(y[evaluation], model.predict(shift(x[evaluation], "composite")))}


def evaluate_update(parent, xcal, ycal, train, fit, old_test, evaluation, path):
    cpu, wall = process_time(), perf_counter()
    model = anchor_update(parent, xcal, ycal, train["x"][fit], train["y"][fit])
    timing = {"cpu_seconds": process_time() - cpu, "wall_seconds": perf_counter() - wall}
    model.save(path)
    return model, {"metrics": scores(model, train["x"], train["y"], old_test, evaluation),
                   "update_cost": timing, "checkpoint_sha256": ex.checkpoint_digest(path),
                   "new_labels": len(ycal), "class_counts": np.bincount(ycal, minlength=6).tolist()}


def run_updates(train, output):
    x, y = train["x"], train["y"]
    records = []
    for fold in (0, 1):
        fit, old_test, pool, evaluation, audit = partitions(x, y, fold)
        parent = StructuredModel.fit(x[fit], y[fit])
        parent_path = output / f"f{fold}_parent.npz"
        parent.save(parent_path)
        baselines = fit_baselines(train, fit) | {"frozen_A": parent}
        record = {"fold": fold, "partition": audit, "parent_sha256": ex.checkpoint_digest(parent_path),
                  "baselines": {name: scores(model, x, y, old_test, evaluation) for name, model in baselines.items()},
                  "budgets": {}, "missing": {}, "orders": {}}
        for count in (0, 1, 2, 4):
            rows = pool[:, :count].ravel()
            _, result = evaluate_update(parent, shift(x[rows], "composite"), y[rows], train, fit, old_test, evaluation,
                                        output / f"f{fold}_budget{count}.npz")
            result["calibration_rows"] = rows.tolist()
            record["budgets"][str(count)] = result
        for name, counts in {"no_Ac": [3, 3, 2, 0, 2, 2], "no_Et": [0, 3, 3, 2, 2, 2]}.items():
            rows = np.concatenate([pool[label, :count] for label, count in enumerate(counts)])
            _, result = evaluate_update(parent, shift(x[rows], "composite"), y[rows], train, fit, old_test, evaluation,
                                        output / f"f{fold}_{name}.npz")
            result["calibration_rows"] = rows.tolist()
            record["missing"][name] = result
        stage_rows = {"clean": pool[:, :2].ravel(), "composite": pool[:, 2:].ravel()}
        for order in (("clean", "composite"), ("composite", "clean")):
            model, previous_path = parent, parent_path
            name = "_then_".join(order)
            updates = []
            for stage in order:
                rows = stage_rows[stage]
                path = output / f"f{fold}_{name}_{stage}.npz"
                model, result = evaluate_update(model, shift(x[rows], stage), y[rows], train, fit, old_test, evaluation, path)
                result.update(stage=stage, calibration_rows=rows.tolist(), parent_sha256=ex.checkpoint_digest(previous_path))
                previous_path = path
                updates.append(result)
            record["orders"][name] = updates
        pooled_x = np.concatenate([shift(x[stage_rows[c]], c) for c in ("clean", "composite")])
        pooled_y = np.concatenate([y[stage_rows[c]] for c in ("clean", "composite")])
        model, record["pooled"] = evaluate_update(parent, pooled_x, pooled_y, train, fit, old_test, evaluation,
                                                   output / f"f{fold}_pooled.npz")
        record["inference_cost"] = cost(model, x[evaluation])
        record["replay_input_bytes"] = int(x[fit].nbytes + y[fit].nbytes)
        records.append(record)
    return records


def reverse_diagnosis(train, checkpoint):
    model = StructuredModel.load(checkpoint)
    x, y = train["x"], train["y"]
    fit = np.arange(len(x)) >= 248
    rebuilt = StructuredModel.fit(x[fit], y[fit])
    for key in ("coef", "center", "scale", "amplitude_scale"):
        np.testing.assert_allclose(getattr(model, key), getattr(rebuilt, key), rtol=1e-9, atol=1e-10)
    design = model.design(x[~fit])
    logits = design @ model.coef
    prediction = logits.argmax(axis=1)
    uncut = (model.features(x[~fit]) - model.center) / model.scale
    blocks = {"shape": slice(1, 113), "amplitude": slice(113, 145)}
    by_class = {}
    for label in (0, 1, 2):
        selected = (y[~fit] == label) & (prediction == 3)
        difference = model.coef[:, 3] - model.coef[:, label]
        by_class[ex.CLASSES[label]] = {"support": int(sum(y[~fit] == label)), "false_ac": int(selected.sum())}
        if selected.any():
            by_class[ex.CLASSES[label]].update({"mean_margin_ac_vs_true": float((logits[selected, 3] - logits[selected, label]).mean()),
                "bias_contribution": float(difference[0]),
                "shape_contribution": float((design[selected, blocks["shape"]] @ difference[blocks["shape"]]).mean()),
                "amplitude_contribution": float((design[selected, blocks["amplitude"]] @ difference[blocks["amplitude"]]).mean()),
                "clipped_feature_fraction": float((np.abs(uncut[selected]) > 6).mean())})
    ablations = {}
    for block, selection in blocks.items():
        changed = design.copy()
        changed[:, selection] = 0
        ablations["center_" + block] = measurement(y[~fit], ex.softmax(changed @ model.coef))
    ablations["no_clip_diagnostic"] = measurement(y[~fit], ex.softmax(np.column_stack((np.ones((~fit).sum()), uncut)) @ model.coef))
    train_design = model.design(x[fit])[:, 1:]
    centers = np.stack([train_design[y[fit] == label].mean(axis=0) for label in range(6)])
    distances = {}
    for label in (0, 1, 2):
        use = y[~fit] == label
        distances[ex.CLASSES[label]] = {ex.CLASSES[c]: float(np.linalg.norm(design[use, 1:] - centers[c], axis=1).mean()) for c in (label, 3)}
    return {"checkpoint_sha256": ex.checkpoint_digest(checkpoint), "training_support": np.bincount(y[fit], minlength=6).tolist(),
            "evaluation_support": np.bincount(y[~fit], minlength=6).tolist(), "metrics": measurement(y[~fit], ex.softmax(logits)),
            "false_ac_by_class": by_class, "feature_counterfactuals": ablations,
            "mean_distance_to_source_class_centroid": distances, "centroid_distances_are_descriptive": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if ex.checkpoint_digest(args.train) != TRAIN_SHA:
        raise ValueError("expected original Batch1")
    train = ex.read_dataset(args.train, "classification", ex.CLASSES)
    if train["sha256"] != TRAIN_SHA:
        raise ValueError("training file changed")
    args.output.mkdir(parents=True, exist_ok=False)
    cpu, wall = process_time(), perf_counter()
    result = {"train_sha256": TRAIN_SHA, "updates": run_updates(train, args.output),
              "reverse": reverse_diagnosis(train, args.reverse_checkpoint),
              "independent_generalization_verified": False, "decision": "no_update", "runtime_llm_calls": 0}
    result["cost"] = {"cpu_seconds": process_time() - cpu, "wall_seconds": perf_counter() - wall,
                      "peak_working_set_bytes": process_peak()}
    result["code_sha256"] = {name: ex.checkpoint_digest(Path(__file__).with_name(name))
                             for name in ("study.py", "recipe.py", "structured_model.py", "AUDIT_PLAN.md")}
    result["checkpoints"] = {p.name: ex.checkpoint_digest(p) for p in args.output.glob("*.npz")}
    ex.save_json(args.output / "results.json", result)
    print(json.dumps(result["cost"]))


if __name__ == "__main__":
    main()

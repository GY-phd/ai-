# -*- coding: utf-8 -*-
"""Protocol-consistent A/B/C audit and the degradation-weight experiment.

The point of this script is to separate three numbers that the research
reports currently mix:

* the **full validation** score (every row of every held-out unit),
* the **hidden-like** score (one row per unit, at a truncation horizon:
  C's test sequences are truncated, and the platform scores the last
  observed cycle), and
* the **deployable** score (a model that only sees ``inputs`` at
  prediction time; no unit id, no elapsed count).

The current research champions for B (two-stage) and C (``elapsed``
feature) are not deployable: the platform's prediction request carries
only ``{task, inputs}``.  This script measures the gap, then tests
degradation-stage sample weights and a median bias correction on the
C protocol.

Run:
    python -m research.tracks.optimize_abc --out artifacts/tracks/opt_abc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tracks import official_baselines as ob  # noqa: E402
from research.tracks.official_baselines import score_a, score_bc  # noqa: E402

# The public packages live under the user-provided data root.  The C
# directory has a "(1)" suffix in that copy; the official loader reads
# the same files either way.
ob.DATA_ROOT = Path(r"C:\harness_review\数据源")
ob.PACKAGES["C"] = "cpu-c-guide-20261110-v1 (1)"

C_HORIZONS = (0, 10, 20, 30, 50, 75, 100)


def _c_rows():
    root = ob.DATA_ROOT / ob.PACKAGES["C"] / "public"
    rows = [json.loads(line) for line in (root / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    engine, cycle, subset = [], [], []
    for row in rows:
        meta = row["meta"]
        engine.append(f"{meta['source_subset']}#{meta['unit']}")
        subset.append(meta["source_subset"])
        cycle.append(int(meta["cycle"]))
    return rows, np.asarray(engine), np.asarray(cycle, dtype=np.float64), np.asarray(subset)


def last_cycle_indices(engine, cycle, horizons=C_HORIZONS):
    """One row per engine at each truncation horizon (cycles before failure).

    The row's own ``answer`` is the true RUL at that cycle, exactly the
    quantity the official test set's ``RUL_FD00x`` provides for the last
    observed cycle.  This is the closest local approximation of the
    hidden prediction protocol.
    """
    out = {}
    last = {}
    for index, name in enumerate(engine):
        last[name] = max(last.get(name, 0), int(cycle[index]))
    for horizon in horizons:
        picked = []
        for name, last_cycle in last.items():
            members = np.nonzero(engine == name)[0]
            target = last_cycle - horizon
            chosen = members[np.argmin(np.abs(cycle[members] - target))]
            if cycle[chosen] <= last_cycle:
                picked.append(int(chosen))
        out[horizon] = np.asarray(sorted(set(picked)), dtype=int)
    return out


def c_metrics(y_true, y_pred):
    return score_bc(y_true, y_pred, True)


def fit_c(x, y, *, hidden=128, epochs=200, weights=None):
    from rulkit.model import Model

    model = Model("regression", [f"f{i}" for i in range(x.shape[1])], [], x, y,
                  hidden=hidden, seed=42, learning_rate=0.003, batch_size=128,
                  feature_transform="raw", target_scaling=True)
    model.fit(x, y, epochs, sample_weight=weights)
    return model


def degradation_weights(y, alpha=1.0):
    """Late-life rows carry more mass; the feature schema is untouched."""
    y = np.asarray(y, dtype=np.float64)
    top = max(float(y.max()), 1.0)
    return 1.0 + float(alpha) * (1.0 - np.clip(y / top, 0.0, 1.0))


def c_section():
    train = ob.load_split("C", "train")
    valid = ob.load_split("C", "validation")
    rows, engine, cycle, subset = _c_rows()
    picks = last_cycle_indices(engine, cycle)
    _, baseline = ob.load_checkpoint("C")
    base_full = c_metrics(valid["y"], baseline.predict(valid["x"]))
    base_last = {str(h): c_metrics(valid["y"][idx], baseline.predict(valid["x"][idx]))
                 for h, idx in picks.items()}

    print("C: fitting input-only h128/e200 (deployable) ...", flush=True)
    model = fit_c(train["x"], train["y"])
    full = c_metrics(valid["y"], model.predict(valid["x"]))
    last = {str(h): c_metrics(valid["y"][idx], model.predict(valid["x"][idx]))
            for h, idx in picks.items()}

    print("C: fitting weighted h128/e200 (alpha=1.0) ...", flush=True)
    weighted = fit_c(train["x"], train["y"], weights=degradation_weights(train["y"], 1.0))
    weighted_last = {str(h): c_metrics(valid["y"][idx], weighted.predict(valid["x"][idx]))
                     for h, idx in picks.items()}
    weighted_full = c_metrics(valid["y"], weighted.predict(valid["x"]))

    # Median bias correction fitted on one half of the validation engines
    # and scored on the other half.  The held-out half never contributes
    # to the offset, so this is an honest post-correction estimate.
    names = sorted(set(engine.tolist()))
    cal = set(names[::2])
    test = set(names[1::2])
    cal_rows = np.asarray([i for i, name in enumerate(engine) if name in cal], dtype=int)
    test_rows = np.asarray([i for i, name in enumerate(engine) if name in test], dtype=int)
    raw = model.predict(valid["x"])
    offset = float(np.median(valid["y"][cal_rows] - raw[cal_rows]))
    corrected = np.clip(raw + offset, 0.0, None)
    corrected_full = c_metrics(valid["y"][test_rows], corrected[test_rows])
    uncorrected_full = c_metrics(valid["y"][test_rows], raw[test_rows])

    return {
        "official_baseline": {"full": base_full, "last_cycle": base_last},
        "input_only_h128_e200": {"full": full, "last_cycle": last},
        "weighted_alpha1": {"full": weighted_full, "last_cycle": weighted_last},
        "bias_correction_holdout": {
            "offset": offset,
            "calibration_engines": len(cal), "test_engines": len(test),
            "before": uncorrected_full, "after": corrected_full,
        },
    }


def b_lobo(x, y, paths, groups, *, hidden=96, epochs=400, alpha=0.0):
    """Leave-one-bearing-out with an input-only MLP (deployable)."""
    from rulkit.model import Model

    per_fold = []
    for name in sorted(set(groups.tolist())):
        test = groups == name
        train = ~test
        weights = None
        if alpha:
            y_train = y[train]
            start = {}
            for unit in np.unique(groups[train]):
                mask = groups[train] == unit
                start[unit] = max(float(y_train[mask].max()), 1.0)
            w = np.asarray([1.0 + alpha * (1.0 - min(yy / start[gg], 1.0))
                            for yy, gg in zip(y_train, groups[train])], dtype=np.float64)
            weights = w
        model = Model("regression", paths, [], x[train], y[train],
                      hidden=hidden, seed=42, learning_rate=0.003, batch_size=128,
                      feature_transform="eng34", target_scaling=False)
        model.fit(x[train], y[train], epochs, sample_weight=weights)
        pred = model.predict(x[test])
        per_fold.append({"bearing": str(name), "rmse": float(np.sqrt(np.mean((pred - y[test]) ** 2))),
                         "mae": float(np.mean(np.abs(pred - y[test])))})
    return {"mean_rmse": float(np.mean([row["rmse"] for row in per_fold])),
            "mean_mae": float(np.mean([row["mae"] for row in per_fold])),
            "folds": per_fold}


def b_section():
    train = ob.load_split("B", "train")
    meta = [json.loads(line)["meta"] for line in
            (ob.DATA_ROOT / ob.PACKAGES["B"] / "public" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    groups = np.asarray([row["bearing"] for row in meta], dtype=object)
    print("B: input-only eng34/h96 LOBO (deployable) ...", flush=True)
    plain = b_lobo(train["x"], train["y"], train["paths"], groups, epochs=400, alpha=0.0)
    print("B: same + late-life weights alpha=1 ...", flush=True)
    weighted = b_lobo(train["x"], train["y"], train["paths"], groups, epochs=400, alpha=1.0)
    two_stage = json.loads((ROOT / "artifacts" / "tracks" / "track_b_final.json").read_text(encoding="utf-8"))
    return {
        "input_only_eng34_h96_lobo": plain,
        "input_only_eng34_h96_lobo_weighted": weighted,
        "two_stage_reference_lobo": two_stage.get("leave_one_bearing_out"),
        "note": "two_stage needs unit grouping at prediction; input-only is the deployable path",
    }


def a_section():
    train = ob.load_split("A", "train")
    valid = ob.load_split("A", "validation")
    # The validation batch is **ordered by class inside the batch** (the
    # A research note says so).  A split by ``source_index`` is therefore
    # a class split, not a time split: the last 40% of batch 2 is almost
    # entirely Acetone and Toluene.  That result is kept only as evidence
    # of the trap, never as a model score.
    rows = [json.loads(line) for line in
            (ob.DATA_ROOT / ob.PACKAGES["A"] / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    order = np.asarray([int(row["meta"]["source_index"]) for row in rows], dtype=np.int64)
    ranked = np.argsort(order, kind="stable")
    cut = int(0.6 * len(ranked))
    early, late = ranked[:cut], ranked[cut:]
    late_classes = np.bincount(valid["y"][late], minlength=6)

    from rulkit.model import Model
    x, y = train["x"], train["y"]
    # Class-balanced oversampling, the deployable version of the
    # rare-class hypothesis (training only; prediction uses inputs only).
    counts = np.bincount(y, minlength=6).astype(np.float64)
    counts[counts < 1] = 1.0
    mass = (counts.max() / counts)[y]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=len(y), replace=True, p=mass / mass.sum())
    model = Model("classification", train["paths"], list(ob.CLASSES_A), x[idx], y[idx],
                  hidden=32, seed=42, learning_rate=0.003, batch_size=128,
                  feature_transform="raw", target_scaling=False)
    model.fit(x[idx], y[idx], 50)

    def macro(indices):
        pred = np.argmax(model.predict(valid["x"][indices]), axis=1)
        return score_a([ob.CLASSES_A[int(i)] for i in valid["y"][indices]],
                       [ob.CLASSES_A[int(i)] for i in pred])

    # A **class-stratified** holdout inside batch 2.  This is the valid
    # way to hold out rows from a class-ordered batch; the strict
    # later-batch question cannot be answered because the package ships
    # only batches 1 and 2.
    holdout = []
    for label in range(6):
        members = np.nonzero(valid["y"] == label)[0]
        rng.shuffle(members)
        keep = max(1, len(members) // 5)
        holdout.extend(members[:keep].tolist())
    holdout = np.asarray(sorted(holdout), dtype=int)

    _, baseline = ob.load_checkpoint("A")
    base_pred = np.argmax(baseline.predict(valid["x"]), axis=1)
    base_labels = [ob.CLASSES_A[int(i)] for i in base_pred]
    return {
        "batch1_only_input_mlp": {
            "full_batch2": macro(np.arange(len(valid["y"]))),
            "stratified_20pct": macro(holdout),
        },
        "official_baseline": {
            "full_batch2": score_a([ob.CLASSES_A[int(i)] for i in valid["y"]], base_labels),
            "stratified_20pct": score_a(
                [ob.CLASSES_A[int(i)] for i in valid["y"][holdout]],
                [base_labels[int(i)] for i in holdout]),
        },
        "invalid_temporal_split": {
            "reason": "batch 2 is class-ordered; the last 40% by source_index is a class split",
            "late_40pct_class_counts": late_classes.tolist(),
            "late_40pct_macro_f1": macro(late)["macro_f1"],
            "late_40pct_accuracy": macro(late)["accuracy"],
        },
        "note": ("the research 0.9707 trains on 75% of batch 2 and scores a "
                 "class-stratified held-out slice; it is an adaptation proxy, "
                 "not a hidden later-batch guarantee. This is the batch-1-only "
                 "deployable number."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/tracks/opt_abc.json")
    parser.add_argument("--only", choices=("A", "B", "C"), default=None)
    args = parser.parse_args()
    results = {}
    if args.only in (None, "C"):
        results["C"] = c_section()
    if args.only in (None, "B"):
        results["B"] = b_section()
    if args.only in (None, "A"):
        results["A"] = a_section()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()

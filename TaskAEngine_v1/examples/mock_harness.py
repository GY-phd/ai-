"""Exercise the update lifecycle with synthetic data and mock label releases."""

import argparse
import json
from pathlib import Path

from task_a_engine import Engine
from task_a_engine.vendor.baselines.model import Model
import numpy as np


PARAMETERS = {"epochs": 4, "learning_rate": 0.02}


def release(stage, purpose, *, independent=True):
    value = {"kind": "mock", "release_id": f"mock-{stage}-{purpose}", "purpose": purpose, "labels": "released"}
    if purpose == "evaluate":
        value["independent"] = independent
    return value


def mock_rows(model, prefix, seed, count, *, rotate_labels=False):
    """Simulate class clusters with a fixed scale shift and no competition rows."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((6, 128))
    for index in range(6):
        centers[index, 20 * index:20 * (index + 1)] = 3.
    labels = np.repeat(np.arange(6), count)
    x = model.mean + model.scale * (centers[labels] + rng.normal(0, .08, (len(labels), 128)))
    targets = (labels + int(rotate_labels)) % 6
    return [{"id": f"{prefix}-{index:03d}", "inputs": {"features": row.tolist()}, "answer": model.meta["classes"][label]}
            for index, (row, label) in enumerate(zip(x, targets))]


def features_only(rows):
    return [{"id": row["id"], "inputs": row["inputs"]} for row in rows]


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    return path


def stable_fixture_model():
    import task_a_engine

    return Model.load(Path(task_a_engine.__file__).parent / "vendor" / "models" / "stable.npz")


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    engine = Engine(output / "engine", mode="mock")
    fixture = stable_fixture_model()
    initial = engine.status()
    stages = []

    unlabeled = features_only(mock_rows(fixture, "stage-1-observed", 6, 4))
    prediction = engine.predict(unlabeled)
    diagnosis = engine.diagnose(unlabeled)
    proposal = engine.propose(diagnosis)
    stages.append({"stage": 1, "data_scope": "unlabeled_stream", "active_model_id": prediction["model_id"],
                   "prediction_resources": prediction["resources"], "diagnosis": diagnosis, "proposal": proposal,
                   "outcome": "keep_without_released_labels"})
    assert engine.status()["revision"] == initial["revision"]

    for stage, rotates in ((2, False), (3, True)):
        train_rows = mock_rows(fixture, f"stage-{stage}-train", 10 * stage, 12, rotate_labels=rotates)
        evaluation_rows = mock_rows(fixture, f"stage-{stage}-evaluate", 10 * stage + 1, 8)
        observed = features_only(evaluation_rows)
        prediction = engine.predict(observed)
        diagnosis = engine.diagnose(observed)
        before = engine.status()
        train_release = release(stage, "train")
        proposal = engine.propose(diagnosis, training_release=train_release)
        train_file = write_rows(output / f"stage-{stage}-train.jsonl", train_rows)
        trained = engine.train(train_file, train_release, parent_id=before["active_model_id"], parameters=PARAMETERS)
        # The independent evaluation labels arrive only after training has completed.
        eval_file = write_rows(output / f"stage-{stage}-evaluate.jsonl", evaluation_rows)
        result = engine.evaluate(trained["model_id"], eval_file, release(stage, "evaluate"))
        action = engine.activate(trained["model_id"], result["evaluation_id"], expected_revision=before["revision"])
        expected = "activated" if stage == 2 else "kept"
        assert action["status"] == expected, action
        stages.append({"stage": stage, "data_scope": "mock_independent_release", "active_before": prediction["model_id"],
                       "prediction_resources": prediction["resources"], "diagnosis": diagnosis, "proposal": proposal,
                       "training": trained, "evaluation": result, "activation": action})

    committed = engine.status()
    rollback = engine.rollback(expected_revision=committed["revision"])
    assert rollback["status"] == "rolled_back" and rollback["active_model_id"] == initial["active_model_id"], rollback
    reopened = Engine(output / "engine", mode="mock").status()
    assert reopened["active_model_id"] == initial["active_model_id"]
    assert reopened["revision"] == rollback["revision"]

    summary = {"scope": "synthetic_mock_lifecycle_only", "independent_generalization_verified": False,
               "competition_score": None, "initial_active_model_id": initial["active_model_id"],
               "stages": stages, "rollback": rollback, "active_after_reopen": reopened["active_model_id"],
               "revision_after_reopen": reopened["revision"]}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the mock releases, ledger, and summary")
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps({"scope": summary["scope"], "summary": str(args.output / "summary.json"),
                      "final_active": summary["active_after_reopen"], "revision": summary["revision_after_reopen"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

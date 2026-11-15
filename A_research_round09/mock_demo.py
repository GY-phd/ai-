"""Run three local stages. All generated rows are mock, never activation evidence."""
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

from engine_recipe import RecipeEngine
from recipe import np
from task_a_engine.engine import CLASSES


def release(name, purpose="train"):
    return {"release_id": name, "purpose": purpose, "labels": "released", "kind": "mock",
            **({"independent": False} if purpose == "evaluate" else {})}


def fixture(folder, name, counts, seed):
    y = np.repeat(np.arange(6), counts)
    x = np.exp(np.random.default_rng(seed).normal(2 + y[:, None] / 4, .2, (len(y), 128)))
    x.reshape(-1, 16, 8)[:, :, 5:] *= -1
    path = folder / f"{name}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for i, (features, label) in enumerate(zip(x, y)):
            stream.write(json.dumps({"id": f"{name}-{i}", "inputs": {"features": features.tolist()}, "answer": CLASSES[label]}) + "\n")
    return path, x


def run(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    engine = RecipeEngine(folder / "store", mode="mock")
    source, _ = fixture(folder, "source", [8] * 6, 1)
    initial = engine.train(source, release("source"), method="structured_bootstrap")
    if initial["status"] != "trained":
        raise RuntimeError(initial)
    parent = initial["model_id"]
    stages = []
    for stage, counts in enumerate(([2] * 6, [3, 3, 2, 0, 2, 2], [0] * 6), 1):
        evaluation, x = fixture(folder, f"eval{stage}", [3] * 6, 10 + stage)
        diagnosis = engine.diagnose(x)
        proposed = engine.propose(diagnosis, training_release=release(f"cal{stage}") if sum(counts) else None)
        if sum(counts):
            path, _ = fixture(folder, f"cal{stage}", counts, stage + 1)
            update = engine.train(path, release(f"cal{stage}"), parent_id=parent, method="C_replay_anchor",
                                  replay_path=source, replay_release=release("source"))
            if update["status"] != "trained":
                raise RuntimeError(update)
            parent = update["model_id"]
        else:
            update = engine.train(method="no_update")
        cpu, wall = process_time(), perf_counter()
        prediction = engine.predict(x, model_id=parent)
        end_to_end = {"cpu_seconds": process_time() - cpu, "wall_seconds": perf_counter() - wall}
        assessed = engine.evaluate(parent, evaluation, release(f"eval{stage}", "evaluate"))
        decision = engine.activate(parent, assessed["evaluation_id"], expected_revision=0)
        assert decision["status"] == "kept" and decision["active_model_id"] == "stable"
        stages.append({"stage": stage, "calibration_counts": counts,
                       "diagnosis": {k: diagnosis[k] for k in ("rows", "mean_shift_l2", "predicted_counts", "labels_used")},
                       "proposed_methods": [r["method"] for r in proposed["candidates"]],
                       "training": update, "prediction_resources": prediction["resources"],
                       "prediction_end_to_end": end_to_end, "evaluation_resources": assessed["resources"],
                       "agent_summary": assessed["agent_summary"], "activation": decision})
    result = {"scope": "synthetic_mock_only", "bootstrap": initial, "stages": stages,
              "rollback": engine.rollback(), "final_active_model": engine.status()["active_model_id"],
              "independent_generalization_verified": False, "llm_calls": 0, "llm_tokens": 0}
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"stages": len(result["stages"]), "active": result["final_active_model"], "scope": result["scope"]}))

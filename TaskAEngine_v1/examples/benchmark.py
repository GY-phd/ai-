"""Measure Task A CPU costs; optionally fit candidates from a legal train release."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

from task_a_engine import Engine
from task_a_engine.data import read_inputs


def run(inputs, store, output, *, repetitions=50, train_release_id=None):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    store = Path(store)
    if store.exists():
        raise ValueError("benchmark requires a fresh store")
    # Features alone enter inference; answers are read only by explicit train().
    data = read_inputs(inputs)
    engine = Engine(store)
    initial = engine.status()
    rows = []
    for spec in initial["models"]:
        name = spec["model_id"]
        engine.predict(data["x"], model_id=name)  # one warmup
        cpu, wall = process_time(), perf_counter()
        compute_cpu = compute_wall = 0.
        for _ in range(repetitions):
            result = engine.predict(data["x"], model_id=name)
            compute_cpu += result["resources"]["cpu_seconds"]
            compute_wall += result["resources"]["wall_seconds"]
        rows.append({"model_id": name, "batch_rows": len(data["x"]), "repetitions": repetitions,
                     "compute_cpu_ms_per_batch": 1000 * compute_cpu / repetitions,
                     "compute_wall_ms_per_batch": 1000 * compute_wall / repetitions,
                     "api_cpu_ms_per_batch": 1000 * (process_time() - cpu) / repetitions,
                     "api_wall_ms_per_batch": 1000 * (perf_counter() - wall) / repetitions,
                     "llm_calls": 0, "llm_tokens": 0})
    training = []
    if train_release_id:
        release = {"purpose": "train", "labels": "released", "kind": "real", "release_id": train_release_id}
        for method in ("mlp_continue", "fusion_refit"):
            cpu, wall = process_time(), perf_counter()
            trained = engine.train(inputs, release, parent_id="stable", method=method)
            trained["api_cpu_seconds"] = process_time() - cpu
            trained["api_wall_seconds"] = perf_counter() - wall
            if trained["status"] != "trained":
                raise RuntimeError(trained)
            engine.predict(data["x"][:6], model_id=trained["model_id"])
            training.append(trained)
    final = engine.status()
    assert final["active_model_id"] == "stable" and final["revision"] == 0
    result = {"scope": "engineering_cost_only", "rows": len(data["x"]), "labels_used_for_inference": False,
              "competition_score": None, "independent_generalization_verified": False,
              "inference": rows, "training": training, "active_model_id": final["active_model_id"],
              "timing_note": "warm batch inference; compute excludes parsing, checkpoint load/hash, SQLite and serialization; API includes these except input-file read; training API includes input-file read"}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--train-release-id", help="explicitly authorize use of this file's legally released training labels")
    args = parser.parse_args()
    result = run(args.inputs, args.store, args.output, repetitions=args.repetitions, train_release_id=args.train_release_id)
    print(json.dumps({"output": args.output, "rows": result["rows"], "candidates_trained": len(result["training"]),
                      "active_model_id": result["active_model_id"]}, ensure_ascii=False))

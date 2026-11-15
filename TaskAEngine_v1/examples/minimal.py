"""Read-only A-task example: python -m examples.minimal --store STORE --inputs INPUT.jsonl."""

import argparse
import json

from task_a_engine import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    args = parser.parse_args()
    engine = Engine(args.store, mode=args.mode)
    prediction = engine.predict(args.inputs)
    diagnosis = engine.diagnose(args.inputs)
    proposal = engine.propose(diagnosis)
    # Keep full per-row predictions in the caller; send only a short summary to its Agent.
    summary = {
        "active_model_id": prediction["model_id"],
        "rows": len(prediction["labels"]),
        "first_labels": prediction["labels"][:5],
        "prediction_resources": prediction["resources"],
        "diagnosis": diagnosis,
        "proposal": proposal,
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

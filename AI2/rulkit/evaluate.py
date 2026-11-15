# -*- coding: utf-8 -*-
"""Local scoring for the B task: RMSE / MAE in minutes, overall and per bearing.

Two modes:
  1. score a checkpoint directly on a labelled JSONL file
        python -m rulkit.evaluate --checkpoint runs/round-1/checkpoint.npz \
            --data public/validation.jsonl
  2. score an already produced prediction file (e.g. platform output)
        python -m rulkit.evaluate --pred preds.jsonl --data public/validation.jsonl

Both print a JSON summary and a per-bearing table; RMSE is the primary metric.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import json_value, metrics, read_dataset
from .model import Model


def read_predictions(path):
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json_value(line)
        if isinstance(row, dict) and "outputs" in row:
            out = row["outputs"]
            values.append(float(out["value"]) if isinstance(out, dict) else float(out))
        elif isinstance(row, (int, float)):
            values.append(float(row))
        else:
            raise ValueError("prediction rows must look like {\"outputs\": {\"value\": number}}")
    return np.asarray(values, dtype=np.float64)


def report(predicted, data):
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(predicted) != len(data["y"]):
        raise ValueError(f"got {len(predicted)} predictions for {len(data['y'])} labelled rows")
    overall = metrics(predicted, data["y"])
    per_bearing = {}
    for group in sorted(set(data["groups"])):
        mask = np.asarray([g == group for g in data["groups"]])
        per_bearing[group] = {"rows": int(mask.sum()), **metrics(predicted[mask], data["y"][mask]),
                              "target_mean": float(data["y"][mask].mean()),
                              "pred_mean": float(predicted[mask].mean())}
    return {"rmse": overall["rmse"], "mae": overall["mae"], "rows": int(len(predicted)),
            "unit": "minutes", "per_bearing": per_bearing}


def print_table(result, label):
    print(f"\n{label}: rows={result['rows']}  RMSE={result['rmse']:.3f} min  MAE={result['mae']:.3f} min")
    print(f"{'bearing':<24}{'rows':>6}{'target_mean':>13}{'pred_mean':>11}{'RMSE':>10}{'MAE':>10}")
    for name, row in result["per_bearing"].items():
        print(f"{name:<24}{row['rows']:>6}{row['target_mean']:>13.1f}{row['pred_mean']:>11.1f}"
              f"{row['rmse']:>10.1f}{row['mae']:>10.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--pred", help="Existing prediction JSONL to score")
    parser.add_argument("--data", required=True, help="Labelled JSONL (public validation)")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable summary only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    data = read_dataset(args.data)
    if args.checkpoint:
        model = Model.load(args.checkpoint)
        predicted = model.predict(data["x"])
        label = f"checkpoint {Path(args.checkpoint).name} on {Path(args.data).name}"
    else:
        predicted = read_predictions(args.pred)
        label = f"{Path(args.pred).name} vs {Path(args.data).name}"
    result = report(predicted, data)
    if not args.quiet:
        print_table(result, label)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"SCORE rmse={result['rmse']:.3f} mae={result['mae']:.3f} rows={result['rows']}")


if __name__ == "__main__":
    main()

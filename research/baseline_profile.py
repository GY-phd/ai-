# -*- coding: utf-8 -*-
"""Score the official static baseline under the leave-one-bearing-out protocol.

The official checkpoint was trained on all nine public training bearings, so
it cannot be re-trained per fold.  What *can* be done fairly is to score it
on every training bearing and on the public validation bearings, which
gives the reference error profile the same shape as the candidate's.

Why this matters
----------------
The official baseline reaches RMSE 681 on the public validation bearings.
Those three bearings are ``Bearing{1,2,3}_4`` - the fourth bearing of each
working condition, i.e. the continuation of a condition the model has seen.
Scoring the baseline on all nine training bearings shows how much of that
681 is the "same condition" advantage, and where the baseline is weakest.
A candidate that is only better on the public validation bearings is
better on *continuations*, not on *unseen conditions*.

Run:
    python -m research.baseline_profile --out artifacts/baseline_profile.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.protocol import bearing_groups, load_split, official_baseline, score_on  # noqa: E402


def baseline_profile(data_root: Path) -> dict:
    """Score the official checkpoint on every public bearing."""
    sys.path.insert(0, str(Path(data_root) / "baselines"))
    try:
        from model import Model  # type: ignore
    finally:
        sys.path.pop(0)
    model = Model.load(Path(data_root) / "checkpoint.npz")

    rows = {}
    for split in ("train", "validation"):
        rows[split] = load_split(data_root, split)

    per_bearing = {}
    for split in ("train", "validation"):
        groups = bearing_groups(rows[split])
        for bearing, indices in sorted(groups.items()):
            scored = score_on(
                {"x": rows[split]["x"][indices], "y": rows[split]["y"][indices],
                 "bearing": rows[split]["bearing"][indices]},
                model.predict)
            per_bearing[bearing] = {
                "split": split,
                "rows": int(len(indices)),
                "rmse": round(scored["rmse"], 3),
                "mae": round(scored["mae"], 3),
            }

    train_rmse = np.asarray([v["rmse"] for k, v in per_bearing.items() if v["split"] == "train"])
    valid_rmse = np.asarray([v["rmse"] for k, v in per_bearing.items() if v["split"] == "validation"])

    # Weighted aggregate over the validation bearings, as the rules measure it.
    validation = official_baseline(data_root)

    return {
        "checkpoint": "official static baseline (contest.cpu-mlp.v1, hidden=24, 865 params)",
        "per_bearing": per_bearing,
        "summary": {
            "train_bearing_rmse_mean": round(float(train_rmse.mean()), 3),
            "train_bearing_rmse_worst": round(float(train_rmse.max()), 3),
            "validation_rmse": round(validation["rmse"], 3),
            "validation_mae": round(validation["mae"], 3),
            "validation_per_bearing": validation["per_bearing"],
            "ratio_worst_to_validation": round(float(train_rmse.max() / validation["rmse"]), 3),
        },
    }


def main(argv=None) -> int:
    """Print the baseline's per-bearing error profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    result = baseline_profile(Path(args.data))
    print("=== official static baseline: per-bearing error profile ===")
    for bearing, row in sorted(result["per_bearing"].items()):
        print("  %-18s %-11s rows=%5d  rmse=%8.2f  mae=%8.2f"
              % (bearing, row["split"], row["rows"], row["rmse"], row["mae"]))
    summary = result["summary"]
    print("\n  train-bearing RMSE mean   %.3f" % summary["train_bearing_rmse_mean"])
    print("  train-bearing RMSE worst  %.3f" % summary["train_bearing_rmse_worst"])
    print("  public validation RMSE     %.3f  MAE %.3f" % (summary["validation_rmse"], summary["validation_mae"]))
    print("  worst / validation ratio   %.3f" % summary["ratio_worst_to_validation"])

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

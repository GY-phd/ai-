# -*- coding: utf-8 -*-
"""Leave-one-bearing-out cross validation for the B task.

Why leave-one-bearing-out
-------------------------
The public data has nine training bearings and three validation bearings.
A single train/validation split gives one number per configuration, and
with three bearings that number is noisy enough to reward a lucky
configuration.  Worse, the three validation bearings have very different
lifetimes (122, 42 and 1515 snapshots), and the official baseline's error
on them differs by a factor of three (242 / 391 / 711 RMSE).  A change
that helps the long bearing and hurts the short one can look like an
improvement on the aggregate while being a regression in the regime the
hidden stream is most likely to punish.

Leave-one-bearing-out rotates over all nine training bearings: train on
eight, evaluate on the one that was held out.  Nine folds, every bearing
evaluated exactly once, and the public validation set stays untouched as
an independent final check.

Three numbers come out of it and all three matter:

``rmse``
    The headline.  But on its own it hides the short-bearing regime.

``rmse_short``
    RMSE restricted to bearings whose lifetime is at or below the median.
    The short-life bearings are where an industrial deployment fails
    first, because that is where a prediction error hurts most.

``worst_bearing_rmse``
    The worst single fold.  The rules require the closed evaluation to
    finish unattended, and a configuration that collapses on one bearing
    is a configuration that will collapse on the hidden stream.

A configuration is only accepted if it improves ``rmse`` **and** does not
regress ``rmse_short`` or ``worst_bearing_rmse`` beyond a stated margin.
That is the same philosophy as the runtime gate, applied offline.

Run:
    python -m research.lobo --out artifacts/lobo.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.protocol import bearing_groups, load_split, score_on, subset  # noqa: E402
from rulkit.model import Model  # noqa: E402


def median_life(bearings: dict) -> float:
    """Median lifetime (in snapshots) across the training bearings."""
    counts = [len(v) for v in bearings.values()]
    return float(np.median(counts))


def evaluate_config(train_rows, eval_rows, config: dict) -> dict:
    """Train one configuration on ``train_rows`` and score it on ``eval_rows``."""
    start = time.time()
    model = Model("regression", train_rows["paths"], [], train_rows["x"], train_rows["y"],
                  hidden=config["hidden"], seed=config.get("seed", 42),
                  learning_rate=config.get("learning_rate", 0.003),
                  batch_size=config.get("batch_size", 128),
                  feature_transform=config.get("feature_transform", "eng34"),
                  activation=config.get("activation", "tanh"),
                  target_scaling=config.get("target_scaling", False))
    model.fit(train_rows["x"], train_rows["y"], config.get("epochs", 400),
              learning_rate=config.get("learning_rate", 0.003),
              weight_decay=config.get("weight_decay", 0.0))
    scored = score_on(eval_rows, model.predict)
    scored["train_seconds"] = round(time.time() - start, 2)
    scored["parameter_count"] = model.meta["parameter_count"]
    return scored


def leave_one_out(data_root: Path, config: dict, verbose: bool = True) -> dict:
    """Run the nine-fold rotation and summarise it."""
    rows = load_split(data_root, "train")
    groups = bearing_groups(rows)
    threshold = median_life(groups)

    folds = []
    for held_out in sorted(groups):
        eval_indices = groups[held_out]
        train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                    for i in idx])
        fold = evaluate_config(subset(rows, train_indices), subset(rows, eval_indices), config)
        fold["held_out"] = held_out
        fold["held_out_rows"] = int(len(eval_indices))
        fold["is_short"] = bool(len(eval_indices) <= threshold)
        folds.append(fold)
        if verbose:
            print("    %-18s rows=%5d  rmse=%8.2f  mae=%8.2f  (%s)"
                  % (held_out, fold["held_out_rows"], fold["rmse"], fold["mae"],
                     "short" if fold["is_short"] else "long"))

    rmses = np.asarray([f["rmse"] for f in folds])
    short = np.asarray([f["rmse"] for f in folds if f["is_short"]])
    long_ = np.asarray([f["rmse"] for f in folds if not f["is_short"]])
    return {
        "config": config,
        "folds": folds,
        "summary": {
            "rmse_mean": round(float(rmses.mean()), 3),
            "rmse_std": round(float(rmses.std()), 3),
            "rmse_median": round(float(np.median(rmses)), 3),
            "rmse_worst": round(float(rmses.max()), 3),
            "rmse_best": round(float(rmses.min()), 3),
            "rmse_short": round(float(short.mean()), 3) if len(short) else None,
            "rmse_long": round(float(long_.mean()), 3) if len(long_) else None,
            "n_short": int(len(short)),
            "n_long": int(len(long_)),
            "median_life_snapshots": threshold,
            "total_train_seconds": round(sum(f["train_seconds"] for f in folds), 2),
        },
    }


def main(argv=None) -> int:
    """Run leave-one-bearing-out for one configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--feature-transform", default="eng34")
    parser.add_argument("--activation", default="tanh", choices=("tanh", "relu"))
    parser.add_argument("--target-scaling", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    config = {
        "hidden": args.hidden,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": 128,
        "feature_transform": args.feature_transform,
        "activation": args.activation,
        "target_scaling": args.target_scaling,
        "seed": 42,
    }
    print("=== leave-one-bearing-out ===")
    print("  config: %s" % json.dumps(config, sort_keys=True))
    result = leave_one_out(Path(args.data), config)
    summary = result["summary"]
    print("\n  rmse mean   %.3f  (std %.3f, median %.3f)" % (
        summary["rmse_mean"], summary["rmse_std"], summary["rmse_median"]))
    print("  rmse worst  %.3f" % summary["rmse_worst"])
    print("  rmse short  %.3f  (n=%d bearings)" % (summary["rmse_short"], summary["n_short"]))
    print("  rmse long   %.3f  (n=%d bearings)" % (summary["rmse_long"], summary["n_long"]))
    print("  train time  %.1f s" % summary["total_train_seconds"])

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

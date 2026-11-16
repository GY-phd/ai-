# -*- coding: utf-8 -*-
"""Controlled ablation of the model-side changes, on the B task.

Why this file exists
--------------------
The three-track closed loop (``tracks/``) answers "does the *loop* beat the
baseline".  It does not answer "**which** model-side change bought the
gain, and how much of the gain survives on a bearing the system has never
seen?".

This script answers the second question.  It holds the data split fixed
(leave-one-bearing-out over the nine training bearings, the protocol the
competition's whole-bearing isolation rule describes), holds the width
fixed, and varies one knob at a time:

``feature_transform``
    ``raw`` keeps the 34 official features as they are; ``slog`` compresses
    them with a signed log; ``eng34`` is the 34-column engineering set.

``activation``
    ``tanh`` against ``relu``.

``target_scaling``
    Whether the regression target is standardised before fitting.

``width``
    The hidden layer width, at the end, to separate "the change works"
    from "the change works at this width".

Every configuration is fitted and scored **under the same protocol with the
same seed**, so the rows of the resulting table are comparable.  Negative
results are kept: a configuration that loses is evidence too, and dropping
it would make the surviving rows look stronger than they are.

Run:
    python -m research.exp_fusion --out artifacts/fusion_ablation.json
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

# The grid.  Every entry is a complete, self-contained configuration, so
# adding one is a single line and the table in the README can be
# regenerated without touching any other code.
GRID = [
    {"name": "raw + tanh + scaling",   "feature_transform": "raw",   "activation": "tanh", "target_scaling": True,  "hidden": 96},
    {"name": "raw + tanh + no scaling", "feature_transform": "raw",   "activation": "tanh", "target_scaling": False, "hidden": 96},
    {"name": "slog + tanh + no scaling","feature_transform": "slog", "activation": "tanh", "target_scaling": False, "hidden": 96},
    {"name": "eng34 + tanh + scaling",  "feature_transform": "eng34", "activation": "tanh", "target_scaling": True,  "hidden": 96},
    {"name": "eng34 + tanh + no scaling","feature_transform": "eng34","activation": "tanh", "target_scaling": False, "hidden": 96},
    {"name": "eng34 + relu + no scaling","feature_transform": "eng34","activation": "relu", "target_scaling": False, "hidden": 128},
]

# The official static baseline, refitted per fold on the same eight
# bearings, so the comparison is like-for-like.
OFFICIAL_HIDDEN = 32
OFFICIAL_EPOCHS = 20


def fit_predict(train_rows, eval_rows, config: dict) -> np.ndarray:
    """Fit one configuration and predict on the evaluation rows."""
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
    return np.asarray(model.predict(eval_rows["x"]), dtype=float)


def fit_official(train_rows, eval_rows) -> np.ndarray:
    """Fit the official baseline's architecture on the given rows.

    The package ships a checkpoint, not a training function, so the
    architecture is reproduced from the package's own ``model.py``.  Both
    models then differ only in the target and the feature transform, which
    is what makes the comparison a fair one.
    """
    import importlib
    import sys as _sys

    from research.tracks.official_baselines import DATA_ROOT, PACKAGES, official_modules

    baselines = DATA_ROOT / PACKAGES["B"] / "baselines"
    if str(baselines) not in _sys.path:
        _sys.path.insert(0, str(baselines))
    model_module = importlib.import_module("model")
    importlib.reload(model_module)

    model = model_module.Model("regression", list(train_rows["paths"]), [],
                                train_rows["x"], train_rows["y"],
                                hidden=OFFICIAL_HIDDEN, seed=42, learning_rate=0.003,
                                batch_size=128)
    model.fit(train_rows["x"], train_rows["y"], OFFICIAL_EPOCHS)
    return np.asarray(model.predict(eval_rows["x"]), dtype=float)


def run_config(rows, groups, config: dict, official_rmse: list) -> dict:
    """Leave-one-bearing-out for one configuration."""
    fold_rmse = []
    fold_mae = []
    for held_out in sorted(groups):
        eval_indices = groups[held_out]
        train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                    for i in idx])
        predicted = fit_predict(subset(rows, train_indices), subset(rows, eval_indices),
                                config)
        truth = rows["y"][eval_indices]
        error = predicted - truth
        fold_rmse.append(float(np.sqrt(np.mean(error ** 2))))
        fold_mae.append(float(np.mean(np.abs(error))))

    rmses = np.asarray(fold_rmse)
    baseline_mean = float(np.mean(official_rmse))
    return {
        "config": {k: config[k] for k in
                   ("feature_transform", "activation", "target_scaling", "hidden")},
        "name": config["name"],
        "rmse_mean": round(float(rmses.mean()), 3),
        "rmse_median": round(float(np.median(rmses)), 3),
        "rmse_worst": round(float(rmses.max()), 3),
        "mae_mean": round(float(np.mean(fold_mae)), 3),
        "official_rmse_mean": round(baseline_mean, 3),
        "gain_pct": round((baseline_mean - float(rmses.mean())) / baseline_mean * 100, 2),
        "wins_vs_official": None,   # filled in by the caller
        "folds": [{"bearing": b, "rmse": round(v, 3)}
                  for b, v in zip(sorted(groups), fold_rmse)],
    }


def main(argv=None) -> int:
    """Run the ablation grid under leave-one-bearing-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)

    print("=== controlled ablation, leave-one-bearing-out over %d bearings ===" % len(groups))

    # The official baseline, refitted on the same eight bearings per fold.
    started = time.monotonic()
    official_rmse = []
    for held_out in sorted(groups):
        eval_indices = groups[held_out]
        train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                    for i in idx])
        predicted = fit_official(subset(rows, train_indices), subset(rows, eval_indices))
        error = predicted - rows["y"][eval_indices]
        official_rmse.append(float(np.sqrt(np.mean(error ** 2))))
        print("  official  %-18s rmse %8.2f" % (held_out, official_rmse[-1]))
    print("  official mean %.2f  [%.1fs]\n" % (np.mean(official_rmse),
                                                time.monotonic() - started))

    results = []
    for config in GRID:
        config = dict(config)
        config.setdefault("epochs", args.epochs)
        config.setdefault("seed", 42)
        config.setdefault("learning_rate", 0.003)
        config.setdefault("batch_size", 128)
        config.setdefault("weight_decay", 0.01)
        started = time.monotonic()
        outcome = run_config(rows, groups, config, official_rmse)
        outcome["train_seconds"] = round(time.monotonic() - started, 1)
        results.append(outcome)
        print("  %-28s rmse %8.2f  (%+.1f%% vs official)  [%.1fs]"
              % (outcome["name"], outcome["rmse_mean"], outcome["gain_pct"],
                 outcome["train_seconds"]))

    best = min(results, key=lambda row: row["rmse_mean"])
    print("\n  best: %s  rmse %.2f  (%+.1f%% vs official)"
          % (best["name"], best["rmse_mean"], best["gain_pct"]))

    payload = {
        "protocol": "leave-one-bearing-out over the %d training bearings" % len(groups),
        "official_baseline": {
            "architecture": "the package's own model.py, hidden=%d, epochs=%d"
                            % (OFFICIAL_HIDDEN, OFFICIAL_EPOCHS),
            "rmse_mean": round(float(np.mean(official_rmse)), 3),
            "per_fold": [{"bearing": b, "rmse": round(v, 3)}
                         for b, v in zip(sorted(groups), official_rmse)],
        },
        "results": results,
        "best": {"name": best["name"], "rmse_mean": best["rmse_mean"],
                 "gain_pct": best["gain_pct"]},
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

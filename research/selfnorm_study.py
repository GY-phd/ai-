# -*- coding: utf-8 -*-
"""Self-normalising features: make the signal relative to the bearing itself.

The diagnosis
-------------
``feature_study`` found that **not one** of the 34 official features has a
consistent relationship with degradation across the nine bearings.  The
strongest feature, ``horizontal.mean_abs``, correlates at -0.38 on one
bearing and +0.87 on another.  The sign flips.

The reason is structural, not statistical.  The official features are
*absolute* vibration statistics.  An amplitude of 3 g might mean "nearly
failed" for a bearing that normally hums at 0.5 g and "brand new" for one
that normally hums at 4 g.  The degradation signal is the *change*, and
the change is only visible relative to the bearing's own baseline.

The fix
-------
Re-express every feature relative to a **reference window** drawn from the
same bearing:

``baseline``
    The first ``k`` snapshots of the bearing.  The feature becomes
    ``x / baseline`` (or ``x - baseline`` for signed features).

``running``
    An expanding reference: at snapshot ``i`` the reference is the mean of
    snapshots ``0..i-1``.  This is what a deployed system can actually do -
    it keeps a running estimate of its own healthy state.

``slope``
    The rate of change over a trailing window, which removes the constant
    offset entirely and keeps only the dynamics.

All three are label-free and computable at inference time from the stream's
own history.  This is the standard industrial answer to "the same sensor
reads differently on different machines": normalise by the machine.

Why the first version still failed
----------------------------------
Normalising by the *first* snapshot only is fragile: the first snapshots
of a bearing can already be noisy, and for the long bearings the first
snapshot is hours after installation.  Using a short **baseline window**
(the first ``k`` snapshots, averaged) is far more stable, and is what this
module implements by default.

Run:
    python -m research.selfnorm_study --out artifacts/selfnorm.json
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

from research.protocol import bearing_groups, load_split, score_on, subset  # noqa: E402
from research.target_study import degradation_proxy, life_fraction  # noqa: E402
from rulkit.model import Model  # noqa: E402


# --------------------------------------------------------------- transforms
def self_normalise(rows, mode: str = "baseline", window: int = 5) -> np.ndarray:
    """Return the self-normalised feature matrix.

    The transform is applied **per bearing**, using only that bearing's own
    snapshots, so it never mixes information across bearings and never
    touches a label.
    """
    x = np.asarray(rows["x"], dtype=np.float64).copy()
    groups = bearing_groups(rows)
    out = np.empty_like(x)

    for _, indices in groups.items():
        order = indices[np.argsort(rows["snapshot"][indices])]
        block = x[order]
        n = len(block)
        if n == 0:
            continue

        if mode == "baseline":
            k = max(1, min(window, n))
            reference = block[:k].mean(axis=0)
            # Ratio for positive-definite quantities; the scale of a
            # vibration statistic is what matters, not its sign.
            scale = np.where(np.abs(reference) < 1e-9, 1.0, np.abs(reference))
            out[order] = block / scale

        elif mode == "running":
            # Expanding mean of everything strictly before the current row.
            cumulative = np.cumsum(block, axis=0)
            counts = np.arange(1, n + 1, dtype=float)[:, None]
            prior_mean = np.vstack([np.zeros((1, block.shape[1])), cumulative[:-1] / np.maximum(counts[:-1], 1)])
            scale = np.where(np.abs(prior_mean) < 1e-9, 1.0, np.abs(prior_mean))
            out[order] = block / scale

        elif mode == "slope":
            # Relative change against the first window, then a centred
            # difference to isolate the dynamics.
            k = max(1, min(window, n))
            reference = block[:k].mean(axis=0)
            scale = np.where(np.abs(reference) < 1e-9, 1.0, np.abs(reference))
            relative = block / scale
            centred = relative.copy()
            if n >= 2:
                centred[1:] = relative[1:] - relative[:-1]
            out[order] = centred

        else:
            raise ValueError("unknown mode %r" % mode)

    return out


def feature_consistency(rows, mode: str, window: int = 5):
    """How consistent is each transformed feature with degradation now?"""
    groups = bearing_groups(rows)
    degradation = degradation_proxy(rows)
    x = self_normalise(rows, mode, window)
    rhos = []
    for index in range(x.shape[1]):
        per_bearing = []
        for _, indices in sorted(groups.items()):
            a = x[indices, index]
            b = degradation[indices]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                per_bearing.append(0.0)
                continue
            per_bearing.append(float(np.corrcoef(a, b)[0, 1]))
        rhos.append(per_bearing)
    return np.asarray(rhos)


def train_and_score(train_rows, eval_rows, config: dict) -> dict:
    """Train on self-normalised features and score in minutes."""
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
    return score_on(eval_rows, model.predict)


def main(argv=None) -> int:
    """Compare self-normalisation modes under leave-one-bearing-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--modes", default="raw,baseline,running,slope")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)
    config = {"hidden": args.hidden, "epochs": args.epochs, "learning_rate": 0.003,
              "weight_decay": 0.01, "batch_size": 128, "feature_transform": "eng34",
              "activation": "tanh", "target_scaling": False, "seed": 42}

    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode == "raw":
            x = np.asarray(rows["x"], dtype=np.float64)
        else:
            x = self_normalise(rows, mode, args.window)

        working = dict(rows)
        working["x"] = x

        rmses = []
        for held_out in sorted(groups):
            eval_indices = groups[held_out]
            train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                        for i in idx])
            scored = train_and_score(subset(working, train_indices), subset(working, eval_indices), config)
            rmses.append(scored["rmse"])

        rmses = np.asarray(rmses)
        results[mode] = {
            "folds": [{"bearing": b, "rmse": round(r, 3)} for b, r in zip(sorted(groups), rmses)],
            "rmse_mean": round(float(rmses.mean()), 3),
            "rmse_median": round(float(np.median(rmses)), 3),
            "rmse_worst": round(float(rmses.max()), 3),
        }
        print("  %-10s mean %8.2f  median %8.2f  worst %8.2f" % (
            mode, results[mode]["rmse_mean"], results[mode]["rmse_median"], results[mode]["rmse_worst"]))

    print("\n=== feature consistency after normalisation ===")
    print("  (number of the 34 features whose degradation correlation keeps one sign on all 9 bearings)")
    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode == "raw":
            x = np.asarray(rows["x"], dtype=np.float64)
        else:
            x = self_normalise(rows, mode, args.window)
        working = dict(rows)
        working["x"] = x
        rhos = feature_consistency(working, mode, args.window)
        consistent = 0
        strong = 0
        for row in rhos:
            signs = np.sign(row[np.abs(row) > 0.15])
            if len(signs) and len(set(signs.tolist())) == 1:
                consistent += 1
                if np.abs(row).mean() > 0.3:
                    strong += 1
        print("  %-10s consistent-sign %2d/34   strong-and-consistent %2d/34" % (mode, consistent, strong))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

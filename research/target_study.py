# -*- coding: utf-8 -*-
"""What actually transfers between bearings?  A target-space study.

The problem
-----------
Leave-one-bearing-out showed that a model trained on eight bearings and
evaluated on the ninth is roughly as good as the official baseline (639 vs
681 RMSE), and that feeding the stream into training makes things *worse*
(275 -> 642 on the stream tail).  Both facts point at the same thing: the
regression target in absolute minutes does not transfer between bearings.

Why absolute minutes cannot transfer
--------------------------------------
The RUL label is ``(N - i)`` minutes.  ``N`` is the bearing's own lifetime,
which is a property of the failure process, not of the vibration signal.
Two bearings at the *same* point in their degradation can have RUL 2500 and
RUL 40 respectively, because one took three times longer to fail.  A model
that predicts absolute minutes is therefore asked to solve two problems at
once - "how degraded is this?" and "how long will this particular bearing
last?" - and the second part is not in the input.

What might transfer
-------------------
Three candidate reparameterisations, each of which is computable at
inference time and uses no label information:

``absolute`` (the current target)
    Predict ``N - i`` in minutes.  The official baseline's choice.

``relative``
    Predict ``(N - i) / N`` - the fraction of life remaining.  This removes
    the lifetime scale entirely, so the question becomes purely "where is
    this bearing in its own life?".  It is computable from the input only
    if the model can also estimate ``N``, so this is not directly
    implementable - but as a *diagnostic* it tells us how much of the
    error comes from the lifetime scale.

``degradation``
    Predict the *observed* degradation level rather than a time.  We use
    the last snapshot's feature vector as the reference for "fully
    degraded" and the first snapshot's as "new", then interpolate.  This is
    a label-free proxy that is available at inference time.

The study measures each target's cross-bearing transfer under the same
leave-one-bearing-out protocol.  The winner is not necessarily usable as-is
- the point is to *locate* where the error comes from, and to tell the
final model design which quantity it should regress.

Run:
    python -m research.target_study --out artifacts/target_study.json
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
from rulkit.model import Model  # noqa: E402


def life_fraction(rows) -> np.ndarray:
    """Fraction of life remaining, computed per bearing: ``(N - i) / N``."""
    groups = bearing_groups(rows)
    out = np.zeros(len(rows["y"]))
    for _, indices in groups.items():
        n = len(indices)
        out[indices] = (n - rows["snapshot"][indices]) / n
    return out


def degradation_proxy(rows) -> np.ndarray:
    """A label-free degradation index in [0, 1].

    For each bearing, the first snapshot is "new" and the last is "fully
    degraded".  For every feature we interpolate between the two and take
    the mean absolute deviation from the "new" reference, normalised by the
    bearing's own spread.  It is available at inference time because it
    only uses the stream's own history - no labels.

    This is what an industrial system can actually deploy: a model that
    watches the drift of its own features.
    """
    groups = bearing_groups(rows)
    out = np.zeros(len(rows["y"]))
    for _, indices in groups.items():
        order = indices[np.argsort(rows["snapshot"][indices])]
        x = rows["x"][order]
        first, last = x[0], x[-1]
        span = np.abs(last - first)
        span[span < 1e-9] = 1.0
        fraction = (np.abs(x - first) / span).mean(axis=1)
        out[order] = fraction
    return out


def train_and_score(train_rows, eval_rows, target_kind: str, config: dict) -> dict:
    """Train with one target space, evaluate in minutes."""
    if target_kind == "absolute":
        y_train = train_rows["y"]
    elif target_kind == "relative":
        y_train = life_fraction(train_rows)
    elif target_kind == "degradation":
        y_train = degradation_proxy(train_rows)
    else:
        raise ValueError(target_kind)

    model = Model("regression", train_rows["paths"], [], train_rows["x"], y_train,
                  hidden=config["hidden"], seed=config.get("seed", 42),
                  learning_rate=config.get("learning_rate", 0.003),
                  batch_size=config.get("batch_size", 128),
                  feature_transform=config.get("feature_transform", "eng34"),
                  activation=config.get("activation", "tanh"),
                  target_scaling=config.get("target_scaling", True))
    model.fit(train_rows["x"], y_train, config.get("epochs", 400),
              learning_rate=config.get("learning_rate", 0.003),
              weight_decay=config.get("weight_decay", 0.0))

    predicted = model.predict(eval_rows["x"])

    # Convert the prediction back to minutes so every target space is
    # compared on the competition's own scale.
    if target_kind == "absolute":
        minutes = predicted
    elif target_kind == "relative":
        minutes = predicted * eval_rows["life_max"]
    else:
        minutes = predicted * eval_rows["life_max"]

    error = minutes - eval_rows["y"]
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error)))}


def main(argv=None) -> int:
    """Compare target spaces under leave-one-bearing-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)

    # ``life_max`` is the lifetime of each bearing; the relative and
    # degradation predictions are multiplied by it to return to minutes.
    life_max = np.zeros(len(rows["y"]))
    for name, indices in groups.items():
        life_max[indices] = len(indices)
    rows["life_max"] = life_max

    config = {"hidden": args.hidden, "epochs": args.epochs, "learning_rate": 0.003,
              "weight_decay": 0.01, "batch_size": 128, "feature_transform": "eng34",
              "activation": "tanh", "target_scaling": True, "seed": 42}

    results = {}
    for kind in ("absolute", "relative", "degradation"):
        rmses = []
        print("=== target space: %s ===" % kind)
        for held_out in sorted(groups):
            eval_indices = groups[held_out]
            train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                        for i in idx])
            scored = train_and_score(subset(rows, train_indices), subset(rows, eval_indices),
                                     kind, config)
            rmses.append(scored["rmse"])
            print("    %-18s rmse=%8.2f" % (held_out, scored["rmse"]))
        results[kind] = {
            "folds": [{"bearing": b, "rmse": round(r, 3)}
                      for b, r in zip(sorted(groups), rmses)],
            "rmse_mean": round(float(np.mean(rmses)), 3),
            "rmse_median": round(float(np.median(rmses)), 3),
            "rmse_worst": round(float(np.max(rmses)), 3),
        }
        print("    mean %.2f  median %.2f  worst %.2f\n" % (
            results[kind]["rmse_mean"], results[kind]["rmse_median"], results[kind]["rmse_worst"]))

    print("=== summary (leave-one-bearing-out, RMSE in minutes) ===")
    for kind, row in results.items():
        print("  %-12s mean %8.2f  median %8.2f  worst %8.2f"
              % (kind, row["rmse_mean"], row["rmse_median"], row["rmse_worst"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

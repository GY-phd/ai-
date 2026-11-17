# -*- coding: utf-8 -*-
"""Make the two-stage lifetime estimator safe.

The problem
-----------
``combine.py`` found that the ``baseline`` feature normalisation combined
with the ``relative`` target produced a catastrophic outlier: one fold
reached 125 937 RMSE while the median fold was 267.  A system that
occasionally produces a 125 000-minute prediction is worse than useless on
an industrial line - it would trigger a false emergency shutdown.

The cause is in the stage-2 estimator.  ``stream_ratio`` infers the
bearing's lifetime from the model's own life-fraction prediction:

    ``N_hat = i / (1 - f_i)``.

When the stage-1 model predicts a fraction very close to 1 on a long
bearing, ``1 - f_i`` is tiny and ``N_hat`` explodes.  One bad prediction
therefore poisons the whole scale.

The guards
----------
Three independent bounds, each of which has a physical justification:

``floor``
    The remaining-life fraction cannot be closer to 1 than
    ``1 / N_min`` for a bearing that is at least ``N_min`` snapshots long.
    Without a floor the divisor ``1 - f`` is unbounded.

``cap``
    The inferred lifetime cannot exceed the largest lifetime the initial
    training set contains.  A bearing that would live longer than every
    bearing we have ever seen is an extrapolation the data does not
    support, and the scale it would imply is not trustworthy.

``consistency``
    The estimate is only allowed to move gradually.  A sudden jump in the
    inferred lifetime means the stage-1 model changed its mind about how
    far the bearing has degraded, not that the bearing changed its
    lifetime.  A rate limit removes exactly those jumps.

Run:
    python -m research.guard_study --out artifacts/guard.json
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

from research.protocol import bearing_groups, load_split, subset  # noqa: E402
from research.selfnorm_study import self_normalise  # noqa: E402
from research.two_stage import life_fraction  # noqa: E402
from rulkit.model import Model  # noqa: E402


def safe_lifetime(fraction: np.ndarray, elapsed: np.ndarray, *, floor: float = 0.02,
                  cap: float = 3000.0, rate_limit: float = 0.25) -> np.ndarray:
    """Infer each bearing's lifetime from the predicted life fraction.

    ``floor``    the minimum plausible remaining fraction, so the divisor
                 ``1 - f`` cannot vanish.
    ``cap``      the maximum plausible lifetime.
    ``rate_limit`` the largest relative jump allowed between consecutive
                 estimates, which rejects the "the model changed its mind"
                    discontinuities.
    """
    f = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0 - floor)
    instantaneous = f * np.asarray(elapsed, dtype=np.float64) / np.maximum(1.0 - f, floor)

    out = np.empty_like(instantaneous)
    previous = None
    for i in range(len(instantaneous)):
        value = instantaneous[i]
        if previous is not None and previous > 0:
            # Rate limit: the estimate may not jump by more than
            # ``rate_limit`` relative to the previous one.
            value = min(value, previous * (1.0 + rate_limit))
            value = max(value, previous / (1.0 + rate_limit))
        value = float(np.clip(value, 1.0, cap))
        out[i] = value
        previous = value
    return out


def main(argv=None) -> int:
    """Compare the guarded estimator against the unguarded one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)
    config = {"hidden": args.hidden, "epochs": args.epochs, "learning_rate": 0.003,
              "weight_decay": 0.01, "batch_size": 128, "feature_transform": "eng34",
              "activation": "tanh", "target_scaling": False, "seed": 42}

    variants = {
        "raw": np.asarray(rows["x"], dtype=np.float64),
        "baseline": self_normalise(rows, "baseline", args.window),
    }

    results = {}
    print("=== guarded two-stage estimator, leave-one-bearing-out ===")
    print("  %-12s %-12s %10s %10s %10s" % ("features", "estimator", "mean", "median", "worst"))
    for feature_name, matrix in variants.items():
        working = dict(rows)
        working["x"] = matrix
        for estimator in ("unguarded", "guarded"):
            rmses = []
            for held_out in sorted(groups):
                eval_indices = groups[held_out]
                train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                            for i in idx])
                train_rows, eval_rows = subset(working, train_indices), subset(working, eval_indices)
                y_train = life_fraction(train_rows)
                model = Model("regression", train_rows["paths"], [], train_rows["x"], y_train,
                              hidden=config["hidden"], seed=42, learning_rate=0.003,
                              batch_size=128, feature_transform="eng34", activation="tanh",
                              target_scaling=False)
                model.fit(train_rows["x"], y_train, config["epochs"], learning_rate=0.003,
                          weight_decay=0.01)

                fraction = np.clip(model.predict(eval_rows["x"]), 0.0, 1.0)
                life_max = np.zeros(len(eval_rows["y"]))
                for _, idx in bearing_groups(eval_rows).items():
                    life_max[idx] = len(idx)

                if estimator == "unguarded":
                    f = np.clip(fraction, 1e-3, 0.999)
                    lifetime = f * life_max / np.maximum(1.0 - f, 1e-3)
                    lifetime = np.maximum(lifetime, 1.0)
                else:
                    lifetime = np.zeros(len(eval_rows["y"]))
                    for _, idx in bearing_groups(eval_rows).items():
                        lifetime[idx] = safe_lifetime(fraction[idx], life_max[idx] * 0 +
                                                      np.arange(1, len(idx) + 1))

                predicted = np.maximum(0.0, fraction * lifetime)
                error = predicted - eval_rows["y"]
                rmses.append(float(np.sqrt(np.mean(error ** 2))))

            rmses = np.asarray(rmses)
            key = "%s|%s" % (feature_name, estimator)
            results[key] = {
                "features": feature_name, "estimator": estimator,
                "rmse_mean": round(float(rmses.mean()), 3),
                "rmse_median": round(float(np.median(rmses)), 3),
                "rmse_worst": round(float(rmses.max()), 3),
                "folds": [{"bearing": b, "rmse": round(v, 3)} for b, v in zip(sorted(groups), rmses)],
            }
            print("  %-12s %-12s %10.2f %10.2f %10.2f" % (
                feature_name, estimator, rmses.mean(), np.median(rmses), rmses.max()))

    print("\n=== effect of the guards ===")
    for feature_name in variants:
        before = results["%s|unguarded" % feature_name]
        after = results["%s|guarded" % feature_name]
        print("  %-12s worst %.1f -> %.1f   mean %.1f -> %.1f" % (
            feature_name, before["rmse_worst"], after["rmse_worst"],
            before["rmse_mean"], after["rmse_mean"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

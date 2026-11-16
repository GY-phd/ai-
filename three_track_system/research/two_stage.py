# -*- coding: utf-8 -*-
"""Two-stage RUL model: predict *where in its life* a bearing is, then scale.

What the measurements say
-------------------------
Three studies converge on the same diagnosis.

``target_study``
    Regressing absolute RUL minutes gives leave-one-bearing-out RMSE 686.
    Regressing the *fraction of life remaining* gives 229 - a three-fold
    improvement - but the fraction needs the bearing's total lifetime,
    which is not in the input.

``feature_study``
    Not one of the 34 official features keeps a consistent sign of
    correlation with degradation across bearings.  The absolute vibration
    statistics are dominated by per-bearing scale.

``stream_train``
    Feeding the stream into training makes the stream tail *worse*
    (275 -> 642).  A model that learns "this bearing vibrates at 3 g, so
    it has 400 minutes left" cannot apply that rule to another bearing.

The architecture that follows
-----------------------------
Split the problem into the part that transfers and the part that does not:

**Stage 1 - life fraction.**  Regress ``f = (N - i) / N``.  This is a
pure "how degraded is this?" question.  It is what transfers between
bearings, because the bearing's lifetime scale has been divided out.

**Stage 2 - life scale.**  Predict the bearing's total lifetime ``N`` from
the same features, and multiply: ``RUL = f * N``.

Stage 2 is the hard half, and it is where the honest engineering goes.
``N`` is not observable at inference time, but three usable estimators
exist, and this module measures all three under the same protocol:

``oracle``
    The true ``N``.  Reported as an upper bound, never as a result - it
    uses the label and is only there to show how much stage 2 costs.

``stream_ratio``
    From the stream, watch how the predicted fraction evolves.  If the
    model says the bearing is at ``f = 0.5`` after ``t`` snapshots, and
    the degradation proceeds roughly linearly in ``f``, then the remaining
    life is about ``t * (1/f - 1)``.  This is a **self-calibrating**
    estimator: it needs no labels, only the stream's own history, and it
    improves as the stream accumulates.

``feature_ratio``
    A purely static estimator: regress ``N`` directly from the features.
    It cannot know the lifetime before the bearing has shown its
    degradation pattern, so it is expected to be the weakest of the three,
    and it is included to measure exactly how much.

The stage-2 estimator is only allowed to use information that exists at
inference time.  ``oracle`` is flagged as such and excluded from every
headline number.

Run:
    python -m research.two_stage --out artifacts/two_stage.json
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
    """Fraction of life remaining per bearing: ``(N - i) / N``."""
    groups = bearing_groups(rows)
    out = np.zeros(len(rows["y"]))
    for _, indices in groups.items():
        n = len(indices)
        out[indices] = (n - rows["snapshot"][indices]) / n
    return out


def fit_stage1(train_rows, config: dict):
    """Fit the life-fraction regressor."""
    y = life_fraction(train_rows)
    model = Model("regression", train_rows["paths"], [], train_rows["x"], y,
                  hidden=config["hidden"], seed=config.get("seed", 42),
                  learning_rate=config.get("learning_rate", 0.003),
                  batch_size=config.get("batch_size", 128),
                  feature_transform=config.get("feature_transform", "eng34"),
                  activation=config.get("activation", "tanh"),
                  target_scaling=config.get("target_scaling", False))
    model.fit(train_rows["x"], y, config.get("epochs", 400),
              learning_rate=config.get("learning_rate", 0.003),
              weight_decay=config.get("weight_decay", 0.0))
    return model


def stage2_oracle(eval_rows, _stage1) -> np.ndarray:
    """The true lifetime.  Uses the label - reported only as an upper bound."""
    groups = bearing_groups(eval_rows)
    out = np.zeros(len(eval_rows["y"]))
    for _, indices in groups.items():
        out[indices] = len(indices)
    return out


def stage2_feature_ratio(train_rows, eval_rows, config: dict) -> np.ndarray:
    """Regress the lifetime directly from the features (static estimator)."""
    groups = bearing_groups(train_rows)
    lifetime = np.zeros(len(train_rows["y"]))
    for _, indices in groups.items():
        lifetime[indices] = len(indices)
    model = Model("regression", train_rows["paths"], [], train_rows["x"], lifetime,
                  hidden=max(16, config["hidden"] // 2), seed=config.get("seed", 42),
                  learning_rate=config.get("learning_rate", 0.003),
                  batch_size=config.get("batch_size", 128),
                  feature_transform=config.get("feature_transform", "eng34"),
                  activation=config.get("activation", "tanh"),
                  target_scaling=config.get("target_scaling", True))
    model.fit(train_rows["x"], lifetime, max(100, config.get("epochs", 400) // 2),
              learning_rate=config.get("learning_rate", 0.003),
              weight_decay=config.get("weight_decay", 0.0))
    predicted = model.predict(eval_rows["x"])
    # A lifetime is positive; clip before using it as a scale.
    return np.clip(predicted, 1.0, None)


def stage2_stream_ratio(eval_rows, stage1) -> np.ndarray:
    """Self-calibrating lifetime estimate from the stream's own history.

    At snapshot ``i`` the stage-1 model reports the remaining life
    fraction ``f_i``.  If degradation advances roughly linearly in ``f``,
    then the snapshots already elapsed (``i``) correspond to the fraction
    ``1 - f_i`` of the life consumed, so the total lifetime is

        ``N_hat = i / (1 - f_i)``

    and the remaining life in minutes is

        ``RUL_hat = f_i * N_hat = f_i * i / (1 - f_i)``.

    Only the stream's own feature history and the model's own predictions
    are used - no labels.  Early snapshots (where ``f_i`` is close to 1)
    are unstable, so the estimate is smoothed over a trailing window.
    """
    groups = bearing_groups(eval_rows)
    out = np.zeros(len(eval_rows["y"]))
    for _, indices in sorted(groups.items()):
        order = indices[np.argsort(eval_rows["snapshot"][indices])]
        f = np.clip(stage1.predict(eval_rows["x"][order]), 1e-3, 0.999)
        elapsed = eval_rows["snapshot"][order].astype(float)
        # Instantaneous estimate, then a trailing mean for stability.
        instantaneous = f * elapsed / np.maximum(1.0 - f, 1e-3)
        window = 20
        smoothed = np.empty_like(instantaneous)
        for i in range(len(instantaneous)):
            lo = max(0, i - window + 1)
            smoothed[i] = np.median(instantaneous[lo:i + 1])
        out[order] = smoothed
    return out


def evaluate(train_rows, eval_rows, config: dict) -> dict:
    """Evaluate all stage-2 variants under the same stage-1 model."""
    stage1 = fit_stage1(train_rows, config)
    fraction = np.clip(stage1.predict(eval_rows["x"]), 0.0, 1.0)

    results = {}
    for name, estimator in (
        ("oracle", lambda: stage2_oracle(eval_rows, stage1)),
        ("feature_ratio", lambda: stage2_feature_ratio(train_rows, eval_rows, config)),
        ("stream_ratio", lambda: stage2_stream_ratio(eval_rows, stage1)),
    ):
        lifetime = estimator()
        predicted = np.maximum(0.0, fraction * lifetime)
        error = predicted - eval_rows["y"]
        results[name] = {
            "rmse": round(float(np.sqrt(np.mean(error ** 2))), 3),
            "mae": round(float(np.mean(np.abs(error))), 3),
        }
    return results


def main(argv=None) -> int:
    """Run the two-stage study under leave-one-bearing-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)
    config = {"hidden": args.hidden, "epochs": args.epochs, "learning_rate": 0.003,
              "weight_decay": 0.01, "batch_size": 128, "feature_transform": "eng34",
              "activation": "tanh", "target_scaling": False, "seed": 42}

    per_fold = {name: [] for name in ("oracle", "feature_ratio", "stream_ratio")}
    print("=== two-stage model, leave-one-bearing-out ===")
    for held_out in sorted(groups):
        eval_indices = groups[held_out]
        train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                    for i in idx])
        scored = evaluate(subset(rows, train_indices), subset(rows, eval_indices), config)
        for name, value in scored.items():
            per_fold[name].append(value["rmse"])
        print("  %-18s  oracle %8.2f | feature %8.2f | stream %8.2f" % (
            held_out, scored["oracle"]["rmse"], scored["feature_ratio"]["rmse"],
            scored["stream_ratio"]["rmse"]))

    print("\n=== summary (RMSE, minutes) ===")
    summary = {}
    for name, values in per_fold.items():
        values = np.asarray(values)
        summary[name] = {
            "rmse_mean": round(float(values.mean()), 3),
            "rmse_median": round(float(np.median(values)), 3),
            "rmse_worst": round(float(values.max()), 3),
            "folds": [{"bearing": b, "rmse": round(v, 3)}
                      for b, v in zip(sorted(groups), values)],
        }
        label = name + (" (upper bound, uses label)" if name == "oracle" else "")
        print("  %-26s mean %8.2f  median %8.2f  worst %8.2f"
              % (label, summary[name]["rmse_mean"], summary[name]["rmse_median"],
                 summary[name]["rmse_worst"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Is Profile B overfitting, and to what?

Profile B regresses a bearing's remaining life in minutes.  The unit is
the **bearing**: the competition's rules keep whole bearings on one side
of every split, so the question "is the model overfitting?" has a
precise meaning here - does the model score well on the bearings it was
fitted on and badly on bearings it has never seen?

This script measures exactly that, for a range of recipes:

* the score on the **training** bearings,
* the score on the **held-back** bearings the loop gates on, and
* the gap between them.

A model that is overfitting shows a wide gap, and the gap **grows** as
the capacity or the schedule grows.  A model that is merely noisy shows
a wide gap that does not move.

The two-stage lifetime model has a second kind of overfitting that a
single score cannot see: the *two-stage* estimate depends on a
per-unit baseline, and if the model memorises that baseline the
per-unit spread collapses.  So the per-unit RMSE spread is reported
alongside.

Run::

    python tools/check_profile_b_overfit.py
    python tools/check_profile_b_overfit.py --seeds 4
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

from tracks.hypotheses import CHAMPION_DEFAULTS  # noqa: E402
from tracks.profiles import get  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

SEEDS = (0, 1, 2, 42, 7, 123)

#: The recipes to compare.  The champion comes first; the rest walk the
#: two axes that could cause overfitting - capacity and schedule.
RECIPES = [
    ("champion", dict(CHAMPION_DEFAULTS["B"]), None),
    ("h32 e400", {"hidden": 32, "epochs": 400, "normalisation": "slope",
                  "target": "life_fraction", "floor": 0.02, "cap": 3000.0,
                  "rate_limit": 0.25, "feature_transform": "eng34",
                  "activation": "tanh", "seed": 42}, None),
    ("h64 e400", {"hidden": 64, "epochs": 400, "normalisation": "slope",
                  "target": "life_fraction", "floor": 0.02, "cap": 3000.0,
                  "rate_limit": 0.25, "feature_transform": "eng34",
                  "activation": "tanh", "seed": 42}, None),
    ("h96 e400 (champion width)", {"hidden": 96, "epochs": 400,
                                    "normalisation": "slope",
                                    "target": "life_fraction", "floor": 0.02,
                                    "cap": 3000.0, "rate_limit": 0.25,
                                    "feature_transform": "eng34",
                                    "activation": "tanh", "seed": 42}, None),
    ("h128 e400", {"hidden": 128, "epochs": 400, "normalisation": "slope",
                   "target": "life_fraction", "floor": 0.02, "cap": 3000.0,
                   "rate_limit": 0.25, "feature_transform": "eng34",
                   "activation": "tanh", "seed": 42}, None),
    ("champion + e800", dict(CHAMPION_DEFAULTS["B"], epochs=800), None),
    ("champion + e1600", dict(CHAMPION_DEFAULTS["B"], epochs=1600), None),
    ("champion + e4000", dict(CHAMPION_DEFAULTS["B"], epochs=4000), None),
]


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64)) ** 2)))


def _per_unit_rmse(y_true, y_pred, groups):
    """The RMSE inside each unit, and the spread of those values."""
    values = []
    for name in sorted(set(groups.tolist())):
        mask = np.asarray(groups) == name
        values.append(_rmse(y_true[mask], y_pred[mask]))
    return np.asarray(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("B")
    seeds = SEEDS[:args.seeds]
    initial, stream, judge, validation, meta = build_stream("B", 4, 2)

    print("profile B  initial %d rows / %d bearings   judge %d rows / %d bearings"
          % (len(initial["y"]), len(set(initial["groups"].tolist())),
             len(judge["y"]), len(set(judge["groups"].tolist()))), flush=True)

    rows = []
    for label, overrides, _ in RECIPES:
        train_scores, judge_scores, spreads = [], [], []
        for seed in seeds:
            kwargs = dict(overrides)
            kwargs["seed"] = seed
            model = profile.make_model(initial["x"], initial["y"],
                                       initial["paths"],
                                       groups=initial["groups"],
                                       order=initial["order"], **kwargs)
            # The training bearings, scored in-sample.
            train_pred = model.predict(initial["x"], initial["groups"],
                                       initial["order"])
            train_scores.append(_rmse(initial["y"], train_pred))
            # The held-back bearings, never trained on.
            judge_pred = model.predict(judge["x"], judge["groups"],
                                       judge["order"])
            judge_scores.append(_rmse(judge["y"], judge_pred))
            spreads.append(_per_unit_rmse(judge["y"], judge_pred,
                                           judge["groups"]).std())
        train_scores = np.asarray(train_scores)
        judge_scores = np.asarray(judge_scores)
        gap = train_scores.mean() - judge_scores.mean()
        rows.append({
            "label": label,
            "train_rmse": float(train_scores.mean()),
            "judge_rmse": float(judge_scores.mean()),
            "judge_rmse_std": float(judge_scores.std()),
            "gap": float(gap),
            "per_unit_spread": float(np.mean(spreads)),
        })
        print("  %-30s train %7.2f   judge %7.2f +- %-6.2f   gap %+8.2f   "
              "per-unit spread %6.2f"
              % (label, train_scores.mean(), judge_scores.mean(),
                 judge_scores.std(), gap, np.mean(spreads)), flush=True)

    print("\n" + "=" * 86)
    print("does the gap grow with the recipe?  (that is what overfitting looks like)")
    print("=" * 86)
    champion = rows[0]
    print("  %-30s gap %+8.2f" % (champion["label"], champion["gap"]))
    for row in rows[1:]:
        flag = "  <-- wider" if row["gap"] < champion["gap"] else ""
        print("  %-30s gap %+8.2f%s" % (row["label"], row["gap"], flag))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_overfit.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_overfit.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

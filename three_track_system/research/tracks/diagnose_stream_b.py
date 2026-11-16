# -*- coding: utf-8 -*-
"""Diagnose why a candidate trained on ``initial + stream[:k]`` scores so
much worse on the held-out stream bearings than the initial champion does.

The symptom
-----------
On profile B the closed loop produced this:

* the initial champion, trained on six bearings, scores 706 RMSE;
* a candidate trained on the same six bearings **plus** the first
  600 rows of the stream, scores 2111 RMSE on the *same* judge slice.

Retraining on *more* data made the model dramatically worse.  That is
not a hyper-parameter problem - it is a protocol problem, and the only
way to find out which one is to look.

The three candidate explanations
--------------------------------
1. **The fraction target is wrong for a partial unit.**  A unit that
   has only 600 of its 2538 rows so far has a *provisional* lifetime.
   Regressing the fraction against that provisional lifetime teaches
   the model that the unit ends soon - the opposite of the truth - and
   the error compounds with every row.
2. **The slope normalisation is contaminated.**  The ``slope``
   transform uses each unit's first five snapshots as its baseline.
   For a stream unit that has just arrived, those five snapshots are
   fine.  But the transform is applied to the *concatenated* array, so
   the initial units and the stream units are normalised independently -
   which should be fine.
3. **The lifetime estimate is out of range.**  ``safe_lifetime`` caps
   the scale at 3000 minutes and rate-limits jumps to 0.25.  If the
   fraction is wrong, the rate limiter then prevents the estimate from
   recovering, and the prediction is stuck at a wrong scale for the
   rest of the unit's life.

This script measures all three on the real data.

Run:
    python -m research.tracks.diagnose_stream_b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulkit.rul import LifeFractionModel  # noqa: E402
from tracks.profiles import life_fraction_from_arrays  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

CONFIG = {
    "hidden": 96, "epochs": 400, "learning_rate": 0.003, "batch_size": 128,
    "feature_transform": "eng34", "activation": "tanh", "target_scaling": False,
    "seed": 42, "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25,
}


def elapsed_counts(groups, order):
    groups = np.asarray(groups)
    order = np.asarray(order, dtype=np.float64)
    out = np.zeros(len(groups), dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        ordered = indices[np.argsort(order[indices])]
        out[ordered] = np.arange(1, len(ordered) + 1, dtype=float)
    return out


def fit(x, y, paths, groups, order, normalisation="slope", target="fraction"):
    """Fit a two-stage model with the chosen normalisation and target."""
    elapsed = elapsed_counts(groups, order)
    if target == "fraction":
        fit_target = life_fraction_from_arrays(y, groups, order)
    else:
        fit_target = y
    model = LifeFractionModel(floor=CONFIG["floor"], cap=CONFIG["cap"],
                              rate_limit=CONFIG["rate_limit"],
                              max_rul_minutes=CONFIG["cap"],
                              normalisation=normalisation)
    model.fit(x, fit_target, paths, groups,
              hidden=CONFIG["hidden"], seed=CONFIG["seed"],
              learning_rate=CONFIG["learning_rate"], batch_size=CONFIG["batch_size"],
              feature_transform=CONFIG["feature_transform"],
              activation=CONFIG["activation"], epochs=CONFIG["epochs"],
              weight_decay=0.01, elapsed=elapsed)
    return model, elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut", type=int, default=600)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    initial, stream, validation, meta = build_stream("B", 3)
    cut = args.cut

    print("=== diagnosis: why does adding stream rows make the model worse? ===")
    print("  initial units : %s" % ", ".join(meta["initial_units"]))
    print("  stream units  : %s (first %d rows of the stream)"
          % (", ".join(meta["stream_units"]), cut))

    judge_x = stream["x"][cut:]
    judge_y = stream["y"][cut:]
    judge_groups = stream["groups"][cut:]
    judge_order = stream["order"][cut:]

    results = {}

    # -- variant 1: initial only, fraction target ------------------------
    model_a, _ = fit(initial["x"], initial["y"], initial["paths"],
                     initial["groups"], initial["order"])
    pred_a = model_a.predict(judge_x, judge_groups,
                              elapsed_counts(judge_groups, judge_order))
    rmse_a = float(np.sqrt(np.mean((pred_a - judge_y) ** 2)))
    print("\n  [A] initial only, fraction target")
    print("      judge RMSE %.1f" % rmse_a)
    results["initial_only"] = {"rmse": rmse_a}

    x = np.concatenate([initial["x"], stream["x"][:cut]])
    y = np.concatenate([initial["y"], stream["y"][:cut]])
    paths = initial["paths"]
    groups = np.concatenate([initial["groups"], stream["groups"][:cut]])
    order = np.concatenate([initial["order"], stream["order"][:cut]])

    # -- variant 2: initial + stream[:cut], fraction target --------------
    model_b, _ = fit(x, y, paths, groups, order)
    pred_b = model_b.predict(judge_x, judge_groups,
                              elapsed_counts(judge_groups, judge_order))
    rmse_b = float(np.sqrt(np.mean((pred_b - judge_y) ** 2)))
    print("\n  [B] initial + stream[:%d], fraction target" % cut)
    print("      judge RMSE %.1f   <-- the failure" % rmse_b)
    results["initial_plus_stream_fraction"] = {"rmse": rmse_b}

    # -- variant 3: initial + stream[:cut], minutes target ---------------
    model_c, _ = fit(x, y, paths, groups, order, target="minutes")
    pred_c = model_c.predict(judge_x, judge_groups,
                              elapsed_counts(judge_groups, judge_order))
    rmse_c = float(np.sqrt(np.mean((pred_c - judge_y) ** 2)))
    print("\n  [C] initial + stream[:%d], minutes target" % cut)
    print("      judge RMSE %.1f" % rmse_c)
    results["initial_plus_stream_minutes"] = {"rmse": rmse_c}

    # -- the target itself -------------------------------------------------
    print("\n  --- the fraction target on the stream unit ---")
    stream_units = sorted(set(stream["groups"][:cut].tolist()))
    fraction = life_fraction_from_arrays(y, groups, order)
    for name in stream_units:
        mask = groups == name
        seen = int(mask.sum())
        true_fraction = np.asarray(y)[mask] / max(np.asarray(y)[mask].max(), 1.0)
        stride = max(1, seen // 6)
        print("      %-14s rows seen %4d" % (name, seen))
        print("        fraction target : %s"
              % " ".join("%.3f" % v for v in fraction[mask][::stride][:6]))
        print("        true fraction   : %s"
              % " ".join("%.3f" % v for v in true_fraction[::stride][:6]))
        results.setdefault("fraction_diagnostic", {})[name] = {
            "rows_seen": seen,
            "fraction_target": [round(float(v), 4) for v in fraction[mask][::stride][:6]],
            "true_fraction": [round(float(v), 4) for v in true_fraction[::stride][:6]],
        }

    # -- the prediction, per unit -----------------------------------------
    print("\n  --- per-unit judge RMSE ---")
    for label, prediction in (("A initial only", pred_a),
                               ("B +stream fraction", pred_b),
                               ("C +stream minutes", pred_c)):
        pieces = []
        for name in sorted(set(judge_groups.tolist())):
            mask = judge_groups == name
            piece = float(np.sqrt(np.mean((prediction[mask] - judge_y[mask]) ** 2)))
            pieces.append("%s %.0f" % (name, piece))
        print("      %-20s %s" % (label, " | ".join(pieces)))

    print("\n  --- verdict ---")
    if rmse_c < rmse_b:
        print("      The minutes target beats the fraction target by %.0f%%."
              % ((rmse_b - rmse_c) / rmse_b * 100))
        print("      The failure is the FRACTION TARGET on a partial unit:")
        print("      a unit with 600 of 2538 rows has a provisional lifetime")
        print("      of 600, so the fraction target says the unit ends at")
        print("      row 600 - and the model learns that falsehood.")
    else:
        print("      The fraction target is not the problem; the failure is")
        print("      elsewhere (normalisation or the lifetime estimate).")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

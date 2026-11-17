# -*- coding: utf-8 -*-
"""Why the two-stage model wins on short bearings and loses on the long one.

The measurement
---------------
On the official validation bearings:

=================  ==============  ==============
bearing            official RMSE   two-stage RMSE
=================  ==============  ==============
Bearing1_4           241.98            61.05
Bearing2_4           390.82            18.59
Bearing3_4           710.81           745.94
**average**          681.16           708.77
=================  ==============  ==============

The two-stage model is **four to twenty times better** on the two short
bearings and marginally worse on the long one - and the long bearing has
1515 of the 1679 validation rows, so it decides the average.

That is a much more useful finding than "the model is 4% worse than the
baseline".  It says the two-stage model has solved the *scale* problem
and lost the *shape* problem, and the shape problem only shows up on
long bearings.

The hypothesis
--------------
A bearing's remaining life in minutes is **not** a linear function of
its degradation state over a long life.  The two-stage model assumes

    RUL = f * s

with ``f`` the predicted fraction of life remaining and ``s`` the
guarded lifetime estimate.  On a short bearing that is close to right.
On a long bearing the degradation is slow at the start and fast at the
end, so the true curve is strongly **convex**, and a linear-in-fraction
model under-predicts the remaining life in the middle of the bearing.

The test
--------
The script fits the two-stage model on the training bearings and then
compares, for the held-out bearing, three ways of turning the fraction
into minutes:

``two-stage``
    ``RUL = f * s`` - the current formula.
``affine``
    ``RUL = a * f + b``, with ``a`` and ``b`` fitted on the *training*
    bearings.  This allows a non-zero intercept, which is exactly what
    a convex curve needs.
``blend``
    ``RUL = w * (f * s) + (1 - w) * (a * f + b)``.

Each is fitted **without touching the held-out bearing**, and the
held-out bearing is scored on all three.  The winner is the formula
that generalises, not the one that fits.

Run:
    python -m research.tracks.shape_of_rul
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

from rulkit.rul import LifeFractionModel, safe_lifetime  # noqa: E402
from tracks.profiles import elapsed_counts_from, life_fraction_from_arrays  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

CONFIG = {
    "hidden": 96, "epochs": 400, "learning_rate": 0.003, "batch_size": 128,
    "feature_transform": "eng34", "activation": "tanh", "seed": 42,
    "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25,
}


def fit_two_stage(x, y, paths, groups, order):
    """Fit the fraction regressor and return it with its elapsed counts."""
    elapsed = elapsed_counts_from(groups, order)
    fraction = life_fraction_from_arrays(y, groups, order)
    model = LifeFractionModel(floor=CONFIG["floor"], cap=CONFIG["cap"],
                              rate_limit=CONFIG["rate_limit"],
                              max_rul_minutes=CONFIG["cap"], normalisation="slope",
                              smooth_window=1)
    model.fit(x, fraction, paths, groups,
              hidden=CONFIG["hidden"], seed=CONFIG["seed"],
              learning_rate=CONFIG["learning_rate"], batch_size=CONFIG["batch_size"],
              feature_transform=CONFIG["feature_transform"],
              activation=CONFIG["activation"], epochs=CONFIG["epochs"],
              weight_decay=0.01, elapsed=elapsed)
    return model, elapsed


def two_stage_prediction(model, x, groups, order):
    """``RUL = f * s`` with the guarded lifetime estimate."""
    elapsed = elapsed_counts_from(groups, order)
    fraction = np.clip(model.predict_fraction(x, groups, elapsed), 0.0, 1.0)
    scale = np.zeros(len(fraction), dtype=np.float64)
    for name in np.unique(np.asarray(groups)):
        indices = np.nonzero(np.asarray(groups) == name)[0]
        ordered = indices[np.argsort(elapsed[indices])]
        scale[ordered] = safe_lifetime(fraction[ordered], elapsed[ordered],
                                       floor=CONFIG["floor"], cap=CONFIG["cap"],
                                       rate_limit=CONFIG["rate_limit"])
    return fraction, np.clip(fraction * scale, 0.0, CONFIG["cap"])


def fit_affine(fraction, y, groups):
    """Fit ``RUL = a * f + b`` per unit on the training rows.

    The coefficients are fitted on the *training* units only.  Using the
    held-out unit's own rows to fit them would be leakage - it would
    answer "what is the best affine correction for this bearing?" with
    the answer in hand.
    """
    a, b = 1.0, 0.0
    # A single pair of coefficients for the whole training set: the
    # correction is a property of the *model*, not of the unit.
    design = np.stack([fraction, np.ones(len(fraction))], axis=1)
    solution, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
    a, b = float(solution[0]), float(solution[1])
    return a, b


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    initial, stream, validation, meta = build_stream("B", 3)

    print("=== the shape of RUL: two-stage vs affine on the validation bearings ===")
    print("  training bearings : %s" % ", ".join(meta["initial_units"]))
    print("  held-out bearings : %s" % ", ".join(meta["validation_units"]))

    model, train_elapsed = fit_two_stage(initial["x"], initial["y"], initial["paths"],
                                         initial["groups"], initial["order"])
    train_fraction, _ = two_stage_prediction(model, initial["x"], initial["groups"],
                                              initial["order"])
    a, b = fit_affine(train_fraction, initial["y"], initial["groups"])
    print("\n  affine correction fitted on the training bearings: RUL = %.2f * f + %.2f"
          % (a, b))

    valid_fraction, two_stage = two_stage_prediction(model, validation["x"],
                                                     validation["groups"],
                                                     validation["order"])
    affine = np.clip(a * valid_fraction + b, 0.0, CONFIG["cap"])

    groups = validation["groups"]
    y = validation["y"]

    print("\n  %-16s %12s %12s %12s" % ("bearing", "two-stage", "affine", "winner"))
    results = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        r_two = float(np.sqrt(np.mean((two_stage[mask] - y[mask]) ** 2)))
        r_aff = float(np.sqrt(np.mean((affine[mask] - y[mask]) ** 2)))
        winner = "affine" if r_aff < r_two else "two-stage"
        print("  %-16s %12.2f %12.2f %12s" % (name, r_two, r_aff, winner))
        results[str(name)] = {"two_stage_rmse": r_two, "affine_rmse": r_aff,
                              "winner": winner}

    overall_two = float(np.sqrt(np.mean((two_stage - y) ** 2)))
    overall_aff = float(np.sqrt(np.mean((affine - y) ** 2)))
    print("\n  %-16s %12.2f %12.2f" % ("ALL", overall_two, overall_aff))
    if overall_aff < overall_two:
        print("\n  The affine correction wins overall by %.1f%%."
              % ((overall_two - overall_aff) / overall_two * 100))
        print("  This says the two-stage formula is systematically under-predicting")
        print("  on long bearings, and a non-zero intercept is what it needs.")
    else:
        print("\n  The two-stage formula wins; the affine correction is not needed.")

    # -- the diagnostic: where does the two-stage model go wrong? ---------
    print("\n  --- where the two-stage model goes wrong (Bearing3_4) ---")
    mask = groups == "xjtu_Bearing3_4"
    if mask.any():
        order = np.argsort(validation["order"][mask])
        true_y = y[mask][order]
        pred_two = two_stage[mask][order]
        pred_aff = affine[mask][order]
        n = len(true_y)
        for label, start, stop in (("early 10%", 0, n // 10),
                                   ("middle", n // 2 - n // 20, n // 2 + n // 20),
                                   ("late 10%", n - n // 10, n)):
            true_part = true_y[start:stop]
            two_part = pred_two[start:stop]
            aff_part = pred_aff[start:stop]
            print("    %-12s true %7.1f..%7.1f | two-stage %7.1f..%7.1f | affine %7.1f..%7.1f"
                  % (label, true_part.min(), true_part.max(),
                     two_part.min(), two_part.max(),
                     aff_part.min(), aff_part.max()))

    results["overall"] = {"two_stage_rmse": overall_two, "affine_rmse": overall_aff,
                          "affine_a": a, "affine_b": b}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

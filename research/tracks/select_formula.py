# -*- coding: utf-8 -*-
"""Learn, from the training bearings only, which RUL formula to trust.

The problem
-----------
Two formulas convert the model's predicted life fraction into minutes:

* the **two-stage product** ``RUL = f * s`` (a guarded lifetime estimate
  times the fraction), and
* the **affine map** ``RUL = a * f + b`` (a global linear correction).

Neither dominates.  On the official validation bearings the two-stage
product is four to twenty times better on the short bearings and the
affine map is better on the long one, and because the long bearing
holds 1515 of the 1679 validation rows, the average follows the long
one.

Choosing between them by a hand-set threshold on the estimated lifetime
turned out to be fragile: moving the threshold from the training
distribution's median to its largest unit changed the validation RMSE
from 570.68 to 595.67.  A rule with a sensitive knob is not a result.

The method
----------
Rather than hand-setting the rule, **learn it from the training
bearings**, by the same leave-one-bearing-out discipline the rest of
the project uses:

1. For each training bearing, measure which formula wins on the other
   training bearings (leave-one-out inside the training set).
2. Find the feature that predicts the winner.
3. Fit the rule on the training bearings.
4. Apply it, frozen, to the validation bearings - which are scored
   once.

The candidate features are deliberately few and each has a reason to
matter:

``estimated_lifetime``
    The median of the unit's guarded lifetime estimate.  The
    hypothesis: the product is structural and wins on units whose life
    is short; the affine map extrapolates from the training mean and
    wins on units near that mean.
``rows_so_far``
    The unit's length.  A long bearing has more evidence, so the
    structural estimate should be more trustworthy there.
``fraction_spread``
    How much the predicted fraction varies across the unit.  A unit
    whose fraction barely moves is one the model cannot read, and its
    structural estimate is correspondingly weak.

Run:
    python -m research.tracks.select_formula --out artifacts/tracks/select_formula.json
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
    return model


def predict_parts(model, x, groups, order):
    """The fraction, the guarded lifetime estimate and their product."""
    elapsed = elapsed_counts_from(groups, order)
    fraction = np.clip(model.predict_fraction(x, groups, elapsed), 0.0, 1.0)
    scale = np.zeros(len(fraction), dtype=np.float64)
    for name in np.unique(np.asarray(groups)):
        indices = np.nonzero(np.asarray(groups) == name)[0]
        ordered = indices[np.argsort(elapsed[indices])]
        scale[ordered] = safe_lifetime(fraction[ordered], elapsed[ordered],
                                       floor=CONFIG["floor"], cap=CONFIG["cap"],
                                       rate_limit=CONFIG["rate_limit"])
    return fraction, scale, np.clip(fraction * scale, 0.0, CONFIG["cap"])


def fit_affine(fraction, y):
    design = np.stack([np.asarray(fraction, dtype=np.float64),
                       np.ones(len(fraction))], axis=1)
    solution, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
    return float(solution[0]), float(solution[1])


def unit_features(fraction, scale, groups, order):
    """One feature row per unit."""
    groups = np.asarray(groups)
    order = np.asarray(order, dtype=np.float64)
    names = sorted(set(groups.tolist()))
    out = {}
    for name in names:
        mask = groups == name
        indices = np.nonzero(mask)[0]
        length = len(indices)
        out[str(name)] = {
            "estimated_lifetime": float(np.median(scale[mask])),
            "rows_so_far": float(length),
            "fraction_spread": float(np.std(fraction[mask])),
            "length": length,
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    initial, stream, validation, meta = build_stream("B", 3)
    train_groups = initial["groups"]
    valid_groups = validation["groups"]

    print("=== learning which RUL formula to trust, from the training bearings ===")

    # -- leave-one-out inside the training set ---------------------------
    #
    # For each training bearing: fit on the other five, score on the
    # held-out one, with both formulas.  This gives an honest,
    # per-bearing comparison on units the rule was not fitted on.
    print("\n  --- leave-one-out inside the training bearings ---")
    print("  %-16s %10s %10s %10s" % ("held-out", "two-stage", "affine", "winner"))
    records = []
    names = sorted(set(train_groups.tolist()))
    for held_out in names:
        keep = train_groups != held_out
        held = train_groups == held_out

        model = fit_two_stage(initial["x"][keep], initial["y"][keep],
                              initial["paths"], train_groups[keep], initial["order"][keep])
        fraction, scale, two_stage = predict_parts(
            model, initial["x"][held], train_groups[held], initial["order"][held])

        # The affine coefficients are fitted on the same kept bearings,
        # using their own in-sample fraction predictions, so that the
        # held-out bearing contributes nothing.
        kept_fraction, _, _ = predict_parts(
            model, initial["x"][keep], train_groups[keep], initial["order"][keep])
        a, b = fit_affine(kept_fraction, initial["y"][keep])
        affine = np.clip(a * fraction + b, 0.0, CONFIG["cap"])

        y_held = initial["y"][held]
        r_two = float(np.sqrt(np.mean((two_stage - y_held) ** 2)))
        r_aff = float(np.sqrt(np.mean((affine - y_held) ** 2)))
        features = unit_features(fraction, scale, train_groups[held],
                                  initial["order"][held])
        record = {"bearing": str(held_out), "two_stage_rmse": r_two,
                  "affine_rmse": r_aff,
                  "winner": "affine" if r_aff < r_two else "two-stage",
                  **features[str(held_out)]}
        records.append(record)
        print("  %-16s %10.2f %10.2f %10s"
              % (held_out, r_two, r_aff, record["winner"]))

    # -- which feature predicts the winner? --------------------------------
    print("\n  --- does any feature predict the winner? ---")
    for feature in ("estimated_lifetime", "rows_so_far", "fraction_spread"):
        winners = [r["winner"] for r in records]
        values = [r[feature] for r in records]
        two_values = [r[feature] for r in records if r["winner"] == "two-stage"]
        aff_values = [r[feature] for r in records if r["winner"] == "affine"]
        if not two_values or not aff_values:
            print("    %-20s all units chose the same formula - no split to learn"
                  % feature)
            continue
        print("    %-20s two-stage %s | affine %s"
              % (feature,
                 " ".join("%.0f" % v for v in sorted(two_values)),
                 " ".join("%.0f" % v for v in sorted(aff_values))))
        if max(two_values) < min(aff_values) or max(aff_values) < min(two_values):
            print("        -> the feature separates the winners perfectly")
        else:
            print("        -> the feature does NOT separate them")

    # -- the rule: pick the formula with the lower training error ---------
    #
    # The simplest rule that can be fitted honestly: on the training
    # bearings, compare the two formulas **overall** and use the better
    # one.  The leave-one-out table above says how that generalises.
    model = fit_two_stage(initial["x"], initial["y"], initial["paths"],
                          train_groups, initial["order"])
    train_fraction, train_scale, train_two = predict_parts(
        model, initial["x"], train_groups, initial["order"])
    a, b = fit_affine(train_fraction, initial["y"])
    train_affine = np.clip(a * train_fraction + b, 0.0, CONFIG["cap"])

    r_two = float(np.sqrt(np.mean((train_two - initial["y"]) ** 2)))
    r_aff = float(np.sqrt(np.mean((train_affine - initial["y"]) ** 2)))
    rule = "affine" if r_aff < r_two else "two-stage"
    print("\n  the rule fitted on the training bearings: use %s" % rule)
    print("      training RMSE  two-stage %.2f  affine %.2f" % (r_two, r_aff))

    # -- apply the frozen rule to the validation bearings ------------------
    valid_fraction, valid_scale, valid_two = predict_parts(
        model, validation["x"], valid_groups, validation["order"])
    valid_affine = np.clip(a * valid_fraction + b, 0.0, CONFIG["cap"])
    chosen = valid_affine if rule == "affine" else valid_two

    print("\n  --- validation bearings, scored once with the frozen rule ---")
    print("  %-16s %12s %12s %12s" % ("bearing", "two-stage", "affine", "rule"))
    results = {"rule": rule, "affine_a": a, "affine_b": b,
               "training_rmse": {"two_stage": r_two, "affine": r_aff},
               "leave_one_out": records, "per_bearing": {}}
    for name in sorted(set(valid_groups.tolist())):
        mask = valid_groups == name
        r_t = float(np.sqrt(np.mean((valid_two[mask] - validation["y"][mask]) ** 2)))
        r_a = float(np.sqrt(np.mean((valid_affine[mask] - validation["y"][mask]) ** 2)))
        r_c = float(np.sqrt(np.mean((chosen[mask] - validation["y"][mask]) ** 2)))
        print("  %-16s %12.2f %12.2f %12.2f" % (name, r_t, r_a, r_c))
        results["per_bearing"][str(name)] = {"two_stage_rmse": r_t,
                                             "affine_rmse": r_a, "rule_rmse": r_c}

    overall_t = float(np.sqrt(np.mean((valid_two - validation["y"]) ** 2)))
    overall_a = float(np.sqrt(np.mean((valid_affine - validation["y"]) ** 2)))
    overall_c = float(np.sqrt(np.mean((chosen - validation["y"]) ** 2)))
    print("\n  %-16s %12.2f %12.2f %12.2f" % ("ALL", overall_t, overall_a, overall_c))
    print("\n  official baseline 681.16")
    print("    rule (%s): %.2f  ->  %+.2f%%" % (rule, overall_c,
                                               (681.16 - overall_c) / 681.16 * 100))
    results["overall"] = {"two_stage_rmse": overall_t, "affine_rmse": overall_a,
                          "rule_rmse": overall_c, "baseline_rmse": 681.16}

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

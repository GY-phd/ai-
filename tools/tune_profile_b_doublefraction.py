# -*- coding: utf-8 -*-
"""Fix Profile B's double-applied fraction, and measure it properly.

The bug
-------
``rulkit.rul.safe_lifetime`` returns the **remaining** life implied by
the predicted fraction::

    instantaneous = f * t / (1 - f)

That is already the answer to "how many minutes are left".  But
``LifeFractionModel.predict`` treats the returned value as the bearing's
**total lifetime** and multiplies by the fraction a second time::

    product = fraction * scale        # = f * (f * t / (1 - f)) = f^2 * t / (1 - f)

The fraction is therefore applied **twice**, and the estimate is low by
a factor of ``f`` - about two-fold at the median fraction.  The
measurement confirms it: under leave-one-bearing-out the estimated
lifetime is 0.32 to 0.59 of the truth on **every** bearing, mean 0.49.

That single defect explains the whole symptom set:

* the product under-predicts by ~50% on the training bearings;
* the affine map - a global straight line with a slope of ~2600, fitted
  on a population that is bimodal (seven bearings of 61-266 minutes and
  two of 1247-2537) - has to be blended in to compensate, and it is
  three to four times worse on the short bearings, which is what makes
  the held-out score bad;
* the blend's ramp is fitted on the longest training bearing, so it
  hands the affine map to bearings that should trust the product.

The fix
-------
Either the scale is the total lifetime ``t / (1 - f)`` and the product
is ``f * scale``, or the scale is the remaining life ``f * t / (1 - f)``
and the product is the scale itself.  The second form is what the
existing guards were designed around - the ``floor`` bounds the divisor
``1 - f`` **inside** the scale, which the first form cannot do - so this
script measures the second form: the product is the scale, unchanged.

This script measures, under the leave-one-bearing-out protocol with the
pooled RMSE that profile B's metric actually uses:

* the current code,
* the product without the double fraction,
* the product without the double fraction, with the affine map blended
  in under the current ramp,
* the product without the double fraction, with the affine map dropped.

Run::

    python tools/tune_profile_b_doublefraction.py
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

from tracks.profiles import get  # noqa: E402
from tracks.run import elapsed_counts  # noqa: E402


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64)) ** 2)))


def _components(model, x, y, groups, order):
    """Everything the two-stage estimate is built from.

    ``scale`` is what ``safe_lifetime`` returns; ``lifetime`` is the
    *total* lifetime implied by the same estimate, obtained by dividing
    the scale back out of its ``f`` factor.
    """
    from rulkit.rul import safe_lifetime
    elapsed = np.asarray(elapsed_counts(groups, order), dtype=np.float64)
    fraction = model.predict_fraction(x, groups, elapsed)
    groups = np.asarray(groups)
    scale = np.zeros(len(fraction), dtype=np.float64)
    for unit in np.unique(groups):
        indices = np.nonzero(groups == unit)[0]
        ordered = indices[np.argsort(elapsed[indices])]
        raw = safe_lifetime(fraction[ordered], elapsed[ordered],
                            floor=model.floor, cap=model.cap,
                            rate_limit=model.rate_limit)
        scale[ordered] = model._smooth(raw)
    if model.affine is None:
        affine = fraction * scale
    else:
        a, b = model.affine
        affine = np.clip(a * fraction + b, 0.0, model.max_rul_minutes)
    return fraction, scale, affine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("B")
    train = profile.load("train")
    groups_all = np.asarray(train["groups"])
    order_all = np.asarray(train["order"])
    y_all = np.asarray(train["y"], dtype=np.float64)
    x_all = train["x"]
    units = sorted(set(groups_all.tolist()))

    folds = []
    for held_out in units:
        keep = groups_all != held_out
        model = profile.make_model(
            x_all[keep], y_all[keep], train["paths"], groups=groups_all[keep],
            order=order_all[keep],
            hidden=96, epochs=400, normalisation="slope",
            target="life_fraction", floor=0.02, cap=3000.0,
            rate_limit=0.25, feature_transform="eng34",
            activation="tanh", seed=42)
        held = ~keep
        fraction, scale, affine = _components(
            model, x_all[held], y_all[held], groups_all[held], order_all[held])
        folds.append({
            "bearing": held_out,
            "rows": int(held.sum()),
            "truth": y_all[held],
            "fraction": fraction,
            "scale": scale,
            "affine": affine,
            "short_cut": model.short_cut,
            "long_cut": model.long_cut,
            "true_life": float(y_all[held].max()),
        })
        est = float(np.median(scale))
        print("  fold %-12s held-out rows %5d   true life %7.1f   est %7.1f  (%.0f%%)"
              % (held_out, held.sum(), y_all[held].max(), est,
                 100 * est / max(y_all[held].max(), 1e-9)), flush=True)

    results = []

    def score(label, predictor):
        total = 0.0
        total_rows = 0
        per_unit = {}
        for fold in folds:
            prediction = predictor(fold)
            error = _rmse(fold["truth"], prediction)
            per_unit[fold["bearing"]] = error
            total += error ** 2 * fold["rows"]
            total_rows += fold["rows"]
        pooled = (total / total_rows) ** 0.5
        results.append({"label": label, "pooled": pooled, "per_unit": per_unit})
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-40s %10.2f   %s" % (label, pooled, detail), flush=True)
        return pooled

    def ramp_weight(fold):
        weight = np.clip((fold["scale"] - fold["short_cut"])
                         / max(fold["long_cut"] - fold["short_cut"], 1e-9),
                         0.0, 1.0)
        return weight

    print("\n" + "=" * 104)
    print("leave-one-bearing-out, pooled RMSE (profile B's metric)")
    print("=" * 104)
    print("%-40s %10s   %s" % ("estimator", "pooled rmse", "per held-out bearing"))
    print("-" * 104)

    # -- the current code --------------------------------------------------
    score("current: f * (f * t/(1-f)) + affine blend",
          lambda f: ramp_weight(f) * f["affine"]
          + (1.0 - ramp_weight(f)) * (f["fraction"] * f["scale"]))
    score("current, product only (f^2 form)",
          lambda f: f["fraction"] * f["scale"])
    score("current, affine only",
          lambda f: f["affine"])

    # -- the fix: the scale already IS the remaining life ------------------
    score("fixed: scale alone (no double f)",
          lambda f: f["scale"])
    score("fixed: scale alone, no affine",
          lambda f: f["scale"])

    # -- the fix, with the affine map blended back in ----------------------
    for ramp_name, weight_fn in (
            ("current ramp", lambda f: ramp_weight(f)),
            ("always affine", lambda f: np.ones(len(f["scale"]))),
            ("never affine", lambda f: np.zeros(len(f["scale"]))),
    ):
        score("fixed: scale + affine under %s" % ramp_name,
              lambda f, weight_fn=weight_fn: weight_fn(f) * f["affine"]
              + (1.0 - weight_fn(f)) * f["scale"])

    # -- the fix, with the lifetime estimate rescaled to the training mean -
    # The product is right in shape but the magnitude of the lifetime
    # estimate may still be off; a single global multiplier fitted on the
    # training bearings fixes that without adding a per-bearing constant.
    print("\n" + "=" * 104)
    print("the same, with a global multiplier on the lifetime estimate")
    print("=" * 104)
    results = []

    def score_scaled(label, predictor):
        total = 0.0
        total_rows = 0
        per_unit = {}
        for fold in folds:
            prediction = predictor(fold)
            error = _rmse(fold["truth"], prediction)
            per_unit[fold["bearing"]] = error
            total += error ** 2 * fold["rows"]
            total_rows += fold["rows"]
        pooled = (total / total_rows) ** 0.5
        results.append({"label": label, "pooled": pooled, "per_unit": per_unit})
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-40s %10.2f   %s" % (label, pooled, detail), flush=True)
        return pooled

    # A multiplier that makes the *median* of the held-out bearing's
    # estimate match its truth would be cheating (it uses the label),
    # so instead the multiplier is fitted on the training bearings only.
    def fit_multiplier(fold, in_sample):
        """The multiplier that best fits the training rows."""
        numerator = 0.0
        denominator = 0.0
        for truth, scale, fraction in in_sample:
            weight = fraction
            numerator += float(np.sum(weight * truth))
            denominator += float(np.sum(weight * weight * scale))
        if denominator <= 0:
            return 1.0
        return numerator / denominator

    # Rebuild the in-sample components for each fold.
    for fold in folds:
        held_out = fold["bearing"]
        keep = groups_all != held_out
        model = profile.make_model(
            x_all[keep], y_all[keep], train["paths"], groups=groups_all[keep],
            order=order_all[keep],
            hidden=96, epochs=400, normalisation="slope",
            target="life_fraction", floor=0.02, cap=3000.0,
            rate_limit=0.25, feature_transform="eng34",
            activation="tanh", seed=42)
        train_fraction, train_scale, _ = _components(
            model, x_all[keep], y_all[keep], groups_all[keep], order_all[keep])
        fold["train_fraction"] = train_fraction
        fold["train_scale"] = train_scale
        fold["train_truth"] = y_all[keep]
        fold["multiplier"] = fit_multiplier(
            fold, [(fold["train_truth"], fold["train_scale"],
                    fold["train_fraction"])])
        print("  fold %-12s multiplier %.4f" % (held_out, fold["multiplier"]),
              flush=True)

    score_scaled("fixed: scale * multiplier (train-fitted)",
                 lambda f: f["scale"] * f["multiplier"])
    score_scaled("fixed: scale * multiplier + affine, current ramp",
                 lambda f: ramp_weight(f) * f["affine"]
                 + (1.0 - ramp_weight(f)) * (f["scale"] * f["multiplier"]))

    results.sort(key=lambda r: r["pooled"])
    print("\n" + "=" * 104)
    print("ranked (the fixed variants only)")
    print("=" * 104)
    for row in results:
        print("  %-40s %10.2f" % (row["label"], row["pooled"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_doublefraction.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_doublefraction.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

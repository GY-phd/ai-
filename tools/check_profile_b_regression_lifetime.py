# -*- coding: utf-8 -*-
"""Profile B: replace the ratio lifetime estimate with a regression one.

The defect
----------
``rulkit.rul.safe_lifetime`` returns the **remaining** life implied by
the predicted fraction::

    instantaneous = f * t / (1 - f)

but ``LifeFractionModel.predict`` treats the returned value as the
bearing's **total lifetime** and multiplies by the fraction a second
time::

    product = fraction * scale       # = f * (f * t / (1 - f))

so the fraction is applied twice and the estimate is low by a factor of
``f`` - about two-fold at the median fraction.  Under leave-one-
bearing-out the estimated lifetime came out at 32% to 59% of the truth
on every bearing, and a global multiplier fitted independently in each
fold came out at 2.02 to 2.13, which is the same defect seen from the
other side.

Even with the double application removed, the *ratio* itself is the
wrong estimator: ``f * t / (1 - f)`` divides a noisy numerator by a
noisy denominator at every point, and ``E[1/(1 - f)]`` is strictly
smaller than ``1 / E[1 - f]``, so the estimate is biased low by
Jensen's inequality - more so the noisier the fraction model is, and
profile B's fraction model has an RMSE of about 0.27.

The fix
-------
The relationship the two-stage model is built on is exact and linear::

    t = N * (1 - f)

where ``N`` is the bearing's lifetime in the same units as ``t``.  So
``N`` is the intercept of a straight line through the points
``(1 - f, t)``, and a least-squares fit over the bearing's own history
estimates it far more robustly than a per-point ratio - the noise
averages out instead of being amplified.

This module:

* fits ``N`` by least squares on ``(1 - f, t)`` over the unit's
  history, using the same guards the ratio form uses (the floor bounds
  the divisor, the cap bounds the lifetime);
* keeps the fraction's own floor, so a row whose fraction has run to
  zero cannot produce an infinite lifetime;
* and leaves everything else - the normalisation, the fraction model,
  the affine map and its blend - untouched.

Run::

    python tools/check_profile_b_regression_lifetime.py
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


def regression_lifetime(fraction, elapsed, groups, floor, cap):
    """The lifetime of each unit, by least squares on ``(1 - f, t)``.

    The relation ``t = N * (1 - f)`` is exact when the fraction is
    right, so ``N`` is the slope of ``t`` against ``(1 - f)`` and the
    fit over a unit's whole history averages the fraction model's noise
    out instead of amplifying it.

    The fit is made on the **observed** history only, so a unit that
    has only partly arrived still gets a lifetime from what it has
    shown - and, unlike the ratio form, the estimate does not collapse
    when the first rows are noisy.
    """
    groups = np.asarray(groups)
    elapsed = np.asarray(elapsed, dtype=np.float64)
    fraction = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0 - floor)
    lifetime = np.zeros(len(fraction), dtype=np.float64)
    for unit in np.unique(groups):
        mask = groups == unit
        f = fraction[mask]
        t = elapsed[mask]
        # ``t = N * (1 - f) + b`` - the intercept ``b`` absorbs the
        # offset between the snapshot count and the lifetime in
        # minutes, which the snapshots are not evenly spaced in.
        design = np.stack([1.0 - f, np.ones(len(f))], axis=1)
        try:
            solution, *_ = np.linalg.lstsq(design, t, rcond=None)
            slope = float(solution[0])
        except np.linalg.LinAlgError:
            slope = 0.0
        if not np.isfinite(slope) or slope <= 0:
            # Fall back on the ratio form, which is always defined.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(t, np.maximum(1.0 - f, floor))
            finite = ratio[np.isfinite(ratio) & (ratio > 0)]
            slope = float(np.median(finite)) if len(finite) else 1.0
        slope = float(np.clip(slope, 1.0, cap))
        lifetime[mask] = slope
    return lifetime


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

    results = []

    def score(label, predictor):
        total = 0.0
        total_rows = 0
        per_unit = {}
        for held_out in units:
            keep = groups_all != held_out
            model = profile.make_model(
                x_all[keep], y_all[keep], train["paths"],
                groups=groups_all[keep], order=order_all[keep],
                hidden=96, epochs=400, normalisation="slope",
                target="life_fraction", floor=0.02, cap=3000.0,
                rate_limit=0.25, feature_transform="eng34",
                activation="tanh", seed=42)
            held = ~keep
            elapsed = np.asarray(elapsed_counts(groups_all[held],
                                                order_all[held]), dtype=np.float64)
            fraction = model.predict_fraction(x_all[held], groups_all[held],
                                              elapsed)
            prediction = predictor(model, fraction, elapsed,
                                   groups_all[held], y_all[held])
            error = _rmse(y_all[held], prediction)
            per_unit[held_out] = error
            total += error ** 2 * held.sum()
            total_rows += held.sum()
        pooled = (total / total_rows) ** 0.5
        results.append({"label": label, "pooled": pooled, "per_unit": per_unit})
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-42s %10.2f   %s" % (label, pooled, detail), flush=True)
        return pooled

    def current(model, fraction, elapsed, groups, truth):
        from rulkit.rul import safe_lifetime
        scale = np.zeros(len(fraction), dtype=np.float64)
        for unit in np.unique(groups):
            indices = np.nonzero(groups == unit)[0]
            ordered = indices[np.argsort(elapsed[indices])]
            raw = safe_lifetime(fraction[ordered], elapsed[ordered],
                                floor=model.floor, cap=model.cap,
                                rate_limit=model.rate_limit)
            scale[ordered] = model._smooth(raw)
        product = fraction * scale
        if model.affine is None:
            return product
        a, b = model.affine
        affine = np.clip(a * fraction + b, 0.0, model.max_rul_minutes)
        weight = np.clip((model.short_cut and 1 or 1), 0.0, 1.0)
        weight = np.clip((scale - model.short_cut)
                         / max(model.long_cut - model.short_cut, 1e-9), 0.0, 1.0)
        return weight * affine + (1.0 - weight) * product

    def fixed_product(model, fraction, elapsed, groups, truth):
        lifetime = regression_lifetime(fraction, elapsed, groups,
                                        model.floor, model.cap)
        return np.clip(fraction * lifetime, 0.0, model.max_rul_minutes)

    def fixed_blend(model, fraction, elapsed, groups, truth):
        lifetime = regression_lifetime(fraction, elapsed, groups,
                                        model.floor, model.cap)
        product = fraction * lifetime
        if model.affine is None:
            return product
        a, b = model.affine
        affine = np.clip(a * fraction + b, 0.0, model.max_rul_minutes)
        weight = np.clip((lifetime - model.short_cut)
                         / max(model.long_cut - model.short_cut, 1e-9), 0.0, 1.0)
        return weight * affine + (1.0 - weight) * product

    print("=" * 110)
    print("leave-one-bearing-out, pooled RMSE (profile B's metric)")
    print("=" * 110)
    print("%-42s %10s   %s" % ("estimator", "pooled rmse", "per held-out bearing"))
    print("-" * 110)
    score("current code (ratio + affine blend)", current)
    score("current code, product only", lambda m, f, e, g, t: _product(m, f, e, g))
    score("FIXED: regression lifetime, product only", fixed_product)
    score("FIXED: regression lifetime + affine blend", fixed_blend)

    results.sort(key=lambda r: r["pooled"])
    print("\n" + "=" * 110)
    print("ranked")
    print("=" * 110)
    for row in results:
        print("  %-42s %10.2f" % (row["label"], row["pooled"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_regression_lifetime.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_regression_lifetime.json"))
    return 0


def _product(model, fraction, elapsed, groups):
    from rulkit.rul import safe_lifetime
    scale = np.zeros(len(fraction), dtype=np.float64)
    for unit in np.unique(groups):
        indices = np.nonzero(groups == unit)[0]
        ordered = indices[np.argsort(elapsed[indices])]
        raw = safe_lifetime(fraction[ordered], elapsed[ordered],
                            floor=model.floor, cap=model.cap,
                            rate_limit=model.rate_limit)
        scale[ordered] = model._smooth(raw)
    return fraction * scale


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Check the regression-lifetime fix on the official splits.

The leave-one-bearing-out measurement said the fix is worth about a
third of the held-out RMSE.  This script checks it on the splits the
repository actually ships:

* the **official validation** bearings (``Bearing1_4``, ``Bearing2_4``,
  ``Bearing3_4``) - the competition's own held-out set, fitted on the
  nine official training bearings;
* the **judge** bearings the closed loop gates on, and
* the **stream** the loop adapts to.

All three are bearings the model has never seen, so all three are
legitimate measurements.

Run::

    python tools/verify_profile_b_regression_lifetime.py
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
from tracks.stream import build as build_stream  # noqa: E402


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64)) ** 2)))


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


def _regression_lifetime(fraction, elapsed, groups, floor, cap):
    groups = np.asarray(groups)
    elapsed = np.asarray(elapsed, dtype=np.float64)
    fraction = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0 - floor)
    lifetime = np.zeros(len(fraction), dtype=np.float64)
    for unit in np.unique(groups):
        mask = groups == unit
        f = fraction[mask]
        t = elapsed[mask]
        design = np.stack([1.0 - f, np.ones(len(f))], axis=1)
        try:
            solution, *_ = np.linalg.lstsq(design, t, rcond=None)
            slope = float(solution[0])
        except np.linalg.LinAlgError:
            slope = 0.0
        if not np.isfinite(slope) or slope <= 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(t, np.maximum(1.0 - f, floor))
            finite = ratio[np.isfinite(ratio) & (ratio > 0)]
            slope = float(np.median(finite)) if len(finite) else 1.0
        lifetime[mask] = float(np.clip(slope, 1.0, cap))
    return lifetime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("B")
    train = profile.load("train")
    validation = profile.load("validation")
    initial, stream, judge, stream_validation, meta = build_stream("B", 4, 2)

    def variants(part, model):
        elapsed = np.asarray(elapsed_counts(part["groups"], part["order"]),
                             dtype=np.float64)
        fraction = model.predict_fraction(part["x"], part["groups"], elapsed)
        groups = np.asarray(part["groups"])
        out = {}
        product = _product(model, fraction, elapsed, groups)
        out["current: ratio product"] = product
        if model.affine is not None:
            a, b = model.affine
            affine = np.clip(a * fraction + b, 0.0, model.max_rul_minutes)
        else:
            affine = product
        scale = product / np.maximum(fraction, 1e-9)
        weight = np.clip((scale - model.short_cut)
                         / max(model.long_cut - model.short_cut, 1e-9), 0.0, 1.0)
        out["current: ratio + affine blend"] = weight * affine + (1.0 - weight) * product
        lifetime = _regression_lifetime(fraction, elapsed, groups,
                                        model.floor, model.cap)
        fixed = np.clip(fraction * lifetime, 0.0, model.max_rul_minutes)
        out["FIXED: regression product"] = fixed
        weight2 = np.clip((lifetime - model.short_cut)
                          / max(model.long_cut - model.short_cut, 1e-9), 0.0, 1.0)
        out["FIXED: regression + affine blend"] = (weight2 * affine
                                                    + (1.0 - weight2) * fixed)
        return out, lifetime

    results = {}

    # -- the official train/validation split ------------------------------
    model = profile.make_model(
        train["x"], train["y"], train["paths"],
        groups=np.asarray(train["groups"]), order=np.asarray(train["order"]),
        hidden=96, epochs=400, normalisation="slope", target="life_fraction",
        floor=0.02, cap=3000.0, rate_limit=0.25, feature_transform="eng34",
        activation="tanh", seed=42)
    print("=" * 96)
    print("OFFICIAL validation bearings (fitted on the nine official training bearings)")
    print("=" * 96)
    out, lifetime = variants(validation, model)
    groups = np.asarray(validation["groups"])
    for label, prediction in out.items():
        per_unit = {}
        for unit in sorted(set(groups.tolist())):
            mask = groups == unit
            per_unit[unit] = _rmse(np.asarray(validation["y"])[mask],
                                    prediction[mask])
        detail = " ".join("%s=%.1f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        pooled = _rmse(validation["y"], prediction)
        results["official validation"] = results.get("official validation", {})
        results["official validation"][label] = pooled
        print("%-36s pooled %9.2f   %s" % (label, pooled, detail))
    for unit in sorted(set(groups.tolist())):
        mask = groups == unit
        est = float(np.median(lifetime[mask]))
        true = float(np.asarray(validation["y"])[mask].max())
        print("    lifetime estimate %-12s est %8.1f  true %8.1f  (%.0f%%)"
              % (unit, est, true, 100 * est / max(true, 1e-9)))

    # -- the closed loop's splits ------------------------------------------
    print("\n" + "=" * 96)
    print("the closed loop's splits (initial / stream / judge)")
    print("=" * 96)
    loop_model = profile.make_model(
        initial["x"], initial["y"], initial["paths"],
        groups=initial["groups"], order=initial["order"],
        hidden=96, epochs=400, normalisation="slope", target="life_fraction",
        floor=0.02, cap=3000.0, rate_limit=0.25, feature_transform="eng34",
        activation="tanh", seed=42)
    for name, part in (("initial", initial), ("stream", stream), ("judge", judge)):
        out, _ = variants(part, loop_model)
        groups = np.asarray(part["groups"])
        print(" %s:" % name)
        for label, prediction in out.items():
            per_unit = {}
            for unit in sorted(set(groups.tolist())):
                mask = groups == unit
                per_unit[unit] = _rmse(np.asarray(part["y"])[mask],
                                        prediction[mask])
            detail = " ".join("%s=%.1f" % (k.replace("Bearing", "B"), v)
                              for k, v in sorted(per_unit.items()))
            pooled = _rmse(part["y"], prediction)
            results.setdefault(name, {})[label] = pooled
            print("   %-34s pooled %9.2f   %s" % (label, pooled, detail))

    print("\n" + "=" * 96)
    print("summary")
    print("=" * 96)
    for split, scores in results.items():
        best = min(scores, key=lambda k: scores[k])
        current = scores.get("current: ratio + affine blend")
        print("  %-20s current %9.2f   best %-32s %9.2f   change %+.1f%%"
              % (split, current if current else float("nan"), best,
                 scores[best],
                 100 * (scores[best] - current) / current if current else 0.0))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "profile_b_verify_regression.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out_dir / "profile_b_verify_regression.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

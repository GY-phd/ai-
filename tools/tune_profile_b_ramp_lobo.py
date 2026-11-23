# -*- coding: utf-8 -*-
"""Choose Profile B's blend ramp under leave-one-bearing-out.

Summary of what the diagnosis established
------------------------------------------
Profile B is **not** overfitting in the classical sense.  The training
RMSE (842) is *worse* than the held-out RMSE (442), and the gap does
not move when the capacity or the schedule grows.  What looks like
overfitting is a **miscalibrated blend**:

* the **product** ``RUL = f * s`` estimates the bearing's own lifetime
  and is excellent on 7 of 9 held-out bearings, by 2.8x to 14.8x;
* the **affine map** ``RUL = a * f + b`` is fitted on the training
  population and only wins on the two *longest* bearings, by 1.4x-1.5x;
* the current ramp mixes the affine map into bearings that should
  trust the product, and the affine map is three to four times worse
  there.

Profile B's metric is a **pooled** RMSE over all rows
(``tracks/task_metrics.py``), so a bearing's rows count in proportion to
its length - which means the two long bearings carry most of the
weight, and the ramp has to be right for them too.

This script sweeps the ramp's two thresholds over the leave-one-
bearing-out protocol - every one of the nine training bearings held out
in turn, the model fitted on the other eight - and reports:

* the pooled RMSE over all nine held-out bearings (the official metric),
* the RMSE on each held-out bearing separately, so a ramp that only
  wins on the long bearings is visible as such.

Run::

    python tools/tune_profile_b_ramp_lobo.py
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


def _parts(model, x, y, groups, order):
    """The fraction, the lifetime estimate, the product and the affine map."""
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
    product = fraction * scale
    if model.affine is None:
        affine = product
    else:
        a, b = model.affine
        affine = np.clip(a * fraction + b, 0.0, model.max_rul_minutes)
    return fraction, scale, product, affine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("B")
    train = profile.load("train")
    groups = np.asarray(train["groups"])
    order = np.asarray(train["order"])
    y = np.asarray(train["y"], dtype=np.float64)
    x = train["x"]
    units = sorted(set(groups.tolist()))

    # Fold once: fit each of the nine leave-one-out models and cache
    # the components of the held-out bearing.
    folds = []
    for held_out in units:
        keep = groups != held_out
        model = profile.make_model(
            x[keep], y[keep], train["paths"], groups=groups[keep],
            order=order[keep],
            hidden=96, epochs=400, normalisation="slope",
            target="life_fraction", floor=0.02, cap=3000.0,
            rate_limit=0.25, feature_transform="eng34",
            activation="tanh", seed=42)
        held = ~keep
        fraction, scale, product, affine = _parts(
            model, x[held], y[held], groups[held], order[held])
        folds.append({
            "bearing": held_out,
            "rows": int(held.sum()),
            "truth": y[held],
            "scale": scale,
            "product": product,
            "affine": affine,
            "short_cut": model.short_cut,
            "long_cut": model.long_cut,
        })
        print("  fitted fold, held out %-12s rows %5d  (short_cut %.1f long_cut %.1f)"
              % (held_out, held.sum(), model.short_cut, model.long_cut), flush=True)

    def evaluate(short_frac, long_frac):
        """Score a ramp whose thresholds are a fraction of the training
        population's largest estimated lifetime."""
        total = 0.0
        total_rows = 0
        per_unit = {}
        for fold in folds:
            largest = fold["short_cut"] / 0.08  # the code's own 8% rule
            short = largest * short_frac
            long = largest * long_frac
            weight = np.clip((fold["scale"] - short)
                             / max(long - short, 1e-9), 0.0, 1.0)
            prediction = weight * fold["affine"] + (1.0 - weight) * fold["product"]
            error = _rmse(fold["truth"], prediction)
            per_unit[fold["bearing"]] = error
            total += error ** 2 * fold["rows"]
            total_rows += fold["rows"]
        return (total / total_rows) ** 0.5, per_unit

    print("\n" + "=" * 92)
    print("the ramp, scored by pooled RMSE over all nine held-out bearings")
    print("=" * 92)
    print("%-30s %12s   %s" % ("ramp", "pooled rmse", "per held-out bearing"))
    print("-" * 92)

    candidates = [("product only", None, None), ("affine only", -1e18, 1e18)]
    for short, long in ((0.08, 0.25), (0.13, 0.41), (0.20, 0.50),
                        (0.30, 0.60), (0.40, 0.80), (0.50, 1.00),
                        (0.60, 1.20), (0.80, 1.60), (1.00, 2.00)):
        candidates.append(("%.2f / %.2f of largest" % (short, long), short, long))

    results = []
    for label, short, long in candidates:
        if short is None:
            # product only / affine only, handled separately
            total = 0.0
            total_rows = 0
            per_unit = {}
            for fold in folds:
                prediction = fold["product"] if short is None else fold["affine"]
                error = _rmse(fold["truth"], prediction)
                per_unit[fold["bearing"]] = error
                total += error ** 2 * fold["rows"]
                total_rows += fold["rows"]
            pooled = (total / total_rows) ** 0.5
        else:
            pooled, per_unit = evaluate(short, long)
        results.append({"label": label, "short": short, "long": long,
                        "pooled": pooled, "per_unit": per_unit})
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-30s %12.2f   %s" % (label, pooled, detail), flush=True)

    results.sort(key=lambda r: r["pooled"])
    print("\n" + "=" * 92)
    print("ranked")
    print("=" * 92)
    for row in results:
        print("  %-30s %12.2f" % (row["label"], row["pooled"]))

    current = next(r for r in results if r["label"].startswith("0.13"))
    best = results[0]
    print("\ncurrent ramp:  pooled %8.2f" % current["pooled"])
    print("best:         %-30s pooled %8.2f" % (best["label"], best["pooled"]))
    print("change:       %+.2f (%.1f%%)"
          % (best["pooled"] - current["pooled"],
             100 * (best["pooled"] - current["pooled"]) / current["pooled"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_ramp_lobo.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_ramp_lobo.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

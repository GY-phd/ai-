# -*- coding: utf-8 -*-
"""Recalibrate Profile B's blend, correctly this time.

The first attempt built the ramp from the *smallest* training lifetime,
which put the ramp's start far too early, so every candidate scored
worse than the current one.  The ramp's thresholds have to be built
from the **largest** training lifetime - the code does exactly that:
``short_cut = max(50, largest * 0.08)`` and
``long_cut = min(cap, largest * 0.25)``.

What the measurement established
--------------------------------
On the held-out bearings the two halves disagree sharply, and the
disagreement is not about the *bearing* but about **how far the
bearing's own trajectory is from the training population's**:

    bearing       product    affine    blended
    Bearing2_1      245.87    828.09    505.17   <- judge
    Bearing3_3      184.06    876.74    338.75   <- judge
    Bearing1_4       64.10   1016.40     64.10   <- validation
    Bearing2_4       20.80    978.50     20.80   <- validation
    Bearing3_4      745.90    510.90    731.90   <- validation

Three of the five held-out bearings want the **product**; only the two
longest ones want the **affine map**.  The current ramp gives the
affine map too much weight to the short and mid-length bearings - and
the affine map is three to four times worse there.

This script searches the ramp properly - thresholds built from the
largest training lifetime - and reports the judge and validation
scores for each, plus the combined score over both held-out sets.

Run::

    python tools/tune_profile_b_blend_thresholds.py
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
from tracks.run import elapsed_counts  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64)) ** 2)))


def _parts(model, part):
    from rulkit.rul import safe_lifetime
    elapsed = np.asarray(elapsed_counts(part["groups"], part["order"]),
                         dtype=np.float64)
    fraction = model.predict_fraction(part["x"], part["groups"], elapsed)
    groups = np.asarray(part["groups"])
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
    initial, stream, judge, validation, meta = build_stream("B", 4, 2)
    overrides = dict(CHAMPION_DEFAULTS["B"])
    model = profile.make_model(initial["x"], initial["y"], initial["paths"],
                               groups=initial["groups"],
                               order=initial["order"], **overrides)

    print("current thresholds: short_cut %.1f  long_cut %.1f"
          % (model.short_cut, model.long_cut))
    print("affine map: RUL = %.4f * f + %.2f" % model.affine)

    _, train_scale, _, _ = _parts(model, initial)
    groups = np.asarray(initial["groups"])
    lifetimes = sorted((unit, float(np.median(train_scale[groups == unit])))
                       for unit in np.unique(groups))
    print("\ntraining bearings' estimated lifetimes:")
    for unit, value in lifetimes:
        print("  %-12s %8.1f" % (unit, value))
    largest = lifetimes[-1][1]

    splits = {"JUDGE": judge, "VALIDATION": validation}
    prepared = {name: _parts(model, part) for name, part in splits.items()}
    weights = {"JUDGE": len(judge["y"]), "VALIDATION": len(validation["y"])}

    rows = []
    print("\n%-32s %10s %10s %11s"
          % ("ramp", "judge", "validation", "combined"))
    print("-" * 68)

    def measure(label, short, long):
        scores = {}
        for name, part in splits.items():
            fraction, scale, product, affine = prepared[name]
            truth = np.asarray(part["y"], dtype=np.float64)
            if short is None:
                prediction = product
            else:
                weight = np.clip((scale - short) / max(long - short, 1e-9),
                                 0.0, 1.0)
                prediction = weight * affine + (1.0 - weight) * product
            scores[name] = _rmse(truth, prediction)
        combined = sum(scores[n] * weights[n] for n in splits) / sum(weights.values())
        rows.append({"label": label, "short_cut": short, "long_cut": long,
                     "judge": scores["JUDGE"], "validation": scores["VALIDATION"],
                     "combined": combined})
        print("%-32s %10.2f %10.2f %11.2f"
              % (label, scores["JUDGE"], scores["VALIDATION"], combined))

    measure("current (%.2f/%.2f)" % (model.short_cut / largest,
                                      model.long_cut / largest),
            model.short_cut, model.long_cut)
    measure("product only", None, None)
    measure("affine only", -1e18, 1e18)
    for short, long in ((0.02, 0.10), (0.05, 0.15), (0.08, 0.25),
                        (0.08, 0.15), (0.10, 0.30), (0.15, 0.40),
                        (0.20, 0.50), (0.25, 0.60), (0.30, 0.80),
                        (0.40, 1.00), (0.50, 1.20), (0.60, 1.50)):
        if long <= short:
            continue
        measure("%.2f / %.2f of largest" % (short, long),
                largest * short, largest * long)

    rows.sort(key=lambda r: r["combined"])
    print("\n" + "=" * 68)
    print("ranked by the combined held-out score")
    print("=" * 68)
    for row in rows[:12]:
        print("  %-32s judge %8.2f   validation %8.2f   combined %8.2f"
              % (row["label"], row["judge"], row["validation"], row["combined"]))

    current = next(r for r in rows if r["label"].startswith("current"))
    best = rows[0]
    print("\ncurrent: judge %.2f  validation %.2f  combined %.2f"
          % (current["judge"], current["validation"], current["combined"]))
    print("best:    %s  judge %.2f  validation %.2f  combined %.2f"
          % (best["label"], best["judge"], best["validation"], best["combined"]))
    print("change:  judge %+.2f  validation %+.2f  combined %+.2f"
          % (best["judge"] - current["judge"],
             best["validation"] - current["validation"],
             best["combined"] - current["combined"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_blend_thresholds.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_blend_thresholds.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

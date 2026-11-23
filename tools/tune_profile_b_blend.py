# -*- coding: utf-8 -*-
"""Does Profile B's blend actually help, or does it hurt?

The decomposition showed the two halves of the two-stage estimate
disagree sharply on the held-out bearings:

    bearing        product RMSE    affine RMSE    blended RMSE
    Bearing2_1           245.87        828.09         505.17
    Bearing3_3           184.06        876.74         338.75

The blend is **worse than the product alone** on both of them.  The
blend weight is a ramp in the *estimated* lifetime, and the estimate is
too generous on a bearing it has never seen - so a bearing that should
trust the product is handed a third of the affine map, and the affine
map is three to four times worse.

This script measures the alternatives under the leave-one-bearing-out
protocol, which is the protocol the competition's rules describe for
this profile:

* the **product** alone,
* the **affine map** alone,
* the current blend,
* the product with the lifetime estimate **clipped** to the training
  population's range, and
* the product with a **per-unit** correction, so a bearing whose own
  trajectory disagrees with the training population is pulled back
  towards the product rather than towards the global map.

Every number here is measured on bearings the model never saw.

Run::

    python tools/tune_profile_b_blend.py
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


def _components(model, part):
    """The fraction, the product and the affine map, for one split."""
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
    weight = np.clip((scale - model.short_cut)
                     / max(model.long_cut - model.short_cut, 1e-9), 0.0, 1.0)
    blend = weight * affine + (1.0 - weight) * product
    return fraction, scale, product, affine, blend, weight


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

    # The training bearings' lifetime range - what the affine map was
    # fitted for.
    _, _, _, _, _, _ = _components(model, initial)
    train_fraction, train_scale, _, _, _, _ = _components(model, initial)
    train_lifetimes = []
    groups = np.asarray(initial["groups"])
    elapsed_train = np.asarray(elapsed_counts(initial["groups"], initial["order"]),
                               dtype=np.float64)
    for unit in np.unique(groups):
        mask = groups == unit
        train_lifetimes.append(float(np.median(train_scale[mask])))
    low, high = min(train_lifetimes), max(train_lifetimes)
    print("training bearings' estimated lifetimes: %s"
          % ", ".join("%.1f" % v for v in sorted(train_lifetimes)))
    print("range: %.1f .. %.1f" % (low, high))

    rows = []
    for name, part in (("TRAIN", initial), ("JUDGE", judge),
                       ("VALIDATION", validation)):
        fraction, scale, product, affine, blend, weight = _components(model, part)
        truth = np.asarray(part["y"], dtype=np.float64)
        groups = np.asarray(part["groups"])

        # The product with the lifetime estimate clipped to the
        # training population's range.
        clipped = fraction * np.clip(scale, low, high)

        # The product with the lifetime estimate clipped to the
        # *training* range, and the blend weight forced to zero
        # whenever the estimate is outside it - a bearing whose own
        # trajectory says it is far from the training population is
        # exactly the one the affine map cannot be trusted on.
        guard_weight = np.where((scale < low) | (scale > high), 0.0, weight)
        guarded = guard_weight * affine + (1.0 - guard_weight) * product

        variants = {
            "product": product,
            "affine": affine,
            "blend (current)": blend,
            "product + clipped lifetime": clipped,
            "blend + lifetime guard": guarded,
        }
        for label, prediction in variants.items():
            per_unit = {}
            for unit in sorted(set(groups.tolist())):
                mask = groups == unit
                per_unit[unit] = _rmse(truth[mask], prediction[mask])
            rows.append({
                "split": name,
                "variant": label,
                "rmse": _rmse(truth, prediction),
                "per_unit": per_unit,
            })
            print("%-11s %-26s rmse %8.2f   per-unit %s"
                  % (name, label, _rmse(truth, prediction),
                     " ".join("%s=%.1f" % (u.replace("Bearing", "B"), v)
                              for u, v in sorted(per_unit.items()))),
                  flush=True)
        print()

    print("=" * 84)
    print("the judge bearings - the ones the loop gates on")
    print("=" * 84)
    judge_rows = [r for r in rows if r["split"] == "JUDGE"]
    judge_rows.sort(key=lambda r: r["rmse"])
    for row in judge_rows:
        print("  %-28s %8.2f" % (row["variant"], row["rmse"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_blend.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_blend.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

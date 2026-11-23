# -*- coding: utf-8 -*-
"""Choose Profile B's blend under the leave-one-bearing-out protocol.

Why this script exists
----------------------
Profile B's two-stage estimate blends two estimators:

* the **product** ``RUL = f * s``, which estimates the bearing's own
  lifetime from its degradation trajectory, and
* the **affine map** ``RUL = a * f + b``, a global linear correction
  fitted on the training bearings.

They disagree sharply, and the disagreement depends on the bearing:

    bearing       product    affine    blended
    Bearing2_1      245.87    828.09    505.17   (judge)
    Bearing3_3      184.06    876.74    338.75   (judge)
    Bearing1_4       64.10   1016.40     64.10   (validation)
    Bearing2_4       20.80    978.50     20.80   (validation)
    Bearing3_4      745.90    510.90    731.90   (validation)

The judge bearings want the **product**; the long validation bearing
wants the **affine map**.  Choosing between them on either split alone
is choosing on one population - and the two splits are structurally
different (the judge's bearings live 370-490 minutes, the validation
set is 90% one 1514-minute bearing).

So the ramp is chosen on the protocol the competition's rules describe
for this profile: **leave-one-bearing-out** over all nine training
bearings, every one of them held out in turn.  A ramp that is only
right for the short bearings is not a ramp, it is a coincidence.

Run::

    python tools/tune_profile_b_lobo.py
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
    print("leave-one-bearing-out over %d training bearings" % len(units),
          flush=True)

    # For each held-out bearing, the model is fitted on the other eight
    # and scored on the one left out.  That is the protocol the
    # competition's rules describe for this profile.
    rows = []
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
        rows.append({
            "bearing": held_out,
            "rows": int(held.sum()),
            "true_median": float(np.median(y[held])),
            "product_rmse": _rmse(y[held], product),
            "affine_rmse": _rmse(y[held], affine),
        })
        print("  %-12s n=%4d  true_med %7.1f   product %8.2f   affine %8.2f"
              % (held_out, held.sum(), np.median(y[held]),
                 _rmse(y[held], product), _rmse(y[held], affine)), flush=True)

    print("\n" + "=" * 82)
    print("which half wins, per held-out bearing?")
    print("=" * 82)
    product_wins = 0
    for row in rows:
        better = "product" if row["product_rmse"] < row["affine_rmse"] else "affine"
        if better == "product":
            product_wins += 1
        print("  %-12s true_med %7.1f   product %8.2f   affine %8.2f   -> %s wins (%.1fx)"
              % (row["bearing"], row["true_median"], row["product_rmse"],
                 row["affine_rmse"], better,
                 max(row["product_rmse"], row["affine_rmse"])
                 / max(min(row["product_rmse"], row["affine_rmse"]), 1e-9)))
    print("\n  the product wins on %d of %d held-out bearings"
          % (product_wins, len(rows)))

    total_rows = sum(r["rows"] for r in rows)
    pooled_product = (sum(r["product_rmse"] ** 2 * r["rows"] for r in rows)
                      / total_rows) ** 0.5
    pooled_affine = (sum(r["affine_rmse"] ** 2 * r["rows"] for r in rows)
                     / total_rows) ** 0.5
    print("\n  pooled over all held-out bearings:")
    print("    product  RMSE %8.2f" % pooled_product)
    print("    affine   RMSE %8.2f" % pooled_affine)
    print("    -> the %s is better by %.1f%%"
          % ("product" if pooled_product < pooled_affine else "affine",
             100 * abs(pooled_affine - pooled_product) / max(pooled_product, pooled_affine)))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_lobo.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_lobo.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

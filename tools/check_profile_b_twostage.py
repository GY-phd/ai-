# -*- coding: utf-8 -*-
"""Which half of Profile B's two-stage estimate is wrong?

Profile B turns a predicted **fraction** of life remaining into minutes
in two stages, and the two halves can fail independently:

* the **product** ``RUL = f * s`` estimates the bearing's total
  lifetime ``s`` from its own degradation trajectory, and
* the **affine map** ``RUL = a * f + b`` is a global linear correction
  fitted by least squares **on the training bearings**.

The blend weight is decided by the estimated lifetime: a short-lived
bearing trusts the product, a long-lived one trusts the affine map.

The measurement above showed the model **under-predicting** the training
bearings by 33-71% and **over-predicting** the held-out ones by 42-71%.
That sign flip is the fingerprint of a global correction fitted on one
population and applied to another - so this script scores each half on
its own, per bearing, and shows where the error actually lives.

Run::

    python tools/check_profile_b_twostage.py
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

    rows = []
    for name, part in (("TRAIN", initial), ("JUDGE", judge)):
        elapsed = elapsed_counts(part["groups"], part["order"])
        fraction = model.predict_fraction(part["x"], part["groups"], elapsed)
        groups = np.asarray(part["groups"])
        elapsed = np.asarray(elapsed, dtype=np.float64)

        # -- the product half ------------------------------------------------
        scale = np.zeros(len(fraction), dtype=np.float64)
        from rulkit.rul import safe_lifetime
        for unit in np.unique(groups):
            indices = np.nonzero(groups == unit)[0]
            ordered = indices[np.argsort(elapsed[indices])]
            raw = safe_lifetime(fraction[ordered], elapsed[ordered],
                                floor=model.floor, cap=model.cap,
                                rate_limit=model.rate_limit)
            scale[ordered] = model._smooth(raw)
        product = fraction * scale

        # -- the affine half -------------------------------------------------
        if model.affine is None:
            affine = None
        else:
            a, b = model.affine
            affine = np.clip(a * fraction + b, 0.0, model.max_rul_minutes)

        # -- the blend -------------------------------------------------------
        blended = model.predict(part["x"], part["groups"], elapsed)

        for unit in sorted(set(groups.tolist())):
            mask = groups == unit
            truth = np.asarray(part["y"], dtype=np.float64)[mask]
            row = {
                "split": name,
                "bearing": unit,
                "rows": int(mask.sum()),
                "true_median": float(np.median(truth)),
                "blended_rmse": _rmse(truth, blended[mask]),
                "blended_bias": float(np.mean(blended[mask] - truth)),
                "product_rmse": _rmse(truth, product[mask]),
                "product_bias": float(np.mean(product[mask] - truth)),
                "estimated_lifetime": float(np.median(scale[mask])),
            }
            if affine is not None:
                row["affine_rmse"] = _rmse(truth, affine[mask])
                row["affine_bias"] = float(np.mean(affine[mask] - truth))
            # The blend weight this bearing actually got.
            weight = np.clip((scale[mask] - model.short_cut)
                             / max(model.long_cut - model.short_cut, 1e-9),
                             0.0, 1.0)
            row["blend_weight"] = float(np.median(weight))
            rows.append(row)

    header = ("%-6s %-12s %5s %9s %9s %9s %9s %9s %7s"
              % ("split", "bearing", "rows", "true_med", "est_life",
                 "blend", "product", "affine", "weight"))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("%-6s %-12s %5d %9.1f %9.1f %9.2f %9.2f %9s %7.2f"
              % (row["split"], row["bearing"], row["rows"],
                 row["true_median"], row["estimated_lifetime"],
                 row["blended_rmse"], row["product_rmse"],
                 ("%.2f" % row["affine_rmse"]) if "affine_rmse" in row else "-",
                 row["blend_weight"]))

    print("\n" + "=" * 92)
    print("the bias of each half, per bearing")
    print("=" * 92)
    for row in rows:
        print("%-6s %-12s  blend %+8.1f   product %+8.1f   affine %+8s"
              % (row["split"], row["bearing"], row["blended_bias"],
                 row["product_bias"],
                 ("%+8.1f" % row["affine_bias"]) if "affine_bias" in row else "-"))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_twostage.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_twostage.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

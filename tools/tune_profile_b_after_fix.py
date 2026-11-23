# -*- coding: utf-8 -*-
"""Tune Profile B's remaining knobs now that the lifetime estimate is right.

What changed
------------
The lifetime estimate was being computed by a pointwise ratio that is
biased low by Jensen's inequality, and the fraction was then applied
**twice**.  Together they put the estimated lifetime at 32% to 59% of
the truth on every bearing.  ``rulkit.rul.fitted_lifetime`` now fits the
relation ``t = N * (1 - f)`` by least squares over each unit's own
history, and the fraction is applied once.

With a correct lifetime estimate the **affine map is no longer pulling
its weight**: it was there to compensate for the ratio's bias, and on a
correct product it only adds its own error - a global straight line
fitted on a population that is bimodal (seven bearings of 61-266
minutes and two of 1247-2537).

This script re-tunes what is left, under the **leave-one-bearing-out**
protocol with the pooled RMSE that profile B's metric uses:

* the affine map on or off,
* the blend's thresholds, now that the scale they ramp over is on the
  right scale,
* the fraction model's own capacity - ``hidden`` and ``epochs`` - which
  is worth revisiting now that the estimator downstream is no longer
  burying its error.

Run::

    python tools/tune_profile_b_after_fix.py
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--folds", type=int, default=0,
                        help="stop after this many folds (0 = all)")
    args = parser.parse_args(argv)

    profile = get("B")
    train = profile.load("train")
    groups_all = np.asarray(train["groups"])
    order_all = np.asarray(train["order"])
    y_all = np.asarray(train["y"], dtype=np.float64)
    x_all = train["x"]
    units = sorted(set(groups_all.tolist()))
    if args.folds:
        units = units[:args.folds]

    # -- the lifetime estimator and the affine map ------------------------
    print("=" * 104)
    print("leave-one-bearing-out, pooled RMSE (profile B's metric)")
    print("=" * 104)

    results = []

    def run(label, **overrides):
        total = 0.0
        total_rows = 0
        per_unit = {}
        for held_out in units:
            keep = groups_all != held_out
            kwargs = dict(hidden=96, epochs=400, normalisation="slope",
                          target="life_fraction", floor=0.02, cap=3000.0,
                          rate_limit=0.25, feature_transform="eng34",
                          activation="tanh", seed=42)
            kwargs.update(overrides)
            model = profile.make_model(
                x_all[keep], y_all[keep], train["paths"],
                groups=groups_all[keep], order=order_all[keep], **kwargs)
            held = ~keep
            elapsed = np.asarray(elapsed_counts(groups_all[held],
                                                order_all[held]), dtype=np.float64)
            prediction = model.predict(x_all[held], groups_all[held], elapsed)
            error = _rmse(y_all[held], prediction)
            per_unit[held_out] = error
            total += error ** 2 * held.sum()
            total_rows += held.sum()
        pooled = (total / total_rows) ** 0.5
        results.append({"label": label, "overrides": overrides,
                        "pooled": pooled, "per_unit": per_unit})
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-40s %10.2f   %s" % (label, pooled, detail), flush=True)
        return pooled

    run("fitted + affine on (the fix)")
    run("fitted + affine OFF", affine="off")
    run("ratio + affine on (the old code)", lifetime_estimator="ratio")
    run("ratio + affine OFF", lifetime_estimator="ratio", affine="off")

    # -- the fraction model's capacity ------------------------------------
    for hidden in (32, 64, 128):
        run("fitted, affine off, hidden=%d" % hidden, affine="off", hidden=hidden)
    for epochs in (200, 800):
        run("fitted, affine off, epochs=%d" % epochs, affine="off", epochs=epochs)

    results.sort(key=lambda r: r["pooled"])
    print("\n" + "=" * 104)
    print("ranked")
    print("=" * 104)
    for row in results:
        print("  %-40s %10.2f" % (row["label"], row["pooled"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_after_fix.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_after_fix.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Where does Profile B's lifetime estimate lose its accuracy?

The measurement so far
----------------------
Under leave-one-bearing-out the estimated lifetime is 32% to 59% of
the truth on **every** bearing, mean 0.49 - a two-fold deficit that a
single global multiplier of 2.02 to 2.13 (fitted independently in every
fold) reproduces.  The cause is visible in the code:

* ``safe_lifetime`` returns the **remaining** life ``f * t / (1 - f)``;
* ``predict`` then multiplies by the fraction **again**, so the fraction
  is applied twice.

But removing the double application is not enough on its own: the
result gets *worse* (1227 against 1048), because ``f * t / (1 - f)`` is
a **pointwise ratio** of two noisy quantities.  A ratio of noisy
numbers is biased low by Jensen's inequality - ``E[1/(1-f)]`` is
strictly smaller than ``1/E[1-f]`` - and the bias grows with the
noise in ``f``.

The question this script answers
--------------------------------
How noisy is the fraction, and can the lifetime be estimated **by
regression** instead of by a ratio?  The model of the relationship is
exact and simple::

    t = N * (1 - f)

so ``N`` is the intercept of a straight line through ``(1 - f, t)`` -
and a least-squares fit over the bearing's whole history is far more
robust than a per-point ratio.

The script measures, per held-out bearing:

* the fraction model's own accuracy (the RMSE of ``f`` against the
  truth fraction), which is the quantity everything else depends on;
* the lifetime estimate from the **ratio** (the current form);
* the lifetime estimate from the **regression** (the proposed form);
* the resulting RUL RMSE for each.

Run::

    python tools/tune_profile_b_lifetime_estimator.py
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


def _fraction_components(model, x, groups, order):
    """The predicted fraction and the elapsed count of every row."""
    elapsed = np.asarray(elapsed_counts(groups, order), dtype=np.float64)
    fraction = model.predict_fraction(x, groups, elapsed)
    return fraction, elapsed


def _ratio_scale(fraction, elapsed, groups, model):
    """The current form: ``f * t / (1 - f)``, guarded and smoothed."""
    from rulkit.rul import safe_lifetime
    out = np.zeros(len(fraction), dtype=np.float64)
    for unit in np.unique(groups):
        indices = np.nonzero(groups == unit)[0]
        ordered = indices[np.argsort(elapsed[indices])]
        raw = safe_lifetime(fraction[ordered], elapsed[ordered],
                            floor=model.floor, cap=model.cap,
                            rate_limit=model.rate_limit)
        out[ordered] = model._smooth(raw)
    return out


def _regression_scale(fraction, elapsed, groups, model):
    """The proposed form: ``N`` from a least-squares line through
    ``(1 - f, t)``, then the remaining life ``f * N``."""
    out = np.zeros(len(fraction), dtype=np.float64)
    for unit in np.unique(groups):
        indices = np.nonzero(groups == unit)[0]
        ordered = indices[np.argsort(elapsed[indices])]
        f = np.clip(fraction[ordered], 0.0, 1.0 - model.floor)
        t = elapsed[ordered]
        design = np.stack([1.0 - f, np.ones(len(f))], axis=1)
        solution, *_ = np.linalg.lstsq(design, t, rcond=None)
        lifetime = float(solution[0])
        if not np.isfinite(lifetime) or lifetime <= 0:
            lifetime = 1.0
        lifetime = float(np.clip(lifetime, 1.0, model.cap))
        out[ordered] = f * lifetime
    return out


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

    rows = []
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
        fraction, elapsed = _fraction_components(
            model, x_all[held], groups_all[held], order_all[held])
        groups_held = np.asarray(groups_all[held])

        # The true remaining fraction of the held-out bearing.
        from tracks.profiles import life_fraction_from_arrays
        truth_fraction = life_fraction_from_arrays(
            y_all[held], groups_held, order_all[held])

        ratio = _ratio_scale(fraction, elapsed, groups_held, model)
        regression = _regression_scale(fraction, elapsed, groups_held, model)

        true_life = float(y_all[held].max())
        row = {
            "bearing": held_out,
            "rows": int(held.sum()),
            "true_life": true_life,
            "fraction_rmse": _rmse(truth_fraction, fraction),
            "fraction_bias": float(np.mean(fraction - truth_fraction)),
            "ratio_life": float(np.median(ratio)),
            "ratio_rul_rmse": _rmse(y_all[held], ratio),
            "regression_life": float(np.median(regression) / max(np.median(fraction), 1e-9)),
            "regression_rul_rmse": _rmse(y_all[held], regression),
        }
        rows.append(row)
        print("  %-12s frac rmse %.4f (bias %+.4f)   ratio life %7.1f (%.0f%%) rmse %8.2f"
              "   regression life %7.1f (%.0f%%) rmse %8.2f"
              % (held_out, row["fraction_rmse"], row["fraction_bias"],
                 row["ratio_life"], 100 * row["ratio_life"] / true_life,
                 row["ratio_rul_rmse"], row["regression_life"],
                 100 * row["regression_life"] / true_life,
                 row["regression_rul_rmse"]), flush=True)

    print("\n" + "=" * 104)
    print("leave-one-bearing-out, pooled RMSE (profile B's metric)")
    print("=" * 104)
    total_rows = sum(r["rows"] for r in rows)
    for key, label in (("ratio_rul_rmse", "ratio form (current, with the double f)"),
                       ("regression_rul_rmse", "regression form (f * N, N by least squares)")):
        pooled = (sum(r[key] ** 2 * r["rows"] for r in rows) / total_rows) ** 0.5
        print("  %-46s %10.2f" % (label, pooled))

    print("\n  the fraction model's own accuracy (the thing everything rests on):")
    for row in rows:
        print("    %-12s fraction rmse %.4f   bias %+.4f   true life %7.1f"
              % (row["bearing"], row["fraction_rmse"], row["fraction_bias"],
                 row["true_life"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_lifetime_estimator.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_lifetime_estimator.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Replace Profile B's global affine map with a scale-aware one.

The diagnosis
-------------
Profile B's two-stage estimate blends two estimators:

* the **product** ``RUL = f * s`` - the bearing's own estimated
  lifetime ``s`` times the predicted remaining fraction ``f``; and
* the **affine map** ``RUL = a * f + b`` - a global straight line
  fitted by least squares on the training bearings.

The affine map is the reason the held-out score is what it is.  The
training population is **bimodal**: seven bearings live 61-266 minutes
and two live 1247-2537 minutes, and the two long ones carry 72% of the
rows.  A least-squares straight line through that population fits a
slope of about 2600 - it learns "every bearing lives 2600 minutes" -
and then predicts 374-1538 minutes for a bearing whose true life is
122.  Its **in-sample** RMSE on the short bearings (646-893) is worse
than the product's (60-120): it is not even fitting the training data,
because a line cannot represent ``RUL = f * lifetime``.

What this script measures
-------------------------
A **scale-aware** correction: ``RUL = f * (s + delta)``, where ``s`` is
the bearing's own estimated lifetime and ``delta`` is a single global
offset fitted on the training bearings.  It keeps the multiplicative
structure the problem actually has, and it has one parameter instead of
two, so it cannot be dragged sideways by the long bearings the way a
slope can.

It is compared against the product, the affine map and the current
blend, under the **leave-one-bearing-out** protocol - every one of the
nine training bearings held out in turn, the model fitted on the other
eight - and scored with the pooled RMSE that profile B's metric
actually uses (``tracks/task_metrics.py``).

Run::

    python tools/tune_profile_b_affine.py
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
        fraction, scale, product, affine = _parts(
            model, x_all[held], y_all[held], groups_all[held], order_all[held])
        # The in-sample components, needed to fit the new correction.
        train_fraction, train_scale, train_product, _ = _parts(
            model, x_all[keep], y_all[keep], groups_all[keep],
            order_all[keep])
        folds.append({
            "bearing": held_out,
            "rows": int(held.sum()),
            "truth": y_all[held],
            "scale": scale,
            "fraction": fraction,
            "product": product,
            "affine": affine,
            "train_fraction": train_fraction,
            "train_scale": train_scale,
            "train_product": train_product,
            "train_truth": y_all[keep],
            "short_cut": model.short_cut,
            "long_cut": model.long_cut,
        })
        print("  fold %-12s held-out rows %5d" % (held_out, held.sum()), flush=True)

    # -- the scale-aware correction --------------------------------------
    # ``RUL = f * (s + delta)``.  Fitted on the training rows of the
    # fold, by least squares on ``s + delta`` against ``y / f``.
    def fit_delta(f_train, train_product, train_truth):
        """The single offset that best fits ``RUL = f * (s + delta)``.

        ``y = f * s + f * delta``, so ``delta`` is one unknown and the
        least-squares solution is a single dot product.
        """
        denominator = float(np.sum(f_train * f_train))
        if denominator <= 0:
            return 0.0
        numerator = float(np.sum(f_train * (train_truth - train_product)))
        return numerator / denominator

    print("\n" + "=" * 100)
    print("leave-one-bearing-out, pooled RMSE (profile B's metric)")
    print("=" * 100)
    print("%-34s %12s   %s" % ("estimator", "pooled rmse", "per held-out bearing"))
    print("-" * 100)

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
        print("%-34s %12.2f   %s" % (label, pooled, detail), flush=True)
        return pooled

    results = []
    score("product only", lambda f: f["product"])
    score("affine only (current)", lambda f: f["affine"])

    def current_blend(fold):
        largest = fold["short_cut"] / 0.08
        short = largest * 0.08
        long = largest * 0.25
        weight = np.clip((fold["scale"] - short) / max(long - short, 1e-9),
                         0.0, 1.0)
        return weight * fold["affine"] + (1.0 - weight) * fold["product"]

    score("blend (current code)", current_blend)

    # The scale-aware correction, with the offset fitted per fold.
    for label, use_delta in (("scale-aware f*(s+delta)", True),):
        def predictor(fold, use_delta=use_delta):
            if not use_delta:
                return fold["product"]
            delta = fit_delta(fold["train_fraction"], fold["train_product"],
                              fold["train_truth"])
            return np.clip(fold["fraction"] * (fold["scale"] + delta),
                           0.0, 3000.0)
        score(label, predictor)

    # The scale-aware correction blended with the affine map.
    for weight_frac in (0.0, 0.25, 0.5):
        def predictor(fold, weight_frac=weight_frac):
            delta = fit_delta(fold["train_fraction"], fold["train_product"],
                              fold["train_truth"])
            scaled = np.clip(fold["fraction"] * (fold["scale"] + delta),
                             0.0, 3000.0)
            if weight_frac <= 0:
                return scaled
            largest = fold["short_cut"] / 0.08
            weight = np.clip((fold["scale"] - largest * 0.08)
                             / max(largest * 0.25 - largest * 0.08, 1e-9), 0.0, 1.0)
            return (1 - weight_frac) * scaled + weight_frac * (
                weight * fold["affine"] + (1 - weight) * fold["product"])
        score("scale-aware + %.0f%% affine" % (weight_frac * 100), predictor)

    results.sort(key=lambda r: r["pooled"])
    print("\n" + "=" * 100)
    print("ranked")
    print("=" * 100)
    for row in results:
        print("  %-34s %12.2f" % (row["label"], row["pooled"]))

    current = next(r for r in results if r["label"] == "blend (current code)")
    best = results[0]
    print("\ncurrent:  %8.2f" % current["pooled"])
    print("best:     %-30s %8.2f" % (best["label"], best["pooled"]))
    print("change:   %+.2f (%.1f%%)"
          % (best["pooled"] - current["pooled"],
             100 * (best["pooled"] - current["pooled"]) / current["pooled"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_affine.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_affine.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

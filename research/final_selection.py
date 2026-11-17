# -*- coding: utf-8 -*-
"""The final candidate: the configuration that goes into the submission.

Everything upstream of this script was a measurement.  This script is the
measurement that decides which configuration ships, and it is deliberately
the strictest protocol available:

* **leave-one-bearing-out** over all nine training bearings - every bearing
  is evaluated exactly once, and the public validation bearings are not
  used at all, so they remain an independent check afterwards;
* **three metrics**, because one number hides the failure mode that
  actually matters:
    - ``rmse_mean`` - the headline;
    - ``rmse_worst`` - the worst single bearing, because a configuration
      that collapses on one bearing will collapse on the hidden stream;
    - ``rmse_median`` - the typical bearing, which is what most of the
      deployed fleet looks like;
* **the official baseline under the same protocol**, so the comparison is
  like for like;
* the configuration is only accepted if it beats the baseline on the mean
  **and** does not regress on the worst bearing by more than 10%.

The candidate grid is small and every cell was chosen because a measurement
pointed at it, not because it was convenient to run.

Run:
    python -m research.final_selection --out artifacts/final_selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.protocol import bearing_groups, load_split, subset  # noqa: E402
from research.selfnorm_study import self_normalise  # noqa: E402
from research.two_stage import life_fraction  # noqa: E402
from rulkit.model import Model  # noqa: E402
from rulkit.rul import safe_lifetime  # noqa: E402


def evaluate_candidate(train_rows, eval_rows, config: dict,
                       feature_matrix_train: np.ndarray,
                       feature_matrix_eval: np.ndarray) -> dict:
    """Train one candidate under the chosen architecture and score it.

    Two architectures are supported, and they differ in the *target*:

    ``absolute``
        Regress the RUL in minutes directly, then clip at zero.  This is
        what the official baseline does.

    ``relative``
        Regress the fraction of life remaining (the transferable half),
        then multiply by a guarded estimate of the bearing's lifetime
        (the non-transferable half).  The guard is what stops a fraction
        prediction near 1 from producing an absurd scale.
    """
    started = time.time()
    target = config.get("target", "relative")

    y_train = train_rows["y"] if target == "absolute" else life_fraction(train_rows)
    model = Model("regression", train_rows["paths"], [], feature_matrix_train, y_train,
                  hidden=config["hidden"], seed=config.get("seed", 42),
                  learning_rate=config.get("learning_rate", 0.003),
                  batch_size=config.get("batch_size", 128),
                  feature_transform=config.get("feature_transform", "eng34"),
                  activation=config.get("activation", "tanh"),
                  target_scaling=False)
    model.fit(feature_matrix_train, y_train, config.get("epochs", 400),
              learning_rate=config.get("learning_rate", 0.003),
              weight_decay=config.get("weight_decay", 0.0))

    if target == "absolute":
        predicted = np.maximum(0.0, model.predict(feature_matrix_eval))
    else:
        # --- stage 1 output: the transferable part -------------------------
        fraction_eval = np.clip(model.predict(feature_matrix_eval), 0.0, 1.0)
        # --- stage 2: the non-transferable part, a guarded lifetime estimate -
        groups = bearing_groups(eval_rows)
        lifetime = np.zeros(len(eval_rows["y"]))
        for _, indices in sorted(groups.items()):
            order = indices[np.argsort(eval_rows["snapshot"][indices])]
            elapsed = np.arange(1, len(order) + 1, dtype=float)
            lifetime[order] = safe_lifetime(fraction_eval[order], elapsed,
                                            floor=config.get("floor", 0.02),
                                            cap=config.get("cap", 3000.0),
                                            rate_limit=config.get("rate_limit", 0.25))
        predicted = np.clip(fraction_eval * lifetime, 0.0, config.get("max_rul_minutes", 3000.0))

    error = predicted - eval_rows["y"]
    return {
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 3),
        "mae": round(float(np.mean(np.abs(error))), 3),
        "seconds": round(time.time() - started, 2),
        "parameter_count": int(model.meta["parameter_count"]),
    }


def main(argv=None) -> int:
    """Run the final selection and print the decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)

    # The feature variants that the measurements supported.
    variants = {
        "raw": np.asarray(rows["x"], dtype=np.float64),
        "slope": self_normalise(rows, "slope", 5),
    }

    # The candidates that the upstream studies pointed at.
    candidates = [
        {"name": "绝对分钟目标 (eng34)", "features": "raw", "target": "absolute",
         "hidden": 96, "epochs": 400, "weight_decay": 0.01, "feature_transform": "eng34",
         "learning_rate": 0.003, "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25},
        {"name": "两阶段 · 原始特征", "features": "raw", "target": "relative",
         "hidden": 96, "epochs": 400, "weight_decay": 0.01, "feature_transform": "eng34",
         "learning_rate": 0.003, "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25},
        {"name": "两阶段 · 变化率特征", "features": "slope", "target": "relative",
         "hidden": 96, "epochs": 400, "weight_decay": 0.01, "feature_transform": "eng34",
         "learning_rate": 0.003, "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25},
        {"name": "两阶段 · 变化率 · 宽", "features": "slope", "target": "relative",
         "hidden": 128, "epochs": 600, "weight_decay": 0.01, "feature_transform": "eng34",
         "learning_rate": 0.003, "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25},
        {"name": "两阶段 · 变化率 · 无归一化", "features": "slope", "target": "relative",
         "hidden": 96, "epochs": 400, "weight_decay": 0.01, "feature_transform": "raw",
         "learning_rate": 0.003, "floor": 0.02, "cap": 3000.0, "rate_limit": 0.25},
    ]

    results = {}
    print("=== final selection: leave-one-bearing-out over 9 bearings ===")
    print("  %-26s %9s %9s %9s %9s" % ("candidate", "mean", "median", "worst", "mae"))
    for candidate in candidates:
        matrix = variants[candidate["features"]]
        config = {k: v for k, v in candidate.items() if k not in ("name", "features", "target")}
        config.setdefault("seed", 42)
        config.setdefault("batch_size", 128)
        config.setdefault("activation", "tanh")

        rmses, maes, seconds = [], [], 0.0
        for held_out in sorted(groups):
            eval_indices = groups[held_out]
            train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                        for i in idx])
            scored = evaluate_candidate(subset(rows, train_indices), subset(rows, eval_indices),
                                        config, matrix[train_indices], matrix[eval_indices])
            rmses.append(scored["rmse"])
            maes.append(scored["mae"])
            seconds += scored["seconds"]

        rmses = np.asarray(rmses)
        results[candidate["name"]] = {
            "config": candidate,
            "folds": [{"bearing": b, "rmse": round(r, 3)}
                      for b, r in zip(sorted(groups), rmses)],
            "rmse_mean": round(float(rmses.mean()), 3),
            "rmse_median": round(float(np.median(rmses)), 3),
            "rmse_worst": round(float(rmses.max()), 3),
            "mae_mean": round(float(np.mean(maes)), 3),
            "train_seconds": round(seconds, 1),
        }
        print("  %-26s %9.2f %9.2f %9.2f %9.2f" % (
            candidate["name"], rmses.mean(), np.median(rmses), rmses.max(), np.mean(maes)))

    # --- the official baseline under the same protocol ---------------------
    # The shipped official checkpoint was trained on **all nine** training
    # bearings, so scoring it on those bearings measures memorisation, not
    # generalisation.  It is reported separately and clearly labelled; the
    # *fair* reference is an official-architecture model retrained inside the
    # same leave-one-bearing-out folds.
    sys.path.insert(0, str(Path(args.data) / "baselines"))
    try:
        from model import Model as OfficialModel  # type: ignore
    finally:
        sys.path.pop(0)

    official = OfficialModel.load(Path(args.data) / "checkpoint.npz")
    baseline_rmse = []
    for held_out in sorted(groups):
        indices = groups[held_out]
        predicted = official.predict(rows["x"][indices])
        error = predicted - rows["y"][indices]
        baseline_rmse.append(float(np.sqrt(np.mean(error ** 2))))
    baseline_rmse = np.asarray(baseline_rmse)
    print("\n  official checkpoint (trained on ALL 9 bearings - in-sample, not a fair reference):")
    print("    mean %.2f  median %.2f  worst %.2f" % (
        baseline_rmse.mean(), np.median(baseline_rmse), baseline_rmse.max()))
    results["official_checkpoint_in_sample"] = {
        "rmse_mean": round(float(baseline_rmse.mean()), 3),
        "rmse_median": round(float(np.median(baseline_rmse)), 3),
        "rmse_worst": round(float(baseline_rmse.max()), 3),
        "note": "the shipped checkpoint was trained on these very bearings; this is an "
                "in-sample measurement and is NOT a fair reference",
        "folds": [{"bearing": b, "rmse": round(r, 3)}
                  for b, r in zip(sorted(groups), baseline_rmse)],
    }

    # --- the FAIR reference: the official architecture, retrained per fold -
    print("\n  official architecture retrained inside each fold (hidden=24, 15 epochs):")
    fair_rmse, fair_mae = [], []
    for held_out in sorted(groups):
        eval_indices = groups[held_out]
        train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                    for i in idx])
        train_rows, eval_rows = subset(rows, train_indices), subset(rows, eval_indices)
        model = OfficialModel("regression", train_rows["paths"], [],
                              train_rows["x"], train_rows["y"],
                              hidden=24, seed=42, learning_rate=0.003, batch_size=128)
        model.fit(train_rows["x"], train_rows["y"], 15)
        predicted = model.predict(eval_rows["x"])
        error = predicted - eval_rows["y"]
        fair_rmse.append(float(np.sqrt(np.mean(error ** 2))))
        fair_mae.append(float(np.mean(np.abs(error))))
        print("    %-18s rmse=%8.2f" % (held_out, fair_rmse[-1]))
    fair_rmse = np.asarray(fair_rmse)
    print("    mean %.2f  median %.2f  worst %.2f" % (
        fair_rmse.mean(), np.median(fair_rmse), fair_rmse.max()))
    results["official_baseline"] = {
        "rmse_mean": round(float(fair_rmse.mean()), 3),
        "rmse_median": round(float(np.median(fair_rmse)), 3),
        "rmse_worst": round(float(fair_rmse.max()), 3),
        "mae_mean": round(float(np.mean(fair_mae)), 3),
        "note": "official architecture (hidden=24, 15 epochs) retrained inside each "
                "leave-one-bearing-out fold - the fair reference",
        "folds": [{"bearing": b, "rmse": round(r, 3)}
                  for b, r in zip(sorted(groups), fair_rmse)],
    }

    # --- the decision ------------------------------------------------------
    print("\n=== decision ===")
    # Sort by RMSE explicitly.  ``min`` over ``(name, dict)`` tuples would
    # compare the *name strings* first, which picks a candidate by its
    # label rather than by its score.
    ranked = sorted((v["rmse_mean"], k) for k, v in results.items()
                    if k not in ("official_baseline", "official_checkpoint_in_sample", "decision")
                    and "rmse_mean" in v)
    best_name, best = ranked[0][1], results[ranked[0][1]]
    baseline = results["official_baseline"]
    gain = (baseline["rmse_mean"] - best["rmse_mean"]) / baseline["rmse_mean"] * 100.0
    worst_ok = best["rmse_worst"] <= baseline["rmse_worst"] * 1.10
    accepted = gain > 0 and worst_ok
    print("  winner: %s" % best_name)
    print("    rmse mean   %.2f  (fair baseline %.2f,  %+.1f%%)" % (
        best["rmse_mean"], baseline["rmse_mean"], gain))
    print("    rmse median %.2f  (fair baseline %.2f)" % (best["rmse_median"], baseline["rmse_median"]))
    print("    rmse worst  %.2f  (fair baseline %.2f,  tolerance 10%%)" % (
        best["rmse_worst"], baseline["rmse_worst"]))
    print("    accepted: %s" % ("YES" if accepted else "NO"))
    results["decision"] = {"winner": best_name, "gain_pct": round(gain, 2),
                           "worst_bearing_ok": bool(worst_ok), "accepted": bool(accepted)}

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

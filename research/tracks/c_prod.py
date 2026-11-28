# -*- coding: utf-8 -*-
"""The production configuration, measured at the points the code uses.

What this adds to ``research/tracks/c_ship.py``
-------------------------------------------------
``c_ship.py`` measured the **joint effect of the epoch budget and the
learning rate** with the degradation weight held at ``deg0.5``, and
``research/tracks/c_tune.py`` swept 24 recipes including the weight -
but the two studies never crossed, so the combination that the
production code actually runs was **never measured**:

* ``c_tune.py`` measured ``alpha 0.25`` only at **40 epochs, 0.004**,
  and ``lr 0.008`` only at **80 epochs, alpha 0.5**.
* ``c_ship.py`` measured the production budgets at **alpha 0.5**.

The production code after the falsifier fix runs the degradation
hypothesis at **42 to 74 epochs** (``strategies.epoch_budget``) with
``learning_rate 0.008`` (``profile_c.learning_rate``) and the weight
at ``adapter.degradation_alpha()``.  This module measures that grid:
every epoch budget the production schedule can actually produce,
crossed with the learning rate and the weight.

Why this matters
----------------
The platform scores "资源效率" (resource efficiency) out of 5 and rated
it **0.000** on the 2026-11-25 run, with C's CPU at 7 996 s against
the v5 run's 7 860 s.  A configuration that is better but twice as
expensive would trade one objective-score line for another, so the
**CPU cost** is measured for every point here too, on the full public
training set, the way the platform's meter sees it.

The protocol
------------
Same as every other C study: leave-one-engine-out over the public
training engines, six folds, two seeds, and the shipped static
checkpoint as the ruler under the identical mask.  The pairing is at
the **fold** level, because the two seeds use the same folds and the
seed-level means are only two numbers.

Run::

    python -m research.tracks.c_prod --folds 6 --seeds 42,7
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tracks.c_holdout import (  # noqa: E402
    baseline_on, engine_folds, evaluate, train_on,
)
from research.tracks.c_protocol import load  # noqa: E402

#: The epoch budgets the production schedule can produce.
#:
#: ``strategies.epoch_budget`` gives the degradation hypothesis a base
#: of 85, a step of 7 and profile C's ``epoch_scale`` of 0.5, so the
#: ten rounds run 42, 46, 49, 53, 56, 60, 63, 67, 70 and 74 epochs.
#: The grid below samples that schedule: its first, middle and last
#: budgets, plus the two neighbours the schedule can produce around
#: them.
BUDGETS = (42, 53, 63, 74)

#: The learning rates the profile can declare.
RATES = (0.004, 0.008)

#: The degradation weights the profile can declare.
ALPHAS = (0.25, 0.5)


def candidates():
    """The grid the production code can actually produce."""
    rows = []
    for epochs in BUDGETS:
        for lr in RATES:
            for alpha in ALPHAS:
                rows.append(("ep%d_lr%g_a%g" % (epochs, lr, alpha),
                             epochs, lr, alpha))
    return rows


def cpu_cost(ctx, spec, epochs, lr):
    """Seconds to train on the **full** public training set.

    This is the number the platform's CPU meter sees for one round, so
    it is measured the way the round measures it: the whole training
    projection, the real batch size, and the real weight shape.
    """
    mask = np.ones(len(ctx["train"]["y"]), dtype=bool)
    started = time.monotonic()
    train_on(ctx, spec, mask, epochs=epochs, lr=lr, batch=256, seed=42)
    return time.monotonic() - started


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_prod.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    grid = candidates()
    print("public train engines %d -> %d folds; %d candidates"
          % (len(ctx["t_engines"]), args.folds, len(grid)), flush=True)

    rows = []
    per_fold = {}
    for name, epochs, lr, alpha in grid:
        spec = dict(w="deg", alpha=alpha)
        for seed in seeds:
            folds = engine_folds(ctx, args.folds, seed=seed)
            scores, bases = [], []
            started = time.time()
            for fold_index, held in enumerate(folds):
                held_mask = np.isin(ctx["tunit"], held)
                train_mask = ~held_mask
                model = train_on(ctx, spec, train_mask, epochs=epochs, lr=lr,
                                 batch=256, seed=seed)
                score = evaluate(ctx, model, held_mask)
                base = baseline_on(ctx, held_mask)
                scores.append(score)
                bases.append(base)
                per_fold.setdefault(name, []).append({
                    "seed": seed, "fold": fold_index,
                    "holdout": score["holdout_full_rmse"],
                    "baseline": base["holdout_full_rmse"],
                    "delta": score["holdout_full_rmse"] - base["holdout_full_rmse"],
                })
            agg = float(np.mean([s["holdout_full_rmse"] for s in scores]))
            bagg = float(np.mean([b["holdout_full_rmse"] for b in bases]))
            rows.append({
                "candidate": name, "epochs": epochs, "lr": lr, "alpha": alpha,
                "seed": seed, "holdout_rmse": agg, "baseline_rmse": bagg,
                "delta": agg - bagg,
                "wins": sum(1 for s, b in zip(scores, bases)
                            if s["holdout_full_rmse"] < b["holdout_full_rmse"]),
                "folds": len(folds),
                "seconds": round(time.time() - started, 1),
            })
            print("%-22s ep%-4d lr%-6g a%-4g s%-3d holdout %8.4f (base %8.4f, "
                  "%+7.4f) win %d/%d  %.0fs"
                  % (name, epochs, lr, alpha, seed, agg, bagg, agg - bagg,
                     rows[-1]["wins"], len(folds), rows[-1]["seconds"]), flush=True)

    # ---- the CPU cost, on the full training set ----------------------
    print("\n=== single-round training cost (full %d rows) ===" % len(ctx["train"]["y"]))
    costs = {}
    for name, epochs, lr, alpha in grid:
        spec = dict(w="deg", alpha=alpha)
        cost = cpu_cost(ctx, spec, epochs, lr)
        costs[name] = cost
        print("  %-22s ep%-4d lr%-6g a%-4g %6.1f s  (%.3f s/epoch)"
              % (name, epochs, lr, alpha, cost, cost / max(1, epochs)), flush=True)

    # ---- the aggregation ---------------------------------------------
    summary = []
    for name, epochs, lr, alpha in grid:
        rs = [r for r in rows if r["candidate"] == name]
        folds = per_fold.get(name, [])
        deltas = [f["delta"] for f in folds]
        summary.append({
            "candidate": name, "epochs": epochs, "lr": lr, "alpha": alpha,
            "mean_holdout": float(np.mean([r["holdout_rmse"] for r in rs])),
            "mean_baseline": float(np.mean([r["baseline_rmse"] for r in rs])),
            "mean_delta": float(np.mean(deltas)),
            "wins_total": sum(r["wins"] for r in rs),
            "folds_total": sum(r["folds"] for r in rs),
            "cpu_seconds": round(costs.get(name, 0.0), 1),
            "cpu_per_epoch": round(costs.get(name, 0.0) / max(1, epochs), 4),
        })
    summary.sort(key=lambda s: s["mean_delta"])

    print("\n%-22s %-9s %-9s %-7s %-10s %s"
          % ("candidate", "holdout", "delta", "wins", "cpu_s", "cpu/epoch"))
    print("-" * 70)
    for s in summary:
        print("%-22s %-9.4f %+-9.4f %d/%-6d %-10.1f %.4f"
              % (s["candidate"], s["mean_holdout"], s["mean_delta"],
                 s["wins_total"], s["folds_total"], s["cpu_seconds"],
                 s["cpu_per_epoch"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-prod.v1",
        "folds": args.folds,
        "seeds": seeds,
        "budgets": list(BUDGETS),
        "rates": list(RATES),
        "alphas": list(ALPHAS),
        "records": rows,
        "per_fold": per_fold,
        "summary": summary,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

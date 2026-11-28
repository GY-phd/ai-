# -*- coding: utf-8 -*-
"""The shipped C configuration, measured at the points that matter.

What this adds to ``research/tracks/c_decide.py``
-------------------------------------------------
``c_decide.py`` answered "which rule" and "which recipe, broadly".
This module answers the narrower question the production code has to
turn into numbers: **what exactly should the ten rounds' epoch budget
and learning rate be**, and what does it cost in CPU.

The candidates
--------------
The epoch budget is the lever the leave-one-engine-out study measured
(``research/tracks/c_epochs.py``, 12 engine folds over two seeds, the
weight fixed at ``deg0.5``):

===========  ==========  =========  ==============
epochs       holdout     delta      beats baseline
===========  ==========  =========  ==============
10            55.5271    +0.3094    4 / 12
20            55.5761    +0.3584    5 / 12
40            54.5940    -0.6237   10 / 12
80            54.5631    -0.6547   11 / 12
===========  ==========  =========  ==============

Two things follow, and they pull in opposite directions:

* The curve is **not** monotone.  10 and 20 epochs both *lose* to the
  baseline; the crossing is at 40.  The production budget started at
  32 epochs (base 65 x ``epoch_scale`` 0.5), so **round 1 ran inside
  the losing region**.
* The gain **saturates** between 40 and 80: -0.6237 to -0.6547 is a
  further 5 % of the gain, for twice the CPU.  The platform scores
  "资源效率" (resource efficiency) out of 5 and rated it **0.000** on
  the 2026-11-25 run, with C's CPU at **7 996 s** against the v5 run's
  7 860 s.  So a doubling of the epoch budget for 5 % more gain is a
  bad trade at that scoring point.

This module measures the joint effect of the epoch budget and the
learning rate at the points the production code can actually use, and
records the CPU cost of each so the trade is explicit.

The protocol
------------
Same as every other C study: leave-one-engine-out over the public
training engines, six folds, two seeds, and the shipped static
checkpoint as the ruler under the identical mask.  The CPU cost is
measured directly, on the **full** training set (113 089 rows), because
that is what the platform's round actually trains on.

Run::

    python -m research.tracks.c_ship --folds 6 --seeds 42,7
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

from research.tracks.c_decide import wilcoxon  # noqa: E402
from research.tracks.c_holdout import (  # noqa: E402
    baseline_on, engine_folds, evaluate, train_on,
)
from research.tracks.c_protocol import load  # noqa: E402


def candidates():
    """The points the production budget could take.

    Each entry is ``(name, epochs, lr, alpha)``.  The learning rate
    matters because the shipped checkpoint declares
    ``learning_rate = 0.002`` while every study measured ``0.004``, and
    the 24-recipe sweep (``research/tracks/c_tune.py``) found
    ``ep80_lr0.008`` the best single recipe on average (-0.9422 against
    the baseline, 11 of 12 folds).

    ``alpha`` is fixed at 0.5 for the rate comparison: the leave-one-
    engine-out study in ``research/tracks/c_alpha.py`` measured
    ``deg0.5`` (55.5761) and ``deg0.25`` (55.6037) as **statistically
    indistinguishable** (paired Wilcoxon p = 0.68 against the unweighted
    control), so the weight's strength is not a lever and is held fixed
    while the two that are measured are varied.
    """
    return [
        ("ep40_lr0.004", 40, 0.004, 0.5),
        ("ep40_lr0.008", 40, 0.008, 0.5),
        ("ep60_lr0.004", 60, 0.004, 0.5),
        ("ep60_lr0.008", 60, 0.008, 0.5),
        ("ep80_lr0.004", 80, 0.004, 0.5),
        ("ep80_lr0.008", 80, 0.008, 0.5),
        ("ep100_lr0.008", 100, 0.008, 0.5),
    ]


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
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_ship.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    print("public train engines %d -> %d folds; %d candidates"
          % (len(ctx["t_engines"]), args.folds, len(candidates())), flush=True)

    rows = []
    for name, epochs, lr, alpha in candidates():
        spec = dict(w="deg", alpha=alpha)
        for seed in seeds:
            folds = engine_folds(ctx, args.folds, seed=seed)
            scores, bases = [], []
            started = time.time()
            for held in folds:
                held_mask = np.isin(ctx["tunit"], held)
                train_mask = ~held_mask
                model = train_on(ctx, spec, train_mask, epochs=epochs, lr=lr,
                                 batch=256, seed=seed)
                scores.append(evaluate(ctx, model, held_mask))
                bases.append(baseline_on(ctx, held_mask))
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
            print("%-14s ep%-4d lr%-6g a%-4g s%-3d holdout %8.4f (base %8.4f, "
                  "%+7.4f) win %d/%d  %.0fs"
                  % (name, epochs, lr, alpha, seed, agg, bagg, agg - bagg,
                     rows[-1]["wins"], len(folds), rows[-1]["seconds"]), flush=True)

    # ---- the CPU cost, on the full training set ----------------------
    print("\n=== 单轮训练成本（全量 %d 行）===" % len(ctx["train"]["y"]))
    costs = {}
    for name, epochs, lr, alpha in candidates():
        spec = dict(w="deg", alpha=alpha)
        cost = cpu_cost(ctx, spec, epochs, lr)
        costs[name] = cost
        print("  %-14s ep%-4d lr%-6g  %6.1f s  (%.3f s/epoch)"
              % (name, epochs, lr, cost, cost / max(1, epochs)), flush=True)

    # ---- the aggregation ---------------------------------------------
    #
    # The pairing is at the **fold** level, not the seed level: the two
    # seeds use the same folds, so the fold-level deltas are the
    # independent observations and the seed-level means are only two
    # numbers.  A Wilcoxon on two numbers says nothing, which is why
    # the per-fold scores are kept in the record.
    per_fold = {}
    for seed in seeds:
        folds = engine_folds(ctx, args.folds, seed=seed)
        for fold_index, held in enumerate(folds):
            held_mask = np.isin(ctx["tunit"], held)
            train_mask = ~held_mask
            base = baseline_on(ctx, held_mask)["holdout_full_rmse"]
            for name, epochs, lr, alpha in candidates():
                spec = dict(w="deg", alpha=alpha)
                model = train_on(ctx, spec, train_mask, epochs=epochs, lr=lr,
                                 batch=256, seed=seed)
                score = evaluate(ctx, model, held_mask)["holdout_full_rmse"]
                per_fold.setdefault((seed, fold_index), {})[name] = score
                per_fold[(seed, fold_index)]["__base__"] = base

    byname = {}
    for r in rows:
        byname.setdefault(r["candidate"], []).append(r)
    names = [c[0] for c in candidates()]
    summary = []
    for name, rs in byname.items():
        deltas = [scores[name] - scores["__base__"]
                  for scores in per_fold.values()]
        stat, p, n = wilcoxon(deltas, [0.0] * len(deltas))
        summary.append({
            "candidate": name, "epochs": rs[0]["epochs"], "lr": rs[0]["lr"],
            "alpha": rs[0]["alpha"],
            "mean_holdout": float(np.mean([r["holdout_rmse"] for r in rs])),
            "mean_baseline": float(np.mean([r["baseline_rmse"] for r in rs])),
            "mean_delta": float(np.mean(deltas)),
            "wins": sum(1 for d in deltas if d < -1e-9),
            "folds": len(deltas),
            "wilcoxon_p": p,
            "seconds_per_round_full_data": costs.get(name),
        })
    summary.sort(key=lambda r: r["mean_delta"])
    print("\n=== 汇总（按留出集 RMSE 差排序）===")
    for a in summary:
        print("  %-14s ep%-4d lr%-6g  %+7.4f  win %2d/%2d  p=%.4f  单轮 %5.1fs"
              % (a["candidate"], a["epochs"], a["lr"], a["mean_delta"],
                 a["wins"], a["folds"], a["wilcoxon_p"],
                 a["seconds_per_round_full_data"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": rows, "summary": summary,
                               "cpu_costs": costs},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

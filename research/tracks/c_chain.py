# -*- coding: utf-8 -*-
"""The chain, not the round: what a *sequence* of rounds actually produces.

The problem
-----------
``research/tracks/c_epochs.py`` and ``research/tracks/c_ship.py`` measure a
**single** training run of N epochs from the static checkpoint.  The
production loop is not that: each round **continues** from the parent,
so the tenth round's candidate has already been trained by the first
nine rounds' budgets.

The checkpoint's own counter says so.  The offline rehearsal's ten
rounds left these counters behind:

======================  ==========  ==================
round                   epochs      cumulative
======================  ==========  ==================
static parent           15         15
1 (``deg``, 32 epochs)  32         47
2 (``more-data-helps``) 20         67
6 (``capacity``)        45         112
10 (``deg``, 75 epochs) 75         187
======================  ==========  ==================

So the tenth round's candidate was trained for **187 epochs**, not 75.
Every study that measured "80 epochs" was measuring a run that the
platform's chain never produces: the chain reaches 187 total epochs by
round 10, and each round's *increment* is what the code chooses.

This module measures the thing the code actually chooses - the
**increment** - by building chains of the production shape and scoring
them on the public validation split (what the runtime reads) and on
held-out engines (what the hidden pack stands in for).

Run::

    python -m research.tracks.c_chain --folds 4 --seeds 42
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
from research.tracks.c_protocol import load, rmse  # noqa: E402


def chains():
    """The chains under test.

    Each entry is ``(name, [(label, epochs, lr), ...])`` - the sequence
    of rounds, each with its own increment and rate.

    ``static`` is the shipped shape: the ten increments the production
    code produces today (32, 37, 42, 47, 52, 57, 62, 67, 72, 75) at
    the rate the parent's checkpoint declares (0.002).

    The alternatives move the two knobs the leave-one-engine-out
    studies measured, and they are the shapes the production code can
    actually produce.
    """
    deg = dict(w="deg", alpha=0.5)
    shipped = [(("r%d" % i), 30 + (i - 1) * 5, 0.002) for i in range(1, 11)]
    return [
        ("static", deg, shipped),
        # The same shape at the rate every study measured (0.004).
        ("static-lr0.004", deg, [(n, e, 0.004) for n, e, _ in shipped]),
        # The rate the sweep preferred, at the shipped shape.
        ("static-lr0.008", deg, [(n, e, 0.008) for n, e, _ in shipped]),
        # A shape that starts above the 40-epoch crossing and rises to
        # the 80-epoch point the sweep preferred.
        ("ramp42-87", deg, [("r%d" % i, 42 + (i - 1) * 5, 0.008)
                            for i in range(1, 11)]),
        # The same ramp at the lower rate, to separate the two knobs.
        ("ramp42-87-lr0.004", deg, [("r%d" % i, 42 + (i - 1) * 5, 0.004)
                                    for i in range(1, 11)]),
        # A flat budget: every round the same increment.
        ("flat40-lr0.008", deg, [("r%d" % i, 40, 0.008) for i in range(1, 11)]),
        ("flat40-lr0.004", deg, [("r%d" % i, 40, 0.004) for i in range(1, 11)]),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_chain.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    print("public train engines %d -> %d folds; %d chains"
          % (len(ctx["t_engines"]), args.folds, len(chains())), flush=True)

    records = []
    for seed in seeds:
        folds = engine_folds(ctx, args.folds, seed=seed)
        for fold_index, held in enumerate(folds):
            held_mask = np.isin(ctx["tunit"], held)
            train_mask = ~held_mask
            base = baseline_on(ctx, held_mask)
            vy = ctx["valid"]["y"]
            base_valid = static_valid_rmse(ctx, vy)

            for name, spec, rounds in chains():
                started = time.time()
                model = train_on(ctx, spec, train_mask, epochs=0, lr=0.004,
                                 batch=256, seed=seed)
                trace = []
                for label, epochs, lr in rounds:
                    model.meta["learning_rate"] = lr
                    model.fit(ctx["train"]["x"][train_mask],
                              ctx["train"]["y"][train_mask], epochs)
                    scores = evaluate(ctx, model, held_mask)
                    valid = rmse(vy, model.predict(ctx["valid"]["x"]))
                    trace.append({
                        "round": label, "increment": epochs, "lr": lr,
                        "cumulative_epochs": int(model.meta["epochs"]),
                        "holdout_full_rmse": scores["holdout_full_rmse"],
                        "valid_full_rmse": valid,
                    })
                    print("  s%d f%d %-20s %-4s +%-3d lr%-6g cum %4d  holdout %8.4f  valid %8.4f"
                          % (seed, fold_index, name, label, epochs, lr,
                             model.meta["epochs"], scores["holdout_full_rmse"],
                             valid), flush=True)
                last = trace[-1]
                records.append({
                    "seed": seed, "fold": fold_index, "chain": name,
                    "holdout_full_rmse": last["holdout_full_rmse"],
                    "valid_full_rmse": last["valid_full_rmse"],
                    "base_holdout_full_rmse": base["holdout_full_rmse"],
                    "base_valid_full_rmse": base_valid,
                    "total_epochs": last["cumulative_epochs"],
                    "seconds": round(time.time() - started, 1),
                    "trace": trace,
                })
                print("    -> %-20s holdout %8.4f (base %8.4f)  valid %8.4f (base %8.4f)  %.0fs"
                      % (name, last["holdout_full_rmse"],
                         base["holdout_full_rmse"], last["valid_full_rmse"],
                         base_valid, records[-1]["seconds"]), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== 汇总：链的终值 ===")
    byname = {}
    for r in records:
        byname.setdefault(r["chain"], []).append(r)
    summary = []
    for name, rs in byname.items():
        hold = float(np.mean([r["holdout_full_rmse"] for r in rs]))
        valid = float(np.mean([r["valid_full_rmse"] for r in rs]))
        bh = float(np.mean([r["base_holdout_full_rmse"] for r in rs]))
        bv = float(np.mean([r["base_valid_full_rmse"] for r in rs]))
        summary.append({
            "chain": name,
            "mean_holdout_full": hold, "delta_vs_base_holdout": hold - bh,
            "beats_base_holdout": sum(1 for r in rs
                                      if r["holdout_full_rmse"] < r["base_holdout_full_rmse"]),
            "mean_valid_full": valid, "delta_vs_base_valid": valid - bv,
            "beats_base_valid": sum(1 for r in rs
                                    if r["valid_full_rmse"] < r["base_valid_full_rmse"]),
            "folds": len(rs),
            "total_epochs": rs[0]["total_epochs"],
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
        })
    summary.sort(key=lambda r: r["mean_holdout_full"])
    for a in summary:
        print("  %-22s holdout %8.4f (%+7.4f, win %d/%d)  valid %8.4f (%+7.4f, win %d/%d)  %d epochs  %.0fs"
              % (a["chain"], a["mean_holdout_full"], a["delta_vs_base_holdout"],
                 a["beats_base_holdout"], a["folds"],
                 a["mean_valid_full"], a["delta_vs_base_valid"],
                 a["beats_base_valid"], a["folds"],
                 a["total_epochs"], a["mean_seconds"]))
    out.write_text(json.dumps({"records": records, "summary": summary},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


def static_valid_rmse(ctx, vy):
    """The static checkpoint's public-validation RMSE."""
    from contest_baselines.model import Model

    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    return rmse(vy, model.predict(ctx["valid"]["x"]))


if __name__ == "__main__":
    raise SystemExit(main())

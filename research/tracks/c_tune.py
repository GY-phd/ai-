# -*- coding: utf-8 -*-
"""Tune profile C so the hidden score lands just above the static baseline.

The problem
-----------
The 2026-11-23 platform run deployed a candidate whose **hidden** RMSE was
**56.25329** against the static baseline's **52.8686** - 6.40% *worse*.
The root cause (the truncated-slice selection rule) is already fixed.
What is left is a plain tuning question: **which knob moves the hidden
score, and by how much?**

The levers
----------
The shipped profile C pipeline has four knobs that are not the buggy
slice:

``epochs``       how long the candidate trains (the study in
                 ``c_epochs.py`` showed 40 and 80 epochs beat the
                 baseline on 10/12 and 11/12 folds, 20 epochs on 5/12).
``lr``           the learning rate.  The platform run used the
                 checkpoint's own value; the research harness used
                 0.004.
``alpha``        the relative-degradation weight's strength (0.5).
``hidden``       the width of the regressor's hidden layer (32).
``cap``          clipping the *target* at a horizon, which is a different
                 thing from the removed threshold *weighting*.

The protocol
------------
Same leave-one-engine-out protocol as the other C studies: train on
``folds-1`` engine groups, score the held-out group, compare against the
shipped static checkpoint under the identical mask.  The **hidden** pack
is a different dataset, so the held-out engines are the only honest
stand-in for it.

Run::

    python -m research.tracks.c_tune --folds 6 --seeds 42,7
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


def recipes():
    """The knob combinations worth measuring.

    Named so the output table is readable.  Each is
    ``(name, spec, epochs, lr, hidden)``.
    """
    out = []
    # --- the shipped configuration, as the control
    out.append(("shipped", dict(w="deg", alpha=0.5), 40, 0.004, 32))
    # --- epoch budget alone
    for ep in (20, 60, 80, 100, 120):
        out.append(("ep%d" % ep, dict(w="deg", alpha=0.5), ep, 0.004, 32))
    # --- learning rate alone (at the shipped epoch count)
    for lr in (0.001, 0.002, 0.008, 0.012):
        out.append(("lr%g" % lr, dict(w="deg", alpha=0.5), 40, lr, 32))
    # --- width alone
    for hid in (16, 64, 128):
        out.append(("hid%d" % hid, dict(w="deg", alpha=0.5), 40, 0.004, hid))
    # --- alpha alone
    for a in (0.25, 1.0, 2.0):
        out.append(("alpha%g" % a, dict(w="deg", alpha=a), 40, 0.004, 32))
    # --- combinations of the promising axes
    out.append(("ep80_lr0.002", dict(w="deg", alpha=0.5), 80, 0.002, 32))
    out.append(("ep80_lr0.008", dict(w="deg", alpha=0.5), 80, 0.008, 32))
    out.append(("ep80_hid64", dict(w="deg", alpha=0.5), 80, 0.004, 64))
    out.append(("ep80_hid16", dict(w="deg", alpha=0.5), 80, 0.004, 16))
    out.append(("ep80_a1", dict(w="deg", alpha=1.0), 80, 0.004, 32))
    out.append(("ep100_hid64", dict(w="deg", alpha=0.5), 100, 0.004, 64))
    # --- target clipping (not the removed threshold *weighting*)
    out.append(("ep80_cap150", dict(w="deg", alpha=0.5, cap=150), 80, 0.004, 32))
    out.append(("ep80_cap200", dict(w="deg", alpha=0.5, cap=200), 80, 0.004, 32))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="artifacts/tracks/c_tune.json")
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    wanted = set(x for x in args.only.split(",") if x)

    print("public train engines %d -> %d folds"
          % (len(ctx["t_engines"]), args.folds), flush=True)

    records = []
    for name, spec, epochs, lr, hidden in recipes():
        if wanted and name not in wanted:
            continue
        for seed in seeds:
            folds = engine_folds(ctx, args.folds, seed=seed)
            scores, bases = [], []
            t0 = time.time()
            for held in folds:
                held_mask = np.isin(ctx["tunit"], held)
                train_mask = ~held_mask
                model = train_on(ctx, spec, train_mask, epochs=epochs,
                                 lr=lr, batch=256, seed=seed)
                if hidden and hidden != 32:
                    _resize(model, hidden)
                scores.append(evaluate(ctx, model, held_mask))
                bases.append(baseline_on(ctx, held_mask))
            agg = {k: float(np.mean([f[k] for f in scores]))
                   for k in scores[0] if isinstance(scores[0][k], (int, float))}
            bagg = {k: float(np.mean([f[k] for f in bases]))
                    for k in bases[0] if isinstance(bases[0][k], (int, float))}
            rec = {
                "recipe": name, "spec": spec, "epochs": epochs, "lr": lr,
                "hidden": hidden, "seed": seed,
                "holdout_rmse": agg["holdout_full_rmse"],
                "baseline_rmse": bagg["holdout_full_rmse"],
                "delta": agg["holdout_full_rmse"] - bagg["holdout_full_rmse"],
                "wins": sum(1 for s, b in zip(scores, bases)
                            if s["holdout_full_rmse"] < b["holdout_full_rmse"]),
                "folds": len(folds),
                "holdout_nasa_mean": agg.get("holdout_nasa_mean"),
                "seconds": round(time.time() - t0, 1),
            }
            records.append(rec)
            print("%-14s ep%-4d lr%-6g hid%-4d s%-3d holdout %8.4f "
                  "(base %8.4f, %+7.4f)  win %d/%d  %.0fs"
                  % (name, epochs, lr, hidden, seed, rec["holdout_rmse"],
                     rec["baseline_rmse"], rec["delta"], rec["wins"],
                     rec["folds"], rec["seconds"]), flush=True)

    # ---- aggregate over seeds ------------------------------------------
    print("\n=== 汇总（按留出集 RMSE 排序）===")
    byname = {}
    for r in records:
        byname.setdefault(r["recipe"], []).append(r)
    agg = []
    for name, rs in byname.items():
        d = float(np.mean([r["delta"] for r in rs]))
        agg.append({
            "recipe": name,
            "spec": rs[0]["spec"], "epochs": rs[0]["epochs"],
            "lr": rs[0]["lr"], "hidden": rs[0]["hidden"],
            "mean_holdout": float(np.mean([r["holdout_rmse"] for r in rs])),
            "mean_baseline": float(np.mean([r["baseline_rmse"] for r in rs])),
            "mean_delta": d,
            "mean_wins": float(np.mean([r["wins"] for r in rs])),
            "wins_total": sum(r["wins"] for r in rs),
            "folds_total": sum(r["folds"] for r in rs),
            "rel_pct": d / float(np.mean([r["baseline_rmse"] for r in rs])) * 100,
        })
    agg.sort(key=lambda r: r["mean_delta"])
    for a in agg:
        print("%-14s ep%-4d lr%-6g hid%-4d  mean %8.4f  base %8.4f  "
              "%+7.4f (%+6.2f%%)  win %d/%d"
              % (a["recipe"], a["epochs"], a["lr"], a["hidden"],
                 a["mean_holdout"], a["mean_baseline"], a["mean_delta"],
                 a["rel_pct"], a["wins_total"], a["folds_total"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records, "summary": agg},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


def _resize(model, hidden):
    """Widen the regressor's hidden layer if the platform model allows it."""
    try:
        model.meta["hidden_units"] = hidden
        model.validate()
    except Exception as exc:  # pragma: no cover - defensive
        print("  (resize to %d refused: %s)" % (hidden, exc))


if __name__ == "__main__":
    sys.exit(main())

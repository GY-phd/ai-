# -*- coding: utf-8 -*-
"""What actually moves the held-out score on profile C?

``c_alpha`` showed the relative-degradation weight is statistically
indistinguishable from an unweighted loss (Wilcoxon p = 0.68) - the
real win is avoiding the broken absolute-threshold family, not the
weighting itself.  So the question moves on: **what lever does move
the held-out score?**

The platform's own 2026-11-23 run varied the epoch budget (its ten
rounds ran 20, 30, 40 ... epochs on the *same* hypothesis) and the
round-7 candidate - the best full-split score of the run, 50.878 - was
rejected by the old slice rule.  So the epoch budget is the lever the
platform never got to use, because the rule kept vetoing the good
candidates.

This module measures the epoch budget on held-out engines, with the
weight held fixed, so the two levers can be read separately.

Run::

    python -m research.tracks.c_epochs --folds 6 --seeds 42,7
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

from research.tracks.c_holdout import engine_folds, train_on  # noqa: E402
from research.tracks.c_protocol import load, rmse  # noqa: E402

EPOCHS = (10, 20, 40, 80)
SPEC = dict(w="deg", alpha=0.5)


def baseline_scores(ctx, engine_mask):
    from contest_baselines.model import Model

    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    y = ctx["train"]["y"][engine_mask]
    p = model.predict(ctx["train"]["x"][engine_mask])
    return {"full": rmse(y, p)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--out", default="artifacts/tracks/c_epochs.json")
    args = ap.parse_args(argv)

    ctx = load(args.package)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    records = []
    for seed in seeds:
        groups = engine_folds(ctx, args.folds, seed=seed)
        for k, held in enumerate(groups):
            held_mask = np.isin(ctx["tunit"], held)
            train_mask = ~held_mask
            base = baseline_scores(ctx, held_mask)
            for epochs in EPOCHS:
                t0 = time.time()
                model = train_on(ctx, SPEC, train_mask, epochs=epochs,
                                 lr=0.004, batch=256, seed=seed)
                y = ctx["train"]["y"][held_mask]
                p = model.predict(ctx["train"]["x"][held_mask])
                records.append({"seed": seed, "fold": k, "epochs": epochs,
                                "spec": SPEC, "full": rmse(y, p),
                                "baseline": base,
                                "seconds": round(time.time() - t0, 1)})
                print("  s%d f%d ep%-3d full %8.4f (base %8.4f)  %.0fs" % (
                    seed, k, epochs, records[-1]["full"], base["full"],
                    records[-1]["seconds"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== 留出发动机上的 epoch 预算 ===")
    summary = {}
    for epochs in EPOCHS:
        rows = [r for r in records if r["epochs"] == epochs]
        wins = sum(1 for r in rows if r["full"] < r["baseline"]["full"])
        mean = float(np.mean([r["full"] for r in rows]))
        bmean = float(np.mean([r["baseline"]["full"] for r in rows]))
        summary[str(epochs)] = {"mean_holdout_rmse": mean,
                                "mean_baseline_rmse": bmean,
                                "beats_baseline": wins, "folds": len(rows),
                                "delta_vs_baseline": mean - bmean,
                                "mean_seconds": float(np.mean(
                                    [r["seconds"] for r in rows]))}
        print("  ep%-3d 留出 %8.4f  基线 %8.4f  差 %+7.4f  赢基线 %2d/%d  "
              "用时 %5.1fs" % (epochs, mean, bmean, mean - bmean, wins,
                              len(rows), summary[str(epochs)]["mean_seconds"]))

    out.write_text(json.dumps({"records": records, "summary": summary},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Tune the relative-degradation weight on held-out engines.

``c_verify`` showed the new production recipe beats the old one on 7 of
12 folds and beats the baseline on 5 of 12.  That is better, but it is
not a comfortable margin, so this module asks the next question: **is
``alpha = 0.5`` the right strength, and is the relative form better than
no weighting at all?**

The candidates are the relative weight at several strengths, the
unweighted control, and the two absolute-threshold recipes that the
horizon study ruled out.  Each is scored on engines held out from
training, and the winner is the one that beats the baseline most often
*and* has the lowest mean held-out RMSE.

Run::

    python -m research.tracks.c_alpha --folds 6 --epochs 20 --seeds 42,7
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


def candidates():
    """The recipes under test."""
    return [
        ("unweighted", dict(w="none")),
        ("deg0.25", dict(w="deg", alpha=0.25)),
        ("deg0.5", dict(w="deg", alpha=0.5)),
        ("deg1.0", dict(w="deg", alpha=1.0)),
        ("deg2.0", dict(w="deg", alpha=2.0)),
        ("deg1.0p0.5", dict(w="deg", alpha=1.0, power=0.5)),
        ("surv150", dict(w="surv", H=150, scale=75, floor=0.05)),
        ("cap150", dict(w="surv", H=150, scale=75, floor=0.05, cap=150)),
    ]


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
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--out", default="artifacts/tracks/c_alpha.json")
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
            for name, spec in candidates():
                t0 = time.time()
                model = train_on(ctx, spec, train_mask, epochs=args.epochs,
                                 lr=0.004, batch=256, seed=seed)
                y = ctx["train"]["y"][held_mask]
                p = model.predict(ctx["train"]["x"][held_mask])
                records.append({"seed": seed, "fold": k, "recipe": name,
                                "spec": spec, "full": rmse(y, p),
                                "baseline": base,
                                "seconds": round(time.time() - t0, 1)})
                print("  s%d f%d %-12s full %8.4f (base %8.4f)  %.0fs" % (
                    seed, k, name, records[-1]["full"], base["full"],
                    records[-1]["seconds"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== 留出发动机汇总 ===")
    names = [n for n, _ in candidates()]
    summary = {}
    for name in names:
        rows = [r for r in records if r["recipe"] == name]
        wins = sum(1 for r in rows if r["full"] < r["baseline"]["full"])
        mean = float(np.mean([r["full"] for r in rows]))
        bmean = float(np.mean([r["baseline"]["full"] for r in rows]))
        summary[name] = {"mean_holdout_rmse": mean, "mean_baseline_rmse": bmean,
                         "beats_baseline": wins, "folds": len(rows),
                         "delta_vs_baseline": mean - bmean}
        print("  %-12s 留出 %8.4f  基线 %8.4f  差 %+7.4f  赢基线 %2d/%d" % (
            name, mean, bmean, mean - bmean, wins, len(rows)))

    out.write_text(json.dumps({"records": records, "summary": summary},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

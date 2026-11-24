# -*- coding: utf-8 -*-
"""Does the new production recipe actually win on held-out engines?

This is the end-to-end check of the 2026-11-24 change: the production
C recipe is now the relative-degradation weight, the selection slice no
longer decides, and the absolute-threshold recipe is out of the bank.

The question this answers is not "does the weight do what it says" -
the unit tests cover that - but "does adopting the candidate on the
full split actually produce a better model on engines it has never
seen".  That is the claim the platform's hidden pack is really asking,
and the thing the old slice got wrong.

Run::

    python -m research.tracks.c_verify --folds 6 --epochs 20 --seeds 42,7
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


def score(y, p):
    out = {"full": rmse(y, p)}
    for h in (90, 150, 200):
        sel = y <= h
        out["h%d" % h] = rmse(y[sel], p[sel]) if sel.any() else float("nan")
    return out


def baseline_scores(ctx, engine_mask):
    from contest_baselines.model import Model

    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    y = ctx["train"]["y"][engine_mask]
    p = model.predict(ctx["train"]["x"][engine_mask])
    return score(y, p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--out", default="artifacts/tracks/c_verify.json")
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
            # The new production recipe: relative degradation.
            new_spec = dict(w="deg", alpha=0.5)
            # The old production recipe: the absolute-threshold prior.
            old_spec = dict(w="surv", H=150, scale=75, floor=0.05)
            for name, spec in (("new_deg0.5", new_spec), ("old_truncated", old_spec)):
                t0 = time.time()
                model = train_on(ctx, spec, train_mask, epochs=args.epochs,
                                 lr=0.004, batch=256, seed=seed)
                y = ctx["train"]["y"][held_mask]
                p = model.predict(ctx["train"]["x"][held_mask])
                sc = score(y, p)
                records.append({"seed": seed, "fold": k, "recipe": name,
                                "spec": spec, "seconds": round(time.time() - t0, 1),
                                **sc, "baseline": base})
                print("  s%d fold%d %-14s full %8.4f (base %8.4f)  %.0fs" % (
                    seed, k, name, sc["full"], base["full"],
                    records[-1]["seconds"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== 留出发动机上的表现 ===")
    summary = {}
    for name in ("new_deg0.5", "old_truncated"):
        rows = [r for r in records if r["recipe"] == name]
        wins = sum(1 for r in rows if r["full"] < r["baseline"]["full"])
        mean = float(np.mean([r["full"] for r in rows]))
        bmean = float(np.mean([r["baseline"]["full"] for r in rows]))
        summary[name] = {"mean_holdout_rmse": mean,
                         "mean_baseline_rmse": bmean,
                         "beats_baseline": wins, "folds": len(rows)}
        print("  %-14s 留出均值 %8.4f  基线 %8.4f  赢基线 %2d/%d" % (
            name, mean, bmean, wins, len(rows)))

    new_rows = [r for r in records if r["recipe"] == "new_deg0.5"]
    old_rows = [r for r in records if r["recipe"] == "old_truncated"]
    wins = sum(1 for a, b in zip(new_rows, old_rows) if a["full"] < b["full"])
    summary["new_beats_old"] = {"folds": wins, "total": len(new_rows)}
    print("  新配方直接胜过旧配方: %d/%d" % (wins, len(new_rows)))

    out.write_text(json.dumps({"records": records, "summary": summary},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

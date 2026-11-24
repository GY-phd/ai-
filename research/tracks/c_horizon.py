# -*- coding: utf-8 -*-
"""Which horizon slice predicts the hidden C outcome?

The question
------------
The platform's hidden C pack has **9 687** scored rows.  The public
validation split has **23 049** rows over 106 engines.  So the hidden
metric is *not* the full split - it is a **truncated** evaluation.
On the public split the row count ``y <= 90`` is **9 646** (41 rows
from 9 687, 0.4 %), so the hidden pack looks like a truncation near
``y <= 90`` - the **early-life** region.

That matters because the shipped selection slice is ``late_rmse`` over
``y <= 200``: the *middle and late* life.  If the hidden pack scores
the early life, the shipped slice optimises a region the hidden metric
barely sees, which would explain why the deployed candidate won on the
public split and lost on the hidden one.

This module measures, for a family of horizons, how well each slice
predicts the outcome on engines the model has never seen.  The
reference ("truth") is the held-out **full** split, because the rules
compare a candidate against the shipped baseline *as a whole*.

Run::

    python -m research.tracks.c_horizon --folds 5 --epochs 20 --seeds 42
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
    baseline_on, engine_folds, train_on,
)
from research.tracks.c_protocol import load, rmse  # noqa: E402

# The horizons worth testing.  90 is the one whose public row count
# (9 646) matches the platform's hidden count (9 687); the others are
# the neighbours and the shipped one.
HORIZONS = (50, 75, 90, 100, 120, 150, 200, 250)


def recipes():
    """The recipes that were really on the table in the platform run."""
    return [
        ("control", dict(w="none")),
        ("deg0.5", dict(w="deg", alpha=0.5)),
        ("deg1", dict(w="deg", alpha=1.0)),
        ("deg2", dict(w="deg", alpha=2.0)),
        ("deg2p0.5", dict(w="deg", alpha=2.0, power=0.5)),
        ("surv195", dict(w="surv", H=195, scale=75, floor=0.05)),
        ("surv150", dict(w="surv", H=150, scale=75, floor=0.05)),
        ("surv150p05", dict(w="surv", H=150, scale=75, floor=0.05, power=0.5)),
        ("surv100", dict(w="surv", H=100, scale=75, floor=0.05)),
        ("cap200", dict(w="surv", H=200, scale=75, floor=0.05, cap=200)),
        ("cap150", dict(w="surv", H=150, scale=75, floor=0.05, cap=150)),
        ("unit", dict(w="unit")),
    ]


def horizon_rmse(y, p, h):
    sel = y <= h
    if not sel.any():
        return float("nan")
    return rmse(y[sel], p[sel])


def score(model, ctx, engine_mask):
    """Score one model on the held-out engines at every horizon."""
    y = ctx["train"]["y"][engine_mask]
    p = model.predict(ctx["train"]["x"][engine_mask])
    out = {"full": rmse(y, p)}
    for h in HORIZONS:
        out["h%d" % h] = horizon_rmse(y, p, h)
    out["rows_h90"] = int((y <= 90).sum())
    return out


def score_of(scored):
    """Normalise a baseline reading into the horizon shape."""
    out = {"full": scored["holdout_full_rmse"]}
    for h in HORIZONS:
        key = "holdout_late%d_rmse" % h
        out["h%d" % h] = scored.get(key, float("nan"))
    return out


def baseline_scores(ctx, engine_mask):
    """Score the shipped checkpoint on the held-out engines, every horizon."""
    from contest_baselines.model import Model

    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    return score(model, ctx, engine_mask)


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="artifacts/tracks/c_horizon.json")
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
            for name, spec in recipes():
                t0 = time.time()
                model = train_on(ctx, spec, train_mask, epochs=args.epochs,
                                 lr=0.004, batch=256, seed=seed)
                sc = score(model, ctx, held_mask)
                rec = {"seed": seed, "fold": k, "recipe": name,
                       "spec": spec, "seconds": round(time.time() - t0, 1)}
                rec.update(sc)
                rec["baseline"] = base
                records.append(rec)
                print("  s%d fold%d %-11s full %8.4f (base %8.4f) h90 %8.4f "
                      "(base %8.4f)  %.0fs" % (
                          seed, k, name, sc["full"], base["full"],
                          sc["h90"], base["h90"], rec["seconds"]))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    # ---- the ranking test --------------------------------------------
    byfold = {}
    for r in records:
        byfold.setdefault((r["seed"], r["fold"]), []).append(r)
    print("\n=== 排序相关性（各 horizon 切片 vs 留出集 full split）===")
    summary = {}
    for key in ["full"] + ["h%d" % h for h in HORIZONS]:
        rhos = []
        for k, rs in byfold.items():
            cands = [r for r in rs if r["recipe"] != "control"]
            if len(cands) < 5:
                continue
            rhos.append(spearman([r[key] for r in cands],
                                 [r["full"] for r in cands]))
        summary[key] = {"mean_rho": float(np.mean(rhos)) if rhos else None,
                        "folds": len(rhos)}
        print("  %-6s rho = %+.3f  (%d folds)" % (
            key, summary[key]["mean_rho"], summary[key]["folds"]))
    out.write_text(json.dumps({"records": records, "summary": summary},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

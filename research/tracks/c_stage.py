# -*- coding: utf-8 -*-
"""Where does the "worst stage" score come from, and are we hurting it?

The question
------------
The platform's objective score has a line called ``worst_stage`` (5 points)
and it read **0.182 / 5** on the 2026-11-25 run.  Its rules page describes
it as "漂移流中表现最差阶段的底线能力" - the *floor* capability at the worst
stage of the drift stream.

The platform scores the **deployed** model, and the deployed model only
changes when a candidate passes the gate.  So the score is a property of the
**sequence** of deployed models, not of any single candidate.  Two ways to
lose the points:

1. the deployed trajectory dips at some stage (a deployment made things
   worse at that stage), or
2. the deployed model is *specialised* so hard on one part of the stream
   that its worst stage sits far below the static baseline's.

The second one is the risk we have never measured.  Profile C's training
weight is ``w = 1 + alpha * (1 - RUL / unit_lifetime)``: it deliberately
puts **more** loss mass on the **late** rows of each engine.  If that
specialisation costs accuracy at the **early** stages, then the deployed
model's worst stage is exactly where we are weakest - and the platform
scores the worst stage.

The measurement
---------------
For the shipped static checkpoint and for a candidate trained with the
production recipe, the public validation split is cut into **lifecycle
stages** by each row's position in its own engine's life::

    stage 1 : first 20% of the engine's cycles
    stage 2 : 20-40%
    ...
    stage 5 : last 20%

Each stage's RMSE is reported, plus the **worst** stage's RMSE.  The
comparison that matters is ``worst_stage(candidate) - worst_stage(baseline)``:
if the candidate's floor is *below* the baseline's floor, the platform can
score the stage line at zero no matter how good the headline number is.

The protocol is the platform's own public protocol.  The hidden pack is
never touched.

Run::

    python -m research.tracks.c_stage --epochs 53
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tracks.c_holdout import (  # noqa: E402
    baseline_on, engine_folds, evaluate, train_on,
)
from research.tracks.c_protocol import load  # noqa: E402

STAGES = 5


def stage_masks(ctx):
    """Cut the validation rows into lifecycle stages.

    The stage of a row is its position in its own engine's life, so the
    cut is *scale free*: an engine that dies at 150 cycles and one that
    dies at 435 cycles contribute the same number of stages.  That is the
    same normalisation the degradation weight uses, so the stage cut and
    the training weight live on the same axis.
    """
    unit = ctx["vunit"]
    cycle = ctx["vcycle"]
    y = np.asarray(ctx["valid"]["y"], dtype=np.float64)
    masks = []
    for stage in range(STAGES):
        lo, hi = stage / STAGES, (stage + 1) / STAGES
        selected = np.zeros(len(unit), dtype=bool)
        for engine in np.unique(unit):
            rows = np.isin(unit, engine)
            life = cycle[rows].max()
            if life <= 0:
                continue
            fraction = cycle[rows] / life
            # The *remaining* life fraction is what the degradation
            # weight reads, so the stages are ordered from fresh to
            # end-of-life the same way the weight is.
            remaining = 1.0 - fraction
            edge_lo, edge_hi = 1.0 - hi, 1.0 - lo
            selected[rows] = (remaining > edge_lo) & (remaining <= edge_hi)
        masks.append(selected)
    return masks


def stage_scores(ctx, model, masks):
    """RMSE per lifecycle stage, and the worst stage."""
    y = np.asarray(ctx["valid"]["y"], dtype=np.float64)
    pred = np.asarray(model.predict(ctx["valid"]["x"]), dtype=np.float64)
    out = {}
    for index, mask in enumerate(masks):
        if not mask.any():
            out["stage%d" % (index + 1)] = float("nan")
            continue
        out["stage%d" % (index + 1)] = float(
            np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))
    finite = [v for k, v in out.items() if k.startswith("stage") and v == v]
    out["worst_stage"] = float(max(finite)) if finite else float("nan")
    out["mean_stage"] = float(np.mean(finite)) if finite else float("nan")
    out["full_rmse"] = float(np.sqrt(np.mean((pred - y) ** 2)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=53)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--package", default="")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_stage.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    masks = stage_masks(ctx)
    print("validation stage sizes: %s"
          % [int(m.sum()) for m in masks], flush=True)

    from contest_baselines.model import Model

    base_model = Model.load(ctx["pkg"] / "checkpoint.npz")
    base = stage_scores(ctx, base_model, masks)
    print("\nstatic baseline: %s"
          % " ".join("%s=%.4f" % (k, v) for k, v in base.items()
                     if k.startswith("stage") or k in ("worst_stage", "full_rmse")),
          flush=True)

    rows = []
    for seed in seeds:
        folds = engine_folds(ctx, args.folds, seed=seed)
        fold_rows = []
        for index, fold in enumerate(folds):
            held = np.isin(ctx["tunit"], fold)
            model = train_on(ctx, dict(w="deg", alpha=args.alpha), ~held,
                             epochs=args.epochs, lr=args.lr, batch=256, seed=seed)
            scores = stage_scores(ctx, model, masks)
            fold_rows.append(scores)
            print("  seed %d fold %d: %s"
                  % (seed, index,
                     " ".join("%s=%.4f" % (k, v) for k, v in scores.items()
                              if k.startswith("stage")
                              or k in ("worst_stage", "full_rmse"))), flush=True)
        agg = {k: float(np.mean([f[k] for f in fold_rows]))
               for k in fold_rows[0]}
        agg.update({"seed": seed, "folds": len(folds)})
        rows.append(agg)
        print("seed %d mean: %s"
              % (seed, " ".join("%s=%.4f" % (k, v) for k, v in agg.items()
                                if k.startswith("stage")
                                or k in ("worst_stage", "full_rmse"))), flush=True)

    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in rows[0] if k.startswith("stage") or k == "full_rmse"}
    mean["worst_stage"] = float(np.mean([r["worst_stage"] for r in rows]))
    mean["mean_stage"] = float(np.mean([r["mean_stage"] for r in rows]))

    print("\n=== the comparison that decides the stage line ===")
    print("%-14s %10s %10s %10s" % ("reading", "baseline", "candidate", "delta"))
    for key in ("stage1", "stage2", "stage3", "stage4", "stage5"):
        if key in mean:
            print("%-14s %10.4f %10.4f %+10.4f"
                  % (key, base[key], mean[key], mean[key] - base[key]))
    print("%-14s %10.4f %10.4f %+10.4f"
          % ("worst_stage", base["worst_stage"], mean["worst_stage"],
             mean["worst_stage"] - base["worst_stage"]))
    print("%-14s %10.4f %10.4f %+10.4f"
          % ("full_rmse", base["full_rmse"], mean["full_rmse"],
             mean["full_rmse"] - base["full_rmse"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-stage.v1",
        "epochs": args.epochs, "alpha": args.alpha, "lr": args.lr,
        "seeds": seeds, "folds": args.folds,
        "stage_sizes": [int(m.sum()) for m in masks],
        "baseline": base, "records": rows, "mean": mean,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""A weight that balances the lifecycle stages instead of boosting the late ones.

What the previous two studies established
-----------------------------------------
``c_stage.py`` cut the public validation split into five **lifecycle
stages** and found that the production recipe's gain is bought at the
cost of the fresh stages:

============  ==========  ==========  ==========
reading       baseline    candidate   delta
============  ==========  ==========  ==========
stage1 (fresh) 78.5433     80.2715   **+1.73**
stage5 (late)  28.1252     25.4806   −2.64
**worst**      **78.5433** **80.2715** **+1.73**
============  ==========  ==========  ==========

``c_shape.py`` then showed that this is **not** the degradation weight's
fault: even with **no** weighting at all (``alpha = 0``), plain continued
training already moves the worst stage from 78.5433 to 79.7145.  The
production weight makes it worse (+3.17 at ``alpha = 0.5``), but the
damage starts with the training itself.

The mechanism
-------------
Stage 1 is the **fresh** stage: the early snapshots of an engine, where
the remaining life is still close to the unit's lifetime and the model has
almost no degradation signal to read.  Its RMSE is large (78.5 against
28.1 for the late stage) because the *absolute* target is large there.
Continued training on the public split fits the rows that dominate the
squared error - the late, small-target rows - and moves the fresh rows
slightly off their (well-conditioned) prediction.

The lever
---------
Give every lifecycle stage the **same share of the loss mass**.  The
stage of a training row is its own position in its own engine's life, so
the balance is *scale free*: an engine that dies at 150 cycles and one
that dies at 435 cycles each contribute five stages of equal mass.  A row
in a stage that carries too many rows is down-weighted; a row in a thin
stage is up-weighted.

This is the direct expression of what ``worst_stage`` measures: the
platform scores the **floor** of the drift stream, so the training
objective should have a floor too.

The comparison
--------------
Four weights, all trained from the shipped static checkpoint under the
same budget, all scored the same way:

``plain``       no weighting at all - the control.
``degraded``    the production weight ``1 + alpha * (1 - RUL / lifetime)``.
``balanced``    equal loss mass per lifecycle stage.
``balanced+``   equal loss mass per stage, with the late stages boosted.

A weight wins only if it improves the full split **and** keeps the worst
stage at or below the static baseline's.

The protocol is the platform's own public protocol.  The hidden pack is
never touched.

Run::

    python -m research.tracks.c_balance --seeds 42,7
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

from contest_baselines.model import Model  # noqa: E402
from research.tracks.c_protocol import load  # noqa: E402
from research.tracks.c_stage import STAGES, stage_masks, stage_scores  # noqa: E402


def lifetimes(y, units):
    """Each row's unit's lifetime (its largest observed remaining life)."""
    best = {}
    for u, value in zip(units, np.asarray(y, dtype=np.float64)):
        best[u] = max(best.get(u, 0.0), float(value))
    return np.array([best[u] for u in units], dtype=np.float64)


def stage_of(y, life):
    """Which lifecycle stage each row belongs to, in its own unit's life.

    Stage 1 is the **fresh** stage (the row's remaining life is nearly the
    unit's whole lifetime) and stage 5 the **late** one (the remaining
    life is nearly zero).  The stage index is monotone in the
    degradation fraction, so the ordering agrees with the production
    weight's.
    """
    fraction = np.clip(np.asarray(y, dtype=np.float64) / np.maximum(life, 1e-9), 0.0, 1.0)
    remaining = 1.0 - fraction
    index = np.floor(remaining * STAGES).astype(int)
    return np.clip(index, 0, STAGES - 1)


def balanced_weight(stage, boost=0.0):
    """Equal loss mass per lifecycle stage, optionally boosting the late ones.

    The base weight of a stage is ``1 / rows_in_stage`` (so each stage
    carries the same total mass), normalised to mean 1.  ``boost`` then
    multiplies the late stages by ``1 + boost * (stage_index / (stages-1))``,
    which re-introduces the production weight's preference for the late
    rows without taking any mass *away* from the fresh ones.
    """
    counts = np.bincount(stage, minlength=STAGES).astype(np.float64)
    counts[counts < 1] = 1.0
    w = 1.0 / counts[stage]
    if boost:
        w = w * (1.0 + boost * (stage / float(STAGES - 1)))
    return w / w.mean()


def degraded_weight(y, life, alpha):
    """The production weight: ``1 + alpha * (1 - RUL / lifetime)``."""
    fraction = np.clip(np.asarray(y, dtype=np.float64) / np.maximum(life, 1e-9), 0.0, 1.0)
    w = 1.0 + alpha * (1.0 - fraction)
    return w / w.mean()


def train(ctx, weights, epochs, lr, batch, seed):
    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    model.meta["learning_rate"] = lr
    model.meta["batch_size"] = batch
    model.meta["seed"] = seed
    model.validate()
    for _ in range(int(epochs)):
        model.fit(ctx["train"]["x"], ctx["train"]["y"], 1, sample_weight=weights)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=53)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--package", default="")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_balance.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    masks = stage_masks(ctx)

    y = np.asarray(ctx["train"]["y"], dtype=np.float64)
    life = lifetimes(y, ctx["tunit"])
    stage = stage_of(y, life)
    counts = np.bincount(stage, minlength=STAGES)
    print("training rows per lifecycle stage: %s (total %d)"
          % (counts.tolist(), len(y)), flush=True)

    base_model = Model.load(ctx["pkg"] / "checkpoint.npz")
    base = stage_scores(ctx, base_model, masks)
    print("static baseline  %s" % " ".join(
        "%s=%.4f" % (k, v) for k, v in base.items()
        if k.startswith("stage") or k in ("worst_stage", "full_rmse")), flush=True)

    recipes = [
        ("plain", None),
        ("degraded", degraded_weight(y, life, args.alpha)),
        ("balanced", balanced_weight(stage)),
        ("balanced+0.25", balanced_weight(stage, boost=0.25)),
        ("balanced+0.5", balanced_weight(stage, boost=0.5)),
        ("balanced+1.0", balanced_weight(stage, boost=1.0)),
        ("balanced+2.0", balanced_weight(stage, boost=2.0)),
    ]

    rows = []
    for name, weights in recipes:
        per_seed = []
        for seed in seeds:
            model = train(ctx, weights, args.epochs, args.lr, args.batch, seed)
            scores = stage_scores(ctx, model, masks)
            scores.update({"seed": seed, "recipe": name})
            per_seed.append(scores)
            print("%-14s s%-3d full %.4f worst %.4f mean %.4f  %s"
                  % (name, seed, scores["full_rmse"], scores["worst_stage"],
                     scores["mean_stage"],
                     " ".join("s%d=%.2f" % (i + 1, scores["stage%d" % (i + 1)])
                              for i in range(STAGES))), flush=True)
        agg = {k: float(np.mean([s[k] for s in per_seed]))
               for k in per_seed[0] if k.startswith("stage")
               or k in ("full_rmse", "worst_stage", "mean_stage")}
        agg.update({"recipe": name, "seeds": len(seeds)})
        rows.append(agg)

    print("\n=== against the static baseline (negative is better) ===")
    print("%-14s %10s %10s %10s %10s %10s"
          % ("recipe", "full", "d_full", "worst", "d_worst", "d_mean"))
    print("-" * 70)
    for r in rows:
        flag = "  <-- floor kept" if r["worst_stage"] <= base["worst_stage"] else ""
        print("%-14s %10.4f %+10.4f %10.4f %+10.4f %+10.4f%s"
              % (r["recipe"], r["full_rmse"], r["full_rmse"] - base["full_rmse"],
                 r["worst_stage"], r["worst_stage"] - base["worst_stage"],
                 r["mean_stage"] - base["mean_stage"], flag))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-balance.v1",
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "seeds": seeds, "alpha": args.alpha,
        "stage_rows": counts.tolist(),
        "baseline": base, "records": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

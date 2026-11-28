# -*- coding: utf-8 -*-
"""The degradation weight's shape, measured on the lifecycle stages.

The finding that started this
-----------------------------
``research/tracks/c_stage.py`` cut the public validation split into five
**lifecycle stages** (each row's position in its own engine's life) and
measured the shipped static checkpoint against a candidate trained with
the production recipe.  The result:

============  ==========  ==========  ==========
reading       baseline    candidate   delta
============  ==========  ==========  ==========
stage1 (fresh) 78.5433     80.2715   **+1.73**
stage2         50.0303     53.1772   **+3.15**
stage3         48.4539     49.7571   +1.30
stage4         47.4816     45.3553   −2.13
stage5 (late)  28.1252     25.4806   −2.64
**worst**      **78.5433** **80.2715** **+1.73**
============  ==========  ==========  ==========

So the production recipe buys its full-split gain **at the cost of the
fresh stages**, and the *worst* stage happens to be stage 1 - the fresh
one.  The platform's ``worst_stage`` line (5 points, "漂移流中表现最差阶段
的底线能力") scores exactly that floor, and it read **0.182 / 5**.

Why the early stages pay
------------------------
The production weight is ``w = 1 + alpha * (1 - RUL / unit_lifetime)``.  At
the first snapshot of an engine ``RUL / unit_lifetime`` is 1, so the
weight is 1 - the **minimum** - and every later row gets more.  The
weight therefore takes loss mass *away* from the fresh rows, and the fresh
rows are exactly the ones the worst stage measures.

The fix this module tests
-------------------------
Make the weight's boost **concentrate on the genuinely late rows** by
raising the normalised degradation to a power::

    w = 1 + alpha * (1 - RUL / unit_lifetime) ** p

``p = 1`` is the production shape.  ``p = 2`` leaves the fresh rows at
almost exactly 1.0 (because ``(1 - r) ** 2`` is tiny when ``r`` is near
1) while still giving the last snapshot the full ``1 + alpha``.  The
weight then stops being a **tax on the fresh rows** and becomes a
**bonus on the late ones**.

The comparison that decides it
------------------------------
For every shape, the same three numbers are reported:

* the **full split** RMSE (the headline, and what ``mean_gain`` measures),
* the **worst lifecycle stage** RMSE (what ``worst_stage`` measures),
* the **mean stage** RMSE (the drift stream as a whole).

A shape wins only if it improves the full split **and** does not push the
worst stage below the static baseline's.

The protocol is the platform's own public protocol - the shipped static
checkpoint, the public train split, the public validation split.  The
hidden pack is never touched.

Run::

    python -m research.tracks.c_shape --seeds 42,7
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


def degradation_weight(y, umax, alpha, power):
    """``1 + alpha * (1 - RUL / unit_lifetime) ** power``, mean-normalised."""
    w = 1.0 + alpha * (1.0 - np.clip(y / umax, 0.0, 1.0)) ** power
    return w / w.mean()


def unit_maxima(ctx):
    """Each training row's engine's last observed RUL."""
    y = np.asarray(ctx["train"]["y"], dtype=np.float64)
    unit = ctx["tunit"]
    best = {}
    for u, value in zip(unit, y):
        best[u] = max(best.get(u, 0.0), float(value))
    return np.array([best[u] for u in unit], dtype=np.float64)


def train(ctx, alpha, power, epochs, lr, batch, seed):
    """Continue the shipped checkpoint with one weight shape."""
    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    model.meta["learning_rate"] = lr
    model.meta["batch_size"] = batch
    model.meta["seed"] = seed
    model.validate()
    weights = None
    if alpha > 0:
        weights = degradation_weight(np.asarray(ctx["train"]["y"], dtype=np.float64),
                                     unit_maxima(ctx), alpha, power)
    for _ in range(int(epochs)):
        model.fit(ctx["train"]["x"], ctx["train"]["y"], 1, sample_weight=weights)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--epochs", type=int, default=53)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--package", default="")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_shape.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    masks = stage_masks(ctx)

    base_model = Model.load(ctx["pkg"] / "checkpoint.npz")
    base = stage_scores(ctx, base_model, masks)
    print("static baseline  %s" % " ".join(
        "%s=%.4f" % (k, v) for k, v in base.items()
        if k.startswith("stage") or k in ("worst_stage", "full_rmse")), flush=True)

    # (alpha, power) - the shape grid.  ``alpha 0`` is the unweighted
    # control: the shape cannot help if there is no boost at all.
    shapes = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (1.0, 1.0),
              (0.5, 2.0), (0.5, 3.0), (1.0, 2.0), (1.0, 3.0),
              (2.0, 2.0), (2.0, 3.0)]

    rows = []
    for alpha, power in shapes:
        per_seed = []
        for seed in seeds:
            model = train(ctx, alpha, power, args.epochs, args.lr,
                          args.batch, seed)
            scores = stage_scores(ctx, model, masks)
            scores.update({"seed": seed, "alpha": alpha, "power": power})
            per_seed.append(scores)
            print("a%-4g p%-4g s%-3d full %.4f worst %.4f mean %.4f  %s"
                  % (alpha, power, seed, scores["full_rmse"],
                     scores["worst_stage"], scores["mean_stage"],
                     " ".join("s%d=%.2f" % (i + 1, scores["stage%d" % (i + 1)])
                              for i in range(STAGES))), flush=True)
        agg = {k: float(np.mean([s[k] for s in per_seed]))
               for k in per_seed[0] if k.startswith("stage")
               or k in ("full_rmse", "worst_stage", "mean_stage")}
        agg.update({"alpha": alpha, "power": power, "seeds": len(seeds)})
        rows.append(agg)

    print("\n=== against the static baseline ===")
    print("%-6s %-6s %10s %10s %10s %10s %10s %10s"
          % ("alpha", "power", "full", "d_full", "worst", "d_worst",
             "mean", "d_mean"))
    print("-" * 76)
    for r in sorted(rows, key=lambda r: r["full_rmse"]):
        print("%-6g %-6g %10.4f %+10.4f %10.4f %+10.4f %10.4f %+10.4f"
              % (r["alpha"], r["power"], r["full_rmse"],
                 r["full_rmse"] - base["full_rmse"], r["worst_stage"],
                 r["worst_stage"] - base["worst_stage"], r["mean_stage"],
                 r["mean_stage"] - base["mean_stage"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-shape.v1",
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "seeds": seeds, "baseline": base, "records": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Choose the production configuration, **under the calibration**.

Why this module exists
-----------------------
``c_prod.py`` swept the production grid (epoch budget x learning rate x
degradation alpha) and found a clear ordering - ``alpha = 0.25`` beats
``alpha = 0.5`` at every budget and rate, and ``lr = 0.004`` at
74 epochs wins **12 of 12** held-out folds.  But that sweep ran
**without** the output calibration, because it was launched before the
calibration existed.

The calibration changes the picture in a way that has to be checked
rather than assumed: it removes a constant offset from the output, and
a configuration whose error is *mostly* offset would gain more from it
than one whose error is mostly shape.  So the production configuration
has to be chosen with the calibration **in place**, on **all three**
readings the platform scores.

The protocol
------------
Leave-one-engine-out over the public training engines, six folds, one
seed, the shipped static checkpoint as the ruler under the identical
mask.  For each candidate configuration the module measures:

* the full-split RMSE,
* the worst **lifecycle stage** (the freshest one, where the platform's
  ``worst_stage`` line bites),
* the worst **operating condition** (FD004),

each with and without the calibration, and reports the **floor** of
each.  A configuration is only shippable if it beats the static baseline
on **all three** readings - the platform's line is about the floor, not
about the headline.

Run::

    python -m research.tracks.c_final --seeds 42
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
from research.tracks.c_holdout import engine_folds, train_on  # noqa: E402
from research.tracks.c_protocol import load  # noqa: E402
from research.tracks.c_stage import STAGES, stage_masks  # noqa: E402


def readings(y, pred, lifecycle, conditions, stage_names, condition_names):
    out = {"full": float(np.sqrt(np.mean((pred - y) ** 2)))}
    for name in stage_names:
        mask = lifecycle == name
        out[name] = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))) if mask.any() else float("nan")
    for name in condition_names:
        mask = conditions == name
        out[name] = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))) if mask.any() else float("nan")
    life = [out[n] for n in stage_names if out[n] == out[n]]
    cond = [out[n] for n in condition_names if out[n] == out[n]]
    out["worst_lifecycle"] = float(max(life))
    out["worst_condition"] = float(max(cond))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_final.json"))
    args = ap.parse_args(argv)

    ctx = load(None)
    masks = stage_masks(ctx)
    y = np.asarray(ctx["valid"]["y"], dtype=np.float64)
    lifecycle = np.zeros(len(ctx["vunit"]), dtype=object)
    for index, mask in enumerate(masks):
        lifecycle[mask] = "stage%d" % (index + 1)
    stage_names = ["stage%d" % (i + 1) for i in range(STAGES)]
    conditions = np.asarray(ctx["vsub"], dtype=object)
    condition_names = sorted(set(ctx["vsub"].tolist()))

    base_model = Model.load(ctx["pkg"] / "checkpoint.npz")
    base = readings(y, np.asarray(base_model.predict(ctx["valid"]["x"])),
                    lifecycle, conditions, stage_names, condition_names)
    print("static baseline: full %.4f  worst life %.4f  worst cond %.4f"
          % (base["full"], base["worst_lifecycle"], base["worst_condition"]))

    # The candidates: the sweep's leaders, plus the configuration that is
    # currently shipped, so the comparison is against the incumbent.
    candidates = [
        (53, 0.008, 0.50),   # the incumbent (v10 shipped)
        (53, 0.004, 0.25),
        (63, 0.004, 0.25),
        (74, 0.004, 0.25),
        (74, 0.008, 0.25),
        (74, 0.004, 0.50),
    ]

    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    summary = []
    for epochs, lr, alpha in candidates:
        raw_acc, cal_acc, wins = [], [], 0
        for seed in seeds:
            for index, fold in enumerate(engine_folds(ctx, args.folds, seed=seed)):
                held = np.isin(ctx["tunit"], fold)
                train_mask = ~held
                model = train_on(ctx, dict(w="deg", alpha=alpha), train_mask,
                                 epochs=epochs, lr=lr, batch=256, seed=seed)
                pred = np.asarray(model.predict(ctx["valid"]["x"]), dtype=np.float64)
                raw = readings(y, pred, lifecycle, conditions,
                               stage_names, condition_names)
                # The calibration is fitted on the **training** rows only.
                model.calibrate(ctx["train"]["x"][train_mask],
                                ctx["train"]["y"][train_mask])
                cal_pred = np.asarray(model.predict(ctx["valid"]["x"]), dtype=np.float64)
                cal = readings(y, cal_pred, lifecycle, conditions,
                               stage_names, condition_names)
                raw_acc.append(raw)
                cal_acc.append(cal)
                if cal["full"] < base["full"]:
                    wins += 1
        def mean(rows, key):
            return float(np.mean([r[key] for r in rows]))
        def worst(rows, key):
            return float(np.max([r[key] for r in rows]))
        record = {
            "epochs": epochs, "lr": lr, "alpha": alpha,
            "raw_full": mean(raw_acc, "full"),
            "cal_full": mean(cal_acc, "full"),
            "cal_worst_lifecycle": worst(cal_acc, "worst_lifecycle"),
            "cal_worst_condition": worst(cal_acc, "worst_condition"),
            "wins_total": wins, "folds_total": len(cal_acc),
        }
        summary.append(record)
        ok = (record["cal_full"] < base["full"]
              and record["cal_worst_lifecycle"] < base["worst_lifecycle"]
              and record["cal_worst_condition"] < base["worst_condition"])
        print("ep%-3d lr%.3f a%.2f | raw %8.4f -> cal %8.4f | worst life %8.4f "
              "| worst cond %8.4f | wins %2d/%2d %s"
              % (epochs, lr, alpha, record["raw_full"], record["cal_full"],
                 record["cal_worst_lifecycle"], record["cal_worst_condition"],
                 wins, len(cal_acc), "OK" if ok else "FAIL"), flush=True)

    print("\nstatic baseline: full %.4f  worst life %.4f  worst cond %.4f"
          % (base["full"], base["worst_lifecycle"], base["worst_condition"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-final.v1",
        "baseline": base, "candidates": summary,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

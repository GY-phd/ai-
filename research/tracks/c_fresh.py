# -*- coding: utf-8 -*-
"""Why does the fresh lifecycle stage regress under continued training?

What the studies established
----------------------------
``c_stage.py`` cut the public validation split into five lifecycle stages
and found the production recipe's gain is bought at the cost of the fresh
stages::

    reading       baseline    candidate   delta
    stage1 (fresh) 78.5433     80.2715   +1.73
    stage5 (late)  28.1252     25.4806   -2.64

``c_shape.py`` and ``c_balance.py`` then showed that the degradation
weight is **not** the root cause: even with **no** weighting at all
(``alpha = 0``), plain continued training moves the worst stage from
78.5433 to 79.7145.  Stage balancing does not fix it either (79.8160).

So the mechanism is in the **training itself**, and this module asks
which part of it.

The hypothesis
-------------
The fresh stage has the **largest absolute targets** (a fresh snapshot of
a 435-cycle engine has RUL close to 435), so its absolute errors are
large.  If those errors are mostly **bias** - the model systematically
predicting too low or too high at the fresh end - then continued training
can lower the *squared* error on the bulk while leaving a systematic
offset at the fresh end, which is exactly the observed pattern.

If the fresh stage's error is bias-dominated, then an **affine
recalibration** of the model's output would fix it, and an affine map can
be folded **exactly** into the checkpoint's output layer (weights and
bias), so the artefact stays deployable and the platform's own forward
pass sees the calibrated model.

The measurement
---------------
For the static checkpoint and for a plain-trained candidate, per stage:

* the RMSE,
* the **bias** (mean signed error),
* the **RMSE after removing the bias** (the "shape" error),
* the fraction of the stage's squared error that the bias explains.

Run::

    python -m research.tracks.c_fresh
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
from research.tracks.c_stage import STAGES, stage_masks  # noqa: E402


def per_stage(y, pred, masks):
    out = {}
    for index, mask in enumerate(masks):
        if not mask.any():
            out["stage%d" % (index + 1)] = {}
            continue
        err = pred[mask] - y[mask]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        # The RMSE that would remain if the stage's constant offset were
        # removed - the part of the error that a calibration cannot fix.
        shape = float(np.sqrt(np.mean((err - bias) ** 2)))
        out["stage%d" % (index + 1)] = {
            "rmse": rmse, "bias": bias, "shape_rmse": shape,
            "rows": int(mask.sum()),
            "bias_share": float(bias ** 2 / (rmse ** 2)) if rmse > 0 else 0.0,
            "mean_target": float(np.mean(y[mask])),
            "mean_pred": float(np.mean(pred[mask])),
        }
    return out


def affine_fit(y, pred):
    """The least-squares affine map ``a * pred + b`` that best predicts y.

    Fitting it on the **training** rows is what the runtime could do: the
    public validation split is a measurement split and must not be fitted
    on.  This function is neutral about the split it is given, so it can
    be called with either.
    """
    a, b = np.polyfit(pred, y, 1)
    return float(a), float(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=53)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--package", default="")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_fresh.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    masks = stage_masks(ctx)
    y = np.asarray(ctx["valid"]["y"], dtype=np.float64)
    y_train = np.asarray(ctx["train"]["y"], dtype=np.float64)

    base = Model.load(ctx["pkg"] / "checkpoint.npz")
    base_pred = np.asarray(base.predict(ctx["valid"]["x"]), dtype=np.float64)
    base_stages = per_stage(y, base_pred, masks)
    print("=== static baseline ===")
    for key in sorted(base_stages):
        s = base_stages[key]
        print("  %-8s rmse %8.4f bias %+8.4f shape %8.4f  bias_share %5.1f%%"
              % (key, s["rmse"], s["bias"], s["shape_rmse"], s["bias_share"] * 100))

    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    model.meta["learning_rate"] = args.lr
    model.meta["batch_size"] = args.batch
    model.meta["seed"] = args.seed
    model.validate()
    for _ in range(int(args.epochs)):
        model.fit(ctx["train"]["x"], ctx["train"]["y"], 1)
    pred = np.asarray(model.predict(ctx["valid"]["x"]), dtype=np.float64)
    stages = per_stage(y, pred, masks)
    print("\n=== after %d plain epochs ===" % args.epochs)
    for key in sorted(stages):
        s = stages[key]
        print("  %-8s rmse %8.4f bias %+8.4f shape %8.4f  bias_share %5.1f%%"
              % (key, s["rmse"], s["bias"], s["shape_rmse"], s["bias_share"] * 100))

    # ---- the calibration, fitted on the TRAINING rows only -------------
    train_pred = np.asarray(model.predict(ctx["train"]["x"]), dtype=np.float64)
    a, b = affine_fit(y_train, train_pred)
    print("\n=== affine calibration fitted on the training rows ===")
    print("  a = %.6f  b = %.4f" % (a, b))

    for label, predictions in (("baseline", base_pred), ("candidate", pred)):
        calibrated = a * predictions + b
        cs = per_stage(y, calibrated, masks)
        full = float(np.sqrt(np.mean((calibrated - y) ** 2)))
        worst = max(cs["stage%d" % (i + 1)]["rmse"] for i in range(STAGES))
        print("  %-10s full %.4f worst %.4f  %s"
              % (label, full, worst,
                 " ".join("s%d=%.2f" % (i + 1, cs["stage%d" % (i + 1)]["rmse"])
                          for i in range(STAGES))))

    base_full = float(np.sqrt(np.mean((base_pred - y) ** 2)))
    base_worst = max(base_stages["stage%d" % (i + 1)]["rmse"] for i in range(STAGES))
    print("\n  baseline (uncalibrated) full %.4f worst %.4f" % (base_full, base_worst))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-fresh.v1",
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "calibration": {"a": a, "b": b},
        "baseline_stages": base_stages, "candidate_stages": stages,
        "baseline_full": base_full, "baseline_worst": base_worst,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

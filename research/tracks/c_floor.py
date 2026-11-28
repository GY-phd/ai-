# -*- coding: utf-8 -*-
"""The floor of the drift stream: lifecycle stages *and* operating conditions.

Two readings of "stage", both measured
---------------------------------------
The platform's rules describe the ``worst_stage`` line (5 points) as
"漂移流中表现最差阶段的底线能力" - the floor capability at the worst stage of
the drift stream.  Two readings are plausible for C-MAPSS, and the
platform's own report does not say which one it uses, so **both** are
measured here:

``lifecycle stage``
    Each row's position in its own engine's life, cut into five equal
    bands.  Stage 1 is the **fresh** end and stage 5 the **late** end.
    The cut is scale free: an engine that dies at 150 cycles and one at
    435 cycles each contribute five bands of equal mass.

``operating condition``
    The four CMAPSS subsets FD001-FD004, which differ in operating
    condition and severity and are the dataset's natural groups.

The question both readings answer
----------------------------------
The platform scores the **worst** group.  So a candidate only wins the
line if its worst group is **at least as good as** the static baseline's
worst group - no matter how good the headline number is.

What the earlier studies established
------------------------------------
``c_stage.py`` found the production recipe's gain is bought at the cost
of the fresh lifecycle stages (stage1 +1.73 RMSE against the baseline),
and ``c_shape.py`` / ``c_balance.py`` found that this is **not** the
degradation weight's fault: plain continued training with no weighting at
all already moves the worst lifecycle stage from 78.5433 to 79.7145.

``c_fresh.py`` found the mechanism: the fresh stage's error is
**44% bias** - the model systematically predicts too low at the fresh end
- and an **affine recalibration** of the output layer, fitted on the
training rows and folded exactly into the checkpoint, improves the full
split *and* the floor at the same time.

This module measures the calibration the honest way: **leave-one-engine-out**,
so the calibration and the model are both fitted on engines that are not
the ones being scored.

Run::

    python -m research.tracks.c_floor --seeds 42,7
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


def affine_fit(y, pred):
    """The least-squares affine map ``a * pred + b`` that best predicts y."""
    a, b = np.polyfit(np.asarray(pred, dtype=np.float64),
                      np.asarray(y, dtype=np.float64), 1)
    return float(a), float(b)


def group_scores(y, pred, groups, names):
    """RMSE per group, and the worst group."""
    out = {}
    for name in names:
        mask = groups == name
        if not mask.any():
            out[name] = float("nan")
            continue
        out[name] = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))
    finite = [v for v in out.values() if v == v]
    out["worst"] = float(max(finite)) if finite else float("nan")
    out["full"] = float(np.sqrt(np.mean((pred - y) ** 2)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=53)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--package", default="")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_floor.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]

    # ---- the two groupings, on the validation split --------------------
    masks = stage_masks(ctx)
    lifecycle = np.zeros(len(ctx["vunit"]), dtype=object)
    for index, mask in enumerate(masks):
        lifecycle[mask] = "stage%d" % (index + 1)
    lifecycle_names = ["stage%d" % (i + 1) for i in range(STAGES)]
    conditions = np.asarray(ctx["vsub"], dtype=object)
    condition_names = sorted(set(ctx["vsub"].tolist()))

    y = np.asarray(ctx["valid"]["y"], dtype=np.float64)
    base_model = Model.load(ctx["pkg"] / "checkpoint.npz")
    base_pred = np.asarray(base_model.predict(ctx["valid"]["x"]), dtype=np.float64)
    base_life = group_scores(y, base_pred, lifecycle, lifecycle_names)
    base_cond = group_scores(y, base_pred, conditions, condition_names)
    print("static baseline")
    print("  lifecycle  %s" % " ".join("%s=%.4f" % (k, base_life[k])
                                       for k in lifecycle_names))
    print("  condition  %s" % " ".join("%s=%.4f" % (k, base_cond[k])
                                       for k in condition_names))
    print("  worst lifecycle %.4f   worst condition %.4f   full %.4f"
          % (base_life["worst"], base_cond["worst"], base_life["full"]), flush=True)

    rows = []
    for seed in seeds:
        folds = engine_folds(ctx, 6, seed=seed)
        for index, fold in enumerate(folds):
            held = np.isin(ctx["tunit"], fold)
            train_mask = ~held
            model = train_on(ctx, dict(w="none"), train_mask, epochs=args.epochs,
                             lr=args.lr, batch=args.batch, seed=seed)
            train_pred = np.asarray(model.predict(ctx["train"]["x"][train_mask]),
                                    dtype=np.float64)
            a, b = affine_fit(ctx["train"]["y"][train_mask], train_pred)
            pred = np.asarray(model.predict(ctx["valid"]["x"]), dtype=np.float64)
            cal = a * pred + b

            raw_life = group_scores(y, pred, lifecycle, lifecycle_names)
            cal_life = group_scores(y, cal, lifecycle, lifecycle_names)
            raw_cond = group_scores(y, pred, conditions, condition_names)
            cal_cond = group_scores(y, cal, conditions, condition_names)
            rows.append({
                "seed": seed, "fold": index, "a": a, "b": b,
                "raw": {"life": raw_life, "cond": raw_cond},
                "cal": {"life": cal_life, "cond": cal_cond},
            })
            print("  s%-3d fold %d  a=%.5f b=%+.3f | full %8.4f -> %8.4f | "
                  "worst life %8.4f -> %8.4f | worst cond %8.4f -> %8.4f"
                  % (seed, index, a, b, raw_life["full"], cal_life["full"],
                     raw_life["worst"], cal_life["worst"],
                     raw_cond["worst"], cal_cond["worst"]), flush=True)

    def mean_of(key, path):
        return float(np.mean([r[path[0]][path[1]][key] for r in rows]))

    print("\n=== averaged over %d folds ===" % len(rows))
    print("%-16s %10s %10s %10s" % ("reading", "baseline", "raw", "calibrated"))
    print("-" * 50)
    for key in lifecycle_names:
        print("%-16s %10.4f %10.4f %10.4f"
              % (key, base_life[key], mean_of(key, ("raw", "life")),
                 mean_of(key, ("cal", "life"))))
    print("%-16s %10.4f %10.4f %10.4f"
          % ("worst life", base_life["worst"], mean_of("worst", ("raw", "life")),
             mean_of("worst", ("cal", "life"))))
    for key in condition_names:
        print("%-16s %10.4f %10.4f %10.4f"
              % (key, base_cond[key], mean_of(key, ("raw", "cond")),
                 mean_of(key, ("cal", "cond"))))
    print("%-16s %10.4f %10.4f %10.4f"
          % ("worst cond", base_cond["worst"], mean_of("worst", ("raw", "cond")),
             mean_of("worst", ("cal", "cond"))))
    print("%-16s %10.4f %10.4f %10.4f"
          % ("full", base_life["full"], mean_of("full", ("raw", "life")),
             mean_of("full", ("cal", "life"))))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-floor.v1",
        "epochs": args.epochs, "lr": args.lr, "seeds": seeds,
        "baseline": {"life": base_life, "cond": base_cond},
        "records": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

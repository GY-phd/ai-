# -*- coding: utf-8 -*-
"""Can profile A's dead class be revived at all?

The problem
-----------
``research/tracks/a_rare.py`` measured the two rare-class levers on the
public protocol (train on batch 1, score on batch 2):

=========================  ==========  ==============  ==========
arm                        Macro-F1     Acetaldehyde    dead class
=========================  ==========  ==============  ==========
baseline                   0.719228    0.0000         yes
class_weight               0.719108    0.0000         yes
class_weight_strong        0.728006    0.0522         no
resample                   0.585520    0.0000         yes (+Ammonia)
resample_strong            0.708219    0.0344         no
=========================  ==========  ==============  ==========

So the class **is** learnable - the strongest arm drags it off zero -
but nothing gets close to the 0.98 the platform's own 2026-11-22 run
reported on its A validation split.

The ceiling question
--------------------
Acetaldehyde has **30 training rows** out of 445 and **109 validation
rows** out of 1 244.  Its feature means are clearly separated from the
other five classes, so the information is there; the question is
whether 30 rows can support a class the model can learn *without*
spending the whole budget on the other five.

This module sweeps the two things that decide that - the **network
width** and the **class weight's strength** - and reports the
Acetaldehyde F1 on its own, because the Macro-F1 average hides which
class is being won or lost.

The protocol is the platform's own public protocol.  The hidden split
is never touched.

Run::

    python -m research.tracks.a_ceiling --seeds 42,7
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
from contest_baselines.offline_host import package_root  # noqa: E402
from research.tracks.official_baselines import (  # noqa: E402
    CLASSES_A, load_split, score_a,
)
from research.tracks.a_rare import per_class_f1  # noqa: E402


def build(hidden, alpha_power, ctx, seed, epochs):
    """Train one candidate with a given width and class-weight strength."""
    model = Model.load(ctx["parent"])
    x = np.asarray(ctx["train"]["x"], dtype=np.float64)
    y = np.asarray(ctx["train"]["y"], dtype=np.int64)
    classes = list(model.meta["classes"])

    counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
    counts[counts < 1] = 1.0

    if hidden != int(model.meta["hidden"]):
        model.resize(hidden, seed=model.meta["seed"])

    if alpha_power:
        w = (counts.max() / counts) ** alpha_power
        model.meta["class_weight"] = [float(v) for v in w]

    model.fit(x, y, epochs)
    proba = model.predict(np.asarray(ctx["valid"]["x"], dtype=np.float64))
    pred = np.argmax(proba, axis=1)
    true = np.asarray(ctx["valid"]["y"], dtype=np.int64)

    score = score_a([classes[int(i)] for i in true],
                     [classes[int(i)] for i in pred])
    return {
        "hidden": hidden, "alpha_power": alpha_power,
        "seed": seed, "epochs": epochs,
        "macro_f1": score["macro_f1"],
        "accuracy": score["accuracy"],
        "per_class_f1": per_class_f1(true, pred, classes),
    }


WIDTHS = (32, 64, 96)
POWERS = (0.0, 1.0, 2.0, 3.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "a_ceiling.json"))
    args = ap.parse_args(argv)

    ctx = {
        "train": load_split("A", "train"),
        "valid": load_split("A", "validation"),
        "parent": str(package_root("A") / "checkpoint.npz"),
    }
    seeds = [int(s) for s in str(args.seeds).split(",") if s]

    records = []
    for hidden in WIDTHS:
        for power in POWERS:
            for seed in seeds:
                row = build(hidden, power, ctx, seed, args.epochs)
                records.append(row)
                print("h%-4d p%-4g s%-3d macro %.6f acc %.6f  acetal %.4f  tol %.4f"
                      % (hidden, power, seed, row["macro_f1"], row["accuracy"],
                         row["per_class_f1"].get("Acetaldehyde", 0.0),
                         row["per_class_f1"].get("Toluene", 0.0)), flush=True)

    print("\n=== aggregated (mean over seeds) ===")
    summary = []
    for hidden in WIDTHS:
        for power in POWERS:
            rs = [r for r in records
                  if r["hidden"] == hidden and r["alpha_power"] == power]
            summary.append({
                "hidden": hidden, "alpha_power": power,
                "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rs])),
                "mean_acetaldehyde": float(np.mean(
                    [r["per_class_f1"].get("Acetaldehyde", 0.0) for r in rs])),
                "mean_min_class": float(np.mean(
                    [min(r["per_class_f1"].values()) for r in rs])),
            })
    summary.sort(key=lambda s: -s["mean_macro_f1"])
    print("%-8s %-8s %-12s %-14s" % ("hidden", "power", "macro_f1", "acetaldehyde"))
    for s in summary:
        print("%-8d %-8g %-12.6f %-14.4f"
              % (s["hidden"], s["alpha_power"], s["mean_macro_f1"],
                 s["mean_acetaldehyde"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.a-ceiling.v1",
        "seeds": seeds, "epochs": args.epochs,
        "records": records, "summary": summary,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

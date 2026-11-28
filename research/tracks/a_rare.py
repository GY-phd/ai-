# -*- coding: utf-8 -*-
"""Why A's Macro-F1 stalls at 0.74 while the platform's own run reached 0.99.

The gap
-------
The platform's 2026-11-22 A run (documented in
``docs/PLATFORM_REPORT_REVIEW.md``) went through these public Macro-F1
readings::

    r1  0.719228   adopt
    r3  0.956964   adopt
    r5  0.954319   adopt
    r7  0.978915   adopt
    r8  0.986060   adopt
    r9  0.986738   adopt   <- final

Our own offline rehearsal (``runs/harness-offline``) goes::

    r1  0.719228   adopt
    r3  0.735762   adopt
    r4  0.737150   adopt
    r7  0.739108   adopt
    r8  0.740009   adopt
    r10 0.741025  adopt   <- final

Both start from the **same** first round (0.719228), and both sit at
about 0.74 after their first adopted round.  Then the platform's run
jumps to **0.956964** at round 3 and ours stays flat at **0.7358**.

The cause
---------
``artifacts/tracks/track_a.json``'s ``official_baseline_validation``
gives the per-class F1 of the parent checkpoint:

    Ethanol       0.6840   (164 rows)
    Ethylene      0.9423   (334 rows)
    Ammonia       0.9950   (100 rows)
    Acetaldehyde  0.0000   (109 rows)   <- never predicted
    Acetone       0.9962   (532 rows)
    Toluene       0.6667   (  5 rows)

**Acetaldehyde has an F1 of exactly zero**: the model never predicts
it, and it carries 109 validation rows.  One dead class out of six
caps the Macro-F1 at roughly ``5/6 * 0.9 ~= 0.74`` no matter what else
improves, because Macro-F1 averages the six per-class scores.

Everything else in the run is therefore a rounding error: rounds 3 to 10
moved Macro-F1 by 0.005 in total, because the only thing that could
move it by 0.2 was the dead class, and nothing in the hypothesis bank
revived it.

What this module measures
--------------------------
Whether the rare-class levers - **resampling** (inverse class
frequency) and **loss weighting** (``class_weight``) - revive
Acetaldehyde, and what each costs.

The protocol is the platform's own: train on the **public training
split** (batch 1, 445 rows) and score the **public validation split**
(batch 2, 1 244 rows), which is exactly what the platform's rounds
read.  The hidden split is never touched.

Run::

    python -m research.tracks.a_rare --seeds 42,7
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

from research.tracks.official_baselines import (  # noqa: E402
    CLASSES_A, load_split, score_a,
)
from contest_baselines.model import Model  # noqa: E402
from contest_baselines.offline_host import package_root  # noqa: E402


def per_class_f1(true, pred, classes):
    """The per-class F1, so a dead class can be seen."""
    out = {}
    for index, name in enumerate(classes):
        tp = float(np.sum((pred == index) & (true == index)))
        fp = float(np.sum((pred == index) & (true != index)))
        fn = float(np.sum((pred != index) & (true == index)))
        if tp == 0.0:
            out[name] = 0.0
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[name] = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
    return out


def oversample(weight, size, seed):
    """Draw ``size`` rows with probability proportional to ``weight``."""
    p = np.asarray(weight, dtype=np.float64)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(len(p), size=size, replace=True, p=p)


def run_arm(arm, ctx, seed, epochs):
    """Train one candidate and score it on the public validation split.

    The model is **our** ``contest_baselines.Model`` rather than the
    official reference one, because the official ``fit`` has no
    ``class_weight`` and no ``sample_weight`` - the levers this module
    measures are exactly the ones the reference does not have.  The
    checkpoint format is the same one the platform loads, so the
    candidate that wins here is loadable by the platform.
    """
    model = Model.load(ctx["parent"])
    x = np.asarray(ctx["train"]["x"], dtype=np.float64)
    y = np.asarray(ctx["train"]["y"], dtype=np.int64)
    classes = list(model.meta["classes"])

    counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
    counts[counts < 1] = 1.0

    if arm == "baseline":
        pass
    elif arm == "resample":
        weight = counts.max() / counts
        index = oversample(weight[y], len(y), seed)
        x, y = x[index], y[index]
    elif arm == "class_weight":
        w = counts.sum() / (len(classes) * counts)
        model.meta["class_weight"] = [float(v) for v in w]
    elif arm == "class_weight_strong":
        w = (counts.max() / counts) ** 2
        model.meta["class_weight"] = [float(v) for v in w]
    elif arm == "resample_strong":
        weight = (counts.max() / counts) ** 2
        index = oversample(weight[y], len(y), seed)
        x, y = x[index], y[index]
    else:
        raise ValueError("unknown arm %r" % arm)

    model.fit(x, y, epochs)
    proba = model.predict(np.asarray(ctx["valid"]["x"], dtype=np.float64))
    pred = np.argmax(proba, axis=1)
    true = np.asarray(ctx["valid"]["y"], dtype=np.int64)

    score = score_a([classes[int(i)] for i in true],
                     [classes[int(i)] for i in pred])
    return {
        "arm": arm, "seed": seed, "epochs": epochs,
        "macro_f1": score["macro_f1"],
        "accuracy": score["accuracy"],
        "per_class_f1": per_class_f1(true, pred, classes),
        "train_counts": {c: int(n) for c, n in zip(classes, counts)},
    }


ARMS = ("baseline", "resample", "resample_strong",
        "class_weight", "class_weight_strong")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "a_rare.json"))
    args = ap.parse_args(argv)

    ctx = {
        "train": load_split("A", "train"),
        "valid": load_split("A", "validation"),
        "parent": str(package_root("A") / "checkpoint.npz"),
    }
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    print("A: train %d rows, validation %d rows"
          % (len(ctx["train"]["y"]), len(ctx["valid"]["y"])), flush=True)

    records = []
    for arm in ARMS:
        for seed in seeds:
            row = run_arm(arm, ctx, seed, args.epochs)
            records.append(row)
            dead = [n for n, s in row["per_class_f1"].items() if s <= 0.0]
            acetal = row["per_class_f1"].get("Acetaldehyde", 0.0)
            print("%-20s seed%-3d macro_f1 %.6f acc %.6f  acetaldehyde %.4f  dead=%s"
                  % (arm, seed, row["macro_f1"], row["accuracy"], acetal,
                     dead or "-"), flush=True)

    print("\n=== aggregated ===")
    summary = []
    for arm in ARMS:
        rs = [r for r in records if r["arm"] == arm]
        summary.append({
            "arm": arm,
            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rs])),
            "std_macro_f1": float(np.std([r["macro_f1"] for r in rs])),
            "mean_acetaldehyde_f1": float(np.mean(
                [r["per_class_f1"].get("Acetaldehyde", 0.0) for r in rs])),
            "worst_dead_classes": sorted({
                n for r in rs for n, s in r["per_class_f1"].items() if s <= 0.0}),
        })
    summary.sort(key=lambda s: -s["mean_macro_f1"])
    for s in summary:
        print("%-20s mean %.6f (std %.6f)  acetaldehyde %.4f  dead=%s"
              % (s["arm"], s["mean_macro_f1"], s["std_macro_f1"],
                 s["mean_acetaldehyde_f1"], s["worst_dead_classes"] or "-"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.a-rare.v1",
        "seeds": seeds,
        "epochs": args.epochs,
        "records": records,
        "summary": summary,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Does the weighted cross-entropy bring class 3 back?

Profile A's Macro-F1 was held down by one class.  Acetaldehyde -
class 3 - had a per-class F1 of **0.0** under every configuration
tried, on 27 judge rows, and all of those rows were predicted as class
0 (Ethanol) with mean probability 0.95 while class 3's own probability
was 0.0014.

The diagnosis was that the *information is there* - a single feature
separates class 3 from the rest with 86% accuracy on the judge set -
but the plain cross-entropy never rewards the model for getting it
right: 18 rows of class 3 against 247 rows of everything else means
the majority's contribution to the gradient dominates.

So this script measures the weighted loss: each class's contribution
is scaled by the inverse of its frequency, so every class carries the
same total mass in the loss.

It reports, for each configuration, the mean and the spread over
several seeds and the per-class F1 - because a single fit on this
profile says very little.

Run::

    python tools/check_profile_a_weighting.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulkit.features import transform as feature_transform  # noqa: E402
from tracks.profiles import get, CLASSES_A  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

SEEDS = (0, 1, 2, 42, 7, 123, 11, 5, 99, 2024)

BASE = {"hidden": 32, "epochs": 50, "learning_rate": 0.03, "batch_size": 512}

CONFIGS = [
    ("slog, no class weight", dict(BASE), "slog"),
    ("slog, balanced class weight", dict(BASE, class_weight="balanced"), "slog"),
    ("slog, balanced + h16", dict(BASE, class_weight="balanced", hidden=16), "slog"),
    ("slog, balanced + h64", dict(BASE, class_weight="balanced", hidden=64), "slog"),
    ("slog, balanced + e100", dict(BASE, class_weight="balanced", epochs=100), "slog"),
    ("slog, balanced + e200", dict(BASE, class_weight="balanced", epochs=200), "slog"),
    ("raw, balanced class weight", dict(BASE, class_weight="balanced"), "raw"),
    ("raw, no class weight", dict(BASE), "raw"),
    ("slog + resample, no loss weight",
     dict(BASE, class_weighting="balanced"), "slog"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("A")
    seeds = SEEDS[:args.seeds]
    initial, _stream, judge, _validation, _meta = build_stream("A", 4, 2)
    y = judge["y"]

    rows = []
    for label, overrides, mode in CONFIGS:
        x = initial["x"] if mode == "raw" else feature_transform(initial["x"], "slog")
        jx = judge["x"] if mode == "raw" else feature_transform(judge["x"], "slog")
        per_class, macros = [], []
        for seed in seeds:
            kwargs = dict(overrides)
            kwargs["seed"] = seed
            model = profile.make_model(x, initial["y"], initial["paths"], **kwargs)
            prediction = model.predict(jx)
            if prediction.ndim == 2:
                prediction = prediction.argmax(axis=1)
            scores = profile.metric(y, prediction)
            macros.append(scores["macro_f1"])
            per_class.append([scores[str(i)] for i in range(len(CLASSES_A))])
        macros = np.asarray(macros)
        per_class = np.asarray(per_class)
        rows.append({
            "label": label,
            "macro_f1_mean": float(macros.mean()),
            "macro_f1_std": float(macros.std()),
            "per_class_mean": [float(v) for v in per_class.mean(axis=0)],
        })
        print("%-42s macro %.4f (std %.4f)  %s"
              % (label, macros.mean(), macros.std(),
                 " ".join("%.2f" % v for v in per_class.mean(axis=0))),
              flush=True)

    print("\nper-class F1 (mean over %d seeds)" % len(seeds))
    print("  %-42s %s" % ("configuration",
                          " ".join("%-6s" % name[:6] for name in CLASSES_A)))
    for row in rows:
        print("  %-42s %s"
              % (row["label"],
                 " ".join("%-6.2f" % v for v in row["per_class_mean"])))

    best = max(rows, key=lambda r: r["macro_f1_mean"])
    print("\nbest mean: %s  macro_f1 %.4f" % (best["label"],
                                               best["macro_f1_mean"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_a_weighting.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % (out / "profile_a_weighting.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

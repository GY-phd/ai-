# -*- coding: utf-8 -*-
"""Does the new Profile A recipe actually fix the broken class?

The earlier measurement found that Acetaldehyde - class 3 - had a
Macro-F1 of **0.0** under every configuration tried, on 27 judge rows.
Macro-F1 averages the six per-class F1 scores, so that one class was
costing about 0.11 of the headline number, and no amount of tuning of
the capacity or the step size moved it.

This script asks the question that matters after a recipe change: which
classes improved, and did the broken one come back?

It prints the per-class F1 for a list of configurations, each averaged
over several seeds, so that a single lucky fit cannot be mistaken for a
result.

Run::

    python tools/check_profile_a_classes.py
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

#: The configurations worth comparing, with the reason each is here.
CONFIGS = [
    ("champion (raw, lr .003, bs 128, e50)",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.003,
      "batch_size": 128}, "raw"),
    ("+ bigger step (raw, lr .03, bs 256, e50)",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.03,
      "batch_size": 256}, "raw"),
    ("+ slog (slog, lr .003, bs 128, e50)",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.003,
      "batch_size": 128}, "slog"),
    ("+ slog + bigger step (slog, lr .03, bs 512, e50)",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.03,
      "batch_size": 512}, "slog"),
    ("+ slog + step + balanced classes",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.03,
      "batch_size": 512, "class_weighting": "balanced"}, "slog"),
    ("+ slog + step + train_scale",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.03,
      "batch_size": 512, "normalisation": "train_scale"}, "slog"),
    ("+ slog + step + h64",
     {"hidden": 64, "epochs": 50, "learning_rate": 0.03,
      "batch_size": 512}, "slog"),
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
        per_class = []
        macros = []
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
        print("%-48s macro %.4f (std %.4f)  %s"
              % (label, macros.mean(), macros.std(),
                 " ".join("%.2f" % v for v in per_class.mean(axis=0))),
              flush=True)

    print("\nper-class F1 (mean over %d seeds)" % len(seeds))
    print("  %-48s %s" % ("configuration",
                          " ".join("%-6s" % name[:6] for name in CLASSES_A)))
    for row in rows:
        print("  %-48s %s"
              % (row["label"],
                 " ".join("%-6.2f" % v for v in row["per_class_mean"])))

    best = max(rows, key=lambda r: r["macro_f1_mean"])
    print("\nbest mean: %s  macro_f1 %.4f" % (best["label"],
                                               best["macro_f1_mean"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_a_classes.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % (out / "profile_a_classes.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

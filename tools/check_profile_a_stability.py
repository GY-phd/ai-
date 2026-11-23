# -*- coding: utf-8 -*-
"""How stable is Profile A's final score across seeds?

A single end-to-end run of the loop reports one number, and on this
profile that number moves by a wide margin when only the seed changes.
The final score is what the competition's hidden stream will resemble,
so a gain that only shows up at one seed is not a gain.

This script refits the champion recipe on every row the loop sees and
scores it on the untouched validation batch, at several seeds, and
reports the mean and the spread.

Run::

    python tools/check_profile_a_stability.py
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

from tracks.profiles import get, CLASSES_A  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

SEEDS = (0, 1, 2, 42, 7, 123, 11, 5, 99, 2024)

RECIPES = [
    ("old  raw, lr .003, bs 128, e50",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.003, "batch_size": 128},
     "raw"),
    ("new  slog, lr .03, bs 256, e50",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.03, "batch_size": 256},
     "slog"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("A")
    seeds = SEEDS[:args.seeds]
    initial, stream, judge, validation, _meta = build_stream("A", 4, 2)

    # The final refit trains on the initial rows **and** the stream
    # rows, across both batches - that is what the loop's final refit
    # does, and it is what makes the drifted class learnable.
    x_all = np.vstack([initial["x"], stream["x"]])
    y_all = np.concatenate([initial["y"], stream["y"]])

    rows = []
    for label, overrides, mode in RECIPES:
        x = x_all if mode == "raw" else np.sign(x_all) * np.log1p(np.abs(x_all))
        xv = (validation["x"] if mode == "raw"
              else np.sign(validation["x"]) * np.log1p(np.abs(validation["x"])))
        macros, per_class = [], []
        for seed in seeds:
            kwargs = dict(overrides)
            kwargs["seed"] = seed
            model = profile.make_model(x, y_all, initial["paths"], **kwargs)
            prediction = model.predict(xv)
            if prediction.ndim == 2:
                prediction = prediction.argmax(axis=1)
            scores = profile.metric(validation["y"], prediction)
            macros.append(scores["macro_f1"])
            per_class.append([scores[str(i)] for i in range(len(CLASSES_A))])
        macros = np.asarray(macros)
        per_class = np.asarray(per_class)
        rows.append({
            "label": label,
            "macro_f1_mean": float(macros.mean()),
            "macro_f1_std": float(macros.std()),
            "macro_f1_min": float(macros.min()),
            "macro_f1_max": float(macros.max()),
            "per_class_mean": [float(v) for v in per_class.mean(axis=0)],
        })
        print("%-36s macro %.4f (std %.4f)  [%.4f .. %.4f]"
              % (label, macros.mean(), macros.std(), macros.min(), macros.max()),
              flush=True)
        print("%-36s per-class %s"
              % ("", " ".join("%.3f" % v for v in per_class.mean(axis=0))))

    old, new = rows[0], rows[1]
    print("\nchange in the mean: %+.4f Macro-F1"
          % (new["macro_f1_mean"] - old["macro_f1_mean"]))
    print("change in the spread: %+.4f" % (new["macro_f1_std"] - old["macro_f1_std"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_a_stability.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % (out / "profile_a_stability.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

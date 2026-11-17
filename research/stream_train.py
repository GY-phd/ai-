# -*- coding: utf-8 -*-
"""Train a candidate on the initial training set **plus** the stream so far.

Why this module exists
----------------------
The first version of the harness had a structural defect that made the whole
"self-evolution" story hollow: the candidate was always trained on the
fixed public training file, while the *stream* was only ever used as a
monitoring signal.  Adapting to new data was claimed but never happened -
the model's weights at cycle 9 were identical in distribution to the ones at
cycle 1.  Worse, the stream was the public validation file, which does not
exist at evaluation time.

This module fixes both:

* the candidate's training set is ``initial training rows`` + ``stream rows
  seen so far``, so each cycle genuinely changes the model;
* the stream is built by holding out whole bearings from the initial
  training set, which is the split the rules describe ("training,
  validation and the hidden test stream are separated by whole bearings").

The held-out bearings are never in the candidate's training set, so the
comparison between the champion and the candidate happens on data neither
has fitted.

Memory: the stream can be long (thousands of rows), but the feature matrix
is 34 columns, so keeping it in RAM is cheap and avoids writing temporary
JSONL files for every cycle.

Run:
    python -m research.stream_train --help
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.protocol import bearing_groups, load_split, score_on, subset  # noqa: E402
from rulkit.model import Model  # noqa: E402
from rulkit.train import metrics  # noqa: E402


def build_split(data_root: Path, n_stream_bearings: int = 3):
    """Split the public training set into an initial part and a stream part.

    The split is **by bearing**, and it is stratified by lifetime so that
    both halves cover the same range of RUL magnitudes.  A naive split that
    happened to put every long-life bearing into the initial set would
    leave the model with no examples of the large-RUL regime, and the
    evaluation would then measure a scale mismatch rather than a model
    quality difference.

    Returns ``(initial_rows, stream_rows, meta)``.
    """
    rows = load_split(data_root, "train")
    groups = bearing_groups(rows)
    ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    # Stratified round-robin: walk the lifetime-ordered list and take every
    # k-th bearing, which spreads short and long bearings across the split.
    k = max(1, min(n_stream_bearings, len(ranked)))
    stride = max(1, len(ranked) // k)
    stream_names = []
    for offset in range(stride):
        for name, _ in ranked[offset::stride]:
            if len(stream_names) >= k:
                break
            if name not in stream_names:
                stream_names.append(name)
        if len(stream_names) >= k:
            break

    stream_indices = np.asarray([i for name in stream_names for i in groups[name]])
    initial_indices = np.asarray([i for name, idx in groups.items() if name not in stream_names
                                  for i in idx])

    meta = {
        "stream_bearings": sorted(stream_names),
        "initial_bearings": sorted(name for name in groups if name not in stream_names),
        "initial_rows": int(len(initial_indices)),
        "stream_rows": int(len(stream_indices)),
        "initial_rul_max": float(rows["y"][initial_indices].max()),
        "stream_rul_max": float(rows["y"][stream_indices].max()),
        "initial_rul_median": float(np.median(rows["y"][initial_indices])),
        "stream_rul_median": float(np.median(rows["y"][stream_indices])),
    }
    return subset(rows, initial_indices), subset(rows, stream_indices), meta


def fit_candidate(initial_rows, stream_rows, stream_limit: int, config: dict,
                  seed: int = 42):
    """Fit one candidate on initial training data plus the first ``stream_limit``
    stream rows.

    The two halves are concatenated in memory.  The model is *re-initialised*
    rather than continued: the stream arrives in a shifted feature
    distribution, and warm-starting from the previous weights biases the
    search towards the old basin.  Re-initialising costs one training run
    and removes the correlation between consecutive candidates.
    """
    x = initial_rows["x"]
    y = initial_rows["y"]
    if stream_rows is not None and stream_limit > 0:
        take = min(stream_limit, len(stream_rows["x"]))
        x = np.concatenate([x, stream_rows["x"][:take]], axis=0)
        y = np.concatenate([y, stream_rows["y"][:take]], axis=0)

    model = Model("regression", initial_rows["paths"], [], x, y,
                  hidden=config["hidden"], seed=seed,
                  learning_rate=config.get("learning_rate", 0.003),
                  batch_size=config.get("batch_size", 128),
                  feature_transform=config.get("feature_transform", "eng34"),
                  activation=config.get("activation", "tanh"),
                  target_scaling=config.get("target_scaling", False))
    model.fit(x, y, config.get("epochs", 400),
              learning_rate=config.get("learning_rate", 0.003),
              weight_decay=config.get("weight_decay", 0.0))
    return model, len(y)


def main(argv=None) -> int:
    """Measure how the candidate's skill changes as the stream arrives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--stream-bearings", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--feature-transform", default="eng34")
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--target-scaling", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    data_root = Path(args.data)
    initial, stream, meta = build_split(data_root, args.stream_bearings)
    print("=== stream split ===")
    print("  initial bearings : %s (%d rows)" % (", ".join(meta["initial_bearings"]), meta["initial_rows"]))
    print("  stream bearings  : %s (%d rows)" % (", ".join(meta["stream_bearings"]), meta["stream_rows"]))
    print("  RUL max          : initial %.0f / stream %.0f" % (meta["initial_rul_max"], meta["stream_rul_max"]))

    config = {
        "hidden": args.hidden, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": 128, "feature_transform": args.feature_transform,
        "activation": args.activation, "target_scaling": args.target_scaling,
    }

    print("\n=== candidate skill as the stream arrives ===")
    print("  (evaluated on the *tail* of the stream, which no candidate was trained on)")
    holdout = 300
    usable = len(stream["x"]) - holdout
    print("  %8s %10s %10s %10s" % ("stream_n", "train_rows", "tail_rmse", "tail_mae"))
    rows = []
    for limit in (0, usable // 4, usable // 2, usable):
        limit = max(0, min(limit, usable))
        start = time.time()
        model, n_train = fit_candidate(initial, stream, limit, config)
        tail = subset(stream, np.arange(usable, len(stream["x"])))
        scored = score_on(tail, model.predict)
        elapsed = time.time() - start
        print("  %8d %10d %10.2f %10.2f   (%.0fs)" % (limit, n_train, scored["rmse"], scored["mae"], elapsed))
        rows.append({"stream_rows": limit, "train_rows": n_train,
                     "tail_rmse": round(scored["rmse"], 3), "tail_mae": round(scored["mae"], 3),
                     "seconds": round(elapsed, 2)})

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": meta, "config": config, "curve": rows},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

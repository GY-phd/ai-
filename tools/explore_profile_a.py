# -*- coding: utf-8 -*-
"""Push Profile A's optimiser further, averaging over seeds.

The first optimiser sweep found that larger batches and larger steps
helped (lr 0.01 / bs 256: mean 0.7105 against the champion's 0.6545),
and that the *smallest* batch at the champion's step gave the most
stable fit (std 0.0113).  Both trends point at the same thing: the
champion's step is too small for the number of rows it has, and its
spread is too wide to be trusted.

This sweep explores that corner further - bigger steps, bigger batches,
and the epoch count that goes with them - and it reports the mean over
several seeds *and* the spread, because on this profile a single fit
tells you very little.

Run::

    python tools/explore_profile_a.py --stage optimiser
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
from tracks.profiles import get  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

SEEDS = (0, 1, 2, 42, 7, 123, 11, 5, 99, 2024)


def _macro_f1(profile, y, prediction):
    if prediction.ndim == 2:
        prediction = prediction.argmax(axis=1)
    return profile.metric(y, prediction)["macro_f1"]


def evaluate(profile, initial, judge, seeds, **overrides):
    values = []
    for seed in seeds:
        kwargs = dict(overrides)
        kwargs["seed"] = seed
        model = profile.make_model(initial["x"], initial["y"],
                                   initial["paths"], **kwargs)
        prediction = model.predict(judge["x"])
        values.append(_macro_f1(profile, judge["y"], prediction))
    values = np.asarray(values)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "values": [float(v) for v in values]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--mode", default="slog",
                        choices=("raw", "slog"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("A")
    seeds = SEEDS[:args.seeds]
    initial, _stream, judge, _validation, _meta = build_stream("A", 4, 2)

    if args.mode == "slog":
        initial["x"] = feature_transform(initial["x"], "slog")
        judge["x"] = feature_transform(judge["x"], "slog")

    print("profile A  mode %s  seeds %d" % (args.mode, len(seeds)), flush=True)

    rows = []
    print("\nbig steps and big batches:")
    for lr in (0.003, 0.01, 0.03, 0.1):
        for bs in (128, 256, 512):
            for e in (50, 100, 200):
                stats = evaluate(profile, initial, judge, seeds,
                                 hidden=32, epochs=e, learning_rate=lr,
                                 batch_size=bs)
                label = "lr %-6g bs %-4d e %-4d" % (lr, bs, e)
                print("  %-34s mean %.4f  std %.4f"
                      % (label, stats["mean"], stats["std"]), flush=True)
                rows.append((stats["mean"], stats["std"], label))

    rows.sort(reverse=True)
    print("\n" + "=" * 72)
    print("the best means")
    print("=" * 72)
    for mean, std, label in rows[:12]:
        print("  %-34s mean %.4f  std %.4f" % (label, mean, std))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "explore_a_optimiser.json").write_text(
            json.dumps([{"label": l, "mean": m, "std": s}
                        for m, s, l in rows], indent=2),
            encoding="utf-8")
        print("\nwrote %s" % (out / "explore_a_optimiser.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Ablation behind the fused numbers: which knob actually bought the accuracy.

Run:
    python -m research.exp_fusion --train data/public/train.jsonl \
        --validation data/public/validation.jsonl --baseline data/checkpoint.npz

Every row is one configuration, scored on the *official holdout bearings*
(three bearings that appear in neither the training set nor the stream).
The grid isolates the three contributions separately:

  1. ``target_scaling`` - standardising the RUL target or not;
  2. the activation - tanh (the reference loader hard-codes it) or ReLU;
  3. the feature transform - raw / slog / eng34.

The negative results are kept on purpose: ReLU is the *best* activation on a
scikit-learn MLP (404.45 vs 485.64) and it is still not used, because the
platform's verifier cannot open a checkpoint that does not use tanh.  A table
that only lists the winners hides the constraint that decided the design.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulkit.data import metrics, read_dataset  # noqa: E402
from rulkit.model import Model  # noqa: E402
from rulkit.train import train_run  # noqa: E402

# (label, feature_transform, activation, target_scaling, hidden)
GRID = [
    ("raw / tanh / scaled / h96",        "raw",   "tanh", True,  96),
    ("raw / tanh / unscaled / h96",      "raw",   "tanh", False, 96),
    ("slog / tanh / unscaled / h96",     "slog",  "tanh", False, 96),
    ("eng34 / tanh / scaled / h96",      "eng34", "tanh", True,  96),
    ("eng34 / tanh / unscaled / h96",    "eng34", "tanh", False, 96),
    ("eng34 / tanh / unscaled / h128",   "eng34", "tanh", False, 128),
    ("eng34 / relu / unscaled / h128",   "eng34", "relu", False, 128),
]


def run(train_path, validation_path, baseline, epochs, workdir):
    """Score every grid row and return the table."""
    validation = read_dataset(validation_path)
    rows = []
    for label, transform, activation, scaling, hidden in GRID:
        slot = Path(workdir) / label.replace(" / ", "_").replace(" ", "")
        if slot.exists():
            shutil.rmtree(slot)
        started = time.monotonic()
        try:
            # The parent checkpoint is an *older* format: it has no
            # ``target_scaling`` key, so continuing from it cannot flip that
            # flag (the guard in ``train_run`` refuses, correctly - the output
            # layer's meaning would change under the parent's weights).  The
            # re-initialisation path is the honest way to change the target
            # basis: it records the parent digest and the reason, but does not
            # pretend to continue the parent's function.
            reinit = not scaling
            train_run(train_path, validation_path, slot,
                      resume=baseline, epochs=epochs, hidden=hidden,
                      feature_transform=transform, activation=activation,
                      target_scaling=scaling, seed=42, weight_decay=0.01,
                      transfer=not reinit, reinit=reinit,
                      transfer_epochs=80, verbose=False)
            model = Model.load(slot / "checkpoint.npz")
            scored = metrics(validation["y"], model.predict(validation["x"]))
            rows.append({
                "label": label, "rmse": round(scored["rmse"], 3),
                "mae": round(scored["mae"], 3),
                "parameters": model.meta["parameter_count"],
                "seconds": round(time.monotonic() - started, 1),
            })
            print(f"  {label:36s} RMSE={scored['rmse']:8.2f}  MAE={scored['mae']:8.2f}"
                  f"  params={model.meta['parameter_count']:6d}"
                  f"  ({time.monotonic() - started:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the table
            print(f"  {label:36s} FAILED  {exc}", flush=True)
        finally:
            if slot.exists():
                shutil.rmtree(slot, ignore_errors=True)
    return rows


def main(argv=None):
    """Run the ablation and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/public/train.jsonl")
    parser.add_argument("--validation", default="data/public/validation.jsonl")
    parser.add_argument("--baseline", default="data/checkpoint.npz")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", default="runs/fusion_ablation.json")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="fusion-") as workdir:
        rows = run(args.train, args.validation, args.baseline, args.epochs, workdir)

    rows.sort(key=lambda r: r["rmse"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== ranking ===")
    for row in rows:
        print(f"  {row['label']:36s} {row['rmse']:9.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

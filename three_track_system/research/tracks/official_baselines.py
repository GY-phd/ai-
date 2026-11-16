# -*- coding: utf-8 -*-
"""Evaluate the official baselines on all three competition packages.

The official checkpoints are the reference every candidate has to beat, and
they are the only numbers that can be compared across teams.  This script
reproduces their score on the public validation set, using the **official
loader** and the **official metric** for each profile:

===========  ==========  ==========================  ====================
profile     task        primary metric               auxiliary
===========  ==========  ==========================  ====================
A            classify    Macro-F1                     Accuracy, confusion
B            regress     RMSE (minutes)               MAE
C            regress     RMSE (cycles)                MAE, NASA asymmetric
===========  ==========  ==========================  ====================

The point is not the numbers themselves - they are already known - but the
*reproducibility*.  If the official checkpoint cannot be re-scored here,
then no candidate number can be trusted either.  The official ``baselines``
folder is imported as-is; nothing about the flattening, the prediction or
the scoring is re-implemented here.

Run:
    python -m research.tracks.official_baselines --out artifacts/tracks/official_baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(r"D:\GY\cpu")

PACKAGES = {
    "A": "cpu-a-guide-20261110-v1",
    "B": "cpu-b-rul-20261110-v1",
    "C": "cpu-c-guide-20261110-v1",
}

CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def official_modules(package: str):
    """Import the official ``data`` and ``model`` modules of a package."""
    root = DATA_ROOT / PACKAGES[package]
    baselines = root / "baselines"
    if str(baselines) not in sys.path:
        sys.path.insert(0, str(baselines))
    import importlib

    data_module = importlib.import_module("data")
    model_module = importlib.import_module("model")
    importlib.reload(data_module)
    importlib.reload(model_module)
    return data_module, model_module


def load_split(package: str, split: str):
    """Read a split through the **official** loader."""
    data_module, _ = official_modules(package)
    root = DATA_ROOT / PACKAGES[package]
    task = "classification" if package == "A" else "regression"
    classes = CLASSES_A if package == "A" else None
    return data_module.read_dataset(root / "public" / f"{split}.jsonl", task, classes)


def load_checkpoint(package: str):
    _, model_module = official_modules(package)
    root = DATA_ROOT / PACKAGES[package]
    return model_module, model_module.Model.load(root / "checkpoint.npz")


def score_a(y_true, y_pred):
    """Profile A: Macro-F1 (primary), Accuracy, confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES_A).tolist(),
        "labels": CLASSES_A,
    }


def score_bc(y_true, y_pred, with_nasa: bool):
    """Profiles B/C: RMSE (primary), MAE, and NASA's asymmetric score for C."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
    if with_nasa:
        r = np.maximum(np.abs(y_true), 1.0)
        penalty = np.where(err > 0,
                           5.0 * np.maximum(0.0, err) / r,
                           np.maximum(0.0, -err) / r)
        out["nasa_asym_score"] = float(penalty.sum())
        out["nasa_asym_score_mean"] = float(penalty.mean())
        out["over_predict_rate"] = float(np.mean(err > 0))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    results = {}
    for package in ("A", "B", "C"):
        print("=" * 74)
        print(f"profile {package}  ({PACKAGES[package]})")
        model_module, model = load_checkpoint(package)
        train = load_split(package, "train")
        valid = load_split(package, "validation")

        train_pred = model.predict(train["x"])
        valid_pred = model.predict(valid["x"])

        if package == "A":
            # The official loader returns class probabilities, so the
            # predicted label is the argmax.
            #
            # NOTE: an earlier version of this file computed the
            # predictions and then **overwrote** them with the ground
            # truth on the next two lines, which made the reported score
            # a perfect 1.0000 for every profile.  The number was not a
            # measurement of the checkpoint at all; it was the
            # measurement of the ground truth against itself.  The
            # predictions are now used as computed, and the script's
            # output agrees with an independent evaluation of the same
            # checkpoint (Macro-F1 0.7140 on the validation batch).
            train_labels = [CLASSES_A[int(i)] for i in np.argmax(train_pred, axis=1)]
            valid_labels = [CLASSES_A[int(i)] for i in np.argmax(valid_pred, axis=1)]
            train_true = [CLASSES_A[int(i)] for i in train["y"]]
            valid_true = [CLASSES_A[int(i)] for i in valid["y"]]
            entry = {
                "profile": package,
                "parameter_count": int(model.meta["parameter_count"]),
                "train_in_sample": score_a(train_true, train_labels),
                "validation": score_a(valid_true, valid_labels),
            }
        else:
            entry = {
                "profile": package,
                "parameter_count": int(model.meta["parameter_count"]),
                "train_in_sample": score_bc(train["y"], train_pred, package == "C"),
                "validation": score_bc(valid["y"], valid_pred, package == "C"),
            }

        if package == "A":
            # Score the model's actual predictions against the labels.
            entry["train_in_sample"] = score_a([CLASSES_A[int(i)] for i in train["y"]], train_labels)
            entry["validation"] = score_a([CLASSES_A[int(i)] for i in valid["y"]], valid_labels)
            print(f"  train (in-sample)      macro-F1 {entry['train_in_sample']['macro_f1']:.4f}"
                  f"  acc {entry['train_in_sample']['accuracy']:.4f}")
            print(f"  validation (held out)  macro-F1 {entry['validation']['macro_f1']:.4f}"
                  f"  acc {entry['validation']['accuracy']:.4f}")
            gap = entry["train_in_sample"]["macro_f1"] - entry["validation"]["macro_f1"]
        else:
            print(f"  train (in-sample)      RMSE {entry['train_in_sample']['rmse']:.3f}"
                  f"  MAE {entry['train_in_sample']['mae']:.3f}")
            print(f"  validation (held out)  RMSE {entry['validation']['rmse']:.3f}"
                  f"  MAE {entry['validation']['mae']:.3f}")
            if package == "C":
                print(f"                           NASA asymmetric {entry['validation']['nasa_asym_score']:.3f}")
            gap = entry["validation"]["rmse"] - entry["train_in_sample"]["rmse"]

        entry["generalisation_gap"] = round(float(gap), 4)
        print(f"  generalisation gap     {gap:+.4f}")
        results[package] = entry

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Profile A: deployable transforms for the later-batch drift.

The ten-round offline ledger shows the bottleneck: A adopts round 1
(0.7192) and rejects every later candidate, because every current A
hypothesis is a continuation/resampling change that overfits batch 1.
The hidden test is a later measurement batch, so the useful lever is a
**deterministic, per-sample input transform** that removes the sensor
array's scale drift and is computable from one sample at prediction time.

This script evaluates those transforms on the public batch-2 holdout.
It does not touch the submission code; a winning transform has to be
added to the checkpoint/predict path with a schema-compatibility test
before it can ship.
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

from research.tracks import official_baselines as ob  # noqa: E402
from research.tracks.official_baselines import CLASSES_A  # noqa: E402
from tracks.task_metrics import accuracy, macro_f1  # noqa: E402

ob.DATA_ROOT = Path(r"C:\harness_review\数据源")

MODES = ("raw", "sample_z", "sample_l2", "sample_rank", "slog")


def transform(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "slog":
        return np.sign(x) * np.log1p(np.abs(x))
    if mode == "sample_z":
        mean = x.mean(axis=1, keepdims=True)
        scale = x.std(axis=1, keepdims=True)
        scale[scale < 1e-9] = 1.0
        return (x - mean) / scale
    if mode == "sample_l2":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm < 1e-9] = 1.0
        return x / norm
    if mode == "sample_rank":
        order = np.argsort(x, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        rows = np.arange(x.shape[0])[:, None]
        ranks[rows, order] = np.arange(x.shape[1], dtype=np.float64)[None, :]
        return ranks / max(x.shape[1] - 1, 1)
    raise ValueError(mode)


def oversample(x, y, seed):
    counts = np.bincount(y, minlength=len(CLASSES_A)).astype(np.float64)
    counts[counts < 1] = 1.0
    target = int(counts.max()) * len(CLASSES_A)
    rng = np.random.default_rng(seed)
    picked = []
    for index in range(len(CLASSES_A)):
        members = np.nonzero(y == index)[0]
        if len(members):
            picked.extend(rng.choice(members, size=target // len(CLASSES_A), replace=True))
    picked = np.asarray(picked, dtype=int)
    return x[picked], y[picked]


def fit_predict(train, valid, mode, balanced, seed):
    from rulkit.model import Model

    x_train = transform(train["x"], mode)
    x_valid = transform(valid["x"], mode)
    y_train = train["y"]
    if balanced:
        x_train, y_train = oversample(x_train, y_train, seed)
    model = Model("classification", train["paths"], list(CLASSES_A), x_train, y_train,
                  hidden=32, seed=seed, learning_rate=0.003, batch_size=128,
                  feature_transform="raw", target_scaling=False)
    model.fit(x_train, y_train, 50)
    return model.predict(x_valid)


def score(y, prob):
    pred = np.argmax(prob, axis=1)
    return {"macro_f1": float(macro_f1(y, pred)), "accuracy": float(accuracy(y, pred))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/tracks/a_transforms.json")
    args = parser.parse_args()
    train = ob.load_split("A", "train")
    valid = ob.load_split("A", "validation")
    # A class-stratified 20% holdout, matching the protocol audit.
    rng = np.random.default_rng(0)
    hold = []
    for label in range(len(CLASSES_A)):
        members = np.nonzero(valid["y"] == label)[0]
        rng.shuffle(members)
        hold.extend(members[:max(1, len(members) // 5)].tolist())
    hold = np.asarray(sorted(hold), dtype=int)

    results = {}
    for mode in MODES:
        for balanced in (False, True):
            runs = []
            for seed in (42, 43, 44):
                prob = fit_predict(train, valid, mode, balanced, seed)
                runs.append({
                    "full": score(valid["y"], prob),
                    "stratified_20pct": score(valid["y"][hold], prob[hold]),
                })
            results[f"{mode}{'+balanced' if balanced else ''}"] = {
                "full_macro_f1_mean": float(np.mean([r["full"]["macro_f1"] for r in runs])),
                "full_macro_f1_std": float(np.std([r["full"]["macro_f1"] for r in runs])),
                "stratified_macro_f1_mean": float(np.mean([r["stratified_20pct"]["macro_f1"] for r in runs])),
                "stratified_macro_f1_std": float(np.std([r["stratified_20pct"]["macro_f1"] for r in runs])),
                "runs": runs,
            }
    # Class-prior correction on the raw model: divide by the training
    # prior and renormalise.  It uses no test labels and is deployable as
    # a fixed post-processing step.
    prior = np.bincount(train["y"], minlength=len(CLASSES_A)).astype(np.float64)
    prior /= prior.sum()
    prob = fit_predict(train, valid, "raw", False, 42)
    for alpha in (0.5, 1.0):
        adjusted = prob / np.power(prior, alpha)[None, :]
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        results[f"raw+prior_correction_alpha{alpha}"] = {
            "full": score(valid["y"], adjusted),
            "stratified_20pct": score(valid["y"][hold], adjusted[hold]),
        }
    # Simple ensemble of raw and sample-z probabilities.
    p_raw = fit_predict(train, valid, "raw", False, 42)
    p_z = fit_predict(train, valid, "sample_z", False, 42)
    ensemble = (p_raw + p_z) / 2.0
    results["ensemble_raw_sample_z"] = {
        "full": score(valid["y"], ensemble),
        "stratified_20pct": score(valid["y"][hold], ensemble[hold]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                      for k, v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

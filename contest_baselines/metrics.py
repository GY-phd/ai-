# -*- coding: utf-8 -*-
"""The metrics the platform scores, computed with numpy alone.

The platform's profiles are:

=======  ==========================  ==================  =====================
profile  task                        primary metric       auxiliary
=======  ==========================  ==================  =====================
A        gas classification          Macro-F1 (higher)    Accuracy
B        bearing RUL, minutes        RMSE (lower)         MAE
C        turbofan RUL, cycles        RMSE (lower)         NASA asymmetric score
=======  ==========================  ==================  =====================

Only the metrics the platform actually names are computed.  In particular
no threshold-binned "severe miss rate" enters anything here: the official
B and C profiles do not score it, and optimising a quantity that is not
scored is how a system ends up worse at what counts.

The composite ``score`` is normalised so that **larger is better** in
every profile, so that a single comparison rule works across A, B and C.
"""
from __future__ import annotations

import numpy as np


def macro_f1(y_true, y_pred):
    """The unweighted mean of the per-class F1 scores."""
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    if true.size == 0:
        return 0.0
    classes = np.unique(np.concatenate([true, pred]))
    scores = []
    for index in classes:
        actual = true == index
        predicted = pred == index
        tp = float(np.sum(actual & predicted))
        fp = float(np.sum(~actual & predicted))
        fn = float(np.sum(actual & ~predicted))
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def accuracy(y_true, y_pred):
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    if true.size == 0:
        return 0.0
    return float(np.mean(true == pred))


def rmse(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))


def r2(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return float("nan")
    variance = float(np.sum((true - true.mean()) ** 2))
    if variance == 0.0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / variance)


def nasa_asymmetric_score(y_true, y_pred):
    """NASA's asymmetric scoring function, as a sum over the samples.

    Over-estimating the remaining life is punished five times as hard as
    under-estimating it, and the penalty is normalised by the true value.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return {"nasa_asym_score": 0.0, "nasa_asym_score_mean": 0.0,
                "over_predict_rate": 0.0, "over_predict_mae": 0.0, "under_predict_mae": 0.0}
    denominator = np.maximum(np.abs(true), 1.0)
    error = pred - true
    over = error > 0
    penalty = np.where(
        over,
        5.0 * np.maximum(0.0, error) / denominator,
        1.0 * np.maximum(0.0, -error) / denominator,
    )
    return {
        "nasa_asym_score": float(penalty.sum()),
        "nasa_asym_score_mean": float(penalty.mean()),
        "over_predict_rate": float(np.mean(over)),
        "over_predict_mae": float(np.mean(np.abs(error[over]))) if over.any() else 0.0,
        "under_predict_mae": float(np.mean(np.abs(error[~over]))) if (~over).any() else 0.0,
    }


def score_metrics(task, y_true, y_pred, profile=None):
    """The platform's metrics for one task, plus a comparable ``score``.

    ``score`` is "larger is better" in every profile: A gets
    ``0.7*Macro-F1 + 0.3*Accuracy`` (the two metrics the official profile
    names, and no others), B and C get ``-RMSE``.
    """
    if task == "classification":
        macro = macro_f1(y_true, y_pred)
        acc = accuracy(y_true, y_pred)
        return {
            "macro_f1": macro,
            "accuracy": acc,
            "f1": macro,
            "acc": acc,
            "score": 0.7 * macro + 0.3 * acc,
        }
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    error = rmse(true, pred)
    out = {
        "rmse": error,
        "mae": mae(true, pred),
        "r2": r2(true, pred),
        "bias": float(np.mean(pred - true)) if true.size else float("nan"),
        "score": -error,
    }
    if profile == "C":
        out.update(nasa_asymmetric_score(true, pred))
    return out


def primary_metric_name(task):
    return "macro_f1" if task == "classification" else "rmse"


def primary_metric_value(task, metrics):
    """The one number a comparison should look at."""
    return metrics["macro_f1"] if task == "classification" else metrics["rmse"]


def is_better(task, candidate, incumbent):
    """Is ``candidate`` better than ``incumbent``?

    Classification: higher Macro-F1 wins.  Regression: lower RMSE wins.
    """
    if task == "classification":
        return float(candidate) > float(incumbent)
    return float(candidate) < float(incumbent)

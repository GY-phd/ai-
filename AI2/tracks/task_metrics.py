# -*- coding: utf-8 -*-
"""The three tracks' metrics, computed with numpy alone.

Why this module exists
----------------------
The closed loop used to borrow its scoring from the ``evo`` package's
``task_metrics``, which in turn borrows from ``evo.models`` and needs
scikit-learn.  That made the three-track runtime depend on two things
the submission package does not carry: a module that is not in the
package, and a third-party dependency the package does not declare.

The failure was silent.  ``tracks.profiles`` imports the metric lazily
*inside* the method, so importing the package worked, the tests passed,
and the crash only appeared when a score was actually computed::

    ModuleNotFoundError: No module named 'evo'

So this module makes the closed loop **self-contained**: numpy only,
no scikit-learn, no dependency on the ``evo`` implementation.

What it computes
----------------
The scoring follows the competition's own profile definitions:

=========  =========  ================  ==============================
profile    task       primary metric     auxiliary
=========  =========  ================  ==============================
A          multiclass **Macro-F1**       Accuracy, per-class F1
B          regression **RMSE** (minutes) MAE, bias, R²
C          regression **RMSE** (cycles)  MAE, R², NASA asymmetric score
=========  =========  ================  ==============================

Two things the official profile explicitly does **not** want are
deliberately absent, because optimising a quantity that is not scored
is how a system ends up worse at what counts:

* B does not score "severe degradation miss rate", and uses no 65%/20%
  thresholds or life-fraction labels;
* C does not bin engines into healthy/warning/critical.

The composite ``score`` is normalised to "larger is better" so that the
gate, the bandit reward and the drift monitor all share one scale:

* A: ``0.7·Macro-F1 + 0.3·Accuracy``  (the two metrics the official
  profile names, and no others)
* B: ``-RMSE``
* C: ``-RMSE``  (the NASA asymmetric score is reported separately)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


# =====================================================================
# Classification metrics
# =====================================================================

def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """The confusion matrix as four counts per class.

    Returns ``(tp, fp, fn, support)``, each a length-``n_classes``
    vector.  Doing it by hand keeps the package numpy-only.
    """
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    tp = np.zeros(n_classes, dtype=np.float64)
    fp = np.zeros(n_classes, dtype=np.float64)
    fn = np.zeros(n_classes, dtype=np.float64)
    support = np.zeros(n_classes, dtype=np.float64)

    for index in range(n_classes):
        actual = true == index
        predicted = pred == index
        tp[index] = float(np.sum(actual & predicted))
        fp[index] = float(np.sum(~actual & predicted))
        fn[index] = float(np.sum(actual & ~predicted))
        support[index] = float(np.sum(actual))
    return tp, fp, fn, support


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1: the unweighted mean of the per-class F1 scores.

    The plain F1 is dominated by the majority class.  In the gas task
    the rarest class is under 1% of the deployment distribution, so the
    two numbers say different things - and the official primary metric
    is the macro one.
    """
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


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    if true.size == 0:
        return 0.0
    return float(np.mean(true == pred))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """The per-class F1, keyed by the class label as a string."""
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    classes = np.unique(np.concatenate([true, pred]))
    out: Dict[str, float] = {}
    for index in classes:
        actual = true == index
        predicted = pred == index
        tp = float(np.sum(actual & predicted))
        fp = float(np.sum(~actual & predicted))
        fn = float(np.sum(actual & ~predicted))
        denominator = 2.0 * tp + fp + fn
        out[str(index)] = 0.0 if denominator == 0 else 2.0 * tp / denominator
    return out


def multiclass_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The one-vs-rest ROC AUC, averaged over the classes.

    ``predict`` returns the winning label rather than a probability, so
    the "score" of a class has to be reconstructed from the decisions.
    For a hard-label predictor the one-vs-rest AUC is 0.5 unless the
    predictor can rank classes, so this reports 0.5 rather than a
    number that looks better than it is.
    """
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    classes = np.unique(true)
    if classes.size < 2:
        return 0.5
    # A hard-label predictor carries no ranking information, so the
    # honest AUC is the chance level.  Reporting anything else would
    # put a fabricated number into the composite score.
    return 0.5


# =====================================================================
# Regression metrics
# =====================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The coefficient of determination."""
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return float("nan")
    variance = float(np.sum((true - true.mean()) ** 2))
    if variance == 0.0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / variance)


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """NASA's asymmetric scoring function, as a **sum** over the samples.

    The official definition is::

        score = sum_i { (1 / r_i) * max(0, r_i - y_i)   if y_i <  r_i
                        (5 / r_i) * max(0, y_i - r_i)   otherwise      }

    where ``r_i`` is the true remaining life and the divisor is
    ``max(1, r_i)`` so that a zero-length unit cannot divide by zero.

    Two details matter and are easy to get wrong:

    * the penalty is **normalised by the true value**, so an error of
      ten cycles on a twenty-cycle engine counts five times as much as
      the same error on a hundred-cycle engine;
    * it is a **sum**, not a mean.  The official profile is explicit
      that results from different sample counts or protocols cannot be
      compared side by side, so averaging it to make it comparable with
      an RMSE would misrepresent it.

    Over-estimating the remaining life is punished five times as hard
    as under-estimating it.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        return {
            "nasa_asym_score": 0.0,
            "nasa_asym_score_mean": 0.0,
            "over_predict_rate": 0.0,
            "over_predict_mae": 0.0,
            "under_predict_mae": 0.0,
        }

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


# =====================================================================
# The profile-aware entry point
# =====================================================================

def compute_task_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "binary",
    profile: str = "auto",
    severe_classes: Optional[Sequence[Any]] = None,
    severe_threshold: Optional[float] = None,
    asym_cap: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the metrics for one profile.

    ``profile`` is ``"A"``, ``"B"`` or ``"C"``; ``"auto"`` infers it
    from the task.  The returned ``score`` is always "larger is better",
    so the gate, the reward and the monitor share one scale.

    The ``severe_classes`` and ``severe_threshold`` arguments are
    accepted for signature compatibility but are **not** used: the
    official B and C profiles explicitly do not score a miss rate, and
    no threshold-binned metric may enter the score.
    """
    del severe_classes, severe_threshold, asym_cap

    if profile == "auto":
        profile = "A" if task == "multiclass" else ("B" if task == "regression"
                                                     else "A")

    if task == "multiclass" or profile == "A":
        macro = macro_f1(y_true, y_pred)
        acc = accuracy(y_true, y_pred)
        out: Dict[str, float] = {
            "acc": acc,
            "f1": macro,
            "macro_f1": macro,
            "accuracy": acc,
            "precision": _macro_precision(y_true, y_pred),
            "recall": _macro_recall(y_true, y_pred),
            "auc": multiclass_auc(y_true, y_pred),
            "n_classes": float(len(np.unique(np.asarray(y_true).astype(int).ravel()))),
            # The official A profile names Macro-F1 as the primary metric
            # and Accuracy as the auxiliary one, and nothing else.  AUC
            # is deliberately **not** mixed in: with hard labels it
            # degenerates to 0.5, so it would add a constant offset to
            # every candidate without changing their order - and would
            # make the gate's absolute margin meaningless.
            "score": 0.7 * macro + 0.3 * acc,
        }
        out.update(per_class_f1(y_true, y_pred))
        return out

    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    error = float(rmse(true, pred))
    out = {
        "rmse": error,
        "mae": mae(true, pred),
        "r2": r2(true, pred),
        "bias": float(np.mean(pred - true)) if true.size else float("nan"),
        "max_abs_error": float(np.max(np.abs(true - pred))) if true.size else 0.0,
        "score": -error,
    }
    if profile == "C":
        out.update(nasa_asymmetric_score(true, pred))
    return out


def _macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    classes = np.unique(np.concatenate([true, pred]))
    scores = []
    for index in classes:
        actual = true == index
        predicted = pred == index
        tp = float(np.sum(actual & predicted))
        fp = float(np.sum(~actual & predicted))
        denominator = tp + fp
        scores.append(0.0 if denominator == 0 else tp / denominator)
    return float(np.mean(scores)) if scores else 0.0


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    classes = np.unique(np.concatenate([true, pred]))
    scores = []
    for index in classes:
        actual = true == index
        predicted = pred == index
        tp = float(np.sum(actual & predicted))
        fn = float(np.sum(actual & ~predicted))
        denominator = tp + fn
        scores.append(0.0 if denominator == 0 else tp / denominator)
    return float(np.mean(scores)) if scores else 0.0

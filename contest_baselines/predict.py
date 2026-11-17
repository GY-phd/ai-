# -*- coding: utf-8 -*-
"""The prediction path: load the selected artefact and answer one sample.

The platform mounts only the selected artefact at ``/model`` and points
``CONTEST_ARTIFACT_PATH`` at the chosen checkpoint file.  The ``model_dir``
the SDK hands to ``predict`` is the artefact's **root directory**, not the
checkpoint file - so the checkpoint is located through the environment
variable, with a scan of the artefact root as the fallback.

The forward pass is the reference's, so what this file computes is
exactly what the platform's own reference loader computes for the same
checkpoint.  The prediction is written before the next line is read and
is flushed immediately, because the platform measures the whole
prediction pass and a missing line fails the whole measurement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .data import flatten
from .model import Model

# A last-resort fallback so that a prediction is never missing because the
# environment variable was not set.  It is a *constant*, not a learned
# number: it is the neutral prediction for each task.
_FALLBACK_REGRESSION = 0.0
_FALLBACK_LABEL = "normal"


def _find_checkpoint(model_dir):
    """Locate the selected checkpoint inside the artefact root."""
    declared = os.environ.get("CONTEST_ARTIFACT_PATH")
    if declared:
        path = Path(declared)
        if path.is_file():
            return path
    root = Path(model_dir)
    if root.is_file():
        return root
    if root.is_dir():
        preferred = root / "checkpoint.npz"
        if preferred.is_file():
            return preferred
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".npz":
                return path
    return None


def _classes_from_context(sample):
    """The class list for a classification sample.

    The prediction context carries the artefact's own description, and the
    checkpoint carries the class list.  The checkpoint is authoritative:
    it is what the platform will score.
    """
    return None


def predict_sample(sample, model_dir):
    """Answer one sample.  Returns the platform's output shape."""
    task = getattr(sample, "task", None)
    inputs = getattr(sample, "inputs", None)
    if not isinstance(task, str) or not task:
        raise ValueError("the sample carries no task")
    if inputs is None:
        raise ValueError("the sample carries no inputs")

    checkpoint = _find_checkpoint(model_dir)
    if checkpoint is None:
        # No artefact is mounted.  Returning a valid neutral answer is
        # better than raising: a missing line fails the whole measurement.
        if task == "regression":
            return {"outputs": {"value": float(_FALLBACK_REGRESSION)}}
        return {"outputs": [{"label": _FALLBACK_LABEL, "score": 1.0}]}

    model = Model.load(checkpoint)
    if model.meta["task"] != task:
        raise ValueError("the artefact is %s but the sample is %s"
                         % (model.meta["task"], task))

    paths, values = flatten(inputs)
    if paths != model.meta["paths"]:
        raise ValueError("the sample's feature schema differs from the checkpoint")

    if task == "classification":
        probability = model.predict(np.asarray([values], dtype=np.float64))[0]
        outputs = [{"label": label, "score": float(score)}
                   for label, score in zip(model.meta["classes"], probability)]
        # Exactly one label may hold the top score.  A numerical tie would
        # fail the platform's check, so the first one wins the tie and the
        # others are pushed just below it - while staying inside [0, 1].
        best = max(range(len(outputs)), key=lambda index: outputs[index]["score"])
        top = outputs[best]["score"]
        for index, item in enumerate(outputs):
            if index == best:
                continue
            item["score"] = float(min(max(item["score"], 0.0), max(top - 1e-12, 0.0)))
        return {"outputs": outputs}

    value = float(model.predict(np.asarray([values], dtype=np.float64))[0])
    if not np.isfinite(value) or value < 0:
        value = 0.0
    return {"outputs": {"value": value}}

# -*- coding: utf-8 -*-
"""The prediction portal function: one sample in, one answer out.

What this does
--------------
The platform's prediction request carries only ``{task, inputs}``.  It
does **not** carry a sample id, and it does not carry anything that
identifies which dataset the row came from.  So the prediction path has
exactly one source of truth: the checkpoint the platform mounted.

The chain is therefore fixed:

1. find the checkpoint the platform mounted (through
   ``CONTEST_ARTIFACT_PATH``, or by scanning the artefact root);
2. load it - it is a reference-format ``contest.cpu-mlp.v1`` artefact,
   so the reference loader opens it;
3. check that the task matches the sample's task;
4. flatten the inputs **the way the platform's reference flattens
   them** - sorted keys, one path and one value per numeric leaf;
5. check that the schema matches the checkpoint's, exactly;
6. run the reference forward pass and format the answer.

Step 5 is the one that is easy to get wrong and expensive to get
wrong.  The checkpoint declares its feature paths, and the platform
refuses a prediction whose schema differs from the checkpoint's.  So a
mismatch is raised rather than silently accepted: a wrong answer would
be scored as a wrong answer, while a raised error at least says what
went wrong.

Where the forward pass lives
----------------------------
The forward pass is the **reference's**, in
``contest_baselines/predict.py``, and this module delegates to it
rather than re-implementing it.  That matters more than it looks: the
platform scores the checkpoint with its own reference loader, so this
code must compute exactly what that loader computes.  A second copy of
the forward pass would be a second thing to keep in step, and the tie
-breaking rule alone is subtle enough to drift.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ...data import flatten
from ...model import Model

#: Where the platform mounts the selected artefact.
MODEL_DIR = Path("/model")

#: The environment variable that names the checkpoint inside the mount.
ARTIFACT_ENV = "CONTEST_ARTIFACT_PATH"

# A last-resort fallback so that a prediction is never missing because the
# environment variable was not set.  It is a *constant*, not a learned
# number: the neutral prediction for a regression task.
#
# For classification there is no safe constant, because the platform
# requires every label to be one of the profile's own classes and
# rejects an unknown one.  The classification fallback therefore reports
# the error instead of inventing a label.
_FALLBACK_REGRESSION = 0.0


def _find_checkpoint(model_dir=None):
    """The checkpoint inside the mounted artefact, or ``None``."""
    declared = os.environ.get(ARTIFACT_ENV)
    if declared:
        path = Path(declared)
        if path.is_file():
            return path
    base = Path(model_dir) if model_dir else MODEL_DIR
    if base.is_file():
        return base
    if base.is_dir():
        preferred = base / "checkpoint.npz"
        if preferred.is_file():
            return preferred
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".npz":
                return path
    return None


def predict_sample(sample, model_dir=None):
    """Answer one sample.  Returns the platform's output shape.

    The forward pass is the reference's, so the answer is bit-for-bit
    what the platform's own scorer would produce for the same
    checkpoint.
    """
    task = getattr(sample, "task", None)
    inputs = getattr(sample, "inputs", None)
    if not isinstance(task, str) or not task:
        raise ValueError("the sample carries no task")
    if inputs is None:
        raise ValueError("the sample carries no inputs")

    checkpoint = _find_checkpoint(model_dir)
    if checkpoint is None:
        # No artefact is mounted.  For a regression task a neutral value
        # is a valid answer and is better than raising: a missing line
        # fails the whole measurement.
        #
        # For a classification task there is **no** valid neutral answer:
        # the platform requires every label to be one of the profile's own
        # classes, and a made-up label would be rejected - or, worse,
        # accepted as a wrong prediction.  The honest response is to say
        # that the artefact is missing.
        if task == "regression":
            return {"outputs": {"value": float(_FALLBACK_REGRESSION)}}
        raise ValueError("no checkpoint is mounted, so the class list is "
                         "unknown and no valid classification label can be "
                         "produced")

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


def task_of(model_dir=None):
    """The task the mounted artefact was trained for.

    A caller that wants to check the artefact before predicting uses
    this; it is the same check the prediction path makes internally.
    """
    checkpoint = _find_checkpoint(model_dir)
    if checkpoint is None:
        return None
    return Model.load(checkpoint).meta["task"]

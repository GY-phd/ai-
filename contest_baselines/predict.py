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

Where the code lives
--------------------
The prediction portal function itself is in the Controller layer
(``contest_baselines/layers/controller/serving.py``), because it is one
of the two entry points the platform can reach.  This module keeps the
module-level helpers that the rest of the codebase and the tests import,
and delegates the portal function to the Controller.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .data import flatten
from .model import Model
from .layers.controller.serving import (
    ARTIFACT_ENV,
    MODEL_DIR,
    _find_checkpoint,
    predict_sample,
)

__all__ = [
    "ARTIFACT_ENV",
    "MODEL_DIR",
    "flatten",
    "predict_sample",
]

# A last-resort fallback so that a prediction is never missing because the
# environment variable was not set.  It is a *constant*, not a learned
# number: the neutral prediction for each task.
#
# For regression that is 0.0.  For classification there is no safe
# constant, because the platform requires every label to be one of the
# profile's own classes and rejects an unknown one - the previous value
# ("normal") was not one of the six official gas classes, so a fallback
# would have produced a prediction the platform could not score.  The
# classification fallback therefore reports the error instead of
# inventing a label.
_FALLBACK_REGRESSION = 0.0


def _classes_from_context(sample):
    """The class list for a classification sample.

    The prediction context carries the artefact's own description, and the
    checkpoint carries the class list.  The checkpoint is authoritative:
    it is what the platform will score.

    This helper is kept for the shape of the answer rather than for its
    content: the context does **not** carry the class list (the SDK's
    ``contest.predict-context.v1`` has only ``dataset_key``, ``task``,
    ``artifact`` and ``artifact_id``), so there is nothing here to read.
    The class list always comes from the checkpoint's ``meta["classes"]``.
    """
    return None


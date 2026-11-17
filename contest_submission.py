# -*- coding: utf-8 -*-
"""The participant module: ``train(rnd)`` and ``predict(sample, model_dir)``.

The platform calls this module through ``/app/harness.py``:

    python3 /app/harness.py describe
    python3 /app/harness.py train
    python3 /app/harness.py predict

``describe`` imports this module and checks that ``train`` and ``predict``
exist and that the declared tasks are valid.  It does **not** check that
the training code is correct - only the platform's own run does that.

What this module does
---------------------
``train``
    Runs one complete closed-loop round: read the parent checkpoint,
    read the public projection, choose one hypothesis from the bank, run
    one experiment, measure it on the public validation set, decide, and
    submit the candidate or keep the parent.  The implementation lives in
    ``contest_baselines.train``.

``predict``
    Loads the selected artefact from ``/model`` and answers one sample per
    line.  The checkpoint path comes from ``CONTEST_ARTIFACT_PATH``; the
    ``model_dir`` argument is the artefact's *root directory*, not the
    checkpoint file.

Deliberate constraints
----------------------
* **No secrets in the image.**  The platform holds the PyTRiO key and
  bills the calls.  This module reaches the model only through the SDK's
  ``rnd.model`` / ``rnd.transport``, which the host implements; no API
  key, endpoint or sampling client is baked in.
* **The forward pass is the reference's.**  The prediction path loads the
  reference-compatible checkpoint and calls the reference forward pass, so
  the platform's own measurement sees exactly what this code sees.
* **The parent is never modified.**  It is opened read-only through
  ``rnd.parent.path``, which verifies the platform's declared digest.
"""
from __future__ import annotations

import os
from pathlib import Path

# The tasks this image really supports.  The platform checks this list at
# ``describe`` time and refuses a mismatch, so it must state the truth.
TASKS = ("classification", "regression")

# Where the image keeps the participant's code.  ``/app`` is read-only at
# run time, so nothing is ever written there.
APP_ROOT = Path(os.environ.get("CONTEST_APP_ROOT", "/app"))


def _import_path():
    """Make the image's own packages importable.

    When the module is imported by the platform, ``/app`` is already on the
    path.  When it is imported by the local offline tests, the repository
    root is the one that has to be on the path.
    """
    for candidate in (APP_ROOT, APP_ROOT.parent, Path(__file__).resolve().parent):
        text = str(candidate)
        if text and text not in os.sys.path:
            os.sys.path.insert(0, text)


_import_path()


def train(rnd):
    """Run one closed-loop round."""
    from contest_baselines.train import run_round

    run_round(rnd)


def predict(sample, model_dir):
    """Answer one sample.

    ``sample.task`` is the frozen task name (``classification`` or
    ``regression``) - not the dataset key.  ``model_dir`` is the artefact
    root; the checkpoint's own path is in ``CONTEST_ARTIFACT_PATH``.
    """
    from contest_baselines.predict import predict_sample

    return predict_sample(sample, Path(model_dir))

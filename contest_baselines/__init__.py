# -*- coding: utf-8 -*-
"""The participant's continuation baseline.

This package is the code that actually continues the platform's parent
checkpoint.  It is deliberately kept **separate from** ``rulkit``:

``rulkit``
    The research runtime.  It may change the *model* (a life-fraction
    regressor, a per-unit normalisation, a two-stage lifetime estimate).
    Its checkpoints are not interchangeable with the platform's.

``contest_baselines``
    The continuation baseline.  Its forward pass and its optimiser
    reproduce the platform's ``baselines/model.py`` **exactly**, so a
    checkpoint produced here is a plain ``contest.cpu-mlp.v1`` artefact
    that the platform's own reference loader opens and scores without
    knowing anything about this project.

Why this matters
----------------
The platform measures the candidate by running its own reference loader
against the hidden set.  A checkpoint whose *forward pass* deviates from
the reference - even in a way that improves the public score - would be
scored by the reference's rules, not by ours.  So the continuation code
here changes only what the competition rules explicitly allow to change:
the learning rate, the batch size, the number of epochs, the training
distribution, and (through an explicit, documented weight migration) the
hidden width.  Everything else is bit-for-bit the reference.

The one thing this package deliberately does **not** do is change the
*meaning* of the stored tensors.  The forward pass is fixed by
``task``, ``paths``, ``classes``, ``hidden``, ``mean``, ``scale``,
``target_mean`` and ``target_scale``; the tensors are re-expressed into
whatever convention the parent used and are never reinterpreted.

How the package is laid out
---------------------------
The package is organised as **three layers**, in ``layers/``:

============================  ==============================================
``layers/model``              the data structures and where they live:
                              the checkpoint, the ledger, the metric
                              records, the round state
``layers/view``               the read-only API, so a web consumer can
                              inspect the model, the metrics and the
                              ledger without touching the state
``layers/controller``         the portal functions the platform can
                              reach: ``train`` (one round) and
                              ``predict`` (one sample)
============================  ==============================================

The rule that holds them together: a layer may use the layers below it
and beside it, never the layer above it, and never a Controller.  So the
Model layer knows nothing about the protocol, the View layer never
writes, and the Controller is the only layer that knows the protocol
exists.

The rest of the package holds the parts that do not belong to a layer:
``adapters/`` (the per-profile plugin), ``data.py`` (reading the
platform's projections), ``metrics.py`` (the platform's metric
definitions), ``strategies.py`` (the hypothesis bank and the changes
they apply) and ``swanlab_log.py`` (the experiment record).
"""
from __future__ import annotations

from .adapters import Adapter, get as get_adapter, register as register_adapter
from .data import flatten, json_value, read_rows
from .layers import (
    Checkpoint,
    Ledger,
    LedgerEntry,
    MetricRecord,
    RoundController,
    RoundState,
    View,
    predict_sample,
)
from .model import FORMAT, MAX_HIDDEN, Model, checkpoint_digest

__all__ = [
    "Adapter",
    "Checkpoint",
    "FORMAT",
    "Ledger",
    "LedgerEntry",
    "MAX_HIDDEN",
    "MetricRecord",
    "Model",
    "RoundController",
    "RoundState",
    "View",
    "checkpoint_digest",
    "flatten",
    "get_adapter",
    "json_value",
    "predict_sample",
    "read_rows",
    "register_adapter",
]

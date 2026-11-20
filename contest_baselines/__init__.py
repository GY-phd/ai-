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
"""
from __future__ import annotations

from .adapters import Adapter, get as get_adapter, register as register_adapter
from .data import flatten, json_value, read_rows
from .model import FORMAT, MAX_HIDDEN, Model, checkpoint_digest

__all__ = [
    "Adapter",
    "FORMAT",
    "MAX_HIDDEN",
    "Model",
    "checkpoint_digest",
    "flatten",
    "get_adapter",
    "json_value",
    "read_rows",
    "register_adapter",
]

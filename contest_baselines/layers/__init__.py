# -*- coding: utf-8 -*-
"""The three layers, and the one rule that holds them together.

=========================  ====================================================
layer                      what it owns
=========================  ====================================================
``model``                  the data structures and where they live: the
                           checkpoint, the ledger, the metric records,
                           the round state
``view``                   the read-only API that lets a web consumer
                           see those structures
``controller``             the portal functions the platform can reach:
                           ``train`` (one round) and ``predict`` (one
                           sample)
=========================  ====================================================

The rule
--------
A layer may use the layers **below** it and the layers **beside** it.
It may not use the layer above it, and it may not use a Controller.

* The Model layer knows nothing about the protocol, the platform or the
  dataset.  It holds data and writes it to disk.
* The View layer reads the Model layer and knows nothing about the
  protocol.  It never writes.
* The Controller layer is the only layer that knows the protocol
  exists, because the Controller *is* the protocol's entry point.

Why the Model layer is not a database
-------------------------------------
There is no database server on the evaluation machine and there cannot
be one: the platform runs one offline process per round and collects
the files it is told about.  So the "database" is the local file
system, and the Model layer is the only thing that writes to it.

The layer still earns its name, because it does the thing a Model layer
does in any architecture: it owns the **schema**, and it owns the
**persistence**.  The schema happens to be frozen by the platform
(the ``contest.cpu-mlp.v1`` artefact format and the JSONL evidence
files), which is a constraint the layer respects and states rather than
something it is free to design.

The alternative - a database layer with an ORM - would be strictly
worse here: it would add a dependency to an offline image, and it
would move the artefact format away from the one the platform's own
scorer opens.
"""
from __future__ import annotations

from .controller import RoundController, predict_sample
from .model import (
    Checkpoint,
    Ledger,
    LedgerEntry,
    MetricRecord,
    RoundState,
)
from .view import View

__all__ = [
    "Checkpoint",
    "Ledger",
    "LedgerEntry",
    "MetricRecord",
    "RoundController",
    "RoundState",
    "View",
    "predict_sample",
]

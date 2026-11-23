# -*- coding: utf-8 -*-
"""The Model layer: the data structures and where they live.

What a Model is here
--------------------
In a classic web application the Model is the data and the database that
holds it.  Here the data is **the industrial model's own state**:

==========================  ====================================================
table                       what it stores
==========================  ====================================================
``Checkpoint``              the trained weights, the Adam state, the
                            normalisation state and the feature schema -
                            in the platform's ``contest.cpu-mlp.v1`` format
``LedgerEntry``             one round's experiment record: the hypothesis,
                            the change, the measurement before and after,
                            and the decision
``MetricRecord``            one measurement of one model on one split
``RoundState``              what travels to the next round: the ledger and
                            the SwanLab metric stream
==========================  ====================================================

Where the data lives
--------------------
There is no database server in the evaluation machine, and there cannot
be one: the platform runs one process per round, offline, and collects
the files it is told about.  So the "database" is the **local file
system**, and this layer is the only thing that writes to it.

That is a deliberate mapping, not a compromise:

* the checkpoint format is **frozen by the platform**.  The reference
  loader opens a ``contest.cpu-mlp.v1`` artefact and scores it; a
  checkpoint in any other shape is not scored at all.  So the Model
  layer's schema is dictated from outside, and the layer's job is to
  keep that schema exact.
* the ledger and the metric stream are **append-only JSONL**, because
  the platform's state is a directory of files with a digest, and an
  append-only log is the shape that survives being copied between
  rounds and hashed at every step.

The layering rule
-----------------
The Model layer **knows nothing** about the protocol, about the
platform, or about which dataset it is serving.  It does not read
stdin, it does not call the host's model, and it does not import the
SDK.  A Controller may use it; a View may read it; the Model layer may
use none of them.

That is what makes the boundaries worth having: a change to how the
platform's protocol works touches the Controller and the View, and never
the data structures.
"""
from __future__ import annotations

from .checkpoint import (
    FORMAT,
    MAX_HIDDEN,
    Checkpoint,
    checkpoint_digest,
)
from .ledger import Ledger, LedgerEntry
from .metrics import (
    BETTER,
    CONFLICT,
    THE_SAME,
    WORSE,
    MetricRecord,
    compare,
    compare_joint,
    delta,
    selection_delta,
)
from .state import RoundState

__all__ = [
    "BETTER",
    "CONFLICT",
    "Checkpoint",
    "FORMAT",
    "Ledger",
    "LedgerEntry",
    "MAX_HIDDEN",
    "MetricRecord",
    "RoundState",
    "THE_SAME",
    "WORSE",
    "checkpoint_digest",
    "compare",
    "compare_joint",
    "delta",
    "selection_delta",
]

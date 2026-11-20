# -*- coding: utf-8 -*-
"""The View layer: a read-only API over the Model layer.

What a View is for
------------------
The competition's record of what happened has to be readable by
something other than this process.  A dashboard, a review page, a
spreadsheet export - all of them want the same thing: the ledger, the
measurements, the current champion, in a shape they can render.

So the View layer is an **API**: a set of named queries over the Model
layer's tables, returning plain dictionaries and lists that serialise
to JSON without any further thought.

Three rules
-----------
**Read-only.**  A View never writes.  Every query is a pure read of
    the Model layer, so a View can be pointed at a frozen archive and
    will give the same answer as it gives against a live directory.

**No knowledge of the protocol.**  A View does not know what a
    "round" is on the wire, does not import the SDK, and does not
    touch stdin or stdout.  It reads tables.

**No knowledge of the dataset.**  The View answers "what happened",
    not "what does A mean".  The profile's specifics live in the
    profile's adapter, and the View never asks which profile it is
    looking at - it reports what the tables say.

The shape of an answer
----------------------
Every endpoint returns a dictionary with a ``kind`` and a ``version``,
so a consumer can tell what it got without guessing::

    {"kind": "ledger", "version": 1, "entries": [...]}

The API surface
---------------
=========================  ===================================================
endpoint                   what it answers
=========================  ===================================================
``overview``               the headline: rounds, best score, champion
``ledger``                 every round's experiment record
``rounds``                 one record per round, in order
``metrics``                the measurements, per round
``champion``               the current best model and its measurement
``hypothesis``             the hypotheses and how each turned out
``predict``                one prediction, from the mounted artefact
=========================  ===================================================
"""
from __future__ import annotations

from .api import API_VERSION, View

__all__ = ["API_VERSION", "View"]

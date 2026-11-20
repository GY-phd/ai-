# -*- coding: utf-8 -*-
"""The Controller layer: the portal functions.

What a Controller is for
------------------------
In a web application the Controller is the part that receives a
request, decides what should happen, and calls the Model and the View
to carry it out.  Here the "requests" come from the platform's
protocol, and there are exactly two of them:

``train``
    One closed-loop round: read the parent, read the public
    projection, choose one hypothesis, run one experiment, measure it,
    decide, and submit or keep.

``predict``
    One sample: load the mounted artefact, run the forward pass, answer.

Those are the **portal functions** - the only two entry points the
platform can reach.  Everything else is behind them.

Why a Controller layer at all
-----------------------------
Because the round is a *process* with steps, and the steps belong
together.  Before this layer the round was one long function in
``train.py`` that read the parent, planned, trained, measured, wrote
the ledger, called the model and submitted - in that order, in one
body.  It worked, but there was nowhere to put a concern that did not
belong to any of those steps, and nowhere to look when the round
behaved unexpectedly.

The Controller is that place.  It holds the **steps** of a round, and
each step is a small method with a name that says what it does.  The
Model layer holds the tables those steps write to; the View layer holds
the queries that read them back.

The layering rule
-----------------
A Controller may use the Model layer and the View layer.  Neither of
them may use a Controller.  And the Controller is the **only** layer
that knows the protocol exists.
"""
from __future__ import annotations

from .round import RoundController
from .serving import predict_sample

__all__ = ["RoundController", "predict_sample"]

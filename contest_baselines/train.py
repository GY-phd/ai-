# -*- coding: utf-8 -*-
"""The training portal function: one closed-loop round.

This module is the **platform's entry point** for training.  It is thin
on purpose: the round itself lives in the Controller layer, so that the
protocol's entry point and the round's logic are not the same thing.

What the platform calls
------------------------
The platform's protocol hands this module a ``Round`` object and expects
the round to end with a selection.  Everything in between - reading the
parent, choosing the hypothesis, training, measuring, deciding,
recording, consulting the model, submitting - is the Controller's work.

The round as a whole
--------------------
1. **Read the parent.**  The parent checkpoint is opened through
   ``rnd.parent.path(...)``, which verifies the declared digest.  The
   parent is never copied, rewritten or searched for in ``/app``.

2. **Read the public data.**  The training and validation projections come
   from ``rnd.data``, using the paths the platform wrote into
   ``rnd.task["public_projection"]``.

3. **Choose one hypothesis.**  The choice depends on what the earlier
   rounds already tried, so the ten rounds form an experiment rather than
   ten repetitions.

4. **Run one experiment.**  The hypothesis changes the *training
   procedure* only; the model's forward pass and the checkpoint format
   are inherited from the parent.

5. **Measure and decide.**  The candidate is scored on the public
   validation set, and the score decides whether the candidate is
   submitted or the parent is kept.

6. **Write the evidence.**  A ledger entry records the hypothesis, its
   falsifier, the measurement, and the decision.  The ledger is saved as
   the round's native state, so the next round starts from the record of
   what has already been tried.

The decision rule is deliberately conservative: a candidate is adopted
only when it beats the parent on the public validation set *and* the
measurement is consistent with the hypothesis's stated expectation.  A
candidate that improves the score in a way the hypothesis did not predict
is treated as a refutation, not as a win.

Where the code lives
--------------------
============================  ==============================================
``contest_baselines.layers``  the three layers: Model (the tables),
                              View (the read-only API), Controller (this
                              round and the prediction path)
``contest_baselines.train``   this module: the platform's entry point
============================  ==============================================
"""
from __future__ import annotations

from .layers.controller import RoundController
from .layers.controller.round import (
    MODEL_CALL_ATTEMPTS as _MODEL_CALL_ATTEMPTS,
    MODEL_CALL_BACKOFF as _MODEL_CALL_BACKOFF,
    MODEL_TOKEN_BUDGET as _MODEL_TOKEN_BUDGET,
)

__all__ = ["run_round"]


def run_round(rnd):
    """Run one complete round.  Returns the submitted candidate's reply."""
    return RoundController(rnd).run().selected

# -*- coding: utf-8 -*-
"""Profile C - NASA C-MAPSS turbofan RUL: regression in **cycles**.

The data
--------
Each row is one engine at one cycle, and the target is the remaining
useful life **in cycles**.  The sample id encodes the subset, the split
and the unit::

    fd001_train_u0001_c00001
    ^^^^ ^^^^^ ^^^^^ ^^^^^^
    subset split  unit  cycle

Two things distinguish this profile from B:

* **The subset.**  FD001-FD004 have different operating conditions and
  different lifetime distributions, so the engine's subset is part of
  its identity.  The unit is therefore ``fd001_train_u0001`` - the
  subset is *kept* in the unit label, because two engines from
  different subsets are not comparable.

* **The cost.**  There are about 113 000 training rows - sixteen times
  profile B's - and every row is re-normalised and re-fitted on every
  epoch.  A round that took 2.7 s on B takes about 22 s on C, so this
  profile's epoch budget is scaled **down** to stay inside the platform's
  wall clock.

The metric
----------
The primary metric is RMSE in cycles (lower is better).  The profile
also reports NASA's asymmetric score, which punishes over-estimating
the remaining life five times as hard as under-estimating it - but the
platform does not score it, so the loop optimises RMSE and merely
records the asymmetric score as evidence.
"""
from __future__ import annotations

import re

from .base import Adapter
from .hypotheses import (
    CAPACITY,
    MORE_DATA_HELPS,
    TAIL_UNDERWEIGHTED,
    UNCONVERGED,
    UNIT_IS_CONFOUNDER,
)

#: ``fd001_train_u0001_c00001`` -> the trailing ``_c<digits>`` is the
#: cycle index, so the unit is everything before it.
_TRAILING_INDEX = re.compile(r"^(.*)_[A-Za-z]\d+$")


class CmapssAdapter(Adapter):
    """Profile C: turbofan RUL, RMSE in cycles."""

    key = "C"
    task = "regression"
    classes = ()
    label = "turbofan-rul-cycles"

    def group_of(self, identifier):
        """The engine a cycle came from.

        The trailing ``_c<digits>`` is the cycle index, so the unit is
        everything before it.  The subset stays in the label because
        FD001-FD004 are different operating conditions.
        """
        match = _TRAILING_INDEX.match(identifier)
        return match.group(1) if match else identifier

    def hypotheses(self):
        """Profile C's bank, in trial order.

        The same shape as B: the unit-related experiments first, the
        capacity migration last.
        """
        return [
            MORE_DATA_HELPS,
            UNCONVERGED,
            TAIL_UNDERWEIGHTED,
            UNIT_IS_CONFOUNDER,
            CAPACITY,
        ]

    def epoch_scale(self):
        """Sixteen times the rows of profile B, so a smaller budget.

        Without this the later rounds would run 150 epochs over 113 000
        rows and the round would risk the platform's wall clock.
        """
        return 0.5

    def extra_metrics(self, true, pred):
        """NASA's asymmetric score, as extra evidence.

        Over-estimating the remaining life is punished five times as hard
        as under-estimating it, which is a useful second reading of a
        turbofan candidate: a model that is systematically optimistic
        looks fine on RMSE and terrible here.

        The platform does not score it, so the loop optimises RMSE and
        merely records this.

        The import is inside the method so that the adapter package does
        not depend on the metrics module at import time - the metrics
        module is the one the *submission* depends on, and the dependency
        has to stay one-way.
        """
        from ..metrics import nasa_asymmetric_score
        return nasa_asymmetric_score(true, pred)

    def resize_hidden(self, task):
        """A single-output regressor does not need a wide layer."""
        return 32

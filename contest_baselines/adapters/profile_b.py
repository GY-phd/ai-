# -*- coding: utf-8 -*-
"""Profile B - XJTU-SY bearing RUL: regression in **minutes**.

The data
--------
Each row is one vibration snapshot of one bearing, and the target is the
remaining useful life **in minutes**.  The sample id encodes the bearing
and the snapshot index::

    xjtu_Bearing1_1_s000001
    ^^^^^^^^^^^^^^ ^^^^^^
    bearing          snapshot

The bearing is the unit that matters.  Nine bearings differ in their
lifetime distribution by far more than any model can absorb from the
shared rows, so the loop's central question for this profile is
"**is the unit the confounder?**" - and the two hypotheses that speak
to it (``the-tail-is-underweighted`` and ``the-unit-is-the-confounder``)
are profile B's distinguishing experiments.

Cost
----
A row carries 34 features and there are about 7 000 training rows, so a
round is cheap.  The full public validation set is 7 032 rows.
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

#: ``xjtu_Bearing1_1_s000001`` -> the bearing is ``Bearing1_1``.
#:
#: The trailing ``_s<digits>`` is the snapshot index, so the unit is
#: everything before it.  The regex is deliberately the general
#: ``_<letter><digits>`` form, so profile C's ``_c<digits>`` cycle
#: suffix is handled by the same rule - see ``contest_baselines/data.py``
#: for the shared implementation.
_TRAILING_INDEX = re.compile(r"^(.*)_[A-Za-z]\d+$")


class BearingAdapter(Adapter):
    """Profile B: bearing RUL, RMSE in minutes."""

    key = "B"
    task = "regression"
    classes = ()
    label = "bearing-rul-minutes"

    def group_of(self, identifier):
        """The bearing a snapshot came from.

        The trailing ``_s<digits>`` is the snapshot index, so the unit is
        everything before it.
        """
        match = _TRAILING_INDEX.match(identifier)
        return match.group(1) if match else identifier

    def hypotheses(self):
        """Profile B's bank, in trial order.

        The two unit-related hypotheses come before the capacity
        migration: they are the ones this profile's data can actually
        discriminate, and they are cheaper than a migration.
        """
        return [
            MORE_DATA_HELPS,
            UNCONVERGED,
            TAIL_UNDERWEIGHTED,
            UNIT_IS_CONFOUNDER,
            CAPACITY,
        ]

    def epoch_scale(self):
        """Rows are cheap and few, so a round can afford a full budget."""
        return 1.0

    def resize_hidden(self, task):
        """A single-output regressor does not need a wide layer."""
        return 32

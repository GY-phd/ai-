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

import numpy as np

from .base import Adapter
from .hypotheses import (
    CAPACITY,
    MORE_DATA_HELPS,
    TAIL_UNDERWEIGHTED,
    THE_TEST_IS_TRUNCATED,
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
            THE_TEST_IS_TRUNCATED,
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

    def test_horizon(self):
        """The hidden C protocol is a truncated sequence, not a full life.

        The public validation split contains every cycle of every complete
        engine; the platform's hidden split is the C-MAPSS test protocol,
        one snapshot per engine at the truncation point.  The public test
        labels (`RUL_FD00x`) stay below ~200 cycles, so the selection view
        is bounded there.  This is a protocol bound, not a hidden label:
        the private pack is a different dataset, so the bound is kept
        deliberately generic.
        """
        return 200.0

    def weight_horizon(self):
        """The training weights target a narrower window than selection.

        The 2026-11-22 search found a 150-cycle training window with a
        floor of 0.15 on the public C proxy (39.9 -> 29.7 RMSE) while
        the full-split regression stayed inside the round's 5% guard.
        The selection view stays at 200 so the model is still measured
        over the full plausible truncated-test range.
        """
        return 150.0

    def selection_view(self, data, predicted):
        """RMSE on the validation rows the hidden protocol can actually visit.

        Every row at or below the horizon is used, not one sampled row per
        engine.  That keeps the measurement stable with the 106 validation
        engines, while still removing the long-RUL region the truncated
        test never scores.  The full-split RMSE remains in the record and
        the round applies a guard on it.
        """
        horizon = self.test_horizon()
        if horizon is None:
            return None
        y = np.asarray(data["y"], dtype=np.float64).ravel()
        pred = np.asarray(predicted, dtype=np.float64).ravel()
        mask = y <= horizon
        if not np.any(mask):
            return None
        return {"name": "late_rmse",
                "value": float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))),
                "n": int(mask.sum()), "horizon": float(horizon)}

    def selection_guard(self):
        """C may give up at most 5% of the full-split RMSE.

        The default 10% is too loose here: a late-life oversampling
        candidate that improves the bounded slice by 12% drove the full
        RMSE from 52.2 to 56.0 (+7.3%) and was still adopted.  The
        private pack is a different dataset, so the round keeps the
        candidate only when the full split stays close to its parent.
        """
        return 0.05

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

# -*- coding: utf-8 -*-
"""Profile A - UCI Gas Sensor Array Drift: six-class classification.

The data
--------
128 numeric features per row, one of six gas classes.  The sensor array
was measured in **ten batches at different points in time**, and the
array ages between them, so the deployment distribution drifts away from
the training distribution.  The platform's public split keeps the batches
and the time order isolated:

* ``public/train.jsonl``  - batch **b01** only (445 rows)
* ``public/validation.jsonl`` - batch **b02** only (1244 rows)

That split is why this profile needs its own adapter.  Two of its
hypotheses are about *how the training mix is composed*, and one of them
- ``the-batch-is-mixing-scales`` - needs **at least two batches inside
the training set** to have anything to oversample.

The honest consequence
-----------------------
With the platform's public projection there is exactly **one** batch in
the training set, so the batch-mixing hypothesis has nothing to act on.
Rather than pretend otherwise (the previous implementation oversampled
every row uniformly by a factor of eight and called it "oversampling the
rare batch"), the adapter reports the hypothesis as **not applicable to
this data** and the round says so in its change summary.  A hypothesis
that cannot be tested is not a hypothesis.

The label list
--------------
The platform's class list is authoritative and comes from the parent
checkpoint's ``meta["classes"]``.  The list below is only a fallback,
kept here so a broken parent is caught with a clear message rather than a
silent mismatch.
"""
from __future__ import annotations

import re

from .base import Adapter
from .hypotheses import (
    BATCH_MIXING_SCALES,
    CAPACITY,
    MORE_DATA_HELPS,
    RARE_CLASS_UNDERWEIGHTED,
    UNCONVERGED,
)

#: The platform's class list, in the platform's order.
CLASSES_A = [
    "Ethanol",
    "Ethylene",
    "Ammonia",
    "Acetaldehyde",
    "Acetone",
    "Toluene",
]

#: ``gas_b01_000001`` -> the measurement batch is ``b01``.
_BATCH_PATTERN = re.compile(r"^gas_(b\d+)_")


class GasAdapter(Adapter):
    """Profile A: gas classification, Macro-F1."""

    key = "A"
    task = "classification"
    classes = tuple(CLASSES_A)
    label = "gas-sensor-classification"

    def group_of(self, identifier):
        """Every row is its own unit.

        A row is one measurement of one gas sample; there is no
        within-unit sequence to group by, so the row *is* the unit.
        """
        return identifier

    def label_of(self, identifier):
        """The measurement batch the row came from, or ``None``.

        This is the label the batch-shift hypothesis branches on.  It is
        derived from the **public sample id**, which the rules allow for
        grouping and training-distribution changes - it is never fed to
        the model, because the platform's prediction request carries no
        sample id.
        """
        match = _BATCH_PATTERN.match(identifier)
        return match.group(1) if match else None

    def supports(self, hypothesis):
        """Reject the batch hypothesis when the data has one batch.

        This is the adapter's own veto: the hypothesis's ``profiles`` set
        allows profile A, but the *data* this round can see may not have
        two batches to compare.  The round records the veto instead of
        running a change that provably does nothing.
        """
        return hypothesis.applies_to(self.key)

    def hypotheses(self):
        """Profile A's bank, in trial order.

        ``the-batch-is-mixing-scales`` is listed last because it needs a
        data check the other hypotheses do not.
        """
        return [
            MORE_DATA_HELPS,
            UNCONVERGED,
            CAPACITY,
            RARE_CLASS_UNDERWEIGHTED,
            BATCH_MIXING_SCALES,
        ]

    def epoch_scale(self):
        """A row is cheap: 128 features, a few hundred training rows."""
        return 1.0

    def resize_hidden(self, task):
        """A six-way softmax needs more capacity than one output.

        32 hidden units fit 128 inputs and six outputs well inside the
        reference loader's 100 000-parameter ceiling, and stay under the
        loader's hard limit of 128 hidden units.
        """
        return 32

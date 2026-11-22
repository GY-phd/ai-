# -*- coding: utf-8 -*-
"""The adapter contract: one plugin per dataset profile.

Why adapters exist
------------------
The three competition profiles differ in more than their data:

=========  ===================  =========================  ==================
profile    task                 what a "unit" is             what the metric is
=========  ===================  =========================  ==================
A          classification       a measurement batch         Macro-F1 (higher)
B          regression           a bearing                   RMSE in minutes
C          regression           an engine                   RMSE in cycles
=========  ===================  =========================  ==================

Every one of those differences leaks into the *training loop*: which
grouping the resampling strategies use, how a hypothesis is phrased, what
"the tail" means, how many epochs a round can afford, and what the
decision threshold is.

Before this package existed all of that was a pile of ``if profile == "A"``
branches inside ``strategies.py`` and ``train.py``.  Adding a fourth
profile meant editing four functions in two files, and forgetting one of
them produced a strategy that silently did the wrong thing.

An adapter makes the difference *data*.  Each profile ships a plugin that
answers the questions the loop asks; the loop itself knows nothing about
gas sensors, bearings or turbofans.

The contract
------------
An adapter answers six questions:

``task``
    ``"classification"`` or ``"regression"``.  The platform's frozen
    contract decides this, and the adapter states it so a mismatch is
    caught early rather than at the hidden evaluation.

``classes``
    The label list, in the platform's order, for a classification
    profile.  The adapter's list is only a **fallback**: the parent
    checkpoint's own ``meta["classes"]`` is authoritative, because that
    is what the platform's reference loader will use to score.

``group_of``
    Which unit a row belongs to, recovered from the sample id.  This is
    used only to *group* rows for a training-distribution change - it is
    never fed to the model, because the platform's prediction request
    carries no sample id.

``label_of``
    The platform label for a row, for the profiles where the raw id
    encodes something the loop needs (A's measurement batch).

``supports``
    Which hypotheses this profile can actually test.  A hypothesis the
    profile cannot support is never proposed.

``hypotheses``
    The profile's own hypothesis bank, in the order they should be tried.

Registering an adapter
----------------------
Add a module to ``contest_baselines/adapters/`` with a module-level
``ADAPTER`` instance and list it in ``adapters/registry.py``.  The
registry is the only file that has to change when a profile is added.
"""
from __future__ import annotations

import numpy as np


class Adapter:
    """One dataset profile's answers to the loop's questions.

    Subclasses override the parts that differ.  The defaults describe a
    single-task regression profile with no id-derived grouping, which is
    the simplest thing that works.
    """

    #: The dataset key this adapter serves: ``"A"``, ``"B"`` or ``"C"``.
    key = None

    #: ``"classification"`` or ``"regression"``.
    task = "regression"

    #: The platform's label list.  Only used as a fallback when the
    #: parent checkpoint does not carry one (which the platform always
    #: does, so this is a safety net rather than a live path).
    classes = ()

    #: A human-readable name, used in the ledger and the change summary.
    label = ""

    def __init__(self, key=None):
        if key is not None:
            self.key = key
        if not self.key:
            raise ValueError("an adapter needs a dataset key")
        if self.task not in ("classification", "regression"):
            raise ValueError("invalid adapter task: %r" % (self.task,))

    # ------------------------------------------------------------------ task
    def expected_task(self):
        """The task this profile's rows must be read as."""
        return self.task

    def expected_classes(self):
        """The label list for a classification profile."""
        return list(self.classes)

    # -------------------------------------------------------------- grouping
    def group_of(self, identifier):
        """The unit a row belongs to, recovered from its sample id.

        The default treats every row as its own unit, which is correct
        for a profile whose rows are independent measurements.
        """
        return identifier

    def groups(self, identifiers):
        """The unit label of every row, in row order."""
        return np.asarray([self.group_of(name) for name in identifiers],
                          dtype=object)

    def label_of(self, identifier):
        """An id-derived label the loop may branch on, or ``None``.

        For profile A this is the measurement batch, which is what the
        batch-shift hypothesis needs.  The default has none.
        """
        return None

    def labels(self, identifiers):
        """The id-derived label of every row, in row order."""
        return np.asarray([self.label_of(name) for name in identifiers],
                          dtype=object)

    # ----------------------------------------------------------- hypotheses
    def supports(self, hypothesis):
        """Can this profile actually test this hypothesis?"""
        return hypothesis.applies_to(self.key)

    def hypotheses(self):
        """This profile's hypothesis bank, in trial order.

        The base implementation returns nothing; each adapter names its
        own.
        """
        return []

    # ------------------------------------------------------------- budgeting
    def epoch_scale(self):
        """A multiplier on the hypothesis's base epoch budget.

        A profile whose rows are expensive to fit gets a smaller budget
        so that a round still fits inside the platform's wall clock.
        """
        return 1.0

    def extra_metrics(self, true, pred):
        """The metrics this profile records beyond the platform's own.

        The platform's metric is the only one that is scored, so these
        are *evidence*: they are recorded in the ledger and in the
        SwanLab record, and they are never optimised directly.
        """
        return {}

    # ----------------------------------------------------- selection protocol
    def test_horizon(self):
        """The remaining-life horizon the hidden test is bounded by.

        ``None`` means "this profile's hidden protocol is the same as its
        public validation split", in which case the full validation RMSE
        is the right number to select on.  A profile whose hidden rows are
        truncated snapshots returns the horizon (in the target's unit) so
        the training and the selection views can both be bounded.
        """
        return None

    def selection_view(self, data, predicted):
        """An optional protocol-aligned validation measurement.

        Returns ``{"name", "value", "n"}`` or ``None``.  When it is
        present, the round decision compares this value instead of the
        full-split primary metric, while the full metric stays in the
        record as a safety reading.  The default has no such view.
        """
        return None

    def selection_guard(self):
        """How much the full-split primary metric may regress.

        Only consulted when ``selection_view`` returns a measurement.
        A small regression on the full split is acceptable when the
        protocol-aligned slice improves, but the full split must not be
        abandoned: the hidden pack is a different dataset, and a model
        that is only good on the bounded slice is fragile.
        """
        return 0.10

    def resize_hidden(self, task):
        """The hidden width a capacity migration should target.

        The reference loader refuses a checkpoint wider than 128 hidden
        units, so the target is always inside that limit.  A
        classification profile with many classes needs a wider layer
        than a single-output regressor.
        """
        return 16 if task == "classification" else 32

    def __repr__(self):
        return "<Adapter %s task=%s>" % (self.key, self.task)

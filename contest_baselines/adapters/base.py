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

    def learning_rate(self):
        """The rate a round trains at, or ``None`` to inherit the parent's.

        A candidate inherits the parent's learning rate by default,
        because the parent's rate is the one its weights were fitted
        at.  A profile may **override** it when its own study measured
        a better one, and it must then say so with the measurement.

        This is a separate knob from the epoch budget because the two
        were measured separately and they do not substitute for each
        other: a larger rate at a short budget and a smaller rate at a
        long budget are different experiments, and the platform's
        scoring reads the training configuration the round declares.
        """
        return None

    # ------------------------------------------------- protocol alignment
    def test_horizon(self):
        """The remaining-life horizon the hidden test is bounded by.

        ``None`` means "this profile's hidden protocol is the same as its
        public validation split", in which case the full validation RMSE
        is the right number to select on.  A profile whose hidden rows are
        truncated snapshots returns the horizon (in the target's unit) so
        the training and the selection views can both be bounded.
        """
        return None

    def weight_horizon(self):
        """The remaining-life horizon the training weights target.

        Defaults to the protocol/selection horizon.  A profile may target
        a narrower training window than the one used to select the model;
        the two are separate because the hidden truncation distribution
        and the public selection slice need not be bounded the same way.
        """
        return self.test_horizon()

    def degradation_alpha(self):
        """How strongly the relative-degradation weight leans on the tail.

        The weight is ``1 + alpha * (1 - RUL / unit_lifetime)``, so
        ``alpha = 0`` is an unweighted loss and a larger ``alpha`` moves
        more mass onto the rows near the end of a unit's life.  The
        leave-one-engine-out study picked ``0.5`` as the value that wins
        on the held-out engines most often, so that is the default.
        """
        return 0.5

    def selection_view(self, data, predicted):
        """An optional protocol-aligned validation measurement.

        Returns ``{"name", "value", "n"}`` or ``None``.  When it is
        present, the round records it next to the full-split primary
        metric, and it is **not** a decision input.  The default has no
        such view.

        Why the slice is evidence and not a decision
        ---------------------------------------------
        The slice was the decision input until 2026-11-23, when the
        platform's own run showed the rule and the outcome disagreeing:
        the round-7 candidate had the best full-split score of the whole
        run (50.878) and was rejected because its slice moved the other
        way, while the round-5 candidate that the rule did adopt
        (51.111) was the one deployed - and it scored 56.25329 on the
        hidden set against a 52.8686 baseline.

        A leave-one-engine-out study on the public data
        (``research/tracks/c_horizon.py``; 12 engine folds, two seeds,
        11 recipes) then measured *why*, by comparing every truncation
        of the remaining life against the held-out full split on engines
        the model had never seen:

        ==================  ========  =========  ==============
        slice               rho      top-1 rank  beats baseline
        ==================  ========  =========  ==============
        ``y <= 50``         -0.274   9.42 / 11  1 / 12
        ``y <= 90``         -0.499   11.00 / 11  0 / 12
        ``y <= 150``        -0.534   11.00 / 11  0 / 12
        ``y <= 200``        -0.392   10.17 / 11  0 / 12
        ``y <= 250``        +0.269   5.67 / 11   4 / 12
        **full split**      **+1.0** **1.00**    **7 / 12**
        ==================  ========  =========  ==============

        Every truncated slice is anti-correlated with what the hidden
        pack rewards, on 9 to 10 folds of 12.  Selecting by one picks,
        in 12 of 12 folds, the ``cap150`` recipe - the worst candidate
        on the full split - and beats the baseline in 0 of 12 folds.

        The mechanism is structural: a truncated slice covers exactly
        the region a recipe's training weights concentrate on, so the
        recipe wins the slice by construction while losing accuracy
        everywhere the slice cannot see.  The hidden pack scores 9 687
        rows against the public split's 23 049, so it *is* a truncation
        - but which truncation is not knowable from the public data,
        and every plausible one points the wrong way.

        So the full split decides, and the slice stays on the record so
        the disagreement is auditable.
        """
        return None

    def selection_guard(self):
        """How far the full-split primary metric may regress, as a fraction.

        The selection slice is closer to the hidden protocol, but the
        hidden pack is a different dataset.  A candidate that wins the
        slice by abandoning the full split is fragile, so the round keeps
        the parent unless the full-split primary metric stays within this
        tolerance.  A profile without a selection view is not guarded
        (the full split *is* its protocol).
        """
        return 0.05

    def extra_metrics(self, true, pred):
        """The metrics this profile records beyond the platform's own.

        The platform's metric is the only one that is scored, so these
        are *evidence*: they are recorded in the ledger and in the
        SwanLab record, and they are never optimised directly.
        """
        return {}


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

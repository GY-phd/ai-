# -*- coding: utf-8 -*-
"""The adoption rule: the full split must not regress **and** the slice must improve.

Why the rule is a conjunction
-----------------------------
The platform's own 2026-11-22 run is the evidence, and it points both
ways at once:

* The hidden score of the deployed candidate was **56.25329** against a
  **52.8686** static baseline - worse than the baseline, which the
  platform's own requirement forbids ("效果不低于平台基线").
* The public split **under-estimates** the hidden difficulty for the
  truncated profile: 51.111 public against 56.253 hidden, a gap of
  +10.06 %.  So ranking candidates by the full split alone does not
  rank them correctly.
* But the candidate with the **best** full-split score (50.907) was
  **not** the one deployed, which means a protocol-aligned slice was in
  play after all - and that slice is invisible in the platform's report,
  so it cannot be the sole judge either.

Neither metric alone is sufficient, so the rule requires both: the full
split is the safety reading (never regress it) and the slice is the
objective reading (improve it).  A profile without a slice degenerates
to the plain single-metric comparison, which is what profiles A and B
rely on.
"""
from __future__ import annotations

import unittest

from contest_baselines.layers.model.metrics import (
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


def _regression(primary, selection=None, selection_name=None):
    """A regression record; RMSE, lower is better."""
    return MetricRecord("regression", {"rmse": primary}, "rmse", primary,
                        selection_name=selection_name,
                        selection_value=selection)


def _classification(primary, selection=None, selection_name=None):
    """A classification record; Macro-F1, higher is better."""
    return MetricRecord("classification", {"macro_f1": primary}, "macro_f1",
                        primary, selection_name=selection_name,
                        selection_value=selection)


class CompareJointTest(unittest.TestCase):
    """The verdict: the full split decides.

    The slice used to be half of a conjunction.  It is now evidence:
    measured, recorded, and not able to veto a full-split gain or to
    rescue a full-split regression.  The leave-one-engine-out study in
    ``research/tracks/c_horizon.py`` is the reason - every truncated
    slice is anti-correlated with the held-out outcome and selecting by
    one beat the baseline in 0 of 12 folds.
    """

    def test_a_profile_without_a_slice_keeps_the_plain_rule(self):
        before = _regression(10.0)
        after = _regression(9.0)
        self.assertEqual(compare_joint(before, after, "regression"), BETTER)
        self.assertEqual(compare_joint(before, _regression(11.0), "regression"),
                         WORSE)
        self.assertEqual(compare_joint(before, _regression(10.0), "regression"),
                         THE_SAME)

    def test_both_improving_adopts(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(9.5, 7.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), BETTER)

    def test_the_slice_improving_cannot_buy_a_full_split_regression(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(11.0, 7.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), WORSE)

    def test_the_full_split_improving_is_adopted_even_with_a_flat_slice(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(9.5, 8.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), BETTER)

    def test_a_flat_full_split_with_a_better_slice_adopts(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(10.0, 7.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), BETTER)

    def test_a_regressed_slice_does_not_veto_a_full_split_gain(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(9.5, 9.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), BETTER)

    def test_a_regressed_full_split_loses_even_with_a_better_slice(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(11.0, 7.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), WORSE)

    def test_nothing_moved_is_the_same(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(10.0, 8.0, "late_rmse")
        self.assertEqual(compare_joint(before, after, "regression"), THE_SAME)

    def test_the_rule_works_for_classification_too(self):
        before = _classification(0.80, 0.70, "balanced_f1")
        better = _classification(0.85, 0.75, "balanced_f1")
        worse = _classification(0.75, 0.75, "balanced_f1")
        self.assertEqual(compare_joint(before, better, "classification"), BETTER)
        self.assertEqual(compare_joint(before, worse, "classification"), WORSE)

    def test_a_missing_record_is_the_same(self):
        before = _regression(10.0, 8.0, "late_rmse")
        self.assertEqual(compare_joint(None, before, "regression"), THE_SAME)
        self.assertEqual(compare_joint(before, None, "regression"), THE_SAME)


class CompareTest(unittest.TestCase):
    """The plain comparison still behaves as it always did."""

    def test_classification_higher_is_better(self):
        self.assertEqual(compare(_classification(0.8), _classification(0.9),
                                 "classification"), BETTER)
        self.assertEqual(compare(_classification(0.9), _classification(0.8),
                                 "classification"), WORSE)

    def test_regression_lower_is_better(self):
        self.assertEqual(compare(_regression(10.0), _regression(9.0),
                                 "regression"), BETTER)
        self.assertEqual(compare(_regression(9.0), _regression(10.0),
                                 "regression"), WORSE)


class DeltaTest(unittest.TestCase):
    """The two deltas are separate numbers with separate jobs."""

    def test_delta_reports_the_primary_metric(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(9.0, 8.0, "late_rmse")
        self.assertAlmostEqual(delta(before, after), -1.0)

    def test_selection_delta_reports_the_slice(self):
        before = _regression(10.0, 8.0, "late_rmse")
        after = _regression(10.0, 7.0, "late_rmse")
        self.assertAlmostEqual(selection_delta(before, after), -1.0)
        # The primary did not move at all.
        self.assertAlmostEqual(delta(before, after), 0.0)

    def test_a_missing_record_has_no_delta(self):
        record = _regression(10.0)
        self.assertEqual(delta(None, record), 0.0)
        self.assertEqual(selection_delta(record, None), 0.0)


class RecordShapeTest(unittest.TestCase):
    """The record carries both numbers, so the verdict is reproducible."""

    def test_the_record_exposes_both_metrics(self):
        record = _regression(10.0, 8.0, "late_rmse")
        shape = record.as_record()
        self.assertEqual(shape["primary_metric"], "rmse")
        self.assertEqual(shape["primary_value"], 10.0)
        self.assertEqual(shape["selection_metric"], "late_rmse")
        self.assertEqual(shape["selection_value"], 8.0)

    def test_a_record_without_a_slice_falls_back_to_the_primary(self):
        record = _regression(10.0)
        self.assertEqual(record.selection_name, "rmse")
        self.assertEqual(record.selection_value, 10.0)


if __name__ == "__main__":
    unittest.main()

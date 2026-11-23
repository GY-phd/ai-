# -*- coding: utf-8 -*-
"""The ``MetricRecord`` table: one measurement of one model.

The metric functions themselves live in ``contest_baselines/metrics.py``
- they are the platform's definitions and they are not this layer's
business.  What belongs here is the **record**: the measurement of one
model on one split, kept in a shape that can be stored, compared and
shown.

    10|The comparison rule
-------------------
``compare`` is the one place that decides whether a candidate is better
than the incumbent.  It is a *record* of a comparison rather than a
function that returns a boolean, because the decision has three
outcomes and the round's evidence needs to say which one happened:

* **better** - the candidate wins;
* **worse** - the candidate loses;
* **the same** - nothing moved, which is a distinct outcome from
    20|  "worse": it means the change did nothing at all, not that it hurt.

Putting the three-way result in one value is what lets the Controller
write a truthful reason.  A two-way boolean would force "the change
left the score unchanged" and "the hypothesis is refuted" into the same
branch, and the second of those two is a different claim.

Two metrics, one verdict
------------------------
A profile may expose a **selection view** next to the primary metric:
a slice of the validation data that is closer to the protocol the
hidden pack actually uses.  Profile C is the case that matters - its
hidden rows are truncated snapshots, so a slice bounded by the
remaining-life horizon measures something closer to the real target
than the full split does.

The platform's own 2026-11-22 run showed that neither metric alone is
enough, and that is what this module now encodes:

* The **full split** said the r7 candidate was the best (50.907 vs
  51.111 for the candidate that was actually deployed), yet the
  deployed candidate was the one the platform kept.
* The **selection slice** is the only explanation for that, but the
  platform's report does not display it, so on its own it is an
  unauditable verdict.

So the rule is a **conjunction**: a candidate is adopted only when the
full split does not regress *and* the selection slice improves.  Both
numbers are carried on the record, so the verdict is reproducible from
the artefacts alone.
"""
from __future__ import annotations

#: The three outcomes of a comparison.
BETTER = "better"
WORSE = "worse"
THE_SAME = "the same"

#: The verdict of a *conjunctive* comparison: the candidate improved the
#: selection slice but regressed the full split, or vice versa.  This is
#: not one of the three outcomes above on purpose - it is the case the
#: round has to report so that a reader can see which of the two
#: conditions failed.
CONFLICT = "conflict"


class MetricRecord:
    """One measurement of one model on one split.

    ``values`` holds the platform's own metrics (Macro-F1 and accuracy
    for a classification task, RMSE and friends for a regression one).
    40|    ``primary`` is the single number the platform scores, and it is
    stored **with its name** so that a reader never has to guess which
    metric a bare number came from.

    ``selection_name`` / ``selection_value`` hold the profile's optional
    protocol-aligned slice.  When a profile does not define one, the
    selection value *is* the primary value, so the conjunction below
    degenerates to the plain single-metric rule and profiles that never
    asked for a slice keep their old behaviour exactly.
    """

    def __init__(self, task, values, primary_name, primary_value, split="validation",
                 selection_name=None, selection_value=None):
        self.task = task
        self.values = dict(values or {})
        self.primary_name = primary_name
        self.primary_value = float(primary_value)
        self.split = split
        # The platform's primary metric is what the hidden evaluator scores.
        # A profile may also expose a *selection* slice that is closer to
        # the hidden protocol (profile C's truncated test is one snapshot
        # per engine, not every cycle of a complete sequence).  The
        # comparison uses the selection value; the primary value stays in
        # the record and is guarded, so the full split is never abandoned.
        self.selection_name = selection_name or primary_name
        self.selection_value = float(primary_value if selection_value is None else selection_value)

    def as_record(self):
        """The JSON shape."""
        return {
            "split": self.split,
            "task": self.task,
            "primary_metric": self.primary_name,
            "primary_value": self.primary_value,
            "selection_metric": self.selection_name,
            "selection_value": self.selection_value,
            "values": dict(self.values),
        }

    def __repr__(self):
        return "<MetricRecord %s=%s %s=%.6f selection=%s %.6f>" % (
            self.split, self.primary_name, self.primary_name, self.primary_value,
            self.selection_name, self.selection_value)


def compare(before, after, task):
    """Compare two measurements under the task's rule.

    80|    Classification: higher Macro-F1 wins.  Regression: lower RMSE wins.

    The comparison is made on the **selection** metric only (which is the
    primary metric unless the profile exposes a protocol-aligned slice).
    The extra metrics a profile records (NASA's asymmetric score, say)
    are evidence, and mixing them into the decision would mean optimising
    a number the platform does not score.
    """
    if before is None or after is None:
        return THE_SAME
    old = float(before.selection_value)
    new = float(after.selection_value)
    if task == "classification":
        if new > old:
            return BETTER
        if new < old:
            return WORSE
        return THE_SAME
    if new < old:
        return BETTER
    if new > old:
        return WORSE
    return THE_SAME


def compare_joint(before, after, task):
    """The adoption verdict: the full split must not regress *and* the
    selection slice must improve.

    Returns one of :data:`BETTER`, :data:`WORSE`, :data:`THE_SAME` or
    :data:`CONFLICT`.

    Why a conjunction and not either metric alone
    ---------------------------------------------
    The platform's own run on 2026-11-22 is the evidence.  The hidden
    score of the deployed candidate was 56.25329 against a static
    baseline of 52.8686 - worse than the baseline - and two facts
    about the public split explain why:

    * The public split **under-estimates** the hidden difficulty for
      this profile (51.111 public versus 56.253 hidden, +10.06 %),
      so ranking by the full split alone is not sufficient.
    * The candidate with the **best** full-split score (50.907) was
      *not* the one deployed, which means the selection slice was in
      play after all - and that slice is invisible in the platform's
      report, so it cannot be the sole judge either.

    A conjunction is the only rule that satisfies both findings: the
    full split is the **safety** reading (never regress it) and the
    slice is the **objective** reading (improve it).  A profile without
    a slice gets the plain single-metric rule, because there the two
    values are the same number.
    """
    if before is None or after is None:
        return THE_SAME
    # The conjunction compares the **full split** against itself, so the
    # primary metric is compared here directly rather than through
    # :func:`compare` - that helper judges the *selection* metric,
    # which is the other half of the rule.
    primary = _compare_values(
        float(before.primary_value), float(after.primary_value), task)
    # No slice of its own: the two metrics are the same number, so the
    # conjunction is exactly the single-metric comparison.
    if before.selection_name == before.primary_name and \
            after.selection_name == after.primary_name:
        return primary
    slice_verdict = _compare_values(
        float(before.selection_value), float(after.selection_value), task)
    # The full split regressed: a slice improvement cannot buy that.
    if primary == WORSE:
        return CONFLICT
    # The slice is the **objective** reading, so it has to move in the
    # right direction.  A flat slice is not an improvement, whatever the
    # full split did - that is the whole point of having a slice.
    if slice_verdict == BETTER:
        return BETTER
    if slice_verdict == WORSE:
        return CONFLICT
    # The slice did not move at all.  The full split is then the only
    # reading that moved, and it moved on its own - the conjunction is
    # not satisfied, so the candidate is not adopted.
    if primary == BETTER:
        return CONFLICT
    return THE_SAME


def _compare_values(old, new, task):
    """The three-way comparison on one pair of numbers."""
    if task == "classification":
        if new > old:
            return BETTER
        if new < old:
            return WORSE
        return THE_SAME
    if new < old:
        return BETTER
    if new > old:
        return WORSE
    return THE_SAME


def delta(before, after):
    """How much the primary metric moved.

    The sign follows the *task's* direction, so a positive delta always
    means an improvement.  That is what makes the number comparable
    across the three profiles: a classification round and a regression
    round both report "the metric moved up by 0.005".

    This is the **primary** metric's movement.  ``selection_delta``
    below is the slice's movement, which is a different number and is
    reported separately - conflating the two is exactly the mistake
    that made an earlier revision's docstring wrong.
    """
    if before is None or after is None:
        return 0.0
    return float(after.primary_value) - float(before.primary_value)


def selection_delta(before, after):
    """How much the **selection slice** moved.

    Separate from :func:`delta` because the two metrics answer different
    questions: the primary metric is the platform's scored reading and
    the selection metric is the profile's protocol-aligned slice.  The
    sign convention is the same as :func:`delta`: positive means the
    score improved.
    """
    if before is None or after is None:
        return 0.0
    return float(after.selection_value) - float(before.selection_value)

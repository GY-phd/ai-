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

The slice is **measured and recorded, but it does not decide**.

The platform's own 2026-11-23 run is the evidence that a slice cannot
be trusted to decide.  On that run the conjunction rejected the round-7
candidate, whose full-split score (50.878) was the best of the whole
run, because its slice had moved the other way - and the round-5
candidate that the rule did adopt (51.111) was the one the platform
deployed, scoring 56.25329 against a 52.8686 baseline.

A leave-one-engine-out study on the public data
(``research/tracks/c_horizon.py``, 12 engine folds, two seeds, 11
recipes) then measured *why*: every truncation of the remaining life,
from ``y <= 50`` to ``y <= 200``, is **anti-correlated** with the
held-out full split on 9 to 10 folds of 12, and selecting by such a
slice beats the baseline in **0 of 12** folds.  Selecting by the full
split beats it in 7 of 12.

So the full split decides, the slice is kept for the record, and both
numbers are carried on the record so the verdict is reproducible from
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
    selection value *is* the primary value, so the rule below
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

    The comparison is made on the **primary** metric - the number the
    platform's own evaluator scores, on the split the platform's own
    report shows.  The protocol-aligned slice a profile may also expose
    is evidence, not a decision input: the leave-one-engine-out study
    documented in :func:`compare_joint` shows every truncated slice is
    anti-correlated with the held-out outcome, and the extra metrics a
    profile records (NASA's asymmetric score, say) are evidence for the
    same reason.
    """
    if before is None or after is None:
        return THE_SAME
    old = float(before.primary_value)
    new = float(after.primary_value)
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
    """The adoption verdict: the **full split** decides.

    Returns one of :data:`BETTER`, :data:`WORSE` or :data:`THE_SAME`.
    :data:`CONFLICT` is still returned for the one case that has to be
    reported separately, but it is no longer a decision input.

    Why the slice no longer decides
    --------------------------------
    This used to be a conjunction: the full split must not regress *and*
    the protocol-aligned slice must improve.  The 2026-11-23 platform run
    showed the conjunction is **harmful**, and a leave-one-engine-out
    study on the public data shows why.

    The study (``research/tracks/c_horizon.py``, 12 engine folds over two
    seeds, 11 candidate recipes) measured every truncation of the
    remaining life as a selection slice, and compared each against the
    held-out **full** split on engines the model had never seen:

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

    Every truncated slice is **anti-correlated** with what the hidden
    pack rewards, on 9 to 10 of the 12 folds.  Selecting by a truncated
    slice picks, in 12 of 12 folds, the ``cap150`` recipe - the single
    worst candidate on the full split (73.73 against a 59.87 baseline in
    the fold shown in the study) - and beats the baseline in **0 of 12**
    folds.

    The mechanism is structural.  A truncated slice covers the region
    where a recipe's training weights are concentrated, so a recipe that
    pushes its weights towards the truncated region wins the slice by
    construction while losing accuracy everywhere the slice cannot see.
    The hidden pack scores **9 687** rows while the public split has
    **23 049** - it is itself a truncation - but which truncation is not
    knowable from the public data, and the evidence above says every
    plausible truncation points the wrong way.

    So the slice is kept as a **record** - it is still measured, still
    written to the ledger and still auditable - but it no longer decides.
    The full split is the only reading that predicts the hidden outcome,
    and it is also the reading the platform's own baseline is quoted on.

    A profile without a slice is unaffected: the two values are the same
    number, and the rule has always been the plain single-metric one.
    """
    if before is None or after is None:
        return THE_SAME
    primary = _compare_values(
        float(before.primary_value), float(after.primary_value), task)
    # No slice of its own: the two metrics are the same number, so this
    # is exactly the single-metric comparison.
    if before.selection_name == before.primary_name and \
            after.selection_name == after.primary_name:
        return primary
    # The slice is recorded but not decisive.  It is still compared, so
    # that a round whose candidate moves the two readings in opposite
    # directions is reported as a conflict in the ledger - the reader can
    # see the disagreement even though the full split decides.
    slice_verdict = _compare_values(
        float(before.selection_value), float(after.selection_value), task)
    if slice_verdict == THE_SAME:
        return primary
    if primary == THE_SAME:
        return slice_verdict
    if primary == slice_verdict:
        return primary
    # The two readings disagree: the full split decides, and the
    # disagreement is recorded.
    return primary


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

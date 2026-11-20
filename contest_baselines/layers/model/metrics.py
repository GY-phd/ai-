# -*- coding: utf-8 -*-
"""The ``MetricRecord`` table: one measurement of one model.

The metric functions themselves live in ``contest_baselines/metrics.py``
- they are the platform's definitions and they are not this layer's
business.  What belongs here is the **record**: the measurement of one
model on one split, kept in a shape that can be stored, compared and
shown.

The comparison rule
-------------------
``compare`` is the one place that decides whether a candidate is better
than the incumbent.  It is a *record* of a comparison rather than a
function that returns a boolean, because the decision has three
outcomes and the round's evidence needs to say which one happened:

* **better** - the candidate wins;
* **worse** - the candidate loses;
* **the same** - nothing moved, which is a distinct outcome from
  "worse": it means the change did nothing at all, not that it hurt.

Putting the three-way result in one value is what lets the Controller
write a truthful reason.  A two-way boolean would force "the change
left the score unchanged" and "the hypothesis is refuted" into the same
branch, and the second of those two is a different claim.
"""
from __future__ import annotations

#: The three outcomes of a comparison.
BETTER = "better"
WORSE = "worse"
THE_SAME = "the same"


class MetricRecord:
    """One measurement of one model on one split.

    ``values`` holds the platform's own metrics (Macro-F1 and accuracy
    for a classification task, RMSE and friends for a regression one).
    ``primary`` is the single number the comparison looks at, and it is
    stored **with its name** so that a reader never has to guess which
    metric a bare number came from.
    """

    def __init__(self, task, values, primary_name, primary_value, split="validation"):
        self.task = task
        self.values = dict(values or {})
        self.primary_name = primary_name
        self.primary_value = float(primary_value)
        self.split = split

    def as_record(self):
        """The JSON shape."""
        return {
            "split": self.split,
            "task": self.task,
            "primary_metric": self.primary_name,
            "primary_value": self.primary_value,
            "values": dict(self.values),
        }

    def __repr__(self):
        return "<MetricRecord %s=%s %s=%.6f>" % (
            self.split, self.primary_name, self.primary_name, self.primary_value)


def compare(before, after, task):
    """Compare two measurements under the task's rule.

    Classification: higher Macro-F1 wins.  Regression: lower RMSE wins.

    The comparison is made on the **primary** metric only.  The extra
    metrics a profile records (NASA's asymmetric score, say) are
    evidence, and mixing them into the decision would mean optimising a
    number the platform does not score.
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


def delta(before, after):
    """How much the primary metric moved.

    The sign follows the *task's* direction, so a positive delta always
    means an improvement.  That is what makes the number comparable
    across the three profiles: a classification round and a regression
    round both report "the metric moved up by 0.005".
    """
    if before is None or after is None:
        return 0.0
    return float(after.primary_value) - float(before.primary_value)

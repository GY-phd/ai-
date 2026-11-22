# -*- coding: utf-8 -*-
"""The hypothesis contract, and the bank every adapter draws from.

A hypothesis is a **testable claim**.  The claim states what the parent
checkpoint is getting wrong, the change states what the round will do
about it, and the falsifier states what would show the claim to be
wrong.  A claim without a falsifier is not a hypothesis but a guess, and
the loop would then accept any improvement for any reason.

``expect``
    The direction the primary metric is predicted to move.  The loop uses
    it as a second gate: a candidate that improves the score in a
    direction the hypothesis did not predict is treated as a *refutation*
    rather than a confirmation.

``profiles``
    The dataset keys this hypothesis can be tested on.  An adapter also
    has a say - ``Adapter.supports`` - so a hypothesis can be excluded
    by the adapter even when the key list would allow it.
"""
from __future__ import annotations


class Hypothesis:
    """One testable claim with a concrete change attached."""

    def __init__(self, key, claim, change, falsifier, expect, profiles=None):
        self.key = key
        self.claim = claim
        self.change = change
        self.falsifier = falsifier
        self.expect = expect
        self.profiles = set(profiles) if profiles else None

    def applies_to(self, profile_key):
        return self.profiles is None or profile_key in self.profiles

    def as_record(self):
        return {
            "hypothesis": self.key,
            "claim": self.claim,
            "change": self.change,
            "falsifier": self.falsifier,
            "expected_effect": self.expect,
        }

    def __repr__(self):
        return "<Hypothesis %s>" % self.key


# ---------------------------------------------------------------- the bank
#
# The order of the bank is the order the hypotheses are tried in.  The
# cheapest and most likely wins come first, because a round has a
# wall-clock budget and the expensive changes (a width migration, a
# resampling) are only worth their cost once the cheap ones have been
# exhausted.

MORE_DATA_HELPS = Hypothesis(
    key="more-data-helps",
    claim="The parent was stopped early, so the model is data-limited "
          "rather than capacity-limited.",
    change="Continue the parent's weights and Adam state for more "
           "epochs on the same public training rows.",
    falsifier="If the held-out score does not improve as the epoch "
              "count grows, the model is not data-limited.",
    expect="RMSE falls monotonically with the epoch count.",
    profiles={"A", "B", "C"},
)

UNCONVERGED = Hypothesis(
    key="the-parent-was-left-unconverged",
    claim="The parent's learning rate was set for a short schedule and "
          "its own validation score was still falling when it stopped, "
          "so a larger learning rate reaches a better point.",
    change="Continue with a larger learning rate.",
    falsifier="If the parent's validation score was already flat at the "
              "end of its own schedule, the schedule - not the rate - "
              "was the limit.",
    expect="RMSE falls more than the same number of epochs at the "
           "parent's own rate.",
    profiles={"A", "B", "C"},
)

CAPACITY = Hypothesis(
    key="capacity-is-overfitting",
    claim="The parent has more capacity than the public rows can "
          "support, so it memorises the units it was trained on.",
    change="Migrate the weights to a narrower hidden layer.  The "
           "migration keeps the surviving units' weights and Adam "
           "state, so it is a continuation and not a re-initialisation.",
    falsifier="If the parent's training score is not near-perfect while "
              "its held-out score is much worse, the problem is not "
              "capacity but the target distribution itself.",
    expect="RMSE falls; the training score stays high.",
    profiles={"A", "B", "C"},
)

TAIL_UNDERWEIGHTED = Hypothesis(
    key="the-tail-is-underweighted",
    claim="The training rows are dominated by a few long-lived units, "
          "so the loss is dominated by the easy rows and the short-lived "
          "units - which are the ones the metric punishes hardest - are "
          "fitted badly.",
    change="Oversample the units with the least remaining life, so "
           "every unit contributes a comparable mass to the loss.",
    falsifier="If the per-unit RMSE spread does not shrink, the tail "
              "was not underweighted.",
    expect="RMSE falls, and the largest gain is on the short-lived units.",
    profiles={"B", "C"},
)

TAIL_WEIGHTED = Hypothesis(
    key="the-tail-is-underweighted-by-weight",
    claim="Late-life rows are a small fraction of the stream but carry "
          "most of the asymmetric penalty, so an unweighted loss fits the "
          "healthy rows at their expense.",
    change="Weight every training row by its unit-relative remaining "
           "life (1 + (1 - RUL/start)).  This is a loss-mass change only: "
           "no row is duplicated, the feature schema is untouched, and "
           "prediction still needs only ``inputs``.",
    falsifier="If the late-life RMSE and the NASA asymmetric score do "
              "not fall, the weighting did not reach the tail.",
    expect="Late-life RMSE falls and the over-prediction rate falls.",
    profiles={"B", "C"},
)

UNIT_IS_CONFOUNDER = Hypothesis(
    key="the-unit-is-the-confounder",
    claim="The model cannot tell which unit a row came from, so it "
          "predicts the population-average lifetime for every unit and "
          "is wrong by the spread of lifetimes.",
    change="Give every unit the same total mass in the loss, so the "
           "population average stops being the objective the model is "
           "fitted to.  The change is a training-distribution change "
           "only: the feature schema, the forward pass and the "
           "checkpoint format are untouched, because the platform's "
           "prediction request carries no sample id and so cannot "
           "reproduce an id-derived column at scoring time.",
    falsifier="If the per-unit RMSE spread is no larger than the "
              "within-unit RMSE, the unit is not the confounder.",
    expect="RMSE falls and the per-unit spread shrinks.",
    profiles={"B", "C"},
)

RARE_CLASS_UNDERWEIGHTED = Hypothesis(
    key="rare-class-is-underweighted",
    claim="The training mix is not the deployment mix, so a class that "
          "is common in training but rare in deployment dominates the "
          "Macro-F1 loss.",
    change="Oversample the rare classes so every class carries the same "
           "total mass in the loss.",
    falsifier="If the rare class's per-class F1 does not rise, the "
              "oversampling did not reach it.",
    expect="Macro-F1 rises; the rare class's F1 rises the most.",
    profiles={"A"},
)

BATCH_MIXING_SCALES = Hypothesis(
    key="the-batch-is-mixing-scales",
    claim="The parent's rows come from batches measured at different "
          "times, and the array ages between them, so the loss is "
          "dominated by the scale shift rather than by the chemistry.",
    change="Oversample the rows of the batch that is rarest in "
           "training, so the minority batch's own decision boundary is "
           "fitted rather than averaged away.",
    falsifier="If the rare batch's own accuracy does not rise, the "
              "batch was not being mixed away.",
    expect="Macro-F1 rises; the rare batch's accuracy rises the most.",
    profiles={"A"},
)

#: Every hypothesis the project knows about.  Adapters pick from this.
BANK = (
    MORE_DATA_HELPS,
    UNCONVERGED,
    TAIL_UNDERWEIGHTED,
    TAIL_WEIGHTED,
    UNIT_IS_CONFOUNDER,
    CAPACITY,
    RARE_CLASS_UNDERWEIGHTED,
    BATCH_MIXING_SCALES,
)

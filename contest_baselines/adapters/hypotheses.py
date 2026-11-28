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

THE_TEST_IS_TRUNCATED = Hypothesis(
    key="the-test-is-truncated",
    claim="The training rows come from complete life sequences, but the "
          "hidden rows are one truncated snapshot per unit.  Most training "
          "mass sits in the long-remaining-life region that the hidden "
          "protocol never scores, so the model is fitted to the wrong part "
          "of the target distribution.",
    change="Re-weight every training row by the importance ratio between a "
           "bounded uniform remaining-life prior (a training window narrower "
           "than the selection view) and the training RUL density, tempered "
           "by a square root.  No row is duplicated, the feature schema is "
           "untouched, and prediction still needs only ``inputs``.",
    falsifier="If the RMSE on the bounded-RUL validation view does not fall, "
              "the long-RUL mass was not the problem.",
    expect="Late-life RMSE falls; full-sequence RMSE may rise slightly.",
    profiles={"C"},
)

DEGRADATION_IS_RELATIVE = Hypothesis(
    key="the-degradation-is-relative",
    claim="What the loss should emphasise is how far a unit has *degraded* "
          "relative to its own lifetime, not how large its remaining life "
          "is on an absolute scale.  An absolute threshold treats the "
          "middle of a 400-cycle engine as end-of-life while treating the "
          "start of a 500-cycle engine as fresh, so it mislabels the very "
          "rows that carry the signal.",
    change="Weight every training row by its unit-relative remaining life: "
           "`1 + alpha * (1 - RUL / unit_lifetime)`.  The weight is a "
           "loss-mass change only - no row is duplicated, the feature schema "
           "is untouched, and prediction still needs only `inputs`.",
    falsifier="If the full-split RMSE on engines held out from training does "
              "not fall, the relative framing is not the better one.",
    expect="Full-split RMSE falls, and the gain holds on held-out engines "
           "rather than only on the training distribution.",
    profiles={"C"},
)

OUTPUT_IS_BIASED = Hypothesis(
    key="the-output-is-biased",
    claim="Part of the model's error is not shape at all but a constant "
          "offset: on profile C the freshest lifecycle stage's error is "
          "44% bias - the model systematically predicts a lower remaining "
          "life than the truth where the absolute targets are largest.  "
          "No amount of extra training removes that offset, because the "
          "fresh rows are a tiny share of the training mass while the "
          "scoring rows are spread over the whole life cycle.",
    change="Fit the least-squares affine map `y ~ a * output + b` on the "
           "**training** rows and fold it into the output layer (`w2` and "
           "`b2`), so the calibrated model *is* the checkpoint and the "
           "platform's own forward pass sees it.  The feature schema, the "
           "target's meaning and the prediction path are untouched.",
    falsifier="If the full-split RMSE does not fall, or if the worst "
              "lifecycle stage's RMSE rises, the output did not carry a "
              "removable offset.",
    expect="Full-split RMSE falls and the worst lifecycle stage's RMSE does "
           "not rise - the floor of the drift stream is kept, not traded "
           "away for the headline number.",
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

SCALE_SWAMPING_SIGNAL = Hypothesis(
    key="the-scale-is-swamping-the-signal",
    claim="The gas array's columns span five orders of magnitude - eight "
          "of the 128 hold 99% of the variance - so the distance the "
          "network measures is dominated by a handful of huge columns "
          "and the class information in the rest is drowned out.",
    change="Compress every column with the sign-preserving log1p before "
           "the network sees it.  The schema does not change: the same "
           "128 columns come out, with different values.",
    falsifier="If the compressed columns do not raise the held-out "
              "Macro-F1, the scale was not what was limiting the model.",
    expect="Macro-F1 rises, and the spread across seeds shrinks.",
    profiles={"A"},
)

RARE_CLASS_NEEDS_THE_LOSS = Hypothesis(
    key="the-rare-class-needs-the-loss-to-notice-it",
    claim="One of the six gases has 18 rows against 247 for the rest, so "
          "the plain cross-entropy never rewards the model for getting "
          "it right and its per-class F1 collapses - which costs a "
          "sixth of the Macro-F1 on its own.",
    change="Weight the cross-entropy by the inverse of each class's "
           "frequency, so every class contributes the same total mass "
           "to the loss.  The rows are not resampled: the rare class's "
           "18 rows keep their own noise.",
    falsifier="If the rare class's per-class F1 does not rise, the "
              "weighting did not reach it.",
    expect="The rare class's per-class F1 rises; Macro-F1 rises with it.",
    profiles={"A"},
)

#: Every hypothesis the project knows about.  Adapters pick from this.
BANK = (
    MORE_DATA_HELPS,
    UNCONVERGED,
    TAIL_UNDERWEIGHTED,
    TAIL_WEIGHTED,
    THE_TEST_IS_TRUNCATED,
    DEGRADATION_IS_RELATIVE,
    OUTPUT_IS_BIASED,
    UNIT_IS_CONFOUNDER,
    CAPACITY,
    RARE_CLASS_UNDERWEIGHTED,
    BATCH_MIXING_SCALES,
    SCALE_SWAMPING_SIGNAL,
    RARE_CLASS_NEEDS_THE_LOSS,
)

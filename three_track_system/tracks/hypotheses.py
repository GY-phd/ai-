# -*- coding: utf-8 -*-
"""The hypothesis bank: what the system is allowed to try, and why.

The point of a self-evolving system is not that it re-fits the same
model more often.  It is that at every cycle it forms an explicit
**hypothesis** about why the champion is losing, proposes a change that
would follow from that hypothesis, and then either confirms or refutes
it with a measurement on unseen data.

The bank below is the system's prior knowledge about *which* hypotheses
are worth testing on this data.  It is deliberately small and explicit:
a large bank of untested ideas is indistinguishable from no bank at all,
because nothing in it has been falsified.

Each entry has four parts.

``hypothesis``
    The claim about the mechanism.
``change``
    The concrete thing to try.
``falsifier``
    What would have to be observed for the claim to be **wrong**.  This
    is the part that makes it science rather than a search.
``expect``
    The direction the metric should move, and by roughly how much.
    If the measurement lands the other way, the hypothesis is refuted
    even if the metric happened to improve for some other reason.

Why the falsifier is not optional
---------------------------------
Without a falsifier, "the metric went up" is the only evidence the
system accepts, and it will happily keep a change that improves the
metric by luck and hurts the mechanism.  With one, a candidate has to
both improve the score *and* exhibit the predicted signature.

Run:
    python -m tracks.hypotheses --profile B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Hypothesis:
    """One testable claim with a concrete change attached."""

    def __init__(self, key, hypothesis, change, falsifier, expect, profiles=None,
                 cost=1.0):
        self.key = key
        self.hypothesis = hypothesis
        self.change = change
        self.falsifier = falsifier
        self.expect = expect
        self.profiles = set(profiles) if profiles else None
        self.cost = cost

    def applies_to(self, profile_key):
        return self.profiles is None or profile_key in self.profiles

    def as_record(self, cycle):
        return {
            "cycle": cycle,
            "hypothesis": self.key,
            "claim": self.hypothesis,
            "change": self.change,
            "falsifier": self.falsifier,
            "expected_effect": self.expect,
            "cost": self.cost,
        }


# --------------------------------------------------------------------- bank
BANK = [
    Hypothesis(
        key="capacity-is-overfitting",
        hypothesis="The champion has more capacity than 445-7032 rows can "
                  "support, so it memorises the units it was trained on and "
                  "loses the ability to generalise to a new unit.",
        change="Shrink the hidden layer and shorten the schedule: the "
               "capacity sweep's smallest configuration.",
        falsifier="If the champion's *training* score is not near-perfect "
                  "while its held-out score is much worse, the problem is "
                  "not capacity but the target distribution itself.",
        expect="Training score stays high; held-out RMSE falls by 10-40%.",
        profiles={"A", "B", "C"},
        cost=0.2,
    ),
    Hypothesis(
        key="units-are-confounded",
        hypothesis="The model cannot tell which unit a row came from, so it "
                  "predicts the population-average lifetime for every unit "
                  "and is wrong by the spread of lifetimes.",
        change="Give the model the unit's elapsed-observation count as a "
               "feature, so the age of the unit is visible.",
        falsifier="If the per-unit RMSE spread is no larger than the "
                  "within-unit RMSE, the unit is not the confounder.",
        expect="Per-unit RMSE spread shrinks; overall RMSE falls.",
        profiles={"B", "C"},
        cost=0.2,
    ),
    Hypothesis(
        key="degradation-is-not-first-order",
        hypothesis="Absolute vibration or sensor levels are dominated by "
                  "the unit's own baseline, so the model tracks the "
                  "baseline instead of the degradation.",
        change="Replace the raw features with the per-unit baseline-relative "
               "slope (the two-stage normalisation).",
        falsifier="If the per-unit feature means differ by more than the "
                  "within-unit variation, the baseline is not dominating.",
        expect="RMSE falls sharply; the effect is largest on the units "
               "with the most extreme baselines.",
        profiles={"B", "C"},
        cost=0.3,
    ),
    Hypothesis(
        key="lifetime-is-multiplicative",
        hypothesis="Remaining life is a fraction of total lifetime, not an "
                  "absolute number of minutes, so the regression target "
                  "should be the lifetime fraction.",
        change="Regress the fraction of lifetime remaining, then convert "
               "back to minutes with the two-stage lifetime estimate.",
        falsifier="If the two-stage conversion's error does not shrink with "
                  "the fraction, the multiplicative model is wrong.",
        expect="RMSE falls, and the largest gains are on the short-lived units.",
        profiles={"B", "C"},
        cost=0.4,
    ),
    Hypothesis(
        key="rare-class-is-underweighted",
        hypothesis="The training mix is not the deployment mix, and a class "
                  "that is common in training but rare in deployment "
                  "dominates the Macro-F1 loss.",
        change="Weight the training loss so that every class contributes "
               "the same total mass.",
        falsifier="If the rare class's per-class F1 does not rise, the "
                  "weighting did not reach it.",
        expect="Macro-F1 rises; the rare class's F1 rises the most.",
        profiles={"A"},
        cost=0.2,
    ),
    Hypothesis(
        key="drift-is-a-scale-shift",
        hypothesis="The stream differs from the training set by a per-feature "
                  "scale, not by a change in the decision boundary.",
        change="Normalise the stream rows by the training set's own "
               "per-feature scale before predicting.",
        falsifier="If the standardised feature shift is small, there is no "
                  "scale shift to remove and the normalisation can only "
                  "add noise.",
        expect="RMSE falls on the drifted batches.",
        profiles={"A", "B", "C"},
        cost=0.1,
    ),
    Hypothesis(
        key="asymmetry-is-a-bias",
        hypothesis="The NASA asymmetric score is high because the model's "
                  "predictions are biased, not because they are noisy.",
        change="Shift the predictions by the median signed error of the "
               "recent stream.",
        falsifier="If the median signed error is already near zero, there "
                  "is no bias to remove.",
        expect="The NASA score falls by more than the RMSE falls.",
        profiles={"C"},
        cost=0.1,
    ),
    Hypothesis(
        key="more-data-helps",
        hypothesis="The model is data-limited rather than capacity-limited.",
        change="Keep the architecture and retrain on every row seen so far, "
               "including the stream.",
        falsifier="If the held-out score does not improve as rows are added, "
                  "the model is not data-limited.",
        expect="The score improves monotonically with the row count.",
        profiles={"A", "B", "C"},
        cost=0.5,
    ),
]


def for_profile(profile_key):
    """The hypotheses that are relevant to one profile."""
    return [h for h in BANK if h.applies_to(profile_key)]


# ------------------------------------------------- hypothesis -> what changes
#
# Each hypothesis is only worth having if testing it changes something.
# The table below says exactly what changes in the fitted model for each
# hypothesis, per profile.  A hypothesis whose entry is empty would be a
# comment, not an experiment.
#
# The entries are deliberately **different from the champion's default
# configuration**.  An earlier version of this table listed
# ``normalisation="slope"`` and ``target="life_fraction"`` for the two
# RUL profiles - which are exactly the defaults - so the candidate and
# the champion were the same model and every cycle measured the
# difference between two identical numbers.  A hypothesis that does not
# change the model is a comment, not an experiment.
#
# The champion's defaults are in ``CHAMPION_DEFAULTS`` below, and every
# override here has to differ from them.
CHAMPION_DEFAULTS = {
    # Profile A: the capacity sweep's winner on the official split -
    # h32/e50 reaches Macro-F1 0.7227, while h32/e400 collapses to
    # 0.5548.  The champion therefore starts *near* the knee of the
    # capacity curve, where the "capacity is overfitting" hypothesis has
    # something real to test.
    "A": {"hidden": 32, "epochs": 50},
    # Profile B: the two-stage model's audited configuration.  The
    # leave-one-bearing-out number this project reports (382.61 min) was
    # measured with exactly this recipe.
    "B": {"hidden": 96, "epochs": 400, "normalisation": "slope",
          "target": "life_fraction", "floor": 0.02, "cap": 3000.0,
          "rate_limit": 0.25, "feature_transform": "eng34",
          "activation": "tanh", "seed": 42},
    # Profile C: the grid sweep's winner - h128/e200 reaches RMSE 50.571
    # on the official validation set.
    "C": {"hidden": 128, "epochs": 200, "target": "cycles"},
}


def champion_defaults(profile_key):
    """The champion's starting recipe for one profile."""
    return dict(CHAMPION_DEFAULTS.get(profile_key, {}))


OVERRIDES = {
    "capacity-is-overfitting": {
        # A *smaller* model than the champion: the sweep says h16/e20
        # scores 0.7076 against the champion's 0.7227, so the test is
        # real - a smaller network might generalise better.
        "A": {"hidden": 16, "epochs": 20},
        "B": {"hidden": 64, "epochs": 200},
        "C": {"hidden": 32, "epochs": 20},
    },
    "units-are-confounded": {
        # ``extra_features="elapsed"`` appends the unit's elapsed-observation
        # count as a feature, so the model can see the age of the unit.
        "B": {"extra_features": "elapsed", "target": "minutes"},
        "C": {"extra_features": "elapsed", "target": "cycles"},
    },
    "degradation-is-not-first-order": {
        # Regress the raw minutes against the *raw* features: if the
        # per-unit baseline really dominates, the un-normalised model
        # should lose badly, and the comparison is the measurement.
        "B": {"normalisation": "raw", "target": "minutes"},
        "C": {"normalisation": "raw", "target": "cycles"},
    },
    "lifetime-is-multiplicative": {
        # The champion already regresses the fraction; this hypothesis
        # tests the *other* branch: the raw minutes, with the two-stage
        # conversion switched off.
        "B": {"target": "minutes"},
        "C": {"target": "cycles"},
    },
    "rare-class-is-underweighted": {
        "A": {"class_weighting": "balanced"},
    },
    "drift-is-a-scale-shift": {
        "A": {"normalisation": "train_scale"},
        "B": {"normalisation": "train_scale"},
        "C": {"normalisation": "train_scale"},
    },
    "asymmetry-is-a-bias": {
        "C": {"bias_correction": "median"},
    },
    "more-data-helps": {
        # The default: same architecture, more rows.
        "A": {}, "B": {}, "C": {},
    },
}


def overrides_for(hypothesis_key, profile_key):
    """The model configuration a hypothesis asks for."""
    by_profile = OVERRIDES.get(hypothesis_key, {})
    return dict(by_profile.get(profile_key, {}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("A", "B", "C"))
    args = parser.parse_args(argv)

    keys = [args.profile] if args.profile else list("ABC")
    for key in keys:
        print("=" * 74)
        print("hypothesis bank for profile %s" % key)
        for hypothesis in for_profile(key):
            print("\n  [%s]  cost %.1f" % (hypothesis.key, hypothesis.cost))
            print("    claim      %s" % hypothesis.hypothesis)
            print("    change     %s" % hypothesis.change)
            print("    falsifier  %s" % hypothesis.falsifier)
            print("    expect     %s" % hypothesis.expect)
    return 0


if __name__ == "__main__":
    sys.exit(main())

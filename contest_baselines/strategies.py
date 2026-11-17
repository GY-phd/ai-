# -*- coding: utf-8 -*-
"""The continuation strategies: what this project changes, and why.

The competition asks for a system that *improves* the parent checkpoint
over ten rounds, with at least three complete
"read the history - run an experiment - decide from the result" records.
That is not a request for one clever trick; it is a request for a
**closed loop** that makes a defensible decision every round.

So the round's logic is split into three parts, and each part is one
module-level concern:

``hypotheses``
    The bank of testable claims, each with a falsifier.  A claim without
    a falsifier is not a hypothesis but a guess, and the loop would then
    accept any improvement for any reason.

``strategies``
    The concrete thing each hypothesis asks for, expressed as a change
    to the *training procedure* only.  Nothing here changes the model's
    forward pass, so every candidate stays a reference-compatible
    ``contest.cpu-mlp.v1`` checkpoint.

``train``
    The driver: read the parent, apply one strategy, fit, measure on the
    public validation set, and write the evidence.

What may change, and what may not
----------------------------------
The rules allow the learning rate, the batch size, the epoch count, the
training distribution, and an explicitly implemented weight migration.
Everything else - the feature schema, the task, the class list, the
target's meaning, the normalisation state - is inherited from the parent
and is **not** touched.  That is the discipline that makes "continue the
parent" true rather than nominal.
"""
from __future__ import annotations

import numpy as np

from .data import unit_ids


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


BANK = [
    Hypothesis(
        key="more-data-helps",
        claim="The parent was stopped early, so the model is data-limited "
              "rather than capacity-limited.",
        change="Continue the parent's weights and Adam state for more "
               "epochs on the same public training rows.",
        falsifier="If the held-out score does not improve as the epoch "
                  "count grows, the model is not data-limited.",
        expect="RMSE falls monotonically with the epoch count.",
        profiles={"A", "B", "C"},
    ),
    Hypothesis(
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
    ),
    Hypothesis(
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
    ),
    Hypothesis(
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
    ),
    Hypothesis(
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
    ),
    Hypothesis(
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
    ),
    Hypothesis(
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
    ),
]


def for_profile(profile_key):
    """The hypotheses relevant to one profile."""
    return [h for h in BANK if h.applies_to(profile_key)]


# --------------------------------------------------------------------- plans
def build_plan(profile_key, round_index, history):
    """Choose the strategy for one round.

    The plan is a *function of what has already been tried*, which is what
    makes the ten rounds an experiment rather than ten repetitions.  Each
    strategy is tried once; the first one that the public validation set
    refuses is recorded as refuted and the next one is used.

    The order is deliberate.  The cheapest and most likely wins come
    first, because a round has a wall-clock budget and the expensive
    changes (a width migration, a resampling) are only worth their cost
    once the cheap ones have been exhausted.
    """
    tried = {record.get("hypothesis") for record in history}
    for hypothesis in for_profile(profile_key):
        if hypothesis.key not in tried:
            return hypothesis
    # Every hypothesis has been tried once; repeat the first one, which by
    # construction is the one with the best evidence behind it.
    return for_profile(profile_key)[0]


# ----------------------------------------------------------------- strategies
def _oversample_indices(weights, count, seed):
    """Resample row indices so the weights become the sampling mass."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return np.arange(count)
    probability = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(count, size=count, replace=True, p=probability)


def _unit_shortness(y, units):
    """How close each row's unit is to its end, as a weight.

    The weight is the *target* of the row, normalised inside its own
    unit: a row whose remaining life is small relative to its unit's
    start is a row from a short-lived unit.  It is used only to decide
    which rows to oversample - it is never fed to the model.
    """
    y = np.asarray(y, dtype=np.float64)
    units = np.asarray(units)
    weight = np.zeros(len(y), dtype=np.float64)
    for name in np.unique(units):
        indices = np.nonzero(units == name)[0]
        values = y[indices]
        start = float(values.max()) if len(values) else 0.0
        if start <= 1e-9:
            weight[indices] = 1.0
            continue
        # 1 at the unit's first snapshot, 0 at its last.
        weight[indices] = np.clip(1.0 - values / start, 0.0, 1.0)
    return weight


def _unit_elapsed(units, order):
    """Each row's elapsed-observation count inside its own unit."""
    units = np.asarray(units)
    order = np.asarray(order, dtype=np.float64)
    out = np.zeros(len(units), dtype=np.float64)
    for name in np.unique(units):
        indices = np.nonzero(units == name)[0]
        ordered = indices[np.argsort(order[indices], kind="stable")]
        out[ordered] = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return out


def prepare(hypothesis, model, train, profile_key, seed):
    """Apply a hypothesis to the training data and to the model's settings.

    Returns a description of what was changed, so the round's
    ``change_summary`` can state it truthfully.

    The **feature schema is inherited from the parent and may not
    change**.  The platform's prediction request carries only
    ``{task, inputs}`` - no sample id - so a column derived from the id
    cannot be reproduced when the hidden rows are scored, and the
    reference loader refuses a checkpoint whose schema differs from the
    data's.  Every hypothesis here therefore changes the *training
    procedure* only: the learning rate, the epoch budget, the training
    distribution, or the hidden width through an explicit migration.
    """
    task = model.meta["task"]
    x = train["x"]
    y = train["y"]
    units = unit_ids(train["ids"])
    parent_paths = list(model.meta["paths"])
    changed = {}

    if hypothesis.key == "the-parent-was-left-unconverged":
        rate = min(0.01, float(model.meta["learning_rate"]) * 2.0)
        model.meta["learning_rate"] = rate
        changed["learning_rate"] = rate

    elif hypothesis.key == "the-tail-is-underweight" or hypothesis.key == "the-tail-is-underweighted":
        weight = _unit_shortness(y, units)
        # Give the short-lived units a comparable mass, without letting any
        # single row dominate.
        weight = np.where(weight <= 0, 0.25, weight)
        indices = _oversample_indices(weight, len(y), seed)
        changed["resampled_rows"] = int(len(indices))
        changed["resample_weight"] = "unit-relative remaining life"
        x, y = x[indices], y[indices]

    elif hypothesis.key == "the-unit-is-the-confounder":
        # The unit's elapsed-observation count would be the natural way to
        # expose the unit's identity, but it **cannot be used as a model
        # input**: the platform's prediction request carries only
        # ``{task, inputs}`` - there is no sample id - so a column that is
        # derived from the id cannot be computed when the hidden rows are
        # scored.  A checkpoint whose schema carries such a column would
        # fail every hidden prediction.
        #
        # So the same claim is tested through the only channel that is
        # available at scoring time: the *training distribution*.  If the
        # model cannot tell the units apart, it predicts the
        # population-average lifetime for all of them, and the units with
        # the fewest rows are the ones that pull the average away from
        # their own truth.  Giving every unit the same total mass in the
        # loss removes the population average from the objective.
        unit_counts = {}
        for name in np.unique(units):
            unit_counts[name] = int(np.sum(units == name))
        mass = np.asarray([1.0 / unit_counts[name] for name in units], dtype=np.float64)
        indices = _oversample_indices(mass, len(y), seed)
        changed["resampled_rows"] = int(len(indices))
        changed["resample_weight"] = "uniform mass per unit"
        changed["units"] = len(unit_counts)
        x, y = x[indices], y[indices]

    elif hypothesis.key == "capacity-is-overfitting":
        target = 16 if task == "classification" else 32
        migration = model.resize(target, seed=model.meta["seed"])
        changed["hidden"] = migration
        # The feature matrix has to be re-standardised for the new width;
        # the parent's normalisation state stays, because the features
        # themselves did not change.
        changed["note"] = "weights migrated, Adam state carried over"

    elif hypothesis.key == "rare-class-is-underweighted":
        classes = model.meta["classes"]
        counts = np.bincount(np.asarray(y), minlength=len(classes)).astype(np.float64)
        counts[counts < 1] = 1.0
        weight = counts.max() / counts
        indices = _oversample_indices(weight[y], len(y), seed)
        changed["resampled_rows"] = int(len(indices))
        changed["resample_weight"] = "inverse class frequency"
        x, y = x[indices], y[indices]

    elif hypothesis.key == "the-batch-is-mixing-scales":
        batches = _batch_of(train["ids"])
        if batches is None:
            changed["skipped"] = "no batch information in the sample ids"
        else:
            counts = {}
            for name in np.unique(batches):
                counts[name] = int(np.sum(batches == name))
            rare = min(counts, key=lambda key: counts[key])
            weight = np.where(batches == rare, 8.0, 1.0)
            indices = _oversample_indices(weight, len(y), seed)
            changed["resampled_rows"] = int(len(indices))
            changed["rare_batch"] = rare
            x, y = x[indices], y[indices]

    else:
        changed["note"] = "continued the parent's weights and schedule"

    model.meta["batch_size"] = int(model.meta.get("batch_size", 128))
    model.meta["learning_rate"] = float(model.meta["learning_rate"])
    # The schema guard.  The feature schema is inherited from the parent
    # and is **not** a thing a round may change: the platform's hidden
    # rows carry the parent's schema, and the reference loader refuses a
    # checkpoint whose schema differs.  Any change that added, dropped or
    # renamed a feature would produce a candidate that scores nothing, so
    # it is caught here rather than in the hidden evaluation.
    if list(model.meta["paths"]) != list(parent_paths):
        raise RuntimeError("the feature schema must not change: %r -> %r"
                           % (parent_paths, model.meta["paths"]))
    if x.shape[1] != len(model.meta["paths"]):
        raise RuntimeError("the training matrix does not match the schema")
    model.validate()
    return x, y, changed


def _batch_of(identifiers):
    """The measurement batch of each row, if the ids carry one."""
    batches = []
    for identifier in identifiers:
        parts = identifier.split("_")
        if len(parts) >= 2 and parts[0] == "gas":
            batches.append(parts[1])
        else:
            batches.append(None)
    if all(item is None for item in batches):
        return None
    return np.asarray(batches, dtype=object)


def epoch_budget(hypothesis, round_index, wall_seconds=None):
    """How many epochs to run for this round.

    The budget grows with the round index so that a strategy that needs
    time to show its effect is not cut short, and it stays bounded so
    that the round always finishes inside the platform's wall clock.
    """
    base = {"more-data-helps": 30, "the-parent-was-left-unconverged": 30,
            "the-tail-is-underweighted": 40, "the-unit-is-the-confounder": 40,
            "capacity-is-overfitting": 40, "rare-class-is-underweighted": 30,
            "the-batch-is-mixing-scales": 30}
    count = base.get(hypothesis.key, 30)
    # Each round gets a larger budget than the last, but never without
    # bound: the tenth round is not allowed to run ten times as long.
    # The growth is **strictly increasing**, because a repeated hypothesis
    # must still be a *different* experiment: the same strategy with the
    # same budget from the same parent would produce the same numbers,
    # and the second round would be a repetition dressed up as an
    # experiment rather than an experiment at all.
    count += (round_index - 1) * 10
    return int(min(count, 150))

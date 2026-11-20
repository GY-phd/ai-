# -*- coding: utf-8 -*-
"""The continuation strategies: what this project changes, and why.

The competition asks for a system that *improves* the parent checkpoint
over ten rounds, with at least three complete
"read the history - run an experiment - decide from the result" records.
That is not a request for one clever trick; it is a request for a
**closed loop** that makes a defensible decision every round.

    10|So the round's logic is split into three parts, and each part is one
module-level concern:

``contest_baselines.adapters``
    The **plugin layer**.  One adapter per dataset profile answers what
    that profile's units are, which hypotheses it can test, how long a
    round may run, and how wide a capacity migration should be.  A
    profile's specifics live in its adapter, not in this file.

``contest_baselines.adapters.hypotheses``
    The bank of testable claims, each with a falsifier.  A claim without
    a falsifier is not a hypothesis but a guess, and the loop would then
    accept any improvement for any reason.

``contest_baselines.strategies``
    The concrete thing each hypothesis asks for, expressed as a change
    to the *training procedure* only.  Nothing here changes the model's
    forward pass, so every candidate stays a reference-compatible
    ``contest.cpu-mlp.v1`` checkpoint.

``contest_baselines.train``
    The driver: read the parent, apply one strategy, fit, measure on the
    public validation set, and write the evidence.

What may change, and what may not
----------------------------------
    30|The rules allow the learning rate, the batch size, the epoch count, the
training distribution, and an explicitly implemented weight migration.
Everything else - the feature schema, the task, the class list, the
target's meaning, the normalisation state - is inherited from the parent
and is **not** touched.  That is the discipline that makes "continue the
parent" true rather than nominal.
"""
from __future__ import annotations

import numpy as np

from .adapters import get as get_adapter
from .adapters.hypotheses import BANK, Hypothesis  # noqa: F401  (re-exported)


def for_profile(profile_key):
    """The hypotheses relevant to one profile.

    The adapter decides.  A hypothesis the adapter does not list is not
    tried, whatever its own ``profiles`` set says - the adapter is the
    profile's own statement of what it can test, and it has the final
    word.
    """
    return list(get_adapter(profile_key).hypotheses())


# --------------------------------------------------------------------- plans
def build_plan(profile_key, round_index, history, adapter=None):
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
    adapter = adapter or get_adapter(profile_key)
    tried = {record.get("hypothesis") for record in history}
    for hypothesis in adapter.hypotheses():
        if hypothesis.key not in tried:
            return hypothesis
    # Every hypothesis has been tried once; repeat the first one, which by
    # construction is the one with the best evidence behind it.
    return adapter.hypotheses()[0]


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


def prepare(hypothesis, model, train, profile_key, seed, adapter=None):
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

    The profile's specifics - what a unit is, how the units are
    weighted, how wide a migration should be - come from the adapter.
    """
    adapter = adapter or get_adapter(profile_key)
    task = model.meta["task"]
    x = train["x"]
    y = train["y"]
    units = adapter.groups(train["ids"])
    parent_paths = list(model.meta["paths"])
    changed = {}

    if hypothesis.key == "the-parent-was-left-unconverged":
        rate = min(0.01, float(model.meta["learning_rate"]) * 2.0)
        model.meta["learning_rate"] = rate
        changed["learning_rate"] = rate

    elif hypothesis.key in ("the-tail-is-underweight", "the-tail-is-underweighted"):
        weight = _unit_shortness(y, units)
        # Give the short-lived units a comparable mass, without letting any
        # single row dominate.
        weight = np.where(weight <= 0, 0.25, weight)
        indices = _oversample_indices(weight, len(y), seed)
        changed["resampled_rows"] = int(len(indices))
        changed["resample_weight"] = "unit-relative remaining life"
        changed["units"] = int(len(np.unique(units)))
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
        target = adapter.resize_hidden(task)
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
        # This hypothesis needs **at least two batches inside the training
        # set** - otherwise "oversample the rarest batch" has no meaning
        # and the resampling would be a uniform no-op.  The platform's
        # public projection for profile A contains a single batch, so
        # this branch records the veto instead of pretending.
        labels = adapter.labels(train["ids"])
        # ``None`` means "this id carries no batch label", which is a
        # different thing from a batch name.  It is counted separately
        # and never treated as a batch.
        known = sorted({str(name) for name in labels if name is not None})
        unlabelled = int(sum(1 for name in labels if name is None))
        if len(known) < 2:
            changed["skipped"] = (
                "the training projection holds %d labelled batch(es), so there "
                "is no minority batch to oversample" % len(known))
            if unlabelled:
                changed["unlabelled_rows"] = unlabelled
        else:
            counts = {name: int(np.sum(labels == name)) for name in known}
            rare = min(counts, key=lambda key: counts[key])
            weight = np.where(labels == rare, 8.0, 1.0)
            indices = _oversample_indices(weight, len(y), seed)
            changed["resampled_rows"] = int(len(indices))
            changed["resample_weight"] = "rare batch x8"
            changed["rare_batch"] = rare
            changed["batches"] = len(known)
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


def epoch_budget(hypothesis, round_index, wall_seconds=None, adapter=None):
    """How many epochs to run for this round.

    The budget grows with the round index so that a strategy that needs
    time to show its effect is not cut short, and it stays bounded so
    that the round always finishes inside the platform's wall clock.

    The adapter scales it: profile C has sixteen times the rows of
    profile B, so its budget is smaller.
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
    count = int(min(count, 150))
    if adapter is not None:
        count = max(1, int(count * float(adapter.epoch_scale())))
    return count

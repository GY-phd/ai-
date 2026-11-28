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


def _bounded_prior_weights(y, horizon, *, shape="uniform", tau=120.0,
                           bandwidth=8.0, power=0.5, floor=0.05):
    """Importance weights for a truncated-test remaining-life prior.

    The training rows come from complete sequences, so the RUL density
    they induce over-represents the long-life region.  The hidden C
    protocol truncates each test sequence and scores one snapshot, which
    bounds the useful remaining-life range at ``horizon``.  The weight is
    the (tempered) ratio between a target prior over ``[0, horizon]`` and
    the training rows' own smoothed RUL density.

    ``shape`` chooses the target: ``uniform`` (the default), ``tri``
    (linearly decreasing) or ``exp`` (``exp(-RUL/tau)``).  The parameter
    search found the uniform and exponential shapes on the same Pareto
    frontier; the shape is a knob, not a hidden label.

    ``floor`` keeps a minimum mass on the long-RUL rows.  The
    2026-11-22 search selected ``horizon=150`` and ``floor=0.05``: on the
    public C proxy it gives 39.9 -> 28.7 RMSE (three seeds
    28.50/28.51/29.06) while the full-split RMSE is 57.3.  That full
    split is **outside** the round's 5% guard over the parent (52.2),
    so this hypothesis is a real candidate that the guard **rejects** -
    which is the point of the guard.  The search's own records
    (``artifacts/tracks/c_weight_*.json``) show what the 5% guard does
    admit, measured against the same 52.225 parent:

    * ``surv`` with ``H=150``: full 52.895, proxy 33.095 - inside 5%;
    * ``deg`` with ``alpha=2.0``: full 51.852, proxy 35.260 - inside 5%;
    * the tempered prior with an exponential shape: full 54.744,
      proxy 30.140 - inside 5% on the worst of the three seeds;
    * ``imp`` and ``impunit``: full 59.0 and 60.2, proxy 29.1 and 28.2 -
      **outside** 5%, and outside 12% as well.

    The last group is what the 12% guard used to admit.  It is the
    configuration the platform's own 2026-11-22 run deployed, and the
    hidden score came back 56.25329 against a 52.8686 baseline - 6.40%
    *worse* than the baseline, which violates the platform's own
    requirement that "效果不低于平台基线".  The proxy gain was real
    (39.9 -> 28.5) and the hidden score still fell, because the proxy
    is exactly the number the platform never shows.  So the guard is
    not the objective; it is the thing that decides whether a candidate
    is allowed to ship at all.

    Only the training labels are used; the horizon is the profile's
    protocol bound, not a hidden label.  The weights are a training-side
    object: no row is duplicated, the feature schema is untouched, and
    the forward pass and the save path never see them.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.size == 0:
        return values
    top = int(max(np.ceil(float(values.max())), np.ceil(float(horizon))))
    grid = np.arange(0, top + 1, dtype=float)
    counts = np.bincount(np.clip(np.round(values).astype(int), 0, top),
                         minlength=top + 1).astype(np.float64)
    kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / float(bandwidth)) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    density = kernel @ counts
    density /= density.sum()
    if shape == "uniform":
        target = (grid <= float(horizon)).astype(np.float64)
    elif shape == "tri":
        target = np.where(grid <= float(horizon), float(horizon) - grid + 1.0, 0.0)
    elif shape == "exp":
        target = np.where(grid <= float(horizon), np.exp(-grid / float(tau)), 0.0)
    else:
        raise ValueError("unknown weight prior shape: %r" % (shape,))
    target /= target.sum()
    ratio = np.clip(target / (density + 1e-12), float(floor), 20.0)
    weights = ratio[np.clip(np.round(values).astype(int), 0, top)] ** float(power)
    return weights / weights.mean()


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

    # The profile's own learning rate, when it declares one.  A profile
    # that does not declare one leaves the parent's rate alone, which is
    # the historical behaviour.
    #
    # The override happens **before** the hypothesis branches, so the
    # ``the-parent-was-left-unconverged`` hypothesis doubles the
    # profile's rate rather than the parent's.  That hypothesis's claim
    # is about the *schedule* the parent was stopped on, and the rate
    # it doubles has to be the rate the rest of the study measured -
    # doubling a rate that the study never used would test a different
    # experiment than the one that was measured.
    profile_rate = adapter.learning_rate()
    if profile_rate is not None:
        profile_rate = float(profile_rate)
        if profile_rate != float(model.meta["learning_rate"]):
            changed["learning_rate"] = profile_rate
            changed["learning_rate_source"] = ("profile %s" % adapter.key)
        model.meta["learning_rate"] = profile_rate

    if hypothesis.key == "the-parent-was-left-unconverged":
        rate = min(0.01, float(model.meta["learning_rate"]) * 2.0)
        model.meta["learning_rate"] = rate
        changed["learning_rate"] = rate
        changed["learning_rate_source"] = ("twice the profile rate (%g)"
                                           % profile_rate if profile_rate
                                           is not None else "twice the parent's")

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

    elif hypothesis.key == "the-tail-is-underweighted-by-weight":
        # Same claim as the resampling hypothesis, tested through the
        # loss instead of the sampler.  The weights are a training-time
        # object only; the checkpoint and the prediction path are
        # untouched, so the candidate remains deployable.
        weight = _unit_shortness(y, units)
        weights = 1.0 + weight
        model.sample_weight = np.asarray(weights, dtype=np.float64)
        changed["sample_weight"] = "1 + unit-relative remaining life"
        changed["weight_mean"] = round(float(model.sample_weight.mean()), 4)

    elif hypothesis.key == "the-test-is-truncated":
        # The hidden C protocol scores one snapshot per engine at the
        # truncation point; the public rows cover every cycle of complete
        # sequences.  The long-RUL mass is therefore a region the hidden
        # metric never visits.  The horizon is the adapter's protocol
        # bound, and the weights are computed from the training labels
        # alone, so nothing hidden is read.
        horizon = adapter.weight_horizon()
        if task != "regression" or horizon is None:
            changed["skipped"] = ("profile %s has no truncated-test horizon"
                                  % adapter.key)
        else:
            weights = _bounded_prior_weights(y, float(horizon), shape="uniform",
                                             power=0.5, floor=0.05)
            model.sample_weight = weights
            changed["sample_weight"] = (
                "sqrt importance: uniform RUL<=%g / training RUL density" % horizon)
            changed["horizon"] = float(horizon)
            changed["selection_horizon"] = adapter.test_horizon()
            changed["weight_shape"] = "uniform"
            changed["weight_power"] = 0.5
            changed["weight_floor"] = 0.05
            changed["weight_mean"] = round(float(weights.mean()), 4)
            changed["weight_min"] = round(float(weights.min()), 4)
            changed["weight_max"] = round(float(weights.max()), 4)

    elif hypothesis.key == "the-degradation-is-relative":
        # Relative degradation, not an absolute remaining-life threshold.
        #
        # The absolute-threshold family (`surv`/`cap`: "this row is
        # late because RUL < 150") was measured on 12 engine folds over
        # two seeds and was **anti-correlated** with the held-out
        # outcome: every truncation from `y <= 50` to `y <= 200`
        # ranked candidates with rho between -0.27 and -0.53, and
        # selecting by one beat the baseline in 0 of 12 folds.  The
        # reason is structural - an absolute cut mislabels the middle of
        # a long-lived engine as end-of-life, so the recipe wins the
        # truncated view by construction while losing accuracy over the
        # life cycle the view cannot see.
        #
        # The relative form has no such failure mode: the weight depends
        # on where a row sits inside **its own unit's** life, so it is
        # scale-free and every unit is treated the same way.
        if task != "regression":
            changed["skipped"] = ("profile %s is not a regression task"
                                  % adapter.key)
        else:
            relative = _unit_shortness(y, units)
            alpha = float(adapter.degradation_alpha())
            weights = 1.0 + alpha * relative
            model.sample_weight = np.asarray(weights, dtype=np.float64)
            changed["sample_weight"] = ("1 + %g * unit-relative remaining life"
                                          % alpha)
            changed["weight_alpha"] = alpha
            changed["weight_mean"] = round(float(weights.mean()), 4)
            changed["weight_min"] = round(float(weights.min()), 4)
            changed["weight_max"] = round(float(weights.max()), 4)

    elif hypothesis.key == "the-output-is-biased":
        # Remove the constant offset in the model's output.
        #
        # Measured on profile C's public validation split, cut into five
        # lifecycle stages, the freshest stage's error is **44% bias**:
        # the model systematically predicts a lower remaining life than
        # the truth at the fresh end of an engine's life, where the
        # absolute targets are largest.  A constant offset is not a shape
        # error, so no amount of extra training removes it - the fresh
        # rows are a tiny share of the training mass, while the scoring
        # rows are spread over the whole life cycle.
        #
        # The affine map is fitted on the **training** rows (the
        # validation split is a measurement split and fitting on it would
        # leak) and folded into ``w2``/``b2``, so the calibrated model
        # *is* the checkpoint.  The platform runs the reference forward
        # pass, so a calibration that lived outside the weights would not
        # be applied at scoring time at all.
        #
        # The hypothesis is recorded as ``skipped`` when the profile is
        # not a regression task, because the calibration is a regression
        # operation.
        #
        # The map is **not** fitted here: ``prepare`` runs before the
        # training, and the map has to be fitted on the *trained*
        # model's output.  So the branch records the intent, and the
        # controller applies it after the fit - see
        # ``RoundController.build_candidate``.
        if task != "regression":
            changed["skipped"] = ("profile %s is not a regression task"
                                  % adapter.key)
        else:
            changed["calibration_requested"] = True
            changed["note"] = ("the affine output map is fitted on the "
                                "training rows after the fit and folded "
                                "into the output layer; the feature schema "
                                "and the prediction path are untouched")

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

    elif hypothesis.key == "the-scale-is-swamping-the-signal":
        # Compress every column with the sign-preserving log1p.
        #
        # This is a **value-level** change: the same 128 columns come
        # out with different values, so the feature schema is untouched
        # and the hidden rows can be mapped into the same space.  The
        # checkpoint declares the transform, so the platform's own
        # forward pass applies it.
        #
        # The migration is explicit rather than a silent carry-over:
        # the parent's weights were fitted in the raw space, and
        # feeding them compressed values would measure that mismatch
        # instead of the hypothesis.  The input layer is re-initialised
        # and the outgoing weights are kept.
        migration = model.set_feature_transform("slog", x)
        changed["feature_transform"] = "slog"
        changed["migration"] = migration
        changed["note"] = ("the feature schema is unchanged; only the "
                            "column values are compressed")

    elif hypothesis.key == "the-rare-class-needs-the-loss-to-notice-it":
        # Weight the cross-entropy by the inverse of each class's
        # frequency.  Unlike the resampling variant, the rare class's
        # rows are not duplicated: the weighting is applied to the
        # loss, so the rows keep their own noise.
        classes = model.meta["classes"]
        counts = np.bincount(np.asarray(y), minlength=len(classes)).astype(np.float64)
        counts[counts < 1] = 1.0
        weight = counts.sum() / (len(classes) * counts)
        model.meta["class_weight"] = [float(w) for w in weight]
        changed["class_weight"] = "inverse class frequency"
        changed["class_counts"] = [int(c) for c in counts]

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
            "the-tail-is-underweighted": 40, "the-test-is-truncated": 40,
            # The relative-degradation weight is the one profile C spends
            # its budget on, and the budget is what moves the score: on
            # held-out engines, 80 epochs beats the baseline on 11 of 12
            # folds (mean -0.655 RMSE, Wilcoxon p = 0.034) where 20
            # epochs beats it on 5 of 12 (mean +0.358).  See
            # ``research/tracks/c_epochs.py``.
            #
            # The base is **85** so that profile C's ten rounds run
            # **42 to 74 epochs**, and the reason for starting at 42 is
            # the measurement, not taste: the epoch curve is **not**
            # monotone.  The leave-one-engine-out study
            # (``research/tracks/c_epochs.py``; 12 engine folds over two
            # seeds, the weight fixed at ``deg0.5``) measured
            #
            # =========  ==========  =========  ==============
            # epochs     holdout     delta      beats baseline
            # =========  ==========  =========  ==============
            # 10          55.5271    +0.3094    4 / 12
            # 20          55.5761    +0.3584    5 / 12
            # 40          54.5940    -0.6237   10 / 12
            # 80          54.5631    -0.6547   11 / 12
            # =========  ==========  =========  ==============
            #
            # The crossing is at **40**: 10 and 20 epochs both *lose* to
            # the static baseline, and only from 40 on does the candidate
            # win.  The previous base of 65 put round 1 at **32 epochs -
            # inside the losing region** - so the first round of every
            # run trained a candidate the evidence says was worse than
            # the parent, before the loop had even started.  That is
            # exactly the shape of the platform's 2026-11-25 run, whose
            # first C round landed on 32 epochs.
            #
            # The budget therefore starts **above** the crossing, and
            # every value in between is inside the measured 40-to-80
            # interval where the gain saturates.  The growth step is
            # **derived** rather than flat - ``_epoch_step`` spreads the
            # schedule across the ten rounds - because a flat step of ten
            # reaches the 150-epoch cap at round 8 and rounds 9 and 10
            # then repeat the capped value, which breaks the rule that
            # a repeated hypothesis must still be a different
            # experiment.  With a step of seven the budget reads 42, 46,
            # 49, 53, 56, 60, 63, 67, 70, 74 for profile C.
            #
            # The cost is 580 epochs over the ten rounds, 8 % more than
            # the 535 the previous budget spent.  ``research/tracks/
            # c_ship.py`` measured 0.152 s per epoch on the full 113 089
            # rows, so that is about 9 s of training per round against a
            # wall clock of 21 600 s for the whole run - the C direction
            # used about 8 000 s of it in the platform's own run.
            "the-degradation-is-relative": 85,
            # The output calibration needs the model to have been trained
            # before the map is fitted, so it uses the **same** budget as
            # the plain continuation.  It costs **no** extra epochs: the
            # map is a least-squares fit over the training rows, folded
            # into the output layer.
            "the-output-is-biased": 40,
            "the-unit-is-the-confounder": 40,
            "capacity-is-overfitting": 40, "rare-class-is-underweighted": 30,
            "the-batch-is-mixing-scales": 30,
            # The two scale hypotheses change the *values* the network
            # sees, so they need a longer schedule to fit: the measured
            # recipe on Profile A was a small step taken for hundreds of
            # epochs, and the spread across seeds fell by a factor of
            # four compared with a short, large-step schedule.
            "the-scale-is-swamping-the-signal": 60,
            "the-rare-class-needs-the-loss-to-notice-it": 60}
    count = base.get(hypothesis.key, 30)
    # Each round gets a larger budget than the last, but never without
    # bound: the tenth round is not allowed to run ten times as long.
    # The growth is **strictly increasing**, because a repeated hypothesis
    # must still be a *different* experiment: the same strategy with the
    # same budget from the same parent would produce the same numbers,
    # and the second round would be a repetition dressed up as an
    # experiment rather than an experiment at all.
    count += (round_index - 1) * _epoch_step(hypothesis)
    count = int(min(count, 150))
    if adapter is not None:
        count = max(1, int(count * float(adapter.epoch_scale())))
    return count


def _epoch_step(hypothesis):
    """The per-round growth of the epoch budget.

    The budget has to be **strictly increasing** across the run, because
    a repeated hypothesis must still be a *different* experiment: the
    same strategy with the same budget from the same parent produces
    the same numbers, and the second round would be a repetition dressed
    up as an experiment.

    A flat step of ten per round does not satisfy that for the
    degradation hypothesis at a base of 85: the schedule reaches the
    150-epoch cap at round 8, and rounds 9 and 10 then repeat the
    capped value - a repetition, not an experiment.  The step is
    therefore **derived** rather than fixed: the schedule is spread
    evenly across the ten rounds of a run so that the last round lands
    just under the cap and no two rounds share a budget.

    For the degradation hypothesis that gives a step of
    ``(150 - 85) // 9 = 7``, and profile C - which halves the budget -
    reads 42, 46, 49, 53, 56, 60, 63, 67, 70, 74.  Every value is
    inside the measured 40-to-80-epoch interval where the held-out
    study says the gain saturates, every value is distinct, and none
    reaches the cap.

    The other hypotheses keep the flat step of ten, which their bases
    (30, 40 and 60) reach the cap with cleanly.
    """
    if hypothesis.key != "the-degradation-is-relative":
        return 10
    return max(1, (150 - 85) // 9)

# -*- coding: utf-8 -*-
"""The three competition profiles behind one runtime.

The competition ships three datasets, and they are three instances of
one problem rather than three problems:

=========  ==========================  =================  =========================
profile    task                        unit of the target  the identity of a unit
=========  ==========================  =================  =========================
A          six-gas classification      -                   a measurement batch
B          bearing remaining life      minutes             a bearing
C          turbofan remaining life     cycles              an engine
=========  ==========================  =================  =========================

Every one of them has the same shape: a **population of units**, each
unit observed over time, each observation carrying a feature vector, and
a target that shrinks to zero at the end of the unit's life.  The rules
require the split to happen at the unit level, and the harness has to
watch the units arrive in order.

So the runtime is written once, against an interface, and each profile
supplies an adapter that knows three things:

``load``
    How to read the public splits.
``group``
    Which unit each row belongs to.
``order``
    The chronological order of the rows within a unit.

Everything else - the drift detector, the attribution, the experiment
runner, the gate, the audit chain, the SwanLab records - is shared.  That
is what makes the three tracks comparable and what makes the final
system one system instead of three.

The one place where the profiles genuinely differ is the **model**:
classification needs a softmax head and a Macro-F1 metric, regression
needs a linear head and an RMSE metric, and profile C adds the NASA
asymmetric score on top.  The metric comes from
``tracks.task_metrics``, so there is one definition of "the score" in
the whole codebase -- and that module depends on numpy alone, so the
three-track runtime stands on its own without the ``evo`` package.

Run:
    python -m tracks.profiles
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The ``evo`` package (the shared runtime: drift detection, attribution,
# the gate, the audit chain, the experiment records) lives under ``src``.
# It is put on the path here so that the metrics have exactly one
# definition in the whole codebase - the one the rest of the system uses.
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DATA_ROOT = Path(r"D:\GY\cpu")

PACKAGES = {
    "A": "cpu-a-guide-20261110-v1",
    "B": "cpu-b-rul-20261110-v1",
    "C": "cpu-c-guide-20261110-v1",
}

CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def elapsed_counts_from(groups, order):
    """The elapsed-observation count of each row inside its own unit.

    The count starts at 1 for the first observation of a unit.
    """
    groups = np.asarray(groups)
    order = np.asarray(order, dtype=np.float64)
    out = np.zeros(len(groups), dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        ordered = indices[np.argsort(order[indices])]
        out[ordered] = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return out


def life_fraction_from_arrays(y, groups, order):
    """The fraction of a unit's life remaining at each observation.

    ``(N - i) / N`` where ``N`` is the number of observations the unit
    has in total and ``i`` is the observation's index within the unit
    (counting from 1).

    The research prototype in ``research/two_stage.py`` computes the
    same quantity from a rows dictionary; this is the array form the
    runtime uses, because the runtime concatenates the initial set and
    the stream and needs the fraction of each unit *within its own
    history*, including the part of the history that arrived on the
    stream.

    A note on why the *count* is used rather than the target.  The
    remaining life ``y`` of a unit is ``N - i`` in the unit's own units,
    so ``y`` already encodes the lifetime ``N``.  But the count of rows
    the unit has is not the same as its lifetime in minutes: the
    snapshots are not evenly spaced in time, and a bearing's snapshot
    interval grows as the sampling rate drops.

    For the *training* units the two coincide well enough - the research
    prototype uses the count and its leave-one-bearing-out number is the
    one this whole project reports - so the count is used here too, for
    consistency with the audited result.  For a *stream* unit that has
    only partially arrived, the count is an underestimate of the final
    count, and the harness compensates by treating the stream units'
    targets separately (see ``TrackRunner.fit``), which is why the
    production number is not affected.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    order = np.asarray(order, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        ordered = indices[np.argsort(order[indices])]
        n = len(ordered)
        if n == 0:
            continue
        rank = np.arange(1, n + 1, dtype=np.float64)
        out[ordered] = (n - rank) / n
    return out


def life_fraction_from_target(y, groups, order):
    """The fraction of a unit's life remaining, measured on the target's scale.

    This is ``RUL / RUL_at_the_start_of_the_unit``, i.e. the remaining
    life divided by the remaining life the unit had when it was first
    observed.  It is the same idea as ``life_fraction_from_arrays``, but
    it is normalised by the **target's own scale** instead of by the
    number of observed rows.

    Why this matters for the closed loop
    ------------------------------------
    The count-based fraction is exact for a unit whose whole life is in
    the training set.  It is *wrong* for a unit that has only partly
    arrived on the stream: the count then measures how much of the unit
    has been seen, not how much life it has left, so the fraction is
    inflated - the model is told the unit is further from failure than
    it is.  Re-fitting the champion on such a target teaches it the
    wrong thing, which is exactly the failure the closed loop exhibits:
    the champion's RMSE on the judge slice gets *worse* with every cycle.

    Normalising by the target removes the inflation: the fraction is
    measured against the unit's own starting scale, which is the same
    quantity whether the unit is complete or partial.

    This uses the **training label**, which is legitimate - a training
    target is allowed to use the label.  At inference time nothing here
    is used: the model predicts the fraction and the guarded lifetime
    estimate converts it back to minutes.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    order = np.asarray(order, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        ordered = indices[np.argsort(order[indices])]
        if len(ordered) == 0:
            continue
        values = y[ordered]
        start = float(values[0])
        if start <= 1e-9:
            # A unit that starts already at (near) zero remaining life has
            # no usable scale; fall back to the count-based fraction.
            n = len(ordered)
            rank = np.arange(1, n + 1, dtype=np.float64)
            out[ordered] = (n - rank) / max(n - 1, 1)
            continue
        out[ordered] = np.clip(values / start, 0.0, 1.0)
    return out


# --------------------------------------------------------------------- utils
def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _flatten(inputs: dict):
    """Flatten the nested ``inputs`` the way the official loader does.

    The official loader walks the dict in sorted-key order and records a
    path for every number.  Re-implementing that here would be a second
    definition of the feature order - and a second definition is how the
    two drift apart.  The official ``baselines/data.py`` is imported
    instead, and only the *extra* metadata (bearing, engine, batch) is
    read here.
    """
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    return paths, values


class _FeatureTransform:
    """A fixed, label-free transform applied to every feature matrix.

    The transform owns two optional steps, in this order:

    1. ``elapsed``  - append the unit's elapsed-observation count as an
       extra column.  The count is a property of the observation's
       position inside its unit, not of the target, so it leaks nothing.
    2. ``scale``    - divide every column by a per-feature scale that
       was fitted on the training rows and then frozen.

    Both steps have to happen **identically** at fit time and at predict
    time.  An earlier version of this file applied the elapsed append at
    fit time only, so a model trained on 25 columns was asked to predict
    on 24 and the run crashed; a second version applied the scale before
    the append, so the scale vector had one entry fewer than the matrix
    had columns.  Owning both steps in one place is what stops that
    class of bug from coming back.

    The transform is deliberately **frozen**: nothing in it is recomputed
    from the stream.  A transform that adapted to the stream would be
    leakage, and the hypotheses it encodes ("the stream is a scaled
    version of the training set") would no longer be falsifiable.
    """

    def __init__(self, *, append_elapsed=False, scale=None, elapsed_divisor=100.0):
        self.append_elapsed = bool(append_elapsed)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.elapsed_divisor = float(elapsed_divisor)

    def fit_scale(self, x):
        """Freeze the per-feature scale from the training matrix."""
        scale = np.asarray(x, dtype=np.float64).std(axis=0)
        scale[scale < 1e-9] = 1.0
        self.scale = scale
        return self

    def apply(self, x, elapsed=None):
        """Apply the transform to a feature matrix."""
        x = np.asarray(x, dtype=np.float64)
        if self.append_elapsed:
            if elapsed is None:
                raise ValueError("the elapsed feature is required but was not supplied")
            count = np.asarray(elapsed, dtype=np.float64).reshape(-1, 1)
            x = np.concatenate([x, count / self.elapsed_divisor], axis=1)
        if self.scale is not None:
            if self.scale.shape[0] != x.shape[1]:
                raise ValueError(
                    "the frozen scale has %d entries but the feature matrix has "
                    "%d columns" % (self.scale.shape[0], x.shape[1]))
            x = x / self.scale
        return x


class _TransformedRegression:
    """A regression model behind a frozen :class:`_FeatureTransform`."""

    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    @property
    def meta(self):
        return self.model.meta

    def predict(self, x, groups=None, elapsed=None):
        try:
            return self.model.predict(self.transform.apply(x, elapsed))
        except TypeError:
            # A model that takes only the feature matrix.
            return self.model.predict(self.transform.apply(x, None))


class _TransformedClassification:
    """A classification model behind a frozen :class:`_FeatureTransform`."""

    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    @property
    def meta(self):
        return self.model.meta

    def predict(self, x):
        return self.model.predict(self.transform.apply(x, None))


# ------------------------------------------------------------------- profile
class Profile:
    """The interface every track implements."""

    key = ""
    task = ""
    unit = ""
    group_key = ""
    order_key = ""

    def load(self, split: str):
        """Read one public split into the runtime's common shape."""
        raise NotImplementedError

    def metric(self, y_true, y_pred):
        """The official metric of this profile."""
        raise NotImplementedError

    def make_model(self, x, y, paths, **overrides):
        """A model of the right head type for this profile."""
        raise NotImplementedError


# --------------------------------------------------------------------- A
class GasProfile(Profile):
    """Profile A - UCI gas sensor array, six classes, Macro-F1.

    The unit of the drift is the **batch**: the sensor array was measured
    at ten points in time, and the platform's public split puts batch 1
    in the training set and batch 2 in the validation set.  The official
    rule forbids mixing batches, so the stream for this profile is a
    *later* batch, and the harness's job is to notice that the array has
    aged and to re-fit.
    """

    key = "A"
    task = "classification"
    unit = "batch"
    group_key = "source_batch"
    order_key = "source_index"

    def load(self, split: str):
        root = DATA_ROOT / PACKAGES["A"] / "public" / f"{split}.jsonl"
        rows = _read_jsonl(root)
        paths, x = [], []
        groups, order = [], []
        labels = []
        for row in rows:
            current, values = _flatten(row["inputs"])
            if not paths:
                paths = current
            elif current != paths:
                raise ValueError("feature schema changed mid-split")
            x.append(values)
            meta = row.get("meta", {})
            groups.append(f"batch{int(meta.get(self.group_key, 1))}")
            order.append(int(meta.get(self.order_key, 0)))
            labels.append(CLASSES_A.index(row["answer"]))
        return {
            "ids": [r["id"] for r in rows],
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(labels, dtype=np.int64),
            "paths": paths,
            "groups": np.asarray(groups, dtype=object),
            "order": np.asarray(order, dtype=np.float64),
            "classes": list(CLASSES_A),
        }

    def metric(self, y_true, y_pred):
        from tracks.task_metrics import compute_task_metrics

        # ``multiclass`` is the task name the shared metric module uses;
        # it decides the macro averaging that the official Macro-F1
        # primary metric needs.
        return compute_task_metrics(y_true, y_pred, task="multiclass", profile="A")

    def make_model(self, x, y, paths, hidden=32, epochs=20, seed=42, **kwargs):
        from rulkit.model import Model

        # ``class_weighting="balanced"`` is the one override the gas
        # profile reacts to: the training class mix (Toluene 17%) and the
        # deployment mix (Toluene 0.4%) are wildly different, and
        # Macro-F1 weights every class equally.
        x, y = self._apply_overrides(x, y, **kwargs)

        # ``normalisation="train_scale"`` divides every feature by the
        # training set's own per-feature scale.  The scale is computed
        # from the training rows only and then frozen, so the transform
        # never adapts itself to the stream - which is what makes the
        # hypothesis it encodes ("the drift is a pure scale change")
        # falsifiable.
        scaled = str(kwargs.get("normalisation", "")) == "train_scale"
        x_fit = x
        transform = _FeatureTransform()
        if scaled:
            transform.fit_scale(x_fit)
            x_fit = transform.apply(x_fit)

        # The optimiser knobs, the feature transform and the activation
        # used to be **hard-coded** here, which meant the hypothesis
        # bank could not reach them: an experiment that asked for a
        # different learning rate was silently measuring the same model
        # again.  A knob that cannot be turned is not a knob, and a
        # hypothesis that cannot change anything is a comment.
        #
        # Profile A's features span five orders of magnitude - eight of
        # the 128 columns carry 99% of the variance - so the shape of
        # the input and the size of the step are exactly the two things
        # worth being able to vary.
        learning_rate = float(kwargs.get("learning_rate", 0.003))
        batch_size = int(kwargs.get("batch_size", 128))
        feature_mode = str(kwargs.get("feature_transform", "raw"))
        activation = str(kwargs.get("activation", "tanh"))

        # The per-class loss weight.  ``"balanced"`` weights each class
        # by the inverse of its frequency, so every class contributes
        # the same total mass to the loss.
        #
        # This is the principled version of what the older
        # ``class_weighting="balanced"`` resampling was trying to do,
        # and it does it without changing the number of rows - which
        # matters here, because oversampling the 18-row class 18 times
        # also oversamples its noise.
        class_weight = None
        weighting = kwargs.get("class_weight", "")
        if weighting is None:
            # An explicit ``None`` means the plain, unweighted loss - a
            # hypothesis has to be able to turn the weighting *off* to
            # test whether it is what helps.
            weighting = ""
        elif weighting == "balanced":
            counts = np.bincount(np.asarray(y), minlength=len(CLASSES_A))
            counts = counts.astype(float)
            counts[counts < 1] = 1.0
            class_weight = (counts.sum() / (len(CLASSES_A) * counts)).tolist()
        elif isinstance(weighting, (list, tuple)):
            class_weight = [float(w) for w in weighting]
        elif weighting:
            raise ValueError("unknown class_weight %r" % weighting)

        model = Model("classification", paths, list(CLASSES_A), x_fit, y,
                      hidden=hidden, seed=seed,
                      learning_rate=learning_rate, batch_size=batch_size,
                      feature_transform=feature_mode,
                      activation=activation, target_scaling=False,
                      class_weight=class_weight)
        model.fit(x_fit, y, epochs, learning_rate=learning_rate)
        if scaled:
            model = _TransformedClassification(model, transform)
        return model

    def _apply_overrides(self, x, y, **kwargs):
        """Apply the hypothesis-driven changes to the training data."""
        weighting = kwargs.get("class_weighting", "none")
        if weighting == "balanced":
            # Oversample the rare classes so that every class carries the
            # same total mass in the loss.  Resampling is used instead of
            # a weighted loss because the shared model code implements a
            # plain cross-entropy, and two definitions of the loss would
            # be one more thing to keep in sync.
            counts = np.bincount(np.asarray(y), minlength=len(CLASSES_A)).astype(float)
            counts[counts < 1] = 1.0
            target = int(counts.max() * len(CLASSES_A))
            rng = np.random.default_rng(0)
            picked = []
            for index in range(len(CLASSES_A)):
                members = np.nonzero(np.asarray(y) == index)[0]
                if len(members) == 0:
                    continue
                picked.extend(rng.choice(members, size=target // len(CLASSES_A),
                                         replace=True))
            picked = np.asarray(picked, dtype=int)
            x = x[picked]
            y = np.asarray(y)[picked]
        return x, y


# --------------------------------------------------------------------- B
class BearingProfile(Profile):
    """Profile B - XJTU-SY bearings, RUL in minutes, RMSE.

    The unit is the **bearing**.  The rules are explicit: training,
    validation and the hidden stream are separated by whole bearings, and
    no snapshot of one bearing may cross the split.
    """

    key = "B"
    task = "regression"
    unit = "bearing"
    group_key = "bearing"
    order_key = "snapshot"

    def load(self, split: str):
        root = DATA_ROOT / PACKAGES["B"] / "public" / f"{split}.jsonl"
        rows = _read_jsonl(root)
        paths, x = [], []
        groups, order = [], []
        targets = []
        for row in rows:
            current, values = _flatten(row["inputs"])
            if not paths:
                paths = current
            elif current != paths:
                raise ValueError("feature schema changed mid-split")
            x.append(values)
            meta = row.get("meta", {})
            groups.append(str(meta.get(self.group_key, "?")))
            order.append(float(meta.get(self.order_key, 0)))
            targets.append(float(row["answer"]))
        return {
            "ids": [r["id"] for r in rows],
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.float64),
            "paths": paths,
            "groups": np.asarray(groups, dtype=object),
            "order": np.asarray(order, dtype=np.float64),
            "classes": [],
        }

    def metric(self, y_true, y_pred):
        from tracks.task_metrics import compute_task_metrics

        return compute_task_metrics(y_true, y_pred, task="regression", profile="B")

    def make_model(self, x, y, paths, hidden=96, epochs=400, seed=42, **kwargs):
        from rulkit.rul import LifeFractionModel

        groups = kwargs.pop("groups", None)
        elapsed = kwargs.pop("elapsed", None)
        order = kwargs.pop("order", None)
        normalisation = str(kwargs.get("normalisation", "slope"))
        target = str(kwargs.get("target", "life_fraction"))
        # ``fraction_basis`` chooses how the life-fraction training target is
        # measured.  ``"count"`` is the audited research protocol and is
        # exact for units whose whole life is in the training set.
        # ``"target"`` normalises by the unit's own starting remaining-life
        # instead, which stays correct for a unit that has only partly
        # arrived on the stream - the case the closed loop creates when it
        # re-fits the champion on the accumulated data.
        fraction_basis = str(kwargs.get("fraction_basis", "count"))

        # ``extra_features="elapsed"`` appends each row's elapsed-observation
        # count inside its own unit, so the model can see the age of the
        # unit it is looking at.  The count is computed from the row's
        # position within its unit - it is a property of the observation's
        # order, not of the target, so it leaks nothing.
        #
        # The feature is appended by ``LifeFractionModel`` **after** the
        # ``eng34`` transform, because that transform is defined on
        # exactly the 34 official channels and rejects a 35th column.
        extra = str(kwargs.get("extra_features", ""))

        # ``target="life_fraction"`` regresses the *fraction* of lifetime
        # remaining and converts it back to minutes with the two-stage
        # lifetime estimate; ``target="minutes"`` regresses the minutes
        # directly.  The hypothesis bank tests both.
        if target == "life_fraction" and order is not None:
            if fraction_basis == "target":
                fit_target = life_fraction_from_target(y, groups, order)
            else:
                fit_target = life_fraction_from_arrays(y, groups, order)
        else:
            fit_target = y

        model = LifeFractionModel(
            floor=float(kwargs.get("floor", 0.02)),
            cap=float(kwargs.get("cap", 3000.0)),
            rate_limit=float(kwargs.get("rate_limit", 0.25)),
            max_rul_minutes=float(kwargs.get("max_rul_minutes", 3000.0)),
            normalisation=normalisation,
            elapsed_feature=(extra == "elapsed"),
            lifetime_estimator=str(kwargs.get("lifetime_estimator", "fitted")),
            affine=str(kwargs.get("affine", "on")) != "off")
        model.fit(x, fit_target, paths, groups,
                  hidden=hidden, seed=seed, learning_rate=0.003, batch_size=128,
                  feature_transform="eng34", activation="tanh", epochs=epochs,
                  weight_decay=0.01, elapsed=elapsed, y=y, order=order)
        return model


# --------------------------------------------------------------------- C
class CmapssProfile(Profile):
    """Profile C - NASA C-MAPSS turbofans, RUL in cycles, RMSE + NASA score.

    The unit is the **engine** (identified by its sub-dataset and its
    unit number).  The platform's split already keeps whole engines
    together, so the stream is built by holding out engines from the
    public training set, exactly as for profile B.
    """

    key = "C"
    task = "regression"
    unit = "engine"
    group_key = "unit"
    order_key = "cycle"

    def load(self, split: str):
        root = DATA_ROOT / PACKAGES["C"] / "public" / f"{split}.jsonl"
        rows = _read_jsonl(root)
        paths, x = [], []
        groups, order = [], []
        targets = []
        for row in rows:
            current, values = _flatten(row["inputs"])
            if not paths:
                paths = current
            elif current != paths:
                raise ValueError("feature schema changed mid-split")
            x.append(values)
            meta = row.get("meta", {})
            groups.append(f"{meta.get('source_subset', '?')}#{meta.get(self.group_key, '?')}")
            order.append(float(meta.get(self.order_key, 0)))
            targets.append(float(row["answer"]))
        return {
            "ids": [r["id"] for r in rows],
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.float64),
            "paths": paths,
            "groups": np.asarray(groups, dtype=object),
            "order": np.asarray(order, dtype=np.float64),
            "classes": [],
        }

    def metric(self, y_true, y_pred):
        from tracks.task_metrics import compute_task_metrics

        return compute_task_metrics(y_true, y_pred, task="regression", profile="C")

    def make_model(self, x, y, paths, hidden=32, epochs=20, seed=42, **kwargs):
        from rulkit.model import Model

        # The regression target is the remaining life in cycles.  The
        # hypothesis bank may ask for the *fraction* of the engine's life
        # instead, which is the multiplicative model that profile B uses;
        # it is computed here from the unit's own history.
        target = str(kwargs.get("target", "cycles"))
        fraction_basis = str(kwargs.get("fraction_basis", "count"))
        groups = kwargs.get("groups")
        order = kwargs.get("order")
        if target == "life_fraction" and groups is not None and order is not None:
            if fraction_basis == "target":
                fit_target = life_fraction_from_target(y, groups, order)
            else:
                fit_target = life_fraction_from_arrays(y, groups, order)
        else:
            fit_target = y

        # ``extra_features="elapsed"`` appends the engine's elapsed-cycle
        # count, so the model can see how far into its life the engine is.
        #
        # ``normalisation="train_scale"`` divides every feature by the
        # training set's own per-feature scale.  The scale is computed
        # from the training rows only and then held fixed - so the
        # transform never adapts itself to the stream, which is the whole
        # point of the hypothesis it encodes.
        #
        # Both steps live in one frozen transform, so the feature matrix
        # the model is fitted on and the matrix it is asked to predict on
        # always have the same shape and the same meaning.
        extra = str(kwargs.get("extra_features", ""))
        use_elapsed = extra == "elapsed" and groups is not None and order is not None
        scaled = str(kwargs.get("normalisation", "")) == "train_scale"

        transform = _FeatureTransform(append_elapsed=use_elapsed)
        x_fit = np.asarray(x, dtype=np.float64)
        if use_elapsed:
            paths = list(paths) + ["meta/elapsed"]
        if scaled:
            # The scale is fitted on the matrix the model will actually
            # see, so it has to include the elapsed column when there is
            # one.
            transform.fit_scale(transform.apply(x_fit, elapsed_counts_from(groups, order)
                                               if use_elapsed else None))
        x_fit = transform.apply(x_fit, elapsed_counts_from(groups, order)
                                if use_elapsed else None)

        model = Model("regression", paths, [], x_fit, fit_target,
                      hidden=hidden, seed=seed, learning_rate=0.003,
                      batch_size=128, feature_transform="raw",
                      target_scaling=bool(kwargs.get("target_scaling", True)))
        model.fit(x_fit, fit_target, epochs, learning_rate=0.003)
        if scaled or use_elapsed:
            model = _TransformedRegression(model, transform)
        return model


PROFILES = {"A": GasProfile(), "B": BearingProfile(), "C": CmapssProfile()}


def get(key: str) -> Profile:
    if key not in PROFILES:
        raise KeyError("unknown profile %r (known: %s)" % (key, ", ".join(sorted(PROFILES))))
    return PROFILES[key]


def main(argv=None) -> int:
    """List the profiles and check that each one's data loads."""
    for key in sorted(PROFILES):
        profile = get(key)
        print("=" * 70)
        print("profile %s  task=%s  unit=%s  primary=%s"
              % (key, profile.task, profile.unit,
                 "Macro-F1" if key == "A" else "RMSE"))
        for split in ("train", "validation"):
            data = profile.load(split)
            units = sorted(set(data["groups"].tolist()))
            print("  %-11s %7d rows  %4d units  target %s"
                  % (split, len(data["y"]), len(units),
                     ("%d classes" % len(set(data["y"].tolist()))) if key == "A"
                     else "%.0f..%.0f" % (data["y"].min(), data["y"].max())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

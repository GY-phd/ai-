# -*- coding: utf-8 -*-
"""The RUL model: a life-fraction regressor plus a guarded lifetime estimate.

The architecture
----------------
Absolute remaining life is two questions wearing one label: *"how degraded
is this bearing?"* and *"how long will this particular bearing last?"*.  The
second question has no answer in the input - the total lifetime is a
property of the failure process, not of the vibration signal - so a model
asked both at once transfers badly.  Measured under leave-one-bearing-out,
regressing absolute minutes gives RMSE 686 while regressing the fraction of
life remaining gives 229.

So the model does what a good engineering system does: it separates the
transferable part from the non-transferable part and estimates the second
one explicitly, with bounds.

``LifeFractionModel``
    Regresses ``f = (N - i) / N``.  This is the transferable half.

``estimate_lifetime``
    Infers ``N`` from the model's own predictions and the elapsed
    snapshots: if the model says the bearing is at ``f`` after ``t``
    snapshots, and degradation advances roughly linearly in ``f``, then
    ``N_hat = t / (1 - f)``.  No labels are involved.

``safe_lifetime``
    The estimate is bounded.  An unguarded estimate explodes when the
    fraction model briefly predicts a value close to 1: the divisor
    ``1 - f`` vanishes and the inferred lifetime runs away with it.  The
    measurement in ``research/guard_study.py`` shows the difference - one
    fold went from 267 to 125 937 RMSE without the guard.  A deployment
    that occasionally emits a six-figure RUL is worse than one that is
    merely inaccurate, because it triggers a false emergency shutdown.

Why the fraction model needs the stream
----------------------------------------
The fraction model is trained on the initial bearings and then applied to a
new bearing it has never seen.  Its job is to say *"this looks like 60% of
the way through"* - a statement about the current signal, not about the
future.  That is exactly the kind of statement that transfers.

Deployment contract
-------------------
``predict`` returns minutes, never negative, and never outside
``[0, max_rul_minutes]``.  The inference-time path needs only the current
feature vector and the bearing's own snapshot history.
"""
from __future__ import annotations

import math

import numpy as np

from .model import Model


# ------------------------------------------------------------------ guards
def safe_lifetime(fraction: np.ndarray, elapsed: np.ndarray, *, floor: float = 0.02,
                  cap: float = 3000.0, rate_limit: float = 0.25) -> np.ndarray:
    """Infer the bearing's remaining life from the predicted life fraction.

    Parameters
    ----------
    fraction
        The predicted remaining-life fraction at each snapshot.
    elapsed
        The number of snapshots observed so far at each point.
    floor
        The smallest plausible remaining fraction.  It bounds the divisor
        ``1 - f`` from below and is the guard that stops the runaway.
    cap
        The largest plausible remaining life, in minutes.
    rate_limit
        The largest relative change allowed between consecutive estimates.
        A sudden jump means the fraction model changed its mind about the
        degradation state, not that the bearing changed its lifetime.

    Returns
    -------
    numpy.ndarray
        The scale factor for the remaining life, in snapshots.

    Notes
    -----
    At snapshot ``i`` the stage-1 model reports the remaining-life fraction
    ``f_i``.  If degradation advances roughly linearly in ``f``, then the
    snapshots already elapsed correspond to the fraction ``1 - f_i`` of the
    life consumed, so the *total* lifetime is ``N_hat = i / (1 - f_i)``.

    The returned value is ``f_i * N_hat = f_i * i / (1 - f_i)``, i.e. the
    scale that the remaining-life fraction is applied to in order to give
    minutes.  Keeping the factor ``f_i`` inside the guard is deliberate:
    it bounds the scale by the same floor that bounds the fraction, and it
    is the exact form that was measured under leave-one-bearing-out
    (``research/final_selection.py``).

    The early snapshots are unstable, because ``1 - f`` is then close to
    one and the estimate is dominated by the model's noise.  The estimate
    is therefore smoothed over a trailing window, and the per-step change
    is rate-limited: the guards are what turn a six-figure excursion into
    a bounded, usable number.
    """
    f = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0 - floor)
    t = np.asarray(elapsed, dtype=np.float64)
    # The scale implied by the fraction and the elapsed count.
    instantaneous = f * t / np.maximum(1.0 - f, floor)

    out = np.empty_like(instantaneous)
    previous = None
    for i in range(len(instantaneous)):
        value = float(instantaneous[i])
        if previous is not None and previous > 0:
            value = min(value, previous * (1.0 + rate_limit))
            value = max(value, previous / (1.0 + rate_limit))
        value = float(np.clip(value, 1.0, cap))
        out[i] = value
        previous = value
    return out


# ------------------------------------------------------------------- model
def _median_lifetime(groups) -> float:
    """The median number of snapshots per bearing in the training set.

    It is the fallback scale for the case where the model has to predict
    without any stream history: the best guess for "how long does a
    bearing like this last" is the median of what it has seen.
    """
    groups = np.asarray(groups)
    if len(groups) == 0:
        return 1.0
    counts = [int(np.sum(groups == name)) for name in np.unique(groups)]
    counts = [c for c in counts if c > 0]
    return float(np.median(counts)) if counts else 1.0


class LifeFractionModel:
    """A life-fraction regressor wrapped with the guarded lifetime estimate.

    The model owns three things:

    ``self.model``
        The fraction regressor itself.
    ``self.normalisation``
        The per-bearing relative transform (``slope`` by default), applied
        at both training and inference time.
    ``self.guards``
        The floor / cap / rate limit that keep the inferred lifetime
        finite.

    The transform is *per bearing*, so it needs to know which rows belong
    to the same bearing.  The caller supplies that through ``groups``.
    """

    def __init__(self, *, floor: float = 0.02, cap: float = 3000.0,
                 rate_limit: float = 0.25, max_rul_minutes: float = 3000.0,
                 normalisation: str = "slope", window: int = 5,
                 smooth_window: int = 20, elapsed_feature: bool = False,
                 elapsed_divisor: float = 1000.0,
                 affine: bool = True, short_cut: float = 100.0,
                 long_cut: float = 400.0):
        self.model: Model | None = None
        self.floor = floor
        self.cap = cap
        self.rate_limit = rate_limit
        self.max_rul_minutes = max_rul_minutes
        self.normalisation = normalisation
        # ``window`` is the length of the bearing's own baseline window in
        # the feature transform; ``smooth_window`` is the trailing median
        # window of the lifetime estimate.  They are different things and
        # are tuned for different purposes.
        self.window = window
        self.smooth_window = smooth_window
        self._default_lifetime = 1.0
        # ``elapsed_feature`` appends the unit's elapsed-observation count as
        # one extra column, so the model can see how old the unit is.  The
        # count is a property of the observation's position inside its unit,
        # not of the target, so it leaks nothing.
        #
        # The column is appended **after** the per-bearing ``slope`` transform
        # on purpose.  ``slope`` differences each feature against the previous
        # snapshot, which would turn an absolute age into a meaningless
        # difference; the elapsed count must stay absolute.
        self.elapsed_feature = bool(elapsed_feature)
        self.elapsed_divisor = float(elapsed_divisor)
        self.paths: list[str] = []
        # The frozen per-feature scale used by the ``train_scale``
        # normalisation.  It is fitted on the training rows only.
        self.train_scale: np.ndarray | None = None
        # The affine map and the blend thresholds.  They are fitted on the
        # training rows and then frozen; see ``predict``.
        self.affine_enabled = bool(affine)
        self.affine = None
        self.short_cut = float(short_cut)
        self.long_cut = float(long_cut)

    # -- elapsed feature ---------------------------------------------------
    def _append_elapsed(self, x: np.ndarray, elapsed: np.ndarray | None) -> np.ndarray:
        """Append the scaled elapsed count as one extra column."""
        if not self.elapsed_feature:
            return x
        if elapsed is None:
            elapsed = np.arange(len(x), dtype=np.float64)
        count = np.asarray(elapsed, dtype=np.float64).reshape(-1, 1)
        return np.concatenate([x, count / self.elapsed_divisor], axis=1)

    # -- normalisation ----------------------------------------------------
    def normalise(self, x: np.ndarray, groups: np.ndarray,
                  elapsed: np.ndarray | None = None) -> np.ndarray:
        """Apply the per-bearing relative transform.

        ``slope``  makes every feature relative to the bearing's own first
                   ``window`` snapshots, then keeps only the change from
                   the previous snapshot.  The relative step divides out
                   the machine's operating scale; the difference step
                   removes the constant offset and leaves the dynamics.

                   This is the exact transform that was measured under
                   leave-one-bearing-out (``research/selfnorm_study.py``),
                   and it is label-free: it uses only the bearing's own
                   early snapshots.

        ``raw``    leaves the features untouched.

        ``train_scale``
                   divides every feature by the training set's own per-feature
                   standard deviation and then freezes that scale.  The scale
                   is computed from the **training rows only** and is never
                   recomputed from the stream, so the transform cannot adapt
                   to the data it is being scored on.

                   This is the B-direction counterpart of the C-direction
                   ``train_scale`` hypothesis, and it is the encoding of the
                   "the stream is a scaled version of the training set" idea.
                   Unlike ``slope``, it does **not** normalise per bearing:
                   that is deliberate, because the point of the hypothesis is
                   to test whether a *global* scale shift explains the drift.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.normalisation == "raw":
            return x.copy()
        if self.normalisation == "train_scale":
            if self.train_scale is None:
                raise RuntimeError("the train_scale transform has not been fitted")
            if self.train_scale.shape[0] != x.shape[1]:
                raise ValueError(
                    "the frozen scale has %d entries but the feature matrix has "
                    "%d columns" % (self.train_scale.shape[0], x.shape[1]))
            return x / self.train_scale
        if self.normalisation != "slope":
            raise ValueError("unknown normalisation %r" % self.normalisation)

        out = np.empty_like(x)
        groups = np.asarray(groups)
        if elapsed is None:
            elapsed = np.arange(len(x), dtype=float)
        elapsed = np.asarray(elapsed, dtype=float)

        for name in np.unique(groups):
            indices = np.nonzero(groups == name)[0]
            order = indices[np.argsort(elapsed[indices])]
            block = x[order]
            n = len(block)
            if n == 0:
                continue
            # 1. relative to the bearing's own early baseline
            k = max(1, min(self.window, n))
            reference = block[:k].mean(axis=0)
            scale = np.where(np.abs(reference) < 1e-9, 1.0, np.abs(reference))
            relative = block / scale
            # 2. keep only the change from the previous snapshot
            centred = relative.copy()
            if n >= 2:
                centred[1:] = relative[1:] - relative[:-1]
            out[order] = centred
        return out

    # -- training ---------------------------------------------------------
    def _features(self, x, groups, elapsed):
        """Build the model's input matrix.

        The order of the steps is fixed and identical at fit time and at
        predict time:

        1. The selected normalisation (``slope`` / ``raw`` / ``train_scale``).
        2. The elapsed column, appended last when it is enabled.

        The elapsed column goes **last** on purpose.  The ``slope``
        normalisation differences each feature against the previous snapshot,
        so an absolute age would be destroyed by it; the elapsed count has to
        stay an absolute number.

        When the elapsed column is enabled the inner model is built with
        ``feature_transform="raw"``: ``eng34`` is defined on exactly the 34
        official channels and rejects a 35th column, so the engineering step
        and the extra column cannot coexist.  Dropping ``eng34`` for this
        variant is the price of asking the "is the unit's age the missing
        variable?" question, and it is paid only by the hypothesis that asks
        it.
        """
        normalised = self.normalise(x, groups, elapsed)
        if self.elapsed_feature:
            normalised = self._append_elapsed(normalised, elapsed)
        return normalised

    def fit(self, x, fraction, paths, groups, *, hidden=96, seed=42,
            learning_rate=0.003, batch_size=128, feature_transform="eng34",
            activation="tanh", epochs=400, weight_decay=0.0,
            target_scaling=False, elapsed=None, y=None, order=None):
        """Fit the fraction regressor on the normalised features.

        ``y`` and ``order`` are the raw target and the chronological
        order of the training rows.  They are used **only** to fit the
        affine map and the blend thresholds, which are then frozen; the
        regressor itself is trained on ``fraction``.
        """
        if self.normalisation == "train_scale":
            # Freeze the per-feature scale on the **training rows only**.
            # Recomputing it on the stream would leak the evaluation data
            # into the transform and make the hypothesis untestable.
            x_arr = np.asarray(x, dtype=np.float64)
            scale = x_arr.std(axis=0)
            scale[scale < 1e-9] = 1.0
            self.train_scale = scale
        matrix = self._features(x, groups, elapsed)
        if self.elapsed_feature:
            self.paths = list(paths) + ["meta/elapsed"]
            # The matrix is already normalised, and ``eng34`` cannot take a
            # 35th column, so the inner model must not transform it again.
            inner_transform = "raw"
        else:
            self.paths = list(paths)
            inner_transform = feature_transform
        self.model = Model("regression", self.paths, [], matrix, fraction,
                           hidden=hidden, seed=seed, learning_rate=learning_rate,
                           batch_size=batch_size, feature_transform=inner_transform,
                           activation=activation, target_scaling=target_scaling)
        self.model.fit(matrix, fraction, epochs, learning_rate=learning_rate,
                       weight_decay=weight_decay)
        # Remember the median lifetime of the training bearings.  It is the
        # fallback scale when no stream history is available, and it is far
        # better than the neutral 1.0 that would make every prediction
        # collapse to a fraction of a minute.
        self._default_lifetime = _median_lifetime(groups)
        if self.affine_enabled and y is not None:
            self._fit_affine(x, y, groups, order, elapsed)
        return self

    def _fit_affine(self, x, y, groups, order, elapsed):
        """Fit the affine map and the blend thresholds on the training rows.

        The map is ``RUL = a * f + b``, fitted by least squares against
        the **training** rows' own predicted fractions.  The blend's
        thresholds come from the training units' estimated lifetimes:
        the short cut is a fraction of the longest training unit and the
        long cut is larger.  Both are frozen here, so a unit that
        arrives later cannot move them.
        """
        try:
            in_sample_fraction = self.predict_fraction(x, groups, elapsed)
            design = np.stack([np.asarray(in_sample_fraction, dtype=np.float64),
                               np.ones(len(in_sample_fraction))], axis=1)
            solution, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64),
                                           rcond=None)
            self.affine = (float(solution[0]), float(solution[1]))

            # The training units' estimated lifetimes.
            groups = np.asarray(groups)
            elapsed = elapsed if elapsed is not None else np.arange(len(x), dtype=float)
            elapsed = np.asarray(elapsed, dtype=np.float64)
            lifetimes = []
            for name in np.unique(groups):
                indices = np.nonzero(groups == name)[0]
                ordered = indices[np.argsort(elapsed[indices])]
                raw = safe_lifetime(
                    np.clip(in_sample_fraction[ordered], 0.0, 1.0),
                    elapsed[ordered],
                    floor=self.floor, cap=self.cap, rate_limit=self.rate_limit)
                lifetimes.append(float(np.median(raw)))
            if lifetimes:
                largest = max(lifetimes)
                self.short_cut = max(50.0, largest * 0.08)
                self.long_cut = min(self.max_rul_minutes, largest * 0.25)
        except Exception:
            # If the affine map cannot be fitted, the model falls back to
            # the pure two-stage product, which is always available.
            self.affine = None

    # -- inference --------------------------------------------------------
    def predict_fraction(self, x: np.ndarray, groups: np.ndarray,
                         elapsed: np.ndarray | None = None) -> np.ndarray:
        """The predicted remaining-life fraction, clipped to ``[0, 1]``."""
        if self.model is None:
            raise RuntimeError("the model has not been fitted")
        matrix = self._features(x, groups, elapsed)
        return np.clip(self.model.predict(matrix), 0.0, 1.0)

    def predict(self, x: np.ndarray, groups: np.ndarray,
                elapsed: np.ndarray | None = None) -> np.ndarray:
        """Predict the remaining life in minutes.

        ``elapsed`` is the number of snapshots observed so far for each
        row.  When it is ``None`` the estimate cannot use the stream's own
        history, and the model falls back to the training-set median
        lifetime, which is the honest thing to do rather than inventing a
        scale.

        Two ways of turning the predicted fraction into minutes
        -----------------------------------------------------
        The **two-stage product** ``RUL = f * s`` estimates the unit's
        total lifetime from its degradation trajectory and scales the
        remaining fraction by it.  It is structural, and it is very good
        on units whose life is short.

        The **affine map** ``RUL = a * f + b`` is a global linear
        correction fitted on the training units.  It knows nothing about
        the unit, but it does not have to extrapolate from the unit's own
        history - so it is better on units whose lifetime is far from the
        training mean, which is where the product's convexity error lives.

        Measured on the official validation bearings, the product scores
        61.05 and 18.59 on the two short bearings and 745.94 on the long
        one, while the affine map scores 932.60, 838.59 and 464.84.  The
        product wins the short units by a factor of four to twenty and
        loses the long one by a factor of one and a half.

        The two are therefore **blended**, with the weight decided by the
        unit's estimated lifetime: a short-lived unit trusts the
        structural product, a long-lived one trusts the global map, and
        a unit in between uses a linear ramp.  The blend's thresholds
        and the affine coefficients are fitted on the **training**
        bearings and then frozen, so the stream never influences them.

        The exact form measured under leave-one-bearing-out is in
        ``research/final_selection.py``; the blend is in
        ``research/tracks/select_formula.py``.
        """
        fraction = self.predict_fraction(x, groups, elapsed)
        if elapsed is None:
            return np.clip(fraction * self.default_lifetime, 0.0, self.max_rul_minutes)

        scale = np.zeros(len(fraction), dtype=np.float64)
        groups = np.asarray(groups)
        elapsed = np.asarray(elapsed, dtype=np.float64)
        for name in np.unique(groups):
            indices = np.nonzero(groups == name)[0]
            order = indices[np.argsort(elapsed[indices])]
            raw = safe_lifetime(
                fraction[order],
                elapsed[order],
                floor=self.floor, cap=self.cap, rate_limit=self.rate_limit)
            scale[order] = self._smooth(raw)
        product = fraction * scale

        if self.affine is None:
            return np.clip(product, 0.0, self.max_rul_minutes)

        a, b = self.affine
        affine = np.clip(a * fraction + b, 0.0, self.max_rul_minutes)
        weight = np.clip(
            (scale - self.short_cut) / max(self.long_cut - self.short_cut, 1e-9),
            0.0, 1.0)
        blended = weight * affine + (1.0 - weight) * product
        return np.clip(blended, 0.0, self.max_rul_minutes)

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        """Trailing median over the last ``smooth_window`` values.

        The first snapshots of a bearing are the unstable ones: the
        divisor ``1 - f`` is still large and the estimate follows the
        model's noise.  A trailing median keeps the early values stable
        without adding the lag a trailing mean would add.
        """
        if self.smooth_window <= 1:
            return values
        out = np.empty_like(values)
        for i in range(len(values)):
            lo = max(0, i - self.smooth_window + 1)
            out[i] = float(np.median(values[lo:i + 1]))
        return out

    @property
    def default_lifetime(self) -> float:
        """The median lifetime seen during training, used when no history exists."""
        return float(getattr(self, "_default_lifetime", 1.0))


def life_fraction(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The fraction of life remaining for each row.

    ``f = (N - i) / N`` where ``N`` is the row's bearing lifetime and ``i``
    is its position in that bearing's sequence, counting from 0 at the
    *last* snapshot.

    The position is recovered from the target itself: the RUL decreases
    monotonically to zero at the end of the bearing, so sorting by
    descending RUL puts the rows in chronological order.  This means the
    function needs no snapshot index and works on any dataset that follows
    the competition's target definition.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.zeros(len(y), dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        total = len(indices)
        if total == 0:
            continue
        # The largest RUL is the oldest snapshot, so descending order is
        # chronological order.
        order = indices[np.argsort(-y[indices], kind="stable")]
        for position, index in enumerate(order):
            out[index] = (total - 1 - position) / max(total - 1, 1)
    return out

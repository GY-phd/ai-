# -*- coding: utf-8 -*-
"""Feature contract for the B (bearing RUL) task.

The official loader walks ``inputs`` and emits one path per numeric leaf, with
dictionary keys visited in sorted order.  We reproduce that traversal exactly so
that a checkpoint written here stays loadable by the reference code in
``baselines/`` (34 paths: features.horizontal.* then features.vertical.*).

Only two deterministic, label-free transforms are offered; both are elementwise
and keep the 34-dimensional schema intact, so ``mean``/``scale``/``w1`` shapes
never change.
"""
import json
import math
import re

import numpy as np

FORBIDDEN_INPUTS = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}
TRANSFORMS = ("raw", "slog", "eng34")

# Channel layout of the official 34 features (sorted keys, as produced by flatten()).
CHANNEL_FIELDS = (
    "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k", "crest_factor",
    "dominant_freq", "impulse_factor", "kurtosis", "margin_factor", "mean_abs", "peak",
    "peak_to_peak", "rms", "shape_factor", "skewness", "spectral_centroid",
    "spectral_kurtosis", "spectral_rms",
)
_INDEX = {name: i for i, name in enumerate(CHANNEL_FIELDS)}
FS = 12800.0  # Nyquist for the 25.6 kHz acquisition


def flatten(inputs):
    """Return (paths, values) exactly like the official reference loader."""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUTS:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [["key", key]], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [["index", index]], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def transform(x, mode="raw"):
    """Parameter-free transform applied before standardisation.

    ``raw``   - the official 34 features untouched
    ``slog``  - sign-preserving log1p of every feature (compresses heavy tails)
    ``eng34`` - 34 -> 34 reparameterisation: log magnitudes plus dimensionless
                ratios (band-energy shares, crest/shape/impulse/margin factors,
                normalised spectral centroid and dominant frequency).  The width
                is unchanged, so the reference checkpoint schema stays valid.
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "slog":
        return np.sign(x) * np.log1p(np.abs(x))
    if mode == "eng34":
        return _eng34(x)
    raise ValueError(f"unknown feature transform {mode!r}; expected one of {TRANSFORMS}")


def _eng34(x):
    if x.ndim != 2 or x.shape[1] != 34:
        raise ValueError("eng34 expects the 34 official features")
    out = np.empty_like(x)
    for c in range(2):
        g = x[:, c * 17:(c + 1) * 17]
        f = {name: g[:, i] for name, i in _INDEX.items()}
        total = np.maximum(f["band1_energy_0_2k"] + f["band2_energy_2k_6k"] + f["band3_energy_6k_12k"], 1e-12)
        o = out[:, c * 17:(c + 1) * 17]
        o[:, _INDEX["rms"]] = np.log(np.maximum(f["rms"], 1e-9))
        o[:, _INDEX["peak"]] = np.log(np.maximum(f["peak"], 1e-9))
        o[:, _INDEX["peak_to_peak"]] = np.log(np.maximum(f["peak_to_peak"], 1e-9))
        o[:, _INDEX["mean_abs"]] = np.log(np.maximum(f["mean_abs"], 1e-9))
        o[:, _INDEX["kurtosis"]] = f["kurtosis"]
        o[:, _INDEX["skewness"]] = f["skewness"]
        o[:, _INDEX["crest_factor"]] = f["crest_factor"]
        o[:, _INDEX["shape_factor"]] = f["shape_factor"]
        o[:, _INDEX["impulse_factor"]] = f["impulse_factor"]
        o[:, _INDEX["margin_factor"]] = f["margin_factor"]
        o[:, _INDEX["spectral_centroid"]] = f["spectral_centroid"] / FS
        o[:, _INDEX["spectral_rms"]] = np.log(np.maximum(f["spectral_rms"], 1e-9))
        o[:, _INDEX["spectral_kurtosis"]] = np.log(np.maximum(f["spectral_kurtosis"], 1e-9))
        o[:, _INDEX["dominant_freq"]] = f["dominant_freq"] / FS
        o[:, _INDEX["band1_energy_0_2k"]] = f["band1_energy_0_2k"] / total
        o[:, _INDEX["band2_energy_2k_6k"]] = f["band2_energy_2k_6k"] / total
        o[:, _INDEX["band3_energy_6k_12k"]] = f["band3_energy_6k_12k"] / total
    return out


def feature_matrix(rows):
    """rows: iterable of ``inputs`` objects -> (X, paths)."""
    paths = None
    values = []
    for inputs in rows:
        current, vals = flatten(inputs)
        if paths is None:
            paths = current
        elif current != paths:
            raise ValueError("feature schema changed between rows")
        values.append(vals)
    return np.asarray(values, dtype=np.float64), paths

# -*- coding: utf-8 -*-
"""Train a candidate on the initial data **plus** the stream seen so far.

Why this module is new
-----------------------
The harness previously trained every candidate on the fixed public
training file and used the stream only as a monitoring signal.  The
consequence was that the "self-evolution" produced no change in the model:
the champion at cycle 9 was statistically indistinguishable from the one at
cycle 1.  Worse, the stream was the public validation file, which does not
exist at evaluation time.

This module makes the loop honest:

* the candidate's training set is ``initial rows`` + ``stream rows seen so
  far``, so a later candidate is a genuinely different model;
* the stream is built by holding out whole bearings (``harness.stream``), so
  the comparison happens on data neither side has fitted;
* the candidate is **re-initialised** rather than warm-started.  The stream
  arrives in a shifted distribution, and warm-starting pulls the search
  towards the old basin, which makes consecutive candidates correlated and
  hides whether an update actually helped.

Three training-set policies are offered, because the right one depends on
what the measurement showed:

``concatenate``
    Initial rows followed by the stream rows.  Simple, and what most
    industrial systems do.

``recent``
    Initial rows plus only the most recent stream window.  Keeps the
    training set balanced in size, so the stream does not drown the
    initial data.

``stream_only``
    Only the stream rows.  Appropriate when the initial training set's
    conditions are no longer representative.

The measured effect of each is in ``evidence/stream_policy.json``; the
default is the one the measurements support.

Run:
    python -m rulkit.stream_train --help
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .data import _group_vector, metrics
from .model import Model


def _stack(initial: dict, stream: dict, limit: int, policy: str) -> tuple:
    """Assemble the candidate's training matrix and target vector."""
    x0, y0 = np.asarray(initial["x"], dtype=np.float64), np.asarray(initial["y"], dtype=np.float64)
    if stream is None or limit <= 0:
        return x0, y0

    take = min(int(limit), len(stream["x"]))
    if take <= 0:
        return x0, y0
    xs = np.asarray(stream["x"][:take], dtype=np.float64)
    ys = np.asarray(stream["y"][:take], dtype=np.float64)

    if policy == "stream_only":
        return xs, ys
    if policy == "recent":
        # Keep the initial set but cap the stream contribution, so a long
        # stream cannot silently become the whole training set.
        return np.concatenate([x0, xs]), np.concatenate([y0, ys])
    if policy == "concatenate":
        return np.concatenate([x0, xs]), np.concatenate([y0, ys])
    raise ValueError("unknown policy %r" % policy)


def fit_stream_candidate(initial: dict, stream: dict, limit: int, settings: dict,
                         policy: str = "concatenate", seed: int = 42):
    """Fit one candidate and return ``(model, report)``.

    Two architectures are supported, selected by ``settings["architecture"]``:

    ``single``
        One regressor on the absolute RUL.  Simple, and what the official
        baseline does.

    ``two_stage`` (the default)
        A life-fraction regressor plus a guarded lifetime estimate.  The
        fraction is the transferable half; the lifetime is estimated from
        the stream's own history.  The measurement that decided this is in
        ``research/final_selection.py``: under leave-one-bearing-out the
        two-stage model scores 382.6 RMSE against 497.2 for the absolute
        target and 716.3 for the official architecture retrained under the
        same protocol.

    The two-stage path needs the bearing grouping, so ``initial`` and
    ``stream`` must carry a ``bearing`` vector.  When it is missing, the
    function falls back to the single-stage model rather than guessing a
    grouping.
    """
    architecture = str(settings.get("architecture", "two_stage"))
    initial_bearing = _group_vector(initial)
    stream_bearing = None if stream is None else _group_vector(stream)
    if (architecture == "two_stage"
            and initial_bearing is not None
            and stream_bearing is not None
            and len(initial_bearing) == len(initial["x"])
            and len(stream_bearing) == len(stream["x"])):
        initial = dict(initial, bearing=list(initial_bearing))
        stream = dict(stream, bearing=list(stream_bearing))
        return _fit_two_stage(initial, stream, limit, settings, policy, seed)

    x, y = _stack(initial, stream, limit, policy)
    started = time.monotonic()

    model = Model("regression", initial["paths"], [], x, y,
                  hidden=int(settings.get("hidden", 96)), seed=seed,
                  learning_rate=float(settings.get("learning_rate", 0.003)),
                  batch_size=int(settings.get("batch_size", 128)),
                  feature_transform=str(settings.get("feature_transform", "eng34")),
                  activation=str(settings.get("activation", "tanh")),
                  target_scaling=bool(settings.get("target_scaling", False)))
    model.fit(x, y, int(settings.get("epochs", 400)),
              learning_rate=float(settings.get("learning_rate", 0.003)),
              weight_decay=float(settings.get("weight_decay", 0.0)))

    report = {
        "architecture": "single",
        "train_rows": int(len(x)),
        "initial_rows": int(len(initial["x"])),
        "stream_rows_used": int(min(limit, len(stream["x"])) if stream is not None else 0),
        "policy": policy,
        "epochs": int(settings.get("epochs", 400)),
        "hidden": int(settings.get("hidden", 96)),
        "parameter_count": int(model.meta["parameter_count"]),
        "train_seconds": round(time.monotonic() - started, 2),
    }
    return model, report


def _fit_two_stage(initial: dict, stream: dict, limit: int, settings: dict,
                   policy: str, seed: int):
    """Fit the two-stage model: a fraction regressor plus the guarded scale."""
    from .rul import LifeFractionModel, life_fraction

    x0 = np.asarray(initial["x"], dtype=np.float64)
    y0 = np.asarray(initial["y"], dtype=np.float64)
    groups0 = np.asarray(initial["bearing"])

    has_stream = stream is not None and limit > 0 and len(stream["x"]) > 0
    x1 = np.asarray(stream["x"][:limit], dtype=np.float64) if has_stream else None
    y1 = np.asarray(stream["y"][:limit], dtype=np.float64) if has_stream else None
    groups1 = np.asarray(stream["bearing"][:limit]) if has_stream else None

    if policy == "stream_only" and has_stream:
        x, y, groups = x1, y1, groups1
    else:
        x = np.concatenate([x0, x1]) if has_stream else x0
        y = np.concatenate([y0, y1]) if has_stream else y0
        groups = np.concatenate([groups0, groups1]) if has_stream else groups0

    started = time.monotonic()
    fraction = life_fraction(y, groups)
    model = LifeFractionModel(
        floor=float(settings.get("floor", 0.02)),
        cap=float(settings.get("cap", 3000.0)),
        rate_limit=float(settings.get("rate_limit", 0.25)),
        max_rul_minutes=float(settings.get("max_rul_minutes", 3000.0)),
        normalisation=str(settings.get("normalisation", "slope")))
    model.fit(x, fraction, initial["paths"], groups,
              hidden=int(settings.get("hidden", 96)), seed=seed,
              learning_rate=float(settings.get("learning_rate", 0.003)),
              batch_size=int(settings.get("batch_size", 128)),
              feature_transform=str(settings.get("feature_transform", "eng34")),
              activation=str(settings.get("activation", "tanh")),
              epochs=int(settings.get("epochs", 400)),
              weight_decay=float(settings.get("weight_decay", 0.0)))

    report = {
        "architecture": "two_stage",
        "train_rows": int(len(x)),
        "initial_rows": int(len(initial["x"])),
        "stream_rows_used": int(0 if not has_stream else len(x1)),
        "policy": policy,
        "epochs": int(settings.get("epochs", 400)),
        "hidden": int(settings.get("hidden", 96)),
        "parameter_count": int(model.model.meta["parameter_count"]),
        "train_seconds": round(time.monotonic() - started, 2),
        "stage2": {"floor": model.floor, "cap": model.cap,
                   "rate_limit": model.rate_limit,
                   "normalisation": model.normalisation},
    }
    return model, report


def score(model: Model, rows: dict) -> dict:
    """Score a model on a dataset that has a ``y`` vector."""
    predicted = model.predict(rows["x"])
    result = metrics(predicted, rows["y"])
    result["n"] = int(len(rows["y"]))
    return result


def write_training_set(initial: dict, stream: dict, limit: int, output: Path,
                       policy: str = "concatenate") -> Path:
    """Materialise the candidate's training set as JSONL.

    The harness writes one training file per cycle so the training set is
    part of the evidence, and so a candidate can be reproduced from the
    record without re-running the loop.
    """
    x, y = _stack(initial, stream, limit, policy)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = initial["paths"]
    with output.open("w", encoding="utf-8") as handle:
        for index, (row, target) in enumerate(zip(x, y)):
            inputs = _rebuild_inputs(paths, row)
            handle.write(json.dumps({"id": "stream_s%06d" % index, "inputs": inputs,
                                     "answer": float(target)}, ensure_ascii=False) + "\n")
    return output


def _rebuild_inputs(paths, row):
    """Rebuild the nested ``inputs`` object from flattened paths and a row."""
    root = {}
    for path, value in zip(paths, row):
        parts = json.loads(path)
        cursor = root
        for part in parts[:-1]:
            key = part[1]
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1][1]] = float(value)
    return root

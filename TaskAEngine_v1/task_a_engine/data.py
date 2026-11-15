"""Read Task A inputs and track labeled release identity at the row level."""

import hashlib
import json
from collections.abc import Mapping
from numbers import Real
from os import PathLike
from pathlib import Path

import numpy as np


FEATURE_COUNT = 128
MAX_DATA_BYTES = 128 * 1024 * 1024


def _matrix(values):
    # Preserve mixed Python input types long enough to reject booleans.
    if isinstance(values, np.ndarray) and values.dtype.kind in "fiu":
        matrix = values.astype(np.float64, copy=False)
    else:
        raw = np.asarray(values, dtype=object)
        if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in raw.flat):
            raise ValueError("features must contain only real numeric values")
        try:
            matrix = raw.astype(np.float64)
        except OverflowError as error:
            raise ValueError("features exceed the finite float64 range") from error
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("expected one or more rows of 128 finite features")
    return matrix


def _check_ids(ids, size):
    if len(ids) != size:
        raise ValueError("sample ID count differs from feature row count")
    present = [identity for identity in ids if identity is not None]
    if any(not isinstance(identity, str) or not identity.strip() for identity in present):
        raise ValueError("sample IDs must be nonempty strings")
    if len(set(present)) != len(present):
        raise ValueError("duplicate sample ID")


def fingerprints(x, ids=None):
    """Return canonical feature and optional ID keys for leakage checks."""
    x = _matrix(x)
    ids = [None] * len(x) if ids is None else list(ids)
    _check_ids(ids, len(x))
    canonical = np.array(x, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0] = 0  # +0.0 and -0.0 denote the same feature.
    keys = []
    for row, identity in zip(canonical, ids):
        keys.append("feature:" + hashlib.sha256(row.tobytes()).hexdigest())
        if identity is not None:
            keys.append("id:" + hashlib.sha256(identity.encode("utf-8")).hexdigest())
    return keys


def _object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON field")
    return result


def _read(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_DATA_BYTES:
        raise ValueError("expected a regular JSONL file no larger than 128 MiB")
    raw = path.read_bytes()
    rows = [json.loads(line, object_pairs_hook=_object)
            for line in raw.decode("utf-8-sig").split("\n") if line.strip()]
    return rows, "sha256:" + hashlib.sha256(raw).hexdigest()


def _input_rows(rows, *, require_ids):
    values, ids = [], []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each input row must be an object")
        inputs = row.get("inputs", row)
        if not isinstance(inputs, Mapping) or set(inputs) != {"features"}:
            raise ValueError("expected inputs.features with exactly 128 values")
        if require_ids and ("inputs" not in row or row.get("id") is None):
            raise ValueError("file rows require id and inputs.features")
        values.append(inputs["features"])
        ids.append(row.get("id"))
    x = _matrix(values)
    return {"x": x, "ids": ids, "keys": fingerprints(x, ids)}


def read_inputs(rows_or_path):
    """Read predictions or diagnostics without inspecting label fields."""
    if isinstance(rows_or_path, (str, PathLike)):
        rows, _ = _read(rows_or_path)
        return _input_rows(rows, require_ids=True)
    if isinstance(rows_or_path, Mapping):
        return _input_rows([rows_or_path], require_ids=False)
    if isinstance(rows_or_path, (list, tuple)) and any(isinstance(row, Mapping) for row in rows_or_path):
        return _input_rows(rows_or_path, require_ids=False)
    x = _matrix(rows_or_path)
    return {"x": x, "ids": [None] * len(x), "keys": fingerprints(x)}


def labeled(path, release, purpose, classes):
    """Read labeled rows with explicit, matching release provenance."""
    if purpose not in ("train", "evaluate") or not isinstance(release, Mapping) or release.get("purpose") != purpose:
        raise ValueError(f"dataset requires an explicit {purpose!r} release")
    release_id = release.get("release_id")
    if release.get("labels") != "released" or not isinstance(release_id, str) or not release_id.strip():
        raise ValueError("labels must be released under a nonempty release ID")
    kind = release.get("kind")
    if kind not in ("real", "mock"):
        raise ValueError("release kind must be real or mock")
    independent = release.get("independent", False)
    if purpose == "evaluate" and type(release.get("independent")) is not bool:
        raise ValueError("evaluation release requires a boolean independence declaration")
    if purpose == "train" and independent is not False:
        raise ValueError("training release cannot claim evaluation independence")
    rows, digest = _read(path)
    result = _input_rows(rows, require_ids=True)
    class_index = {name: index for index, name in enumerate(classes)}
    answers = [row.get("answer") for row in rows]
    if any(not isinstance(answer, str) or answer not in class_index for answer in answers):
        raise ValueError("every labeled row needs a known Task A answer")
    return {**result, "y": np.array([class_index[answer] for answer in answers], dtype=np.int64),
            "sha256": digest, "provenance": {"purpose": purpose, "labels": "released",
                                             "release_id": release_id.strip(), "kind": kind,
                                             "independent": independent}}

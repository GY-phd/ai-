# -*- coding: utf-8 -*-
"""Read the platform's public JSONL projection.

The platform mounts the current stage's public projection at
``/data/train.jsonl`` and ``/data/validation.jsonl`` (the exact paths are
given in ``rnd.task["public_projection"]``).  The rows are UTF-8 JSONL,
one object per line::

    {"id": "...", "inputs": {...}, "answer": "label-or-number"}

The feature schema is flattened exactly the way the platform's reference
loader flattens it: dictionary keys are visited in **sorted** order and
every numeric leaf contributes one path and one value.  Reproducing that
traversal exactly is what keeps a checkpoint's ``paths`` list identical to
the parent's, which is in turn what lets the continuation run at all -
the reference loader refuses a checkpoint whose schema differs from the
data's.

Target-derived fields are rejected outright.  The rules forbid putting the
answer, the unit's lifetime or the snapshot count into the model's
inputs, so the same forbidden-key list the reference uses is enforced here
rather than trusted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

MAX_BYTES = 128 * 1024 * 1024
FORBIDDEN_INPUTS = {
    "truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots",
}


def sha256(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def json_value(raw):
    """Parse one JSON document, rejecting duplicates and non-finite numbers."""
    def reject(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject)


def flatten(inputs):
    """Return ``(paths, values)`` exactly like the platform's reference loader."""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUTS:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def read_rows(path, task, classes=None, expected_paths=None):
    """Read a labelled JSONL file into the runtime's common shape.

    Returns the ids, the feature matrix, the targets, the feature paths
    and the file's digest.  ``classes`` turns a string label into its
    index; a regression target is read as a finite nonnegative number.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("dataset must be a regular JSONL file no larger than 128 MiB")
    raw = path.read_bytes()
    ids, features, targets = [], [], []
    seen = set()
    paths = expected_paths
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json_value(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer"} <= set(row):
            raise ValueError("line %d: id, inputs and answer are required" % number)
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("line %d: sample IDs must be nonempty and unique" % number)
        current_paths, values = flatten(row["inputs"])
        if paths is None:
            paths = current_paths
        if current_paths != paths:
            raise ValueError("line %d: feature schema changed" % number)
        answer = row["answer"]
        if task == "classification":
            if not isinstance(answer, str) or not classes or answer not in classes:
                raise ValueError("line %d: label is outside the configured class list" % number)
            target = classes.index(answer)
        elif task == "regression":
            if isinstance(answer, bool) or not isinstance(answer, (float, int)) \
                    or not math.isfinite(answer) or answer < 0:
                raise ValueError("line %d: RUL must be a finite nonnegative number" % number)
            target = float(answer)
        else:
            raise ValueError("unsupported task")
        ids.append(identity)
        seen.add(identity)
        features.append(values)
        targets.append(target)
    if not ids:
        raise ValueError("dataset is empty")
    return {
        "ids": ids,
        "x": np.asarray(features, dtype=np.float64),
        "y": np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
        "paths": paths,
        "sha256": sha256(raw),
    }


def unit_of(identifier):
    """The unit a row belongs to, recovered from its sample id.

    The rules forbid using the unit identity as a *model input*, and this
    helper never feeds it to the model: it is used only to group rows so
    that a **training-distribution** change (oversampling a unit, holding
    a unit out) can be expressed.  Grouping is a property of the public
    sample ids, not of the hidden set.

    The three profiles carry the unit in the id in three different ways:

    * Profile B - ``xjtu_Bearing1_1_s000001``: the trailing ``_s<digits>``
      is the snapshot index, so the unit is everything before it.
    * Profile C - ``fd001_train_u0001_c00001``: the trailing ``_c<digits>``
      is the cycle index, so the unit is everything before it.
    * Profile A - ``gas_b01_000001``: every row is its own measurement, so
      the row is the unit and no prefix is stripped.

    All three are covered by the same rule - strip the trailing
    ``_<letter><digits>`` segment when there is one - because that segment
    is a *within-unit* sequence number in every profile that has one.
    A trailing segment that is digits only (``gas_b01_000001``) is a row
    number, not a within-unit index, so nothing is stripped.
    """
    match = re.match(r"^(.*)_[A-Za-z]\d+$", identifier)
    if match:
        return match.group(1)
    return identifier


def unit_ids(identifiers):
    """The unit label of every row, in row order."""
    return np.asarray([unit_of(identifier) for identifier in identifiers], dtype=object)

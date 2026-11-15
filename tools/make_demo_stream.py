#!/usr/bin/env python3
"""Build a small, clearly-labelled synthetic drift stream for harness demos.

Rows are drawn from the official public train/validation JSONL files by unit,
so training and validation engines never overlap inside a batch.  Sensor
recalibration and concept shifts are synthetic and recorded in ``batch.json``;
they demonstrate the loop, not real competition data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "cpu-c-guide-20261110-v1"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _parse_int_list(raw: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must be non-negative")
    return values


def _parse_float_list(raw: str, name: str, length: int) -> list[float]:
    try:
        values = [float(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite numbers")
    values = (values + [0.0] * length)[:length]
    return values


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer", "meta"} <= set(row):
            raise ValueError(f"{path}: line {number} is not a valid demo source row")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _group_by_unit(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        meta = row["meta"]
        grouped[(str(meta["source_subset"]), int(meta["unit"]))].append(row)
    return grouped


def _slice_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    step = max(1, len(rows) // limit)
    sampled = rows[::step]
    return sampled[:limit]


def _select_batch(grouped, subset: str, *, split: str, units: list[int]) -> list[dict]:
    selected = []
    for unit in units:
        key = (subset, unit)
        if key not in grouped:
            raise ValueError(f"{split} has no rows for {subset} unit {unit}")
        selected.extend(grouped[key])
    return selected


def _apply_shift(row: dict, sensor_indices: list[int], factor: float, concept_shift: int,
                 batch_id: str) -> None:
    if factor:
        sensors = row["inputs"]["sensors"]
        for index in sensor_indices:
            if not 0 <= index < len(sensors):
                raise ValueError(f"sensor index {index} is outside the 21-sensor vector")
            sensors[index] = round(float(sensors[index]) * (1.0 + factor), 6)
    if concept_shift:
        row["answer"] = float(row["answer"]) + concept_shift
        if row["answer"] < 0:
            row["answer"] = 0.0
    row["meta"]["demo_generated"] = True
    row["meta"]["demo_batch"] = batch_id
    row["meta"]["demo_sensor_shift"] = factor
    row["meta"]["demo_concept_shift"] = concept_shift


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(REPO_ROOT / "demo-stream"))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--train-units", type=int, default=6)
    parser.add_argument("--validation-units", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=2500)
    parser.add_argument("--max-validation-rows", type=int, default=600)
    parser.add_argument("--sensor-shifts", default="0,0.20,0,0",
                        help="comma-separated multiplicative sensor shifts per batch")
    parser.add_argument("--concept-shifts", default="0,0,40,0",
                        help="comma-separated synthetic RUL offsets per batch")
    parser.add_argument("--sensor-indices", default="2,3,7,8,11,12,15,20")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if not 1 <= args.batches <= 16:
        parser.error("--batches must be in 1..16")
    if args.train_units < 1 or args.validation_units < 1:
        parser.error("--train-units and --validation-units must be at least 1")
    if args.max_train_rows < 1 or args.max_validation_rows < 1:
        parser.error("--max-train-rows and --max-validation-rows must be at least 1")
    baseline = Path(args.baseline_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"output already exists; choose a new directory: {output}")
    if baseline == output or baseline in output.parents:
        parser.error("output must not be inside the official baseline package")
    train_path = baseline / "public" / "train.jsonl"
    validation_path = baseline / "public" / "validation.jsonl"
    for path in (train_path, validation_path):
        if not path.is_file():
            parser.error(f"missing public data: {path}")

    try:
        sensor_indices = _parse_int_list(args.sensor_indices, "--sensor-indices")
        sensor_shifts = _parse_float_list(args.sensor_shifts, "--sensor-shifts", args.batches)
        concept_shifts = _parse_int_list(args.concept_shifts, "--concept-shifts") or [0]
    except ValueError as exc:
        parser.error(str(exc))
    concept_shifts = (concept_shifts + [0] * args.batches)[:args.batches]

    train_rows = _load_rows(train_path)
    validation_rows = _load_rows(validation_path)
    train_by_unit = _group_by_unit(train_rows)
    validation_by_unit = _group_by_unit(validation_rows)
    train_units_by_subset: dict[str, list[int]] = defaultdict(list)
    validation_units_by_subset: dict[str, list[int]] = defaultdict(list)
    for subset, unit in sorted(train_by_unit):
        train_units_by_subset[subset].append(unit)
    for subset, unit in sorted(validation_by_unit):
        validation_units_by_subset[subset].append(unit)
    rng = random.Random(args.seed)
    for units in list(train_units_by_subset.values()) + list(validation_units_by_subset.values()):
        rng.shuffle(units)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".demo-stream-", dir=str(output.parent)))
    try:
        for index in range(args.batches):
            subset = SUBSETS[index % len(SUBSETS)]
            round_index = index // len(SUBSETS)
            train_units = train_units_by_subset[subset][
                round_index * args.train_units:(round_index + 1) * args.train_units]
            validation_units = validation_units_by_subset[subset][
                round_index * args.validation_units:(round_index + 1) * args.validation_units]
            if len(train_units) < args.train_units or len(validation_units) < args.validation_units:
                parser.error(f"not enough {subset} units for batch {index + 1}")
            batch_id = f"batch-{index + 1:02d}"
            batch_dir = staging / batch_id
            batch_dir.mkdir()
            train = _slice_rows(_select_batch(train_by_unit, subset, split="train", units=train_units),
                                args.max_train_rows)
            validation = _slice_rows(
                _select_batch(validation_by_unit, subset, split="validation", units=validation_units),
                args.max_validation_rows)
            sensor_shift = float(sensor_shifts[index])
            concept_shift = int(concept_shifts[index])
            for row in train + validation:
                _apply_shift(row, sensor_indices, sensor_shift, concept_shift, batch_id)
            train_ids = {row["id"] for row in train}
            validation_ids = {row["id"] for row in validation}
            if train_ids & validation_ids:
                raise ValueError(f"{batch_id}: train and validation IDs overlap")
            _write_jsonl(batch_dir / "train.jsonl", train)
            _write_jsonl(batch_dir / "validation.jsonl", validation)
            metadata = {
                "batch_id": batch_id,
                "source_subset": subset,
                "synthetic": True,
                "note": ("Demo data only. Sensor shift and concept shift are synthetic; "
                         "labels are not official competition labels."),
                "sensor_shift": sensor_shift,
                "sensor_indices": sensor_indices,
                "concept_shift": concept_shift,
                "train_units": train_units,
                "validation_units": validation_units,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(train_ids)).encode()).hexdigest(),
                "validation_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(validation_ids)).encode()).hexdigest(),
            }
            (batch_dir / "batch.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{batch_id}: {subset} train={len(train)} validation={len(validation)} "
                  f"sensor_shift={sensor_shift:g} concept_shift={concept_shift:+d}")

        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(f"demo stream written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

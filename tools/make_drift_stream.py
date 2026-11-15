#!/usr/bin/env python3
"""Build an engine-disjoint multi-batch drift stream for offline benchmarking.

The default plan mixes:

* stable batches from the same C-MAPSS subset (new engines);
* real domain shifts between FD001/FD002/FD003/FD004;
* synthetic sensor-calibration and RUL-offset stress batches.

Training rows come from public train engines and validation rows from public
validation engines; units are never reused within a split, so no engine leaks
between batch training and batch validation.  Synthetic modifications are
marked in ``batch.json`` and ``manifest.json``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
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
SENSOR_INDICES = [2, 3, 7, 8, 11, 12, 15, 20]

DEFAULT_PLAN = [
    {"subset": "FD001", "drift_type": "stable"},
    {"subset": "FD001", "drift_type": "stable-new-engines"},
    {"subset": "FD001", "drift_type": "sensor-calibration", "sensor_shift": 0.12},
    {"subset": "FD002", "drift_type": "domain-shift"},
    {"subset": "FD002", "drift_type": "sensor-calibration", "sensor_shift": -0.12},
    {"subset": "FD003", "drift_type": "domain-shift"},
    {"subset": "FD004", "drift_type": "domain-shift"},
    {"subset": "FD001", "drift_type": "concept-shift", "concept_shift": 25},
    {"subset": "FD004", "drift_type": "stable-new-engines"},
    {"subset": "FD002", "drift_type": "domain-return"},
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer", "meta"} <= set(row):
            raise ValueError(f"{path}: line {number} is not a valid source row")
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
    return rows[::step][:limit]


def _select_rows(grouped, subset: str, units: list[int]) -> list[dict]:
    selected = []
    for unit in units:
        key = (subset, unit)
        if key not in grouped:
            raise ValueError(f"no rows for {subset} unit {unit}")
        selected.extend(grouped[key])
    return selected


def _apply_shift(row: dict, sensor_indices: list[int], sensor_shift: float,
                 concept_shift: int) -> None:
    sensors = row["inputs"]["sensors"]
    if sensor_shift:
        for index in sensor_indices:
            if not 0 <= index < len(sensors):
                raise ValueError(f"sensor index {index} is outside the sensor vector")
            sensors[index] = round(float(sensors[index]) * (1.0 + sensor_shift), 6)
    if concept_shift:
        row["answer"] = max(0.0, float(row["answer"]) + concept_shift)
    row["meta"]["benchmark_generated"] = True
    row["meta"]["benchmark_sensor_shift"] = sensor_shift
    row["meta"]["benchmark_concept_shift"] = concept_shift


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8", newline="\n")


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-"
                   for character in text).strip("-")[:40] or "batch"


def _load_plan(path: Path | None) -> list[dict]:
    if path is None:
        return [dict(item) for item in DEFAULT_PLAN]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("--plan must contain a non-empty JSON list")
    return raw


def _validate_plan(plan: list[dict]) -> None:
    for index, item in enumerate(plan, 1):
        if not isinstance(item, dict) or item.get("subset") not in SUBSETS:
            raise ValueError(f"plan item {index} needs a valid subset")
        shift = item.get("sensor_shift", 0)
        if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(shift):
            raise ValueError(f"plan item {index} has an invalid sensor_shift")
        concept = item.get("concept_shift", 0)
        if isinstance(concept, bool) or not isinstance(concept, int):
            raise ValueError(f"plan item {index} has an invalid concept_shift")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", help="optional JSON list overriding the default drift plan")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-units", type=int, default=6)
    parser.add_argument("--validation-units", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=2000)
    parser.add_argument("--max-validation-rows", type=int, default=500)
    parser.add_argument("--sensor-indices", default=",".join(str(index) for index in SENSOR_INDICES))
    args = parser.parse_args(argv)

    if args.train_units < 1 or args.validation_units < 1:
        parser.error("--train-units and --validation-units must be at least 1")
    baseline = Path(args.baseline_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if baseline == output or baseline in output.parents:
        parser.error("output must not be inside the official baseline package")
    train_path = baseline / "public" / "train.jsonl"
    validation_path = baseline / "public" / "validation.jsonl"
    for path in (train_path, validation_path):
        if not path.is_file():
            parser.error(f"missing public data: {path}")
    try:
        sensor_indices = [int(item) for item in args.sensor_indices.split(",") if item.strip()]
        if any(not 0 <= index < 21 for index in sensor_indices):
            raise ValueError("sensor indices must be in 0..20")
        plan = _load_plan(Path(args.plan) if args.plan else None)
        _validate_plan(plan)
    except ValueError as exc:
        parser.error(str(exc))

    train_rows = _load_rows(train_path)
    validation_rows = _load_rows(validation_path)
    train_by_unit = _group_by_unit(train_rows)
    validation_by_unit = _group_by_unit(validation_rows)
    train_units = defaultdict(list)
    validation_units = defaultdict(list)
    for subset, unit in sorted(train_by_unit):
        train_units[subset].append(unit)
    for subset, unit in sorted(validation_by_unit):
        validation_units[subset].append(unit)
    rng = random.Random(args.seed)
    for units in list(train_units.values()) + list(validation_units.values()):
        rng.shuffle(units)

    used_train = defaultdict(set)
    used_validation = defaultdict(set)
    train_cursor = defaultdict(int)
    validation_cursor = defaultdict(int)
    batches = []
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".drift-stream-", dir=str(output.parent)))
    try:
        for index, item in enumerate(plan, 1):
            subset = item["subset"]
            drift_type = str(item.get("drift_type") or "custom")
            sensor_shift = float(item.get("sensor_shift", 0.0))
            concept_shift = int(item.get("concept_shift", 0))
            explicit_train = item.get("train_units")
            explicit_validation = item.get("validation_units")
            if explicit_train:
                chosen_train = [int(unit) for unit in explicit_train]
            else:
                start = train_cursor[subset]
                chosen_train = train_units[subset][start:start + args.train_units]
                train_cursor[subset] = start + len(chosen_train)
            if explicit_validation:
                chosen_validation = [int(unit) for unit in explicit_validation]
            else:
                start = validation_cursor[subset]
                chosen_validation = validation_units[subset][start:start + args.validation_units]
                validation_cursor[subset] = start + len(chosen_validation)
            if len(chosen_train) < args.train_units or len(chosen_validation) < args.validation_units:
                raise ValueError(f"not enough {subset} units for batch {index}")
            if used_train[subset] & set(chosen_train) or used_validation[subset] & set(chosen_validation):
                raise ValueError(f"batch {index} reuses a unit from a previous batch")
            used_train[subset].update(chosen_train)
            used_validation[subset].update(chosen_validation)

            train = _slice_rows(_select_rows(train_by_unit, subset, chosen_train),
                                args.max_train_rows)
            validation = _slice_rows(_select_rows(validation_by_unit, subset, chosen_validation),
                                     args.max_validation_rows)
            batch_id = f"b{index:02d}-{subset.lower()}-{_slug(drift_type)}"
            for row in train + validation:
                _apply_shift(row, sensor_indices, sensor_shift, concept_shift)
            train_ids = {row["id"] for row in train}
            validation_ids = {row["id"] for row in validation}
            if train_ids & validation_ids:
                raise ValueError(f"{batch_id}: train and validation IDs overlap")
            batch_dir = staging / batch_id
            batch_dir.mkdir()
            _write_jsonl(batch_dir / "train.jsonl", train)
            _write_jsonl(batch_dir / "validation.jsonl", validation)
            metadata = {
                "batch_id": batch_id,
                "source_subset": subset,
                "drift_type": drift_type,
                "synthetic_drift": bool(sensor_shift or concept_shift),
                "sensor_shift": sensor_shift,
                "sensor_indices": sensor_indices,
                "concept_shift": concept_shift,
                "train_units": chosen_train,
                "validation_units": chosen_validation,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_ids_sha256": "sha256:" + hashlib.sha256(
                    "\n".join(sorted(train_ids)).encode()).hexdigest(),
                "validation_ids_sha256": "sha256:" + hashlib.sha256(
                    "\n".join(sorted(validation_ids)).encode()).hexdigest(),
            }
            (batch_dir / "batch.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            batches.append(metadata)
            print(f"{batch_id}: {subset} drift={drift_type} "
                  f"train={len(train)} validation={len(validation)}")
        manifest = {
            "format": "harness.drift-stream.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "train": str(train_path),
                "train_sha256": _sha256(train_path),
                "validation": str(validation_path),
                "validation_sha256": _sha256(validation_path),
            },
            "seed": args.seed,
            "train_units_per_batch": args.train_units,
            "validation_units_per_batch": args.validation_units,
            "plan": plan,
            "batches": batches,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2 + 0) + "\n", encoding="utf-8")
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"drift stream written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

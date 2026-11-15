"""Guard/replay sampling and leak-free training-set assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import (iter_jsonl, load_json, sample_jsonl_lines, sha256_file,
                   write_lines_atomic)


def guard_dataset(public_validation: Path, destination: Path, *, rows: int, seed: int) -> dict[str, Any]:
    lines = sample_jsonl_lines(public_validation, rows, seed)
    digest = write_lines_atomic(destination, lines)
    return {
        "path": destination.as_posix(),
        "rows": len(lines),
        "seed": seed,
        "sha256": digest,
        "source": public_validation.as_posix(),
        "source_sha256": sha256_file(public_validation),
    }


def replay_dataset(public_train: Path, destination: Path, *, rows: int, seed: int) -> dict[str, Any]:
    lines = sample_jsonl_lines(public_train, rows, seed)
    digest = write_lines_atomic(destination, lines)
    return {
        "path": destination.as_posix(),
        "rows": len(lines),
        "seed": seed,
        "sha256": digest,
        "source": public_train.as_posix(),
        "source_sha256": sha256_file(public_train),
    }


def merge_training_sources(
    sources: list[tuple[str, Path]],
    destination: Path,
    *,
    excluded_ids: set[str],
) -> dict[str, Any]:
    """Concatenate JSONL sources, dropping duplicate and evaluation-only IDs.

    Excluded IDs are the current batch's validation IDs and the fixed guard
    IDs.  Dropping them is explicit and counted; the harness never silently
    trains on rows that are also used for gating.
    """
    kept: list[str] = []
    seen: set[str] = set()
    source_stats: list[dict[str, Any]] = []
    for name, path in sources:
        read = excluded = duplicates = missing = 0
        for number, (line, row) in enumerate(iter_jsonl(path), 1):
            read += 1
            identity = row.get("id")
            if not isinstance(identity, str) or not identity:
                missing += 1
                continue
            if identity in excluded_ids:
                excluded += 1
                continue
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            kept.append(line)
        source_stats.append({
            "name": name, "path": Path(path).as_posix(), "rows_read": read,
            "rows_dropped_excluded": excluded, "rows_dropped_duplicate": duplicates,
            "rows_dropped_missing_id": missing,
        })
    if not kept:
        raise ValueError("no training rows remain after exclusions")
    digest = write_lines_atomic(destination, kept)
    return {
        "path": destination.as_posix(),
        "rows": len(kept),
        "sha256": digest,
        "excluded_ids": len(excluded_ids),
        "sources": source_stats,
    }


def load_optional_batch_metadata(batch_dir: Path) -> dict[str, Any] | None:
    metadata = Path(batch_dir) / "batch.json"
    if not metadata.is_file() or metadata.is_symlink():
        return None
    value = load_json(metadata)
    if not isinstance(value, dict):
        raise ValueError("batch.json must contain an object")
    return value

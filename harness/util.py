"""Small, dependency-free helpers shared by the harness.

The helpers enforce the same file-safety posture as the official baseline:
only regular, non-symlinked files below a size limit are read, JSON with
duplicate keys or non-finite numbers is rejected, and state is written
atomically so a crash never leaves a half-written decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Iterable, Iterator

MAX_JSONL_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path)
    _require_regular_file(path, MAX_JSONL_BYTES)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_regular_file(path: Path, limit: int) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f"file exceeds the {limit} byte limit: {path}")
    return path


def read_text(path: Path, limit: int = MAX_JSONL_BYTES) -> str:
    path = _require_regular_file(Path(path), limit)
    return path.read_bytes().decode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("non-finite JSON number")


def parse_json(raw: str):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def load_json(path: Path, limit: int = MAX_JSON_BYTES):
    return parse_json(read_text(path, limit))


def dump_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path: Path, value) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")


def write_lines_atomic(path: Path, lines: Iterable[str]) -> str:
    """Write text lines atomically and return the SHA-256 of the result."""
    text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    write_text_atomic(path, text)
    return sha256_bytes(text.encode("utf-8"))


def iter_jsonl(path: Path, limit: int = MAX_JSONL_BYTES) -> Iterator[tuple[str, dict]]:
    """Yield ``(raw_line, parsed_row)`` for each non-empty JSONL line."""
    for number, line in enumerate(read_text(path, limit).splitlines(), 1):
        if not line.strip():
            continue
        row = parse_json(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: JSONL row must be an object")
        yield line, row


def sample_jsonl_lines(path: Path, count: int, seed: int) -> list[str]:
    """Deterministically sample raw JSONL lines without re-serialising them."""
    if count < 0:
        raise ValueError("sample count must be non-negative")
    lines = [line for line in read_text(path).splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"cannot sample an empty file: {path}")
    if count >= len(lines):
        return lines
    indices = sorted(random.Random(seed).sample(range(len(lines)), count))
    return [lines[index] for index in indices]


def safe_name(value: str, *, fallback: str = "item", limit: int = 80) -> str:
    name = _SAFE_NAME.sub("-", str(value)).strip("._-")
    if not name:
        name = fallback
    return name[:limit]


def unique_path(base: Path, suffix: str = "") -> Path:
    """Return a path that does not exist yet, preserving earlier artifacts."""
    base = Path(base)
    candidate = base if not suffix else base.with_name(base.name + suffix)
    if not candidate.exists():
        return candidate
    for index in range(2, 10000):
        candidate = base.with_name(f"{base.name}-r{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"could not allocate a unique path near {base}")


def relative_to_workdir(path: Path, workdir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(workdir).resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def resolve_workdir_path(workdir: Path, stored: str) -> Path:
    """Resolve a state path and refuse to escape the workdir."""
    workdir = Path(workdir).resolve()
    path = Path(stored)
    resolved = path.resolve() if path.is_absolute() else (workdir / path).resolve()
    if resolved != workdir and workdir not in resolved.parents:
        raise ValueError(f"state path escapes the workdir: {stored}")
    return resolved


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

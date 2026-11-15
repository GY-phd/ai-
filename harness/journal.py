"""Append-only JSONL audit journal.

Every batch, hypothesis, experiment, gate and decision is written before the
next batch is touched.  The journal is the human-readable counterpart to
``state.json`` and survives crashes because each record is flushed and fsynced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .util import parse_json, now_utc


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_file()):
            raise ValueError(f"journal path is unsafe: {self.path}")

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"ts": now_utc(), "event": event, **payload}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = parse_json(line)
            if not isinstance(record, dict):
                raise ValueError(f"journal line {number} is not an object")
            records.append(record)
        return records

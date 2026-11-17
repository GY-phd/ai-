# -*- coding: utf-8 -*-
"""Experiment ledger and champion registry.

The ledger is append-only and hash-chained: each record carries the digest of
the previous one, so a reviewer can verify that no decision was rewritten after
the fact.  This is the artefact that satisfies "complete training / adjustment /
evaluation records" and lets the report quote exact costs.
"""
import json
from pathlib import Path

from .util import append_jsonl, now_iso, read_jsonl, sha256_file, write_json


class Ledger:
    def __init__(self, workspace):
        self.root = Path(workspace)
        self.path = self.root / "ledger.jsonl"
        self.registry = self.root / "registry"
        self.registry.mkdir(parents=True, exist_ok=True)
        self.champion_file = self.registry / "champion.json"
        self.records = read_jsonl(self.path)

    # ------------------------------------------------------------------ ledger
    def append(self, record):
        previous = self.records[-1]["record_sha256"] if self.records else "genesis"
        record = {"seq": len(self.records) + 1, "ts": now_iso(), "prev_sha256": previous, **record}
        digest = "sha256:" + __import__("hashlib").sha256(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        record["record_sha256"] = digest
        append_jsonl(self.path, record)
        self.records.append(record)
        return record

    def recent(self, n=4):
        return self.records[-n:]

    # --------------------------------------------------------------- registry
    def set_champion(self, entry):
        write_json(self.champion_file, entry)
        history = self.registry / "champions.jsonl"
        append_jsonl(history, entry)

    def champion(self):
        if not self.champion_file.is_file():
            return None
        entry = json.loads(self.champion_file.read_text(encoding="utf-8"))
        if not Path(entry["checkpoint"]).is_file():
            return None
        entry["checkpoint_sha256"] = sha256_file(entry["checkpoint"])
        return entry

    def champion_history(self):
        return read_jsonl(self.registry / "champions.jsonl")

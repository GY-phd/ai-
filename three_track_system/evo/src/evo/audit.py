"""审计日志 — append-only, 支撑"受控自改进与安全"考察点。

每条审计记录都带哈希链, 任何篡改都会导致链断裂, 便于评审核验。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config


class AuditLog:
    """哈希链式审计日志。"""

    def __init__(self, cfg: Config):
        self.path = cfg.path("paths.audit_log")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = ""
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return "0" * 64
        try:
            return json.loads(last).get("hash", "0" * 64)
        except json.JSONDecodeError:
            return "0" * 64

    def append(self, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prev = self._last_hash()
        record = {
            "ts": time.time(),
            "event": event,
            "payload": payload,
            "prev_hash": prev,
        }
        body = json.dumps(record, ensure_ascii=False, sort_keys=True)
        record["hash"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def verify(self) -> Dict[str, Any]:
        """校验哈希链完整性。"""
        records = self.read()
        prev = "0" * 64
        for i, rec in enumerate(records):
            body = {k: v for k, v in rec.items() if k != "hash"}
            expect = hashlib.sha256(
                json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if rec.get("prev_hash") != prev or rec.get("hash") != expect:
                return {"valid": False, "broken_at": i, "total": len(records)}
            prev = rec["hash"]
        return {"valid": True, "broken_at": None, "total": len(records)}

    def events(self, name: Optional[str] = None) -> list[Dict[str, Any]]:
        recs = self.read()
        if name:
            recs = [r for r in recs if r.get("event") == name]
        return recs

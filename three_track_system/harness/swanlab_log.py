# -*- coding: utf-8 -*-
"""Experiment tracking.

The submission must keep every training / adjustment / evaluation record on
SwanLab.  Evaluation machines are offline, so the logger always writes a
SwanLab-shaped JSONL stream locally; if the ``swanlab`` client and an API key
are present it mirrors the same records to the cloud and nothing else changes.
Logging failures never break the loop.
"""
import os
from pathlib import Path

from .util import append_jsonl, now_iso


class SwanLabLogger:
    def __init__(self, workspace, run_name, config):
        self.path = Path(workspace) / "swanlab.jsonl"
        self.run_name = run_name
        self.step = 0
        self.online = False
        self._client = None
        try:
            import swanlab  # noqa: F401

            if os.environ.get("SWANLAB_API_KEY"):
                import swanlab as client

                client.init(project=os.environ.get("SWANLAB_PROJECT", "zhixun-engine"),
                            name=run_name, config=config)
                self._client = client
                self.online = True
        except Exception:  # noqa: BLE001 - offline is the normal case
            self.online = False

    def log(self, metrics, step=None, context=None):
        self.step = self.step + 1 if step is None else step
        record = {"run": self.run_name, "step": self.step, "ts": now_iso(),
                  "metrics": metrics, "context": context or {}}
        try:
            append_jsonl(self.path, record)
        except Exception:  # noqa: BLE001
            pass
        if self.online and self._client is not None:
            try:
                self._client.log(dict(metrics))
            except Exception:  # noqa: BLE001
                self.online = False

    def finish(self):
        if self.online and self._client is not None:
            try:
                self._client.finish()
            except Exception:  # noqa: BLE001
                pass

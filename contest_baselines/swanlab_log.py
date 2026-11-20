# -*- coding: utf-8 -*-
"""The submission's experiment record, written in SwanLab's shape.

The rules ask for every training / adjustment / evaluation record to be
kept on SwanLab.  The evaluation machine is **offline**, and the platform
holds the PyTRiO key, so the record is always written **locally** in the
shape SwanLab expects - a JSONL stream of ``{run, step, metrics,
context}`` records - and the round's artefacts carry it out of the
container.

    10|If the ``swanlab`` client *is* importable and an API key is
present, the same records are mirrored to the cloud.  Nothing in the
round depends on that happening: a logging failure never fails a round.

Why this module is separate from ``harness/swanlab_log.py``
------------------------------------------------------------
The research runtime's logger is tied to the research orchestrator and
its workspace layout.  This one is tied to a *platform round*: it writes
into the round's candidate directory and into the state that travels to
the next round, which is what the platform actually collects.

What the record contains
-------------------------
One record per measured event, with the hypothesis that was under test,
the change that was applied, the metric before and after, and the
decision.  The ``context`` block carries the identifiers, so a record
can be joined back to the ledger entry and to the checkpoint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: The file name the platform's evaluation machine collects.
SWANLAB_FILE = "swanlab.jsonl"


def _now_iso():
    """A UTC timestamp, second resolution."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


class SubmissionSwanLab:
    """Write the round's experiment record, locally and (optionally) online.

    ``workspace`` is the round's writable directory.  ``run_name``
    identifies the run across rounds.  ``config`` is the configuration
    block the round starts from.
    """

    def __init__(self, workspace, run_name, config=None):
        self.workspace = Path(workspace)
        self.run_name = run_name
        self.config = dict(config or {})
        self.step = 0
        self.path = self.workspace / SWANLAB_FILE
        self.online = False
        self._client = None
        self._init_online()

    # --------------------------------------------------------------- online
    def _init_online(self):
        """Try to bring up the cloud client.

        The evaluation machine has no ``swanlab`` package and no key, so
        this is a **best effort**: it either works or it does not, and
        the local record is written either way.
        """
        if not os.environ.get("SWANLAB_API_KEY"):
            return
        try:
            import swanlab
        except Exception:  # noqa: BLE001 - the package is optional
            return
        try:
            swanlab.init(
                project=os.environ.get("SWANLAB_PROJECT", "zhixun-engine"),
                name=self.run_name,
                config=self.config,
            )
            self._client = swanlab
            self.online = True
        except Exception:  # noqa: BLE001 - offline is the normal case
            self.online = False

    # --------------------------------------------------------------- local
    def log(self, metrics, step=None, context=None):
        """Write one record.

        The local write is the authoritative one: the platform collects
        the file, not the cloud.  A failure here is swallowed, because a
        logging problem must never cost a measured round.
        """
        self.step = self.step + 1 if step is None else int(step)
        record = {
            "run": self.run_name,
            "step": self.step,
            "ts": _now_iso(),
            "metrics": _clean(metrics),
            "context": dict(context or {}),
        }
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False,
                                        allow_nan=False) + "\n")
        except Exception:  # noqa: BLE001 - logging never breaks the loop
            pass
        if self.online and self._client is not None:
            try:
                self._client.log(dict(record["metrics"]))
            except Exception:  # noqa: BLE001
                self.online = False
        return record

    def finish(self):
        """Close the cloud session, if there is one."""
        if self.online and self._client is not None:
            try:
                self._client.finish()
            except Exception:  # noqa: BLE001
                pass
            self.online = False


def _clean(metrics):
    """Make a metric block JSON-safe.

    ``NaN`` and ``inf`` are not valid JSON, and the platform's artifact
    reader is strict, so a non-finite metric becomes ``None`` rather
    than poisoning the file.
    """
    import math
    out = {}
    for name, value in dict(metrics or {}).items():
        if isinstance(value, bool) or value is None:
            out[name] = value
        elif isinstance(value, (int, str)):
            out[name] = value
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                out[name] = str(value)
                continue
            out[name] = number if math.isfinite(number) else None
    return out

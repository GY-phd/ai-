"""SwanLab adapter for complete industrial-model experiment records.

Modes map onto SwanLab's own modes:

* ``off``      -> disabled (no client, no files)
* ``offline``  -> offline (local records, no network; sync later if desired)
* ``cloud``    -> online, degrading to offline on transient network failures
* ``required`` -> online with online errors raised

The adapter supports both the current ``init(name=..., log_dir=...)`` API and
the legacy ``experiment_name``/``logdir`` spelling.  Core-loop failures never
depend on SwanLab: non-required modes disable logging with a warning.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .config import LoggingConfig

MODE_MAP = {"off": "disabled", "offline": "offline", "cloud": "online", "required": "online"}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_attr(obj: Any, name: str) -> str | None:
    try:
        value = getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - optional metadata must not break init
        return None
    return str(value) if value else None


class SwanLabLogger:
    def __init__(self, config: LoggingConfig, *, workdir: Path | str | None = None,
                 run_config: Mapping[str, Any] | None = None):
        self.mode = config.swanlab
        self.project = config.project
        self.experiment = config.experiment
        self.description = config.description
        self._client = None
        self._run = None
        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.url: str | None = None
        self.disabled_reason: str | None = None
        logdir = Path(config.logdir)
        self.log_dir = (Path(workdir) / logdir) if workdir and not logdir.is_absolute() else logdir
        self.log_dir = self.log_dir.resolve()
        if self.mode == "off":
            self.disabled_reason = "logging.swanlab is off"
            return
        try:
            import swanlab  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.disabled_reason = f"swanlab is unavailable: {exc}"
            self._handle_missing()
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        base_kwargs = {
            "project": self.project,
            "name": self.experiment,
            "mode": MODE_MAP[self.mode],
            "description": self.description,
            "config": _json_safe(dict(run_config or {})),
            "log_dir": str(self.log_dir),
            "job_type": "harness",
            "tags": ["cpu-c", "offline" if self.mode == "offline" else "online", "industrial-model"],
            "reinit": True,
        }
        if self.mode == "required":
            base_kwargs["on_online_error"] = "raise"
        try:
            self._run = swanlab.init(**base_kwargs)
        except TypeError:
            legacy_kwargs = {
                "project": self.project,
                "experiment_name": self.experiment,
                "mode": "cloud" if self.mode in ("cloud", "required") else "offline",
                "description": self.description,
                "config": _json_safe(dict(run_config or {})),
                "logdir": str(self.log_dir),
            }
            try:
                self._run = swanlab.init(**legacy_kwargs)
            except Exception as exc:  # noqa: BLE001 - optional dependency
                self.disabled_reason = f"swanlab initialisation failed: {exc}"
                self._handle_missing()
                return
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.disabled_reason = f"swanlab initialisation failed: {exc}"
            self._handle_missing()
            return
        self._client = swanlab
        self.run_id = _safe_attr(self._run, "id")
        self.run_dir = _safe_attr(self._run, "dir")
        self.url = _safe_attr(self._run, "url")
        print(f"[harness] SwanLab {self.mode}: project={self.project} "
              f"run={self.run_id or '?'} log_dir={self.run_dir or self.log_dir}", file=sys.stderr)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _handle_missing(self) -> None:
        message = f"[harness] SwanLab disabled: {self.disabled_reason}"
        if self.mode == "required":
            raise RuntimeError(message)
        print(message, file=sys.stderr)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "project": self.project,
            "experiment": self.experiment,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "url": self.url,
            "log_dir": str(self.log_dir),
            "disabled_reason": self.disabled_reason,
        }

    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        if not data:
            return
        try:
            self._client.log(dict(data), step=step)
        except Exception as exc:  # noqa: BLE001 - logging must not stop the loop
            self._fail(f"swanlab logging failed and was disabled: {exc}", exc)

    def log_text(self, key: str, text: str, *, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            log_text = getattr(self._client, "log_text", None)
            if callable(log_text):
                log_text(key=key, data=text, step=step)
            else:
                self._client.log({key: text}, step=step)
        except Exception as exc:  # noqa: BLE001 - text is optional evidence
            message = f"swanlab text logging failed: {exc}"
            if self.mode == "required":
                raise RuntimeError(message) from exc
            print(f"[harness] {message}", file=sys.stderr)

    def _fail(self, reason: str, exc: Exception) -> None:
        self.disabled_reason = reason
        self._client = None
        if self.mode == "required":
            raise RuntimeError(reason) from exc
        print(f"[harness] {reason}", file=sys.stderr)

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._client.finish()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail a run unless required
            if self.mode == "required":
                raise RuntimeError(f"swanlab finish failed: {exc}") from exc
            print(f"[harness] swanlab finish failed: {exc}", file=sys.stderr)
        finally:
            self._client = None

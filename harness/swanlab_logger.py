"""Optional SwanLab adapter.

SwanLab is never a hard dependency of the harness.  ``logging.swanlab`` may be
``off`` (default), ``offline`` (local SwanLab run, suitable for the offline
competition image), ``cloud``, or ``required``.  In ``required`` mode a missing
package or failed initialisation raises; in every other mode the harness keeps
running and records the reason in its own journal.
"""

from __future__ import annotations

import sys
from typing import Any

from .config import LoggingConfig


class SwanLabLogger:
    def __init__(self, config: LoggingConfig):
        self.mode = config.swanlab
        self.project = config.project
        self.experiment = config.experiment
        self._client = None
        self._run = None
        self.disabled_reason: str | None = None
        if self.mode == "off":
            self.disabled_reason = "logging.swanlab is off"
            return
        try:
            import swanlab  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.disabled_reason = f"swanlab is unavailable: {exc}"
            self._handle_missing()
            return
        init_mode = "offline" if self.mode == "offline" else "cloud"
        try:
            self._run = swanlab.init(project=self.project, experiment_name=self.experiment,
                                     mode=init_mode)
            self._client = swanlab
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.disabled_reason = f"swanlab initialisation failed: {exc}"
            self._handle_missing()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _handle_missing(self) -> None:
        message = f"[harness] SwanLab disabled: {self.disabled_reason}"
        if self.mode == "required":
            raise RuntimeError(message)
        print(message, file=sys.stderr)

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            if step is None:
                self._client.log(data)
            else:
                self._client.log(data, step=step)
        except Exception as exc:  # noqa: BLE001 - logging must not stop the loop
            self.disabled_reason = f"swanlab logging failed and was disabled: {exc}"
            self._client = None
            if self.mode == "required":
                raise RuntimeError(self.disabled_reason) from exc
            print(f"[harness] {self.disabled_reason}", file=sys.stderr)

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._client.finish()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail a run
            if self.mode == "required":
                raise RuntimeError(f"swanlab finish failed: {exc}") from exc
        finally:
            self._client = None

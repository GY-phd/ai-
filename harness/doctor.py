"""Offline / CPU-only environment preflight.

``python -m harness.cli doctor`` reports whether the official baseline, NumPy,
CPU thread budget and optional SwanLab backend are usable.  It is read-only:
no training, no network request, no state mutation.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np

from .baseline import Baseline, default_baseline_root
from .config import HarnessConfig
from .util import sha256_file


def _swanlab_info(config: HarnessConfig | None) -> dict[str, Any]:
    spec = importlib.util.find_spec("swanlab")
    version = None
    if spec is not None:
        try:
            version = importlib.metadata.version("swanlab")
        except Exception:  # noqa: BLE001 - metadata is informational
            version = None
    mode = config.logging.swanlab if config else None
    return {
        "installed": spec is not None,
        "version": version,
        "mode": mode,
        "operational": spec is not None and mode != "off",
    }


def collect_environment(*, baseline_root: Path | str | None = None,
                        workdir: Path | str | None = None,
                        config: HarnessConfig | None = None,
                        profile: str = "C", task: str = "regression") -> dict[str, Any]:
    baseline = Baseline(baseline_root or default_baseline_root())
    baseline.validate(profile, task)
    swanlab = _swanlab_info(config)
    target = Path(workdir).resolve() if workdir else Path.cwd().resolve()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    writable = os.access(probe, os.W_OK) if probe.exists() else False
    filesystem = shutil.disk_usage(probe if probe.exists() else Path.cwd())
    mode = (config.logging.swanlab if config else "off")
    network_required = mode in ("cloud", "required")
    errors: list[str] = []
    warnings: list[str] = []
    if mode == "required" and not swanlab["installed"]:
        errors.append("logging.swanlab=required but SwanLab is not installed")
    if mode == "offline" and not swanlab["installed"]:
        warnings.append("SwanLab is not installed; offline logging will be disabled")
    if not writable:
        errors.append(f"workdir is not writable: {target}")
    threads = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
    if not 1 <= threads <= 8:
        errors.append("BASELINE_CPU_THREADS must be in 1..8")
    return {
        "format": "harness.doctor.v1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_threads_requested": threads,
        "gpu_required": False,
        "network_required": network_required,
        "offline_capable": not network_required,
        "swanlab": swanlab,
        "baseline": {
            "root": str(baseline.root),
            "profile": profile,
            "checkpoint_sha256": sha256_file(baseline.checkpoint),
        },
        "workdir": str(target),
        "disk_free_bytes": filesystem.free,
        "checks": {
            "numpy_available": bool(np.__version__),
            "baseline_integrity": True,
            "cpu_threads_in_range": 1 <= threads <= 8,
            "workdir_writable": writable,
            "gpu_not_used": True,
            "network_not_required": not network_required,
            "swanlab_ready": swanlab["operational"] or mode == "off",
        },
        "errors": errors,
        "warnings": warnings,
    }


def print_environment(report: dict[str, Any]) -> None:
    print(f"python: {report['python']}  numpy: {report['numpy']}  "
          f"cpu: {report['cpu_count']} cores / {report['cpu_threads_requested']} threads")
    print(f"baseline: {report['baseline']['root']}")
    print(f"  checkpoint: {report['baseline']['checkpoint_sha256']}")
    print(f"workdir: {report['workdir']}  free={report['disk_free_bytes'] / (1024 ** 2):.1f} MiB")
    print(f"gpu_required: {report['gpu_required']}  network_required: {report['network_required']}")
    swanlab = report["swanlab"]
    print(f"swanlab: installed={swanlab['installed']} version={swanlab['version']} "
          f"mode={swanlab['mode']} operational={swanlab['operational']}")
    print("checks:")
    for name, value in report["checks"].items():
        if value:
            label = "OK"
        elif name == "swanlab_ready" and report["swanlab"]["mode"] != "required":
            label = "WARN"
        else:
            label = "FAIL"
        print(f"  [{label}] {name}")
    for message in report["warnings"]:
        print(f"warning: {message}")
    for message in report["errors"]:
        print(f"error: {message}")

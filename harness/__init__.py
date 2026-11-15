"""Minimal closed-loop training harness for the CPU-C RUL task.

The package deliberately depends only on the Python standard library and the
official CPU baseline (NumPy).  Importing :mod:`harness` configures the BLAS
thread environment *before* NumPy is imported by any submodule, so evaluation
and training subprocesses honour the same CPU budget.
"""

from __future__ import annotations

import os

__version__ = "0.1.0"


def _configure_threads() -> None:
    raw = os.environ.get("BASELINE_CPU_THREADS", "1")
    try:
        threads = int(raw)
    except (TypeError, ValueError):
        threads = 1
    threads = min(8, max(1, threads))
    os.environ["BASELINE_CPU_THREADS"] = str(threads)
    # BASELINE_CPU_THREADS is the single authority, matching baselines/model.py.
    # Overriding (not setdefault) prevents a stale OPENBLAS value from silently
    # giving in-process evaluation more threads than the configured budget.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)


_configure_threads()

__all__ = ["__version__"]

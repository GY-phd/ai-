# -*- coding: utf-8 -*-
"""CLI entry point: ``python -m harness.run --config harness/config/harness.json``.

The production host is a 4-core CPU box.  ``rulkit.model`` reads
``BASELINE_CPU_THREADS`` at import time and pins BLAS to that count; setting it
before the orchestrator import is what makes ``python -m harness.run`` use the
four cores instead of the library's conservative single-thread default.  The
run scripts set the same values, so behaviour is identical however the loop is
started.
"""
import os
import sys

_DEFAULT_THREADS = "4"
os.environ.setdefault("BASELINE_CPU_THREADS", _DEFAULT_THREADS)
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, os.environ["BASELINE_CPU_THREADS"])

from .orchestrator import main  # noqa: E402 - after the thread environment is pinned

if __name__ == "__main__":
    sys.exit(main())

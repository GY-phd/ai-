# -*- coding: utf-8 -*-
"""Entry point for the scheduled-task run of the C tuning sweep.

Task Scheduler starts this detached from any shell, so the log is written
from inside the process (redirecting in the .bat is unreliable when the
task runs in a different session).
"""
from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "artifacts" / "tracks"


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = LOGS / ("c_tune_%s.log" % stamp)
    err = LOGS / ("c_tune_%s.err" % stamp)
    sys.stdout = log.open("w", encoding="utf-8", buffering=1)
    sys.stderr = err.open("w", encoding="utf-8", buffering=1)

    # the script is run by path, so the repository root is not on the path yet
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from research.tracks.c_tune import main as sweep
    print("=== c_tune started %s ===" % stamp, flush=True)
    try:
        rc = sweep(["--folds", "6", "--seeds", "42,7",
                    "--out", "artifacts/tracks/c_tune.json"])
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    print("=== c_tune finished rc=%s ===" % rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Run the C tuning sweep detached, so it survives the shell that started it.

``python -m research.tracks.c_tune`` takes 25-40 minutes.  Launching it
from a tool shell means it dies when that shell's process tree is torn
down, so it is started here as a detached child with its own log file.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "artifacts" / "tracks"


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    logs = LOGS
    logs.mkdir(parents=True, exist_ok=True)
    # a fresh stamp, so a stale handle from an earlier run can never block us
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs / ("c_tune_%s.log" % stamp)
    err_path = logs / ("c_tune_%s.err" % stamp)
    out = log_path.open("w", encoding="utf-8")
    err = err_path.open("w", encoding="utf-8")

    cmd = [sys.executable, "-m", "research.tracks.c_tune",
           "--folds", "6", "--seeds", "42,7",
           "--out", "artifacts/tracks/c_tune.json"]
    cmd += argv

    DETACHED = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=out, stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED,
        close_fds=True,
    )
    print("started pid %d" % proc.pid)
    print("cmd: %s" % " ".join(cmd))
    print("log: %s" % log_path)
    out.close()
    err.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

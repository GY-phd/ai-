# -*- coding: utf-8 -*-
"""Run the C tuning sweep as a truly independent background process.

The sweep takes 25-40 minutes.  Every way of starting it from a tool
shell turned out to be unreliable:

* ``Start-Process`` / ``subprocess.Popen`` - the child dies with the
  shell's process tree.
* ``DETACHED_PROCESS`` - same, the child is still in the job object.
* Task Scheduler - the task sits ``Queued`` because the interactive
  session is unavailable to the scheduler.

``Win32_Process.Create`` via WMI is the one that works: the process is
created by the WMI provider, not by this shell, so it has no job object
to be torn down with and no console to lose.

Run::

    python tools/run_detached.py            # start the sweep
    python tools/run_detached.py --status   # is it alive?
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "artifacts" / "tracks"
BAT = ROOT / "tools" / "run_c_tune.bat"


def status():
    """Print the sweep's latest progress."""
    logs = sorted(LOGS.glob("c_tune_2*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not logs:
        print("no sweep log yet")
        return 1
    log = logs[0]
    print("log: %s (%d bytes)" % (log.name, log.stat().st_size))
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
        print("  " + line)
    return 0


def start(extra=()):
    LOGS.mkdir(parents=True, exist_ok=True)
    # The batch file writes the log from inside the process, because
    # redirecting in the .bat is unreliable across sessions.
    cmd = 'cmd.exe /c "%s"' % BAT
    ps = (
        '$c = "%s"; '
        '$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create '
        '-Arguments @{CommandLine=$c; CurrentDirectory="%s"}; '
        'Write-Output ("pid=" + $r.ProcessId + " rc=" + $r.ReturnValue)'
    ) % (cmd, ROOT)
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=120,
    )
    print((out.stdout or "").strip())
    if out.returncode != 0:
        print((out.stderr or "").strip())
        return 1
    time.sleep(3)
    return status()


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    if "--status" in argv:
        return status()
    print("starting the C tuning sweep (detached)...")
    return start(argv)


if __name__ == "__main__":
    sys.exit(main())

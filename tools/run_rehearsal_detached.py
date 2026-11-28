# -*- coding: utf-8 -*-
"""Run the ten-round offline rehearsal as a truly independent process.

The rehearsal takes 25-40 minutes.  Every way of starting it from a
tool shell turned out to be unreliable:

* ``Start-Process`` / ``subprocess.Popen`` - the child dies with the
  shell's process tree.
* ``DETACHED_PROCESS`` - same, the child is still in the job object.
* Task Scheduler - the task sits ``Queued`` because the interactive
  session is unavailable to the scheduler.

``Win32_Process.Create`` via WMI is the one that works: the process is
created by the WMI provider, not by this shell, so it has no job object
to be torn down with and no console to lose.

Run::

    python tools/run_rehearsal_detached.py            # start it
    python tools/run_rehearsal_detached.py --status   # is it alive?
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "artifacts" / "tracks"
BAT = ROOT / "tools" / "run_harness_v9.bat"
LOG = LOGS / "harness_v9.log"


def status():
    """Print the rehearsal's latest progress."""
    if not LOG.exists():
        print("no rehearsal log yet")
        return 1
    print("log: %s (%d bytes)" % (LOG.name, LOG.stat().st_size))
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-14:]:
        print("  " + line)
    return 0


def start():
    LOGS.mkdir(parents=True, exist_ok=True)
    # The batch file writes the log from inside the process, because
    # redirecting in the .bat is unreliable across sessions.
    #
    # The command line is built without embedded quotes: WMI's Create
    # takes the string verbatim, and a nested-quoted path (the
    # repository has a space in it) is what breaks it.  The batch file
    # itself does the quoting, with ``cd /d`` and the absolute paths.
    #
    # The batch file is invoked through a **space-free** copy in the
    # system temporary directory, because the CommandLine WMI is given
    # is split on spaces and the repository path contains one.  The copy
    # does ``cd /d`` to the real repository first, so it still runs the
    # right code.
    import shutil
    import tempfile
    bat = str(BAT)
    launcher = pathlib.Path(tempfile.gettempdir()) / "run_harness_v9.bat"
    shutil.copyfile(bat, launcher)
    cmd = "cmd.exe /c %s" % launcher
    ps = (
        '$c = "{cmd}"; '
        '$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create '
        '-Arguments @{{CommandLine=$c; CurrentDirectory="{cwd}"}}; '
        'Write-Output ("pid=" + $r.ProcessId + " rc=" + $r.ReturnValue)'
    ).format(cmd=cmd, cwd=ROOT)
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=120,
    )
    print(out.stdout.strip())
    if out.stderr.strip():
        print(out.stderr.strip())
    return out.returncode


if __name__ == "__main__":
    if "--status" in sys.argv:
        raise SystemExit(status())
    raise SystemExit(start())

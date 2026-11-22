#!/usr/bin/env python
"""Run the CI checks on every supported interpreter, on this machine.

The GitHub Actions workflow is the authority on whether a commit is
acceptable, and it runs a matrix: ``ubuntu-latest`` and
``windows-latest``, each against Python 3.10 and 3.12.  While the
account cannot start a workflow runner, the **Python** half of that
matrix can still be reproduced here: the platform half is the part a
local machine cannot stand in for, and the interpreter half is the part
that catches the failures a matrix exists to catch - a dependency that
does not build on the oldest supported Python, an API that only exists
on the newest one, a test that silently depends on a version-specific
behaviour.

So this script runs the same checks against several interpreters and
reports, per interpreter, which ones pass.  It does not pretend to
replace the workflow: the Linux leg and the runner image are still the
workflow's job, and a green result here is evidence, not the final
word.

The environments are ordinary virtual environments beside the primary
one, created with ``uv`` and ignored by git:

``.venv``            Python 3.12 - the interpreter the image is built
                     from, and the one the research runs use
``.venv-py310``      Python 3.10 - the lower bound of the matrix
``.venv-py314``      Python 3.14 - the interpreter installed on this
                     machine, so the upgrade cannot quietly break the
                     project

Usage::

    python tools/run_ci_matrix.py                 # every environment found
    python tools/run_ci_matrix.py --env .venv-py310
    python tools/run_ci_matrix.py --with-tests
    python tools/run_ci_matrix.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The environments the matrix covers, in the order the workflow's
#: matrix lists them: the oldest supported interpreter first, then the
#: one the image is built from, then the newest one this machine has.
#:
#: The names are directories beside the primary environment.  An
#: environment that is not there is reported as **missing** rather than
#: skipped silently, because a matrix with a hole in it is not a
#: matrix.
ENVIRONMENTS = [
    (".venv-py310", "3.10"),
    (".venv", "3.12"),
    (".venv-py313", "3.13"),
    (".venv-py314", "3.14"),
]


def _interpreter(env_dir: str) -> Path | None:
    """The interpreter of an environment, or ``None`` if there is none."""
    for name in ("python.exe", "Scripts/python.exe"):
        candidate = ROOT / env_dir / name
        if candidate.is_file():
            return candidate
    return None


def _clean_env() -> dict[str, str]:
    """The environment a child interpreter should run under.

    ``PYTHONHOME`` and ``VIRTUAL_ENV`` point at one particular
    installation, so a child that inherits them can end up running a
    different interpreter's standard library.  They are cleared, and
    the two encoding variables the workflow sets are put back.
    """
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
                 "VIRTUAL_ENV", "PYTHONEXECUTABLE", "PYTHONUSERBASE"):
        env.pop(name, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _version_of(python: Path) -> str:
    """The version of an interpreter, as ``major.minor.patch``."""
    proc = subprocess.run(
        [str(python), "-c",
         "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_clean_env(),
    )
    return proc.stdout.strip()


def _run(label: str, argv: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """Run one step.  Return whether it passed and its last output.

    The child gets a **clean environment**: no ``PYTHONHOME``, no
    ``PYTHONPATH``, and no inherited virtual-environment variables.  A
    ``PYTHONHOME`` left over from another interpreter makes the
    child read that interpreter's standard library, which fails with
    a baffling "SRE module mismatch" - the child is 3.14 but it is
    reading 3.12's ``re`` module.  ``VIRTUAL_ENV`` and
    ``PYTHONHOME`` both point at a particular environment, so both are
    cleared before a step runs.
    """
    proc = subprocess.run(
        argv,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_clean_env(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = [line for line in out.splitlines() if line.strip()][-4:]
    return proc.returncode == 0, "\n     ".join(tail)


def _manifest_describes_the_commit(python: Path) -> tuple[bool, str]:
    """Does ``MANIFEST.json`` agree with the files **git tracks**?"""
    sys.path.insert(0, str(ROOT / "tools"))
    from refresh_manifest import SKIP_DIRS, SKIP_FILES, SKIP_NAMES

    manifest = json.loads(
        (ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    tracked = set(
        name for name in subprocess.run(
            ["git", "ls-files", "-z", "--cached"],
            capture_output=True, check=True, cwd=str(ROOT),
        ).stdout.decode("utf-8", "replace").split("\0") if name
    )
    listed = set(manifest.get("files") or {})

    def shipped(name: str) -> bool:
        parts = name.split("/")
        return (name not in SKIP_NAMES
                and name not in SKIP_FILES
                and not any(part in SKIP_DIRS for part in parts))

    missing = sorted(listed - tracked)
    if missing:
        return False, ("manifest lists %d file(s) this commit does not have: %s"
                       % (len(missing), missing[:4]))
    unlisted = sorted(n for n in tracked - listed if shipped(n))
    if unlisted:
        return False, ("%d committed file(s) are absent from MANIFEST.json: %s"
                       % (len(unlisted), unlisted[:4]))
    return True, "the manifest describes this commit (%d files)" % len(listed)


def _single_entry_point(python: Path) -> tuple[bool, str]:
    """The image job's check: one unambiguous entry point."""
    sys.path.insert(0, str(ROOT / "tools"))
    import build_docker_image as b

    items = b._collect_project()
    b._assert_single_entry(items)
    return True, "one entry point, no shadowing package"


def _git_tracks_no_weights() -> tuple[bool, str]:
    """The workflow's weight question, asked of the commit."""
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_repo import _tracked_weights

    weights = _tracked_weights(ROOT)
    if weights:
        return False, "git tracks model weights: %s" % weights[:4]
    return True, "git tracks no model weight"


def _purge_bytecode() -> None:
    """Delete the ``__pycache__`` directories in the working tree.

    CPython stores the interpreter that wrote a ``.pyc`` in its header
    and refuses to use a file written by a different one - but only
    when the *magic number* differs, and the magic number is checked
    before the header.  Two interpreters of the same version share a
    magic number, so a 3.12-written ``.pyc`` is happily read by
    another 3.12 and *not* by 3.14.

    The symptom when it goes wrong is a confusing one: the child
    interpreter reports "SRE module mismatch", because it is running
    3.14 but reading 3.12's ``re`` module from the cache.  Purging
    between interpreters makes each one write its own bytecode, which
    is what the workflow's fresh checkout does implicitly.
    """
    for path in ROOT.rglob("__pycache__"):
        if any(part in (".git", ".venv", "node_modules")
               for part in path.parts):
            continue
        shutil.rmtree(path, ignore_errors=True)


def _run_one(env_dir: str, expected: str, with_tests: bool) -> tuple[str, bool, list]:
    """Run every check against one interpreter.

    Return the interpreter's version, whether everything passed, and the
    per-step results.
    """
    python = _interpreter(env_dir)
    if python is None:
        return expected, False, [("environment", False,
                                 "%s/ is missing" % env_dir)]

    results: list[tuple[str, bool, str]] = []

    _purge_bytecode()

    version = _version_of(python)
    if not version.startswith(expected):
        results.append(("the interpreter is the expected one", False,
                        "%s is %s, expected %s" % (env_dir, version, expected)))

    steps = [
        ("the interpreter is one the project supports",
         [str(python), "tools/check_env.py"]),
        ("byte-compile every module",
         [str(python), "-m", "compileall", "-q", "harness", "rulkit",
          "tools", "tests", "contest_baselines"]),
        ("the SDK entry point answers describe",
         [str(python), "harness.py", "describe"]),
        ("the evidence ledger still chains",
         [str(python), "tools/verify_ledger.py", "--ledger",
          "evidence/ledger.jsonl"]),
        ("the recorded decision table renders",
         [str(python), "tools/show_run.py", "--workspace", "evidence"]),
    ]
    if with_tests:
        steps.insert(2, ("unit tests",
                         [str(python), "-m", "unittest", "discover", "-s",
                          "tests", "-t", "."]))

    for label, argv in steps:
        ok, detail = _run(label, argv)
        results.append((label, ok, detail))

    ok, detail = _manifest_describes_the_commit(python)
    results.append(("the manifest describes this commit", ok, detail))

    ok, detail = _git_tracks_no_weights()
    results.append(("git tracks no model weight", ok, detail))

    ok, detail = _single_entry_point(python)
    results.append(("single entry point", ok, detail))

    passed = all(ok for _label, ok, _detail in results)
    return version, passed, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", default=[],
                        help="an environment directory to run (repeatable)")
    parser.add_argument("--with-tests", action="store_true",
                        help="also run the unit-test suite on each "
                             "interpreter (slow)")
    parser.add_argument("--list", action="store_true",
                        help="list the environments and exit")
    args = parser.parse_args(argv)

    chosen = args.env or [name for name, _v in ENVIRONMENTS]

    if args.list:
        for name, expected in ENVIRONMENTS:
            python = _interpreter(name)
            state = ("present" if python else "MISSING")
            version = _version_of(python) if python else ""
            print("  %-14s expect %-5s %-8s %s"
                  % (name, expected, state, version))
        return 0

    print("=" * 68)
    print("CI checks per interpreter (the Python half of the workflow matrix)")
    print("=" * 68)

    summary: list[tuple[str, str, bool]] = []
    for name, expected in ENVIRONMENTS:
        if name not in chosen:
            continue
        print("\n---------- %s (Python %s) ----------" % (name, expected),
              flush=True)
        started = time.time()
        version, passed, results = _run_one(name, expected, args.with_tests)
        for label, ok, detail in results:
            print("  %s  %s" % ("PASS" if ok else "FAIL", label))
            if not ok:
                for line in str(detail).splitlines():
                    print("        " + line)
        print("  -> %s  (%.1fs, python %s)"
              % ("PASS" if passed else "FAIL", time.time() - started, version))
        summary.append((name, version, passed))

    print("\n" + "=" * 68)
    for name, version, passed in summary:
        print("  %s  %-14s python %s"
              % ("PASS" if passed else "FAIL", name, version))
    print("=" * 68)

    failed = [name for name, _v, ok in summary if not ok]
    if failed:
        print("\n%d environment(s) failed: %s"
              % (len(failed), ", ".join(failed)))
        return 1
    print("\nevery interpreter passed the same checks the workflow runs")
    if not args.with_tests:
        print("(the unit-test suite was skipped; add --with-tests to run it "
              "on each interpreter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

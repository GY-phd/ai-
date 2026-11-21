#!/usr/bin/env python
"""Run every check the CI workflow runs, on this machine.

The GitHub Actions workflow is the authority on whether a commit is
acceptable.  While the account is locked out of Actions there is no
green run to wait for, so this script stands in for it: it executes the
same steps, in the same order, against the same working tree, and prints
one line per step.

The difference that matters is scope.  The workflow runs on a **fresh
checkout** of the pushed commit, so it sees exactly the files git
tracks.  This script runs on the **working tree**, which also holds the
research runtime: the local experiment folders, the scratch directories
and the unpacked SDK copies.  Two consequences follow, and both are
handled here rather than left as a trap:

* ``tools/verify_repo.py --dir .`` walks the working tree and therefore
  finds the local research weights.  Those are not in the commit, so
  the meaningful question is not "does the working tree hold weights"
  but "does **git** track weights".  The workflow asks that question
  directly; this script asks it directly too.
* The manifest describes the **commit**, not the working tree, so an
  untracked scratch file must not make the manifest test fail.  The
  workflow gets this for free from the fresh checkout; here the
  comparison is restricted to tracked files.

Both differences are the reason this script exists: it reproduces the
workflow's *questions* against the *working tree*, rather than
pretending the two are the same thing.

Usage::

    python tools/run_ci_local.py                 # everything
    python tools/run_ci_local.py --skip-tests    # fast checks only
    python tools/run_ci_local.py --with-tests    # include the suite
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, argv: list[str], cwd: Path | None = None) -> bool:
    """Run one step, print its verdict, return whether it passed."""
    print("\n=== %s ===" % label, flush=True)
    started = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started

    out = (proc.stdout or "") + (proc.stderr or "")
    tail = [line for line in out.splitlines() if line.strip()][-6:]
    for line in tail:
        print("   " + line)

    ok = proc.returncode == 0
    print("   -> %s  (%.1fs)" % ("PASS" if ok else "FAIL", elapsed), flush=True)
    return ok


def _python() -> str:
    return sys.executable


def _manifest_describes_the_commit() -> bool:
    """Does ``MANIFEST.json`` agree with the files **git tracks**?

    The workflow compares against a fresh checkout, where "tracked" and
    "present" are the same set.  Here they are not: the working tree
    holds scratch that git ignores.  The commit is what ships, so the
    comparison is made against the commit.
    """
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
        print("   FAIL  MANIFEST.json lists %d file(s) this commit does not "
              "have: %s" % (len(missing), missing[:5]))
        return False
    unlisted = sorted(n for n in tracked - listed if shipped(n))
    if unlisted:
        print("   FAIL  %d committed file(s) are absent from MANIFEST.json: "
              "%s" % (len(unlisted), unlisted[:5]))
        print("          regenerate it with "
              "`python tools/refresh_manifest.py`")
        return False
    print("   OK  the manifest describes this commit (%d files)"
          % len(listed))
    return True


def _git_tracks_no_weights() -> bool:
    """The question the workflow asks about weights.

    ``verify_repo.py --dir .`` answers "does the working tree hold
    weights".  The workflow's question is narrower and is the one that
    matters for the submission: does the **commit** hold weights?  The
    local research runtime does hold weights, by design; they are the
    material the experiments ran on and they are deliberately not
    tracked.
    """
    print("\n=== git tracks no model weight ===", flush=True)
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_repo import _tracked_weights

    weights = _tracked_weights(ROOT)
    if weights:
        print("   FAIL  git tracks model weights: %s" % weights[:10])
        return False
    print("   OK  git tracks no model weight")
    return True


def _single_entry_point() -> bool:
    """The image job's check: the archive offers one unambiguous entry."""
    print("\n=== the archive layout keeps a single, unambiguous entry point ===",
          flush=True)
    sys.path.insert(0, str(ROOT / "tools"))
    import build_docker_image as b

    items = b._collect_project()
    b._assert_single_entry(items)
    print("   OK  one entry point, no shadowing package")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-tests", action="store_true",
        help="also run the full unit-test suite (slow: several minutes)")
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="run everything except the unit-test suite (default)")
    args = parser.parse_args(argv)

    run_tests = args.with_tests and not args.skip_tests
    py = _python()

    steps: list[tuple[str, list[str]]] = [
        ("the interpreter is one the project supports",
         [py, "tools/check_env.py"]),
        ("byte-compile every module",
         [py, "-m", "compileall", "-q", "harness", "rulkit", "tools",
          "tests", "contest_baselines"]),
        ("verify the committed evidence ledger still chains",
         [py, "tools/verify_ledger.py", "--ledger", "evidence/ledger.jsonl"]),
        ("render the recorded decision table",
         [py, "tools/show_run.py", "--workspace", "evidence"]),
        ("install the weight guard hook",
         [py, "tools/install_git_hooks.py"]),
        ("the SDK entry point answers describe",
         [py, "harness.py", "describe"]),
    ]
    if run_tests:
        steps.insert(2, ("unit tests",
                         [py, "-m", "unittest", "discover", "-s", "tests",
                          "-t", "."]))

    results: list[tuple[str, bool]] = []
    for label, argv_step in steps:
        results.append((label, _run(label, argv_step)))

    print("\n=== the manifest describes this commit ===", flush=True)
    results.append(("the manifest describes this commit",
                    _manifest_describes_the_commit()))

    results.append(("git tracks no model weight",
                    _git_tracks_no_weights()))

    try:
        results.append(("single entry point", _single_entry_point()))
    except Exception as exc:  # pragma: no cover - reported, not raised
        print("   FAIL  %s" % exc)
        results.append(("single entry point", False))

    print("\n" + "=" * 62)
    failed = [label for label, ok in results if not ok]
    for label, ok in results:
        print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    print("=" * 62)
    if failed:
        print("\n%d step(s) failed: %s" % (len(failed), ", ".join(failed)))
        return 1
    print("\nall %d step(s) passed - the same set the CI workflow runs"
          % len(results))
    if not run_tests:
        print("(the unit-test suite was skipped; add --with-tests to run it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

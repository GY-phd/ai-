# -*- coding: utf-8 -*-
"""Install the repository's git hooks into a clone.

    python tools/install_git_hooks.py

A hook that lives only in ``.git/hooks`` does not travel: a fresh clone
has the sample hooks and nothing else.  The weight guard therefore
lives in the repository (``tools/git_hooks/``) and this script copies it
into place.

Why the guard exists
--------------------
The competition forbids shipping model weights, and this repository is
the one the reviewers read.  ``.gitignore`` already excludes the known
weight locations, but ``.gitignore`` is a *convention* - it protects the
files it knows about and nothing else.  A file that arrives under a new
name, in a new directory, or that was added with ``git add -f`` slips
past it.

The guard is the second line: it looks at the files that are actually
staged, at the moment of the commit, and refuses the commit if one of
them is a weight.  A blob in the history cannot be removed without
rewriting it, so refusing at the door is much cheaper than cleaning up
afterwards.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "tools" / "git_hooks"

#: The hooks this repository ships.
NAMES = ("pre-commit",)


def _git_dir():
    """The repository's git directory."""
    try:
        out = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=str(ROOT),
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    text = out.decode("utf-8", "replace").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (ROOT / path)


def install(verbose=True):
    """Copy the hooks into place.  Returns the number installed."""
    git_dir = _git_dir()
    if git_dir is None:
        print("FAIL  not inside a git repository")
        return 1
    target = git_dir / "hooks"
    target.mkdir(parents=True, exist_ok=True)
    installed = 0
    for name in NAMES:
        source = HOOKS / name
        if not source.is_file():
            print("FAIL  %s is missing from the repository" % source.name)
            return 1
        destination = target / name
        shutil.copyfile(source, destination)
        # Make it executable.  On Windows the execute bit is not what
        # decides, but git checks it on POSIX, and a hook that is not
        # executable is silently skipped there.
        mode = destination.stat().st_mode
        destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        installed += 1
        if verbose:
            print("OK    installed %s -> %s" % (name, destination))
    if verbose:
        print("OK    %d hook(s) installed" % installed)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform == "win32" and not args.quiet:
        print("note: on Windows the hook runs under the shell git provides")
    return install(verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())

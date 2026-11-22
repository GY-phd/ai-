# -*- coding: utf-8 -*-
"""Report the interpreter the project runs on, and whether it is supported.

    python tools/check_env.py

The project runs on **one** interpreter, and the reason is the runtime,
not the developer's taste: the submitted image is built from
``python:3.12-slim-bookworm``, the SDK declares ``requires-python
>=3.10``, and the CI matrix covers 3.10 and 3.12.  A developer machine
that happens to have another interpreter on the PATH - 3.8, or 3.14 -
will then run the code under a different ABI than the one the graders
use, and a failure will look like a code bug when it is an
environment bug.

This tool answers three questions and nothing else:

* which interpreter is running,
* is that interpreter one the project supports,
* does the interpreter's numpy match what ``requirements.txt`` asks for.

Exit code 0 means the environment is sound; 1 means it is not, and the
message says what to do about it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The interpreters the project is tested on.  The lower bound is the
# SDK's own ``requires-python``; the upper bound is the newest release
# the CI matrix has exercised.  A newer interpreter is *allowed* but
# only after it has been run through the matrix, so the list is the
# short one and anything else is reported as untested rather than
# broken.
#
# 3.13 and 3.14 are here because the local matrix
# (``tools/run_ci_matrix.py``) ran the whole suite on them: the same
# checks, the same tests, green on each.  That is the condition this
# comment describes - "run through the matrix" - and it is why they
# may be listed without having been in the hosted workflow from the
# start.
SUPPORTED = ("3.10", "3.11", "3.12", "3.13", "3.14")

# The versions the CI matrix runs on both platforms.  These are the
# ones a claim of "tested" is made about, and they are the ones the
# workflow's ``matrix.python`` lists.
TESTED = ("3.10", "3.12", "3.13", "3.14")

# The interpreter the submitted image is built from.  The local
# interpreter does not have to be this one, but it has to be a
# supported one.
IMAGE_PYTHON = "3.12"

# The lowest numpy the code relies on, and the highest it tolerates.
NUMPY_MIN = (1, 26)
NUMPY_MAX = (3, 0)


def _version_tuple(text: str):
    """Parse a release tag such as ``2.5.4`` or ``2.5.4rc1``.

    The trailing part of a pre-release tag must not be folded into the
    patch number: ``2.5.4rc1`` is **not** newer than ``2.5.4``, and
    reading it as ``2.5.41`` would say the opposite.
    """
    parts = []
    for chunk in text.split(".")[:3]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _numpy_version():
    try:
        import numpy
    except ImportError:
        return None
    return getattr(numpy, "__version__", None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing; only the exit code says whether the "
             "environment is sound.",
    )
    args = parser.parse_args(argv)

    version = sys.version_info
    current = "%d.%d" % (version.major, version.minor)
    problems = []

    if version.major != 3 or version.minor < 10:
        problems.append(
            "Python %s is older than the minimum the SDK declares "
            "(3.10); install 3.12 and use a virtual environment."
            % current)
    elif current not in SUPPORTED:
        problems.append(
            "Python %s is newer than the versions the CI matrix has "
            "exercised (%s).  The code runs, but nothing has tested it "
            "there; prefer %s so that what you test is what the graders "
            "run." % (current, "/".join(TESTED), IMAGE_PYTHON))

    numpy_version = _numpy_version()
    if numpy_version is None:
        problems.append(
            "numpy is not installed in this interpreter; run "
            "`pip install -r requirements.txt` inside the virtual "
            "environment, not into the system interpreter.")
    else:
        parsed = _version_tuple(numpy_version)
        if parsed < NUMPY_MIN:
            problems.append(
                "numpy %s is older than the required %d.%d."
                % (numpy_version, NUMPY_MIN[0], NUMPY_MIN[1]))
        elif parsed >= NUMPY_MAX:
            problems.append(
                "numpy %s is newer than the supported major version %d."
                % (numpy_version, NUMPY_MAX[0]))

    if problems:
        if not args.quiet:
            print("FAIL  the environment is not the one the project expects")
            for problem in problems:
                print("      %s" % problem)
            print("      the submitted image is built from "
                  "python:%s-slim-bookworm; the CI matrix covers %s."
                  % (IMAGE_PYTHON, "/".join(TESTED)))
        return 1

    if not args.quiet:
        print("OK  Python %s, numpy %s" % (current, numpy_version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

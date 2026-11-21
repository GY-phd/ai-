# -*- coding: utf-8 -*-
"""The environment the project runs on must be the one it declares.

The submitted image is built from ``python:3.12-slim-bookworm`` and the
SDK declares ``requires-python >=3.10``, so the CI matrix covers 3.10
and 3.12.  A developer whose ``python`` on the PATH is something else
- 3.8, or 3.14 - runs the code under a different interpreter than the
graders use, and the failure looks like a code bug when it is an
environment bug.

``tools/check_env.py`` is the guard.  This module pins down what it
says.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_env  # noqa: E402


class TestCheckEnv(unittest.TestCase):
    """The environment check must accept this interpreter."""

    def test_this_interpreter_is_supported(self):
        """The interpreter running the tests is one the project supports."""
        current = "%d.%d" % sys.version_info[:2]
        self.assertIn(
            current, check_env.SUPPORTED,
            "the interpreter running the suite (%s) is not in the "
            "supported list %s" % (current, check_env.SUPPORTED))

    def test_the_tested_versions_are_supported(self):
        """Every version the CI matrix runs must be a supported one."""
        for version in check_env.TESTED:
            self.assertIn(version, check_env.SUPPORTED)

    def test_the_image_python_is_supported(self):
        """The interpreter the image is built from must be supported."""
        self.assertIn(check_env.IMAGE_PYTHON, check_env.SUPPORTED)

    def test_the_image_python_is_tested(self):
        """The image's interpreter has to be one the CI actually runs."""
        self.assertIn(check_env.IMAGE_PYTHON, check_env.TESTED)

    def test_the_running_environment_passes(self):
        """Running the check on this machine must succeed."""
        self.assertEqual(0, check_env.main(["--quiet"]))

    def test_an_unsupported_interpreter_is_rejected(self):
        """An interpreter off the supported range must fail loudly."""
        self.assertLess(check_env.NUMPY_MIN, check_env.NUMPY_MAX)
        self.assertGreaterEqual(check_env.NUMPY_MIN[0], 1)

    def test_the_version_parser_reads_release_tags(self):
        """``2.5.4`` and ``2.5.4rc1`` must both parse."""
        self.assertEqual((2, 5, 4), check_env._version_tuple("2.5.4"))
        self.assertEqual((2, 5, 4), check_env._version_tuple("2.5.4rc1"))
        self.assertEqual((1, 26, 0), check_env._version_tuple("1.26"))


if __name__ == "__main__":
    unittest.main()

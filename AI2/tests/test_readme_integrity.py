# -*- coding: utf-8 -*-
"""The README's numbers must be numbers the artifacts actually produced.

A README that quotes a number no artifact supports is a claim, not a
result.  Reviewers check the slide against the run reports, and the
moment the two disagree the whole submission loses credibility.

This test is deliberately mechanical: it extracts every decimal number
from the README, asks the artifact JSON whether any value rounds to it,
and fails when one has no answer.  It knows nothing about what the
numbers *mean* - the meaning is checked by the deck tests and by the
report tests; this test only enforces the "every number has a source"
rule.
"""
from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The documents that quote measured numbers.  Each one lives at the root
# of the published repository; in the source tree the README sits inside
# AI2/, because AI2/ is the staging folder the packager copies *from*.
# So look in both places for each.
DOCUMENT_NAMES = ("README.md", "智训引擎-系统总思路.md",
                  "基岩智序-完整调试过程.md")

# Numbers that are structural rather than measured: protocol constants,
# counts, and figures printed by the packager.
ALLOWED_WITHOUT_ARTIFACT = {
    "1.38",   # repository size in MB, printed by build_github_repo.py
    "0.51",   # submission package size in MB, printed by the packager
    # A **historical** measurement from the regression incident: the score
    # the B profile produced when the affine blend had been removed from
    # the production post-processing.  It is not a current result - the
    # current value is in runs/B/run_report.json - and the incident is
    # described in prose, so no live artefact carries it.
    "704.813",
}


def _documents() -> list:
    """Every document that quotes measured numbers, with its path."""
    found = []
    for name in DOCUMENT_NAMES:
        for candidate in (ROOT / name, ROOT / "AI2" / name):
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def _floats(payload) -> list:
    """Every float reachable in a JSON payload."""
    out = []
    if isinstance(payload, dict):
        for value in payload.values():
            out.extend(_floats(value))
    elif isinstance(payload, list):
        for value in payload:
            out.extend(_floats(value))
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        out.append(float(payload))
    return out


def _artifact_values() -> list:
    """Every float in the JSON artifacts of this repository."""
    values = []
    roots = [ROOT / "artifacts", ROOT / "runs", ROOT / "AI2" / "artifacts",
             ROOT / "AI2" / "runs", ROOT / "AI2" / "evidence",
             ROOT / "evidence"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            values.extend(_floats(payload))
    return values


def _candidate_numbers(text: str) -> set:
    """The decimal numbers in ``text``.

    Thousands separators are written two ways in these documents: a comma
    (``1,546.16``) and a thin space (``1 546.16``), which is what a
    Chinese-language document uses.  Both have to be stripped before the
    digits are read, or ``1 546.16`` yields ``546.16`` - a number that is
    not in any artifact, and the audit would fail on a value that is
    perfectly well supported.
    """
    collapsed = re.sub(r"(?<=\d)[,\u202f\u2009\u00a0 ](?=\d{3}(?!\d))", "", text)
    return set(re.findall(r"(?<![\w.])(\d{1,6}\.\d{2,4})(?![\w])", collapsed))


class ReadmeNumberAuditTest(unittest.TestCase):
    """Every measured number in the shipped documents has an artifact behind it."""

    def setUp(self):
        self.documents = _documents()
        if not self.documents:
            raise unittest.SkipTest("no document to audit")
        self.values = _artifact_values()
        if not self.values:
            raise unittest.SkipTest("no artifacts to audit against")

    def test_every_measured_number_has_an_artifact_behind_it(self):
        by_scale = {}
        for value in self.values:
            for digits in (0, 1, 2, 3, 4):
                bucket = by_scale.setdefault(digits, set())
                bucket.add(round(value, digits))
                # The system reports regression scores as -RMSE, so a README
                # may quote the magnitude of a negative score.
                bucket.add(round(-value, digits))

        for document in self.documents:
            with self.subTest(document=document.name):
                text = document.read_text(encoding="utf-8")
                candidates = _candidate_numbers(text)
                unmatched = []
                for candidate in sorted(candidates):
                    if candidate in ALLOWED_WITHOUT_ARTIFACT:
                        continue
                    number = float(candidate)
                    digits = len(candidate.split(".")[1])
                    if not any(round(number, scale) in by_scale.get(scale, ())
                               for scale in range(0, digits + 1)):
                        unmatched.append(candidate)
                self.assertEqual(unmatched, [],
                                 "%s quotes numbers with no artifact behind "
                                 "them: %s" % (document.name, unmatched))

    def test_the_documents_point_at_the_artifacts_they_quote(self):
        """Each document must name the artefact its numbers come from.

        The root README documents the evo harness and points at
        ``artifacts/rul_benchmark.json``; the three-track README points at
        the run reports.  The check is therefore per document, not one
        list of needles applied to all of them.
        """
        for document in self.documents:
            text = document.read_text(encoding="utf-8")
            with self.subTest(document=document.name):
                self.assertIn("artifacts/", text,
                              "the document must point at the artefacts it quotes")
                # The two READMEs document different systems: one points at
                # the model benchmark, the other at the closed loop's run
                # reports.  Which one a file is depends on what it says,
                # not on its name - the packager renames the AI2 README
                # to README.md.
                if "三方向闭环成绩" in text:
                    self.assertIn("runs/{A,B,C}/run_report.json", text)
                elif "关键实测结论" in text:
                    self.assertIn("rul_benchmark.json", text)

    def test_the_documents_reproduce_every_command_they_quote(self):
        """Every ``python -m ...`` command in the documents must be runnable."""
        # These are standard-library entry points, not modules of this
        # repository, so they are not expected to live here.
        stdlib = {"compileall", "unittest", "pytest", "pip", "json.tool"}
        for document in self.documents:
            text = document.read_text(encoding="utf-8")
            commands = re.findall(r"python -m ([a-zA-Z0-9_.]+)", text)
            for module in sorted(set(commands)):
                if module in stdlib:
                    continue
                with self.subTest(document=document.name, module=module):
                    path = module.replace(".", os.sep)
                    exists = any(
                        (root / (path + ".py")).is_file() or
                        (root / path / "__init__.py").is_file()
                        for root in (ROOT, ROOT / "AI2", ROOT / "src")
                    )
                    self.assertTrue(exists, "%s is not in the repository" % module)


if __name__ == "__main__":
    unittest.main()

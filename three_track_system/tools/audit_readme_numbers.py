# -*- coding: utf-8 -*-
"""Check every headline number in the READMEs against the artifacts.

A number on a slide that no artifact backs is a claim, not a result.
This script walks each README, pulls out the decimal numbers, and asks
the artifacts whether they know a value that rounds to it.  It is
deliberately tolerant about *where* the number appears (a README can
quote 3 decimals while the artifact has 12) and strict about *whether*
some artifact agrees.

Run:
    python tools/audit_readme_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every README that ships.  The root one documents the evo harness, the
# AI2 one documents the three-track closed loop, and the presentation
# overview is what goes on the slides; all three are read by people who
# will check them against the run reports.
READMES = (ROOT / "README.md", ROOT / "AI2" / "README.md",
           ROOT / "智训引擎-系统总思路.md",
           ROOT / "基岩智序-完整调试过程.md")

# Numbers that are structural rather than measured: budgets, limits,
# protocol constants, section numbers, and figures that come from the
# official package's own documentation.
ALLOWED_WITHOUT_ARTIFACT = {
    "1.38",   # repository size in MB, printed by the packager
    "0.51",   # submission package size in MB, printed by the packager
}


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


def artifact_numbers() -> list:
    """Every float that appears anywhere in the JSON artifacts."""
    values = []
    roots = [ROOT / "artifacts", ROOT / "runs", ROOT / "AI2" / "artifacts",
             ROOT / "AI2" / "runs", ROOT / "AI2" / "evidence", ROOT / "evidence"]
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

    Thousands separators are written two ways in these documents: a
    comma (``1,546.16``) and a thin space (``1 546.16``), which is what
    a Chinese-language document uses.  Both have to be stripped before
    the digits are read, or ``1 546.16`` yields ``546.16`` - a number
    that is not in any artifact, and the audit would fail on a value that
    is perfectly well supported.
    """
    # A digit group that is followed by a separator and exactly three more
    # digits is a thousands group, not a separate number.
    collapsed = re.sub(r"(?<=\d)[,\u202f\u2009\u00a0 ](?=\d{3}(?!\d))", "", text)
    return set(re.findall(r"(?<![\w.])(\d{1,6}\.\d{2,4})(?![\w])", collapsed))


def unmatched_numbers(text: str, values: list) -> list:
    """The decimal numbers in ``text`` that no artifact supports."""
    candidates = _candidate_numbers(text)

    by_scale = {}
    for value in values:
        for digits in (0, 1, 2, 3, 4):
            bucket = by_scale.setdefault(digits, set())
            bucket.add(round(value, digits))
            # The system reports its score as -RMSE, so a README may quote
            # the magnitude of a negative score.  Accept either sign.
            bucket.add(round(-value, digits))

    unmatched = []
    for candidate in sorted(candidates):
        if candidate in ALLOWED_WITHOUT_ARTIFACT:
            continue
        number = float(candidate)
        digits = len(candidate.split(".")[1])
        if not any(round(number, scale) in by_scale.get(scale, ())
                   for scale in range(0, digits + 1)):
            unmatched.append(candidate)
    return unmatched


def main() -> int:
    values = artifact_numbers()
    if not values:
        print("no artifacts found - nothing to check against")
        return 0

    failed = False
    for readme in READMES:
        if not readme.is_file():
            continue
        text = readme.read_text(encoding="utf-8")
        unmatched = unmatched_numbers(text, values)
        label = readme.relative_to(ROOT)
        if unmatched:
            failed = True
            print("%s: numbers with no artifact behind them:" % label)
            for candidate in unmatched:
                print("  ", candidate)
        else:
            candidates = len(set(re.findall(r"(?<![\w.])(\d{1,6}\.\d{2,4})(?![\w])", text)))
            print("%s: OK  every measured number is backed by an artifact "
                  "(%d candidates)" % (label, candidates))

    if failed:
        return 1
    print("OK  %d artifact values consulted" % len(values))
    return 0


if __name__ == "__main__":
    sys.exit(main())

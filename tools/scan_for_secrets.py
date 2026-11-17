# -*- coding: utf-8 -*-
"""Scan the shipped source for credentials and external client libraries.

The rules forbid putting an API key in the image, and the platform handles
the model access itself: the participant asks the host over the protocol
and never talks to a model provider.  This script is the check that the
code which ships inside the image really has neither.

Run:

    python tools/scan_for_secrets.py
"""
from __future__ import annotations

import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = [
    (re.compile(r"(api[_-]?key|secret|token|password|passwd|bearer|authorization)"
                r"\s*[:=]\s*[\"'][^\"']{6,}", re.I), "a credential assignment"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "an OpenAI-style key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "an AWS key"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "a GitHub token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
    (re.compile(r"\b(openai|anthropic|dashscope|zhipu|cohere|google\.generativeai)\b"),
     "an external model client"),
    (re.compile(r"^\s*import\s+requests\b|^\s*from\s+requests\b", re.M),
     "an outbound HTTP client"),
    (re.compile(r"^\s*import\s+urllib\.request|^\s*from\s+urllib\.request", re.M),
     "an outbound HTTP client"),
]

ROOTS = ["contest_submission.py", "harness.py", "contest_baselines", "tools",
         "tests/test_harness_integration.py", "tests/test_migration.py",
         "tests/test_data_projection.py", "tests/test_strategies.py",
         "tests/test_model_call_fallback.py"]

# The scanner itself mentions the very names it looks for, and the
# build tool pulls base-image layers off a registry - which is a
# development tool, not runtime code, and is kept out of the image by
# ``tools/build_docker_image.py``.  The test file mentions the names it
# refuses to find in the runtime package.  Both are excluded from the
# scan itself.
EXCLUDED = {
    "tools/scan_for_secrets.py",
    "tools/build_docker_image.py",
    "tests/test_harness_integration.py",
}

# The runtime package is what the platform actually executes, so it is
# scanned strictly, with no exclusions at all.
RUNTIME_ROOTS = ["contest_submission.py", "harness.py", "contest_baselines"]


def _files(roots):
    for root in roots:
        path = ROOT / root
        if path.is_file():
            yield path
        elif path.is_dir():
            for item in sorted(path.rglob("*.py")):
                yield item


def _scan(roots, excluded):
    """Return ``(hits, scanned_count)`` for one set of roots."""
    hits = []
    scanned = 0
    for path in _files(roots):
        relative = path.relative_to(ROOT).as_posix()
        if relative in excluded:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in PATTERNS:
            for match in pattern.finditer(text):
                line = text[:match.start()].count("\n") + 1
                snippet = match.group(0)[:60].replace("\n", " ")
                hits.append("%s:%d [%s] %s" % (relative, line, label, snippet))
    return hits, scanned


def main():
    print("== the runtime package (what the platform executes) ==")
    runtime_hits, runtime_scanned = _scan(RUNTIME_ROOTS, set())
    print("scanned %d file(s): %s" % (runtime_scanned, ", ".join(RUNTIME_ROOTS)))
    for hit in runtime_hits:
        print("  LEAK:", hit)

    print()
    print("== the wider tree (development code) ==")
    tree_hits, tree_scanned = _scan(ROOTS, EXCLUDED)
    print("scanned %d file(s) under: %s" % (tree_scanned, ", ".join(ROOTS)))
    for hit in tree_hits:
        print("  REVIEW:", hit)

    if runtime_hits:
        print()
        print("FAIL  the runtime package carries a credential or an external client")
        return 1
    print()
    print("OK  the runtime package carries no credential, no private key and no "
          "external model client")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

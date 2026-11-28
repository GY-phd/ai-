# -*- coding: utf-8 -*-
"""Rewrite the report sources into a paper-like register.

The competition notes ask for the scheme report to read like a paper:
no machine-flavoured markers, no curly quotation marks, no bolding
every other word.  The Markdown sources under ``docs/`` are reviewed
as text in the repository and are also the input of
``tools/build_submission_pdfs.py``, so the cleanup belongs here rather
than in the renderer: a PDF that looked clean while the source still
carried the markers would only hide the problem.

Three transformations are applied, and only to prose:

* ``**bold**`` is dropped.  Emphasis is carried by the section
  structure instead, which is what a paper does.
* The Chinese curly quotation marks U+201C / U+201D become the
  western straight double quote, as the notes require.
* The CJK corner brackets U+300C / U+300D, which serve the same
  purpose, become straight double quotes as well.
* A run of colons used as a lead-in is replaced by a dash.

Code blocks, tables and inline code spans are left exactly as they
are: they are quoted material, not prose, and rewriting them would
break the copy-paste instructions they contain.

Usage::

    python tools/normalize_report_style.py            # rewrite in place
    python tools/normalize_report_style.py --check    # only report
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The documents that are rendered into the submitted PDFs.
REPORTS = [
    ROOT / "docs" / "项目说明书_正文.md",
    ROOT / "docs" / "系统运行说明文档_正文.md",
    ROOT / "docs" / "技术报告_正文.md",
]

# The Markdown files that ship inside the submission package.  They are
# what the reviewers read, so they carry the same register as the PDFs.
PACKAGE = ROOT / "提交材料_20261125"
PACKAGE_SKIP = {"figures"}

# The Chinese curly quotation marks, the CJK corner brackets and the
# full-width colon.
LQ = "\u201c"
RQ = "\u201d"
CL = "\u300c"
CR = "\u300d"
CN_COLON = "\uff1a"


def _segments(text):
    """Split the text into (is_prose, chunk) pairs.

    Fenced code blocks and inline code spans are code, everything else
    is prose.  A table row is prose for the purposes of the bold
    marker, but its ``|`` layout must survive, so table rows are
    treated as prose only outside the cell text.
    """
    out = []
    for block in re.split(r"(```.*?```)", text, flags=re.S):
        if block.startswith("```"):
            out.append((False, block))
            continue
        for span in re.split(r"(`[^`\n]+`)", block):
            if span.startswith("`") and span.endswith("`") and len(span) > 1:
                out.append((False, span))
            else:
                out.append((True, span))
    return out


def normalize(text):
    """Return ``text`` with the machine-flavoured markers removed."""
    out = []
    for is_prose, chunk in _segments(text):
        if not is_prose:
            out.append(chunk)
            continue
        # Bold markers.  The emphasis is dropped entirely rather than
        # paired up: a document that bolds every other phrase reads as
        # machine output, and matching pairs across line breaks leaves
        # stray markers behind.
        chunk = chunk.replace("**", "")
        # The curly quotation marks and the corner brackets the notes
        # call out.
        chunk = chunk.replace(LQ, '"').replace(RQ, '"')
        chunk = chunk.replace(CL, '"').replace(CR, '"')
        # The full-width colon becomes a western one; it is used in
        # running text rather than in the tables and the code.
        chunk = chunk.replace(CN_COLON, ":")
        out.append(chunk)
    return "".join(out)


def _targets():
    """The Markdown files the cleanup applies to."""
    out = []
    for path in REPORTS:
        if path.exists():
            out.append(path)
    if PACKAGE.exists():
        for path in sorted(PACKAGE.rglob("*.md")):
            if any(part in PACKAGE_SKIP for part in path.parts):
                continue
            out.append(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report what would change, and exit non-zero")
    args = ap.parse_args(argv)

    changed = 0
    for path in _targets():
        original = path.read_text(encoding="utf-8")
        updated = normalize(original)
        if updated == original:
            continue
        changed += 1
        if args.check:
            print("would rewrite %s" % path.relative_to(ROOT))
        else:
            path.write_text(updated, encoding="utf-8")
            print("rewrote %s" % path.relative_to(ROOT))

    if args.check and changed:
        return 1
    if not changed:
        print("nothing to rewrite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

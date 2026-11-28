# -*- coding: utf-8 -*-
"""Render the cloud experiment record into a PDF.

Rule 36 of the submission table asks for the cloud experiment record
"链接或查询依据" in the format pdf or txt.  The record itself is a
set of JSONL files, and JSONL is not one of the accepted formats, so
the record has to be presented as a document.  This script turns the
query-basis note into a PDF that goes in the same upload box.

The JSONL files stay in the package as the raw evidence; the PDF is
the human-readable statement of what those files contain and where to
find them.

Usage::

    python tools/build_experiment_record_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_submission_pdfs import render, styles  # noqa: E402

SRC = ROOT / "提交材料_20261125" / "03_云端实验记录" / "云端实验记录说明.md"
DST = ROOT / "提交材料_20261125" / "03_云端实验记录" / "云端实验记录.pdf"


def main():
    if not SRC.exists():
        print("missing source: %s" % SRC, file=sys.stderr)
        return 1
    render(SRC.read_text(encoding="utf-8"), DST, "智训引擎 · 云端实验记录", styles())
    size = DST.stat().st_size
    print("wrote %s (%d bytes, %.2f MB)" % (DST, size, size / 1048576.0))
    if size > 50 * 1024 * 1024:
        print("  !! exceeds the 50 MB limit", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

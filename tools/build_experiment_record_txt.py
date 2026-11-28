# -*- coding: utf-8 -*-
"""Render the cloud experiment record into a plain-text file.

Rule 36 accepts pdf or txt.  The PDF is what most reviewers will
read, but a plain-text copy matters for two reasons: the platform's
"本机小于 100 字" alternative wants short text, and a reviewer who
wants to grep the record cannot grep a PDF.

The text version is the same content with the Markdown table markup
flattened, so it stays readable in a terminal.

Usage::

    python tools/build_experiment_record_txt.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "提交材料_20261125" / "03_云端实验记录" / "云端实验记录说明.md"
DST = ROOT / "提交材料_20261125" / "03_云端实验记录" / "云端实验记录.txt"


def to_text(markdown):
    """Flatten the Markdown into plain text."""
    out = []
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # A table becomes aligned columns, so it survives in a
        # terminal as a table and not as a row of pipes.
        if stripped.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                widths = [max(len(r[c]) for r in rows if c < len(r))
                          for c in range(max(len(r) for r in rows))]
                for r_i, row in enumerate(rows):
                    cells = [(row[c] if c < len(row) else "").ljust(widths[c])
                             for c in range(len(widths))]
                    out.append("  ".join(cells).rstrip())
                    if r_i == 0:
                        out.append("  ".join("-" * w for w in widths))
            out.append("")
            continue

        # Headings lose their hashes but keep their weight.
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            out.append("")
            out.append(m.group(2).strip())
            out.append("=" * max(8, len(m.group(2).strip())))
            out.append("")
            i += 1
            continue

        # A fenced block is quoted material; keep it verbatim.
        if stripped.startswith("```"):
            out.append("")
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                out.append(lines[i].rstrip())
                i += 1
            i += 1
            out.append("")
            continue

        # A block quote keeps its marker.
        m = re.match(r"^>\s?(.*)$", stripped)
        if m:
            out.append("  " + m.group(1))
            i += 1
            continue

        # A bullet keeps its marker.
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            out.append("  - " + m.group(1))
            i += 1
            continue

        if not stripped:
            out.append("")
            i += 1
            continue

        out.append(stripped)
        i += 1

    text = "\n".join(out)
    # The bold markers are a Markdown convention; a text file has no
    # way to render them and they read as noise.
    text = text.replace("**", "")
    return text


def main():
    if not SRC.exists():
        print("missing source: %s" % SRC, file=sys.stderr)
        return 1
    DST.write_text(to_text(SRC.read_text(encoding="utf-8")), encoding="utf-8",
                   newline="\n")
    size = DST.stat().st_size
    print("wrote %s (%d bytes, %.2f MB)" % (DST, size, size / 1048576.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

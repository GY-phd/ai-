# -*- coding: utf-8 -*-
"""Render the submission PDFs (项目说明书 / 系统运行说明文档).

The rules ask for two PDF documents, each at most 10 MB:

* 37 - 系统运行说明文档 (pdf, <= 10M)
* 38 - 项目说明书 (pdf, <= 10M)

This module turns a Markdown-ish source into a paginated PDF with a CJK
font, so the documents are generated from the same text that is reviewed
in the repository.  Nothing here is used at run time inside the image:
the two documents are deliverables, not part of the closed loop.

Usage::

    python tools/build_submission_pdfs.py
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "提交材料_20261125"

# A CJK-capable TrueType font.  Windows always ships one, and the
# repository is developed on Windows; the search list keeps the tool
# usable elsewhere.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

BODY = "Helvetica"
MONO = "Courier"
ACCENT = colors.HexColor("#1F4E79")
MUTED = colors.HexColor("#595959")
RULE = colors.HexColor("#BFBFBF")
CODEBG = colors.HexColor("#F4F6F8")


def _register_fonts():
    """Register the CJK body font and return ``(body, mono)`` names."""
    for path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            if path.lower().endswith(".ttc"):
                pdfmetrics.registerFont(TTFont("CJK", path, subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont("CJK", path))
            return "CJK", "Courier"
        except Exception:  # noqa: BLE001 - a font that will not load is skipped
            continue
    return "Helvetica", "Courier"


FONT_BODY, FONT_MONO = _register_fonts()


def styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName=FONT_BODY, fontSize=22,
        leading=30, textColor=ACCENT, spaceAfter=4, alignment=TA_CENTER)
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"], fontName=FONT_BODY, fontSize=11,
        leading=16, textColor=MUTED, alignment=TA_CENTER, spaceAfter=16)
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName=FONT_BODY, fontSize=15,
        leading=21, textColor=ACCENT, spaceBefore=14, spaceAfter=7)
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName=FONT_BODY, fontSize=12.5,
        leading=18, textColor=ACCENT, spaceBefore=11, spaceAfter=5)
    s["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontName=FONT_BODY, fontSize=11,
        leading=16, textColor=colors.HexColor("#2E5F8A"), spaceBefore=8,
        spaceAfter=4)
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName=FONT_BODY, fontSize=9.6,
        leading=15.2, alignment=TA_JUSTIFY, spaceAfter=5, textColor=colors.black)
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=13, bulletIndent=3, spaceAfter=2.5)
    s["quote"] = ParagraphStyle(
        "quote", parent=s["body"], leftIndent=11, rightIndent=8,
        textColor=colors.HexColor("#333333"), borderPadding=0, spaceAfter=6)
    # A code block is rendered in the CJK font, not in Courier: Courier has
    # no box-drawing glyph, so an ASCII diagram drawn with ``─``/``│`` would
    # come out as a row of "n" (the .notdef glyph).  Microsoft YaHei covers
    # the box characters and keeps the CJK labels in the same diagram
    # readable.
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontName=FONT_BODY, fontSize=7.4,
        leading=10.6, leftIndent=7, rightIndent=5, backColor=CODEBG,
        borderPadding=5, spaceBefore=4, spaceAfter=7)
    s["cell"] = ParagraphStyle(
        "cell", parent=base["BodyText"], fontName=FONT_BODY, fontSize=8.4,
        leading=11.4, alignment=TA_LEFT, spaceAfter=0)
    s["cellhead"] = ParagraphStyle(
        "cellhead", parent=s["cell"], textColor=colors.white, fontSize=8.6)
    s["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"], fontName=FONT_BODY, fontSize=8.2,
        leading=11, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3,
        spaceAfter=9)
    return s


# ---------------------------------------------------------------- inline
_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def _esc(text):
    return "".join(_ESC.get(ch, ch) for ch in text)


def inline(text):
    """Markdown-ish inline markup -> reportlab inline markup.

    Supported: ``code``, **bold**, *italic*.
    """
    out = []
    for part in re.split(r"(``[^`]+``|\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("``") and part.endswith("``"):
            # The inline code font is the CJK font as well: a span like
            # ``rnd.model`` is ASCII, but spans such as ``the-degradation-is-relative``
            # are fine in Courier while any CJK inside one would be lost.
            out.append('<font face="%s" size="7.8">%s</font>'
                       % (FONT_BODY, _esc(part[2:-2])))
        elif part.startswith("**") and part.endswith("**"):
            out.append("<b>%s</b>" % _esc(part[2:-2]))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append("<i>%s</i>" % _esc(part[1:-1]))
        else:
            out.append(_esc(part))
    return "".join(out)


def _table(rows, s, widths=None):
    data = []
    for i, row in enumerate(rows):
        style = s["cellhead"] if i == 0 else s["cell"]
        data.append([Paragraph(inline(str(c)), style) for c in row])
    avail = A4[0] - 36 * mm
    if widths is None:
        widths = [avail / len(rows[0])] * len(rows[0])
    else:
        total = float(sum(widths))
        widths = [avail * w / total for w in widths]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F7F9FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
    ]))
    return table


def render(markdown, path, title, s):
    """Turn the Markdown source into a paginated PDF at ``path``."""
    story = []
    lines = markdown.split("\n")
    i = 0
    n = len(lines)

    def flush_paragraph(buf):
        if buf:
            story.append(Paragraph(inline(" ".join(buf)), s["body"]))
            story.append(Spacer(1, 1))

    para = []
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph(para)
            para = []
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph(para)
            para = []
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            body = "\n".join(code) or " "
            story.append(Preformatted(body, s["code"], maxLineLength=104))
            continue

        if stripped.startswith("|") and i + 1 < n and re.match(
                r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush_paragraph(para)
            para = []
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                story.append(_table(rows, s))
                story.append(Spacer(1, 7))
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            flush_paragraph(para)
            para = []
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                story.append(Paragraph(inline(text), s["title"]))
            else:
                story.append(Paragraph(inline(text), s["h%d" % level]))
            i += 1
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            flush_paragraph(para)
            para = []
            story.append(Paragraph(inline(m.group(1)), s["bullet"],
                                   bulletText="•"))
            i += 1
            continue

        m = re.match(r"^>\s?(.*)$", stripped)
        if m:
            flush_paragraph(para)
            para = []
            story.append(Paragraph(inline(m.group(1)), s["quote"]))
            i += 1
            continue

        if re.match(r"^---+$", stripped):
            flush_paragraph(para)
            para = []
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_paragraph(para)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT_BODY, 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2.0, 13 * mm,
                                 "%s  ·  第 %d 页" % (title, doc.page))
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.restoreState()

    doc = BaseDocTemplate(str(path), pagesize=A4,
                          leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=17 * mm, bottomMargin=19 * mm,
                          title=title, author="智训引擎 参赛队")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=decorate)])
    doc.build(story)
    return path


def main():
    s = styles()
    jobs = []
    spec_src = ROOT / "docs" / "项目说明书_正文.md"
    run_src = ROOT / "docs" / "系统运行说明文档_正文.md"
    spec_out = OUT / "05_项目说明书" / "项目说明书.pdf"
    run_out = OUT / "04_系统运行说明文档" / "系统运行说明文档.pdf"

    if spec_src.exists():
        jobs.append((spec_src, spec_out, "智训引擎 · 项目说明书"))
    if run_src.exists():
        jobs.append((run_src, run_out, "智训引擎 · 系统运行说明文档"))
    rep_src = ROOT / "docs" / "技术报告_正文.md"
    if rep_src.exists():
        jobs.append((rep_src, ROOT / "提交材料_20261125" / "09_技术报告与附件"
                     / "技术报告.pdf", "智训引擎 · 技术报告"))

    if not jobs:
        print("no source documents found", file=sys.stderr)
        return 1
    for src, dst, title in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        render(src.read_text(encoding="utf-8"), dst, title, s)
        size = dst.stat().st_size
        print("wrote %s (%d bytes, %.2f MB)" % (dst, size, size / 1048576.0))
        if size > 10 * 1024 * 1024:
            print("  !! exceeds the 10 MB limit", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

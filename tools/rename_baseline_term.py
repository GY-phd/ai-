# -*- coding: utf-8 -*-
"""Rename 官方基线 -> 静态基线 across the deliverables.

The platform's own wording is 静态基线 (the *static* baseline = the
shipped checkpoint that never trains).  Calling it 官方基线 was our own
invention and it collides with "the official baseline" a reader would
assume means the platform-wide best.  This script normalises the term.

Run::

    python tools/rename_baseline_term.py [--check]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The deliverables and the analysis documents that ship with them.
TARGETS = [
    # --- submitted materials
    "docs/最新技术成果与技术总览.md",
    "docs/答辩PPT内容大纲.md",
    "docs/答辩PPT配图索引.md",
    "docs/演示视频脚本.md",
    "docs/技术报告_正文.md",
    "docs/项目说明书_正文.md",
    "docs/来源声明.md",
    "docs/平台规则与评分要点_20261124.md",
    # --- the root-cause analysis and its predecessors
    "docs/SELECTION_SLICE_ROOT_CAUSE_20261124.md",
    "docs/SELECTION_RULE_DECISION_20261124.md",
    "docs/PLATFORM_REPORT_REVIEW.md",
    "docs/PLATFORM_RUN_20261124_V8_FAILURE.md",
    "docs/PLATFORM_RUN_20261123_ANALYSIS.md",
    "docs/TECHNICAL_REPORT.md",
    "docs/训练凭证.md",
    "docs/RUNBOOK.md",
    "docs/INDUSTRIAL_CPU_CHECK.md",
    "docs/SOLUTION.md",
    "docs/ABC_OPTIMIZATION.md",
    "docs/C_WEIGHTING.md",
    "docs/作品文档.md",
    "docs/ANSWER_TO_TEAMMATE.md",
    # --- the figure generator
    "tools/build_ppt_figures.py",
]

# Longer forms first so the shorter replacement does not eat them.
REPLACEMENTS = [
    ("官方静态基线", "静态基线"),
    ("官方基线", "静态基线"),
    ("官方 baseline", "静态基线"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only report what would change")
    args = ap.parse_args(argv)

    total = 0
    touched = 0
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        orig = s
        for old, new in REPLACEMENTS:
            s = s.replace(old, new)
        if s == orig:
            continue
        n = sum(orig.count(old) for old, _ in REPLACEMENTS)
        total += n
        touched += 1
        print("  %-52s %d" % (rel, n))
        if not args.check:
            p.write_text(s, encoding="utf-8")
    print("\n%d occurrence(s) in %d file(s)" % (total, touched))
    if args.check and total:
        print("(run without --check to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

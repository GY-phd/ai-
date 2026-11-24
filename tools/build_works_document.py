# -*- coding: utf-8 -*-
"""Build the platform's "作品文档" (works document).

The platform's submission form has exactly one file field for the
documents the rules table lists: 作品文档.  The rules table row 34 calls
it 模型配置 (txt, <= 50 MB), but the frontend enforces **2 MiB**:

    if (s.size > 2097152) { setError('作品文档不能超过 2 MiB。'); return; }

The platform's own wording of the required materials is:

    镜像和源码、模型调用信息、实验记录链接，以及设计说明、技术报告
    （不超过 10 页）、演示视频（不超过 5 分钟）和来源声明。
    平台没有单独入口的材料请随作品文档提交或提供链接。

So the works document has to carry the model configuration **and** the
materials that have no field of their own: 设计说明, 技术报告, 演示视频
说明 and 来源声明.  This script concatenates them into one plain-text
file and checks it against the 2 MiB limit.

Usage::

    python tools/build_works_document.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "提交材料_20261125" / "01_模型配置" / "作品文档.txt"

# (marker title, source path).  The order follows the platform's own
# sentence: model configuration first, then the materials that have no
# form field of their own.
SECTIONS = [
    ("模型配置", ROOT / "提交材料_20261125" / "01_模型配置" / "模型配置.txt"),
    ("设计说明（即项目说明书正文）", ROOT / "docs" / "项目说明书_正文.md"),
    ("技术报告", ROOT / "docs" / "技术报告_正文.md"),
    ("来源声明", ROOT / "docs" / "来源声明.md"),
    ("云端实验记录链接与查询依据",
     ROOT / "提交材料_20261125" / "03_云端实验记录" / "云端实验记录说明.md"),
    ("平台规则与评分要点", ROOT / "docs" / "平台规则与评分要点_20261124.md"),
]

RULE = "=" * 78
LIMIT = 2 * 1024 * 1024


def main():
    parts = []
    header = (
        RULE + "\n"
        "智训引擎 — 作品文档（模型配置 + 设计说明 + 技术报告 + 来源声明）\n"
        + RULE + "\n"
        "赛题    : “智训引擎”——工业制造场景下的模型训练与超参数优化\n"
        "队伍    : gy_65511\n"
        "版本    : v8\n"
        "生成日期: 2026-11-24\n"
        "\n"
        "本文件是平台提交表单唯一的文件入口（“作品文档”）的完整内容。\n"
        "平台规则原文：\n"
        "  “镜像和源码、模型调用信息、实验记录链接，以及设计说明、技术报告\n"
        "   （不超过 10 页）和来源声明。\n"
        "   平台没有单独入口的材料请随作品文档提交或提供链接。”\n"
        "\n"
        "其它材料各自独立成文件，见同目录的《提交清单与填写指引.md》：\n"
        "  镜像与源码  -> 02_Harness系统镜像/ 与 06_源码与依赖/\n"
        "  模型调用信息 -> 本文件第一节\n"
        "  实验记录链接 -> 本文件第五节\n"
        + RULE + "\n\n"
    )
    parts.append(header)

    for idx, (title, path) in enumerate(SECTIONS, start=1):
        if not path.exists():
            print("  skip (missing): %s" % path.name, file=sys.stderr)
            continue
        body = path.read_text(encoding="utf-8")
        parts.append(RULE + "\n第 %d 节  %s\n" % (idx, title) + RULE + "\n")
        parts.append(body)
        parts.append("\n\n")

    text = "".join(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")

    size = OUT.stat().st_size
    print("wrote %s" % OUT)
    print("  sections: %d" % len(SECTIONS))
    print("  size    : %d bytes (%.2f MB)" % (size, size / 1048576.0))
    print("  limit   : 2 MiB -> %s" % ("OK" if size <= LIMIT else "EXCEEDED"))
    if size > LIMIT:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

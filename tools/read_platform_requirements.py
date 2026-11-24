# -*- coding: utf-8 -*-
"""Print the platform's own submission-requirement text, unabridged.

The rules table in the screenshot lists the deliverables.  The platform
bundle carries the same text (plus the fields that have no table row),
so the package is built from the platform's wording rather than from a
reading of a screenshot.
"""
from __future__ import annotations

import re
import urllib.request

BUNDLE = "https://rsi-index.115.zone/assets/index-C-tzYOyK.js"

INTERESTING = [
    "评分对象",
    "平台没有单独入口",
    "技术报告",
    "演示视频",
    "来源声明",
    "不超过 10 页",
    "不超过 5 分钟",
    "设计说明",
    "续训后的工业模型",
    "各自不超过 4000",
    "64KiB",
]


def main():
    req = urllib.request.Request(BUNDLE, headers={"User-Agent": "Mozilla/5.0"})
    text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    for needle in INTERESTING:
        for match in list(re.finditer(re.escape(needle), text))[:3]:
            start = max(0, match.start() - 500)
            end = min(len(text), match.start() + 700)
            print("=== %s ===" % needle)
            print(text[start:end].replace("\\n", "\n"))
            print()


if __name__ == "__main__":
    main()

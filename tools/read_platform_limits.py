# -*- coding: utf-8 -*-
"""Read the platform's own bundle and report the upload limits it enforces.

The submission rules (rules item 34/35) say the model configuration is at
most 50 MB and the Harness image at most 50 MB, with a note that a larger
image should be delivered through a network-drive link.  The platform's
frontend enforces a *different*, tighter limit on the "works document"
(the model configuration): 2 MiB.  This script reads the real numbers
out of the shipped bundle instead of guessing.
"""
from __future__ import annotations

import re
import urllib.request

BUNDLE = "https://rsi-index.115.zone/assets/index-C-tzYOyK.js"


def fetch():
    req = urllib.request.Request(BUNDLE, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def main():
    text = fetch()
    print("bundle length: %d" % len(text))
    print()
    print("=== size guards (size > N) ===")
    seen = set()
    for match in re.finditer(r"size\s*>\s*(\d+)", text):
        limit = match.group(1)
        if limit in seen:
            continue
        seen.add(limit)
        segment = text[max(0, match.start() - 200): match.start() + 260]
        segment = segment.replace("\n", " ")
        print("  limit=%s" % limit)
        print("    %s" % segment)
        print()

    print("=== human-readable limits ===")
    for pattern in (r"[^\"`]{0,40}(?:不能超过|最大|不超过|不得超过)[^\"`]{0,40}",
                    r"[^\"`]{0,60}\d+\s*(?:MiB|MB)[^\"`]{0,60}"):
        for match in list(re.finditer(pattern, text))[:14]:
            segment = match.group(0).strip()
            if len(segment) < 12:
                continue
            print("  %s" % segment.replace("\n", " "))
    print()

    print("=== image-archive endpoints ===")
    for match in list(re.finditer(r"/api/v1/image-archives[^\s\"`]{0,50}", text))[:10]:
        print("  %s" % match.group(0))
    print()

    print("=== 网盘 / drive link mentions ===")
    for match in list(re.finditer(r"网盘", text))[:6]:
        print("  %s" % text[max(0, match.start() - 200): match.start() + 240]
              .replace("\n", " "))
        print()


if __name__ == "__main__":
    main()

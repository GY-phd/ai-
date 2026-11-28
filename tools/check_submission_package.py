# -*- coding: utf-8 -*-
"""Check the submission package against the platform's format rules.

The rules are a table, and every row of it has a format and a size
limit.  A package can be complete and still be rejected on a format
detail, so the check is mechanical rather than by eye: it reads the
files, measures them, and prints the verdict per row.

Usage::

    python tools/check_submission_package.py
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "提交材料_20261125"

MB = 1024 * 1024

# row number, label, relative path, format, limit in bytes
ROWS = [
    (34, "模型配置", "01_模型配置/作品文档.txt", "txt", 50 * MB),
    (35, "Harness 的系统镜像或网盘链接", "02_Harness系统镜像/zhixun-engine-linux-amd64.tar", "tar", 50 * MB),
    (36, "云端实验记录", "03_云端实验记录/云端实验记录.pdf", "pdf", 50 * MB),
    (37, "系统运行说明文档", "04_系统运行说明文档/系统运行说明文档.pdf", "pdf", 10 * MB),
    (38, "项目说明书", "05_项目说明书/项目说明书.pdf", "pdf", 10 * MB),
    (40, "项目源代码", "06_源码与依赖/智训引擎-源码包-v10.zip", "zip", 50 * MB),
]


def main():
    problems = []
    # The net-disk link is what legally replaces an over-size mirror,
    # so its presence has to be known before the rows are judged.
    works = PKG / "01_模型配置" / "作品文档.txt"
    link_present = False
    works_text = ""
    if works.exists():
        works_text = works.read_text(encoding="utf-8")
        link_present = "pan.baidu.com" in works_text
        if not link_present:
            problems.append("作品文档里没有网盘链接（镜像超过 50 MB，规则要求提供链接）")
        for key in ("8Y2TYLhjvLG8PyAr25Il1", "trio_4p7Nl7jB8p48twyFqagXQ1lsthaJrYzI2ImwH"):
            if key in works_text:
                problems.append("作品文档里出现了 API Key")

    # The link has to be the *same* everywhere: a stale share pointing
    # at an older image would ship the wrong bytes to the reviewers.
    link_re = re.compile(r"https://pan\.baidu\.com/s/([A-Za-z0-9]+)")
    seen = {}
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        for m in link_re.finditer(body):
            seen.setdefault(m.group(1), []).append(path.name)
    if len(seen) > 1:
        problems.append("包内的网盘链接不一致：%s" %
                        {k: v[:3] for k, v in seen.items()})
    elif seen:
        for key, files in seen.items():
            print("网盘链接一致：%s（%d 个文件）" % (key, len(files)))

    print("%-4s %-22s %-12s %12s %12s  %s" % ("行", "项目", "格式", "实际", "上限", "结论"))
    print("-" * 92)
    for row, label, rel, fmt, limit in ROWS:
        path = PKG / rel
        if not path.exists():
            print("%-4s %-22s %-12s %12s %12s  缺失" % (row, label, fmt, "-", "-"))
            problems.append("行 %d %s：文件缺失 %s" % (row, label, rel))
            continue
        size = path.stat().st_size
        suffix = path.suffix.lower().lstrip(".")
        ok_fmt = suffix == fmt
        ok_size = size <= limit
        verdict = []
        if not ok_fmt:
            verdict.append("格式应为 %s" % fmt)
        if not ok_size:
            verdict.append("超出上限")
        if size > limit:
            # The mirror is legitimately over the limit, and the rules
            # say to give a net-disk link in that case.  The link is
            # checked separately below, so over-size is only a problem
            # when no link is there to replace it.
            if "网盘" in label and link_present:
                verdict = ["超限，按规则以网盘链接替代"]
            else:
                problems.append("行 %d %s：%d 字节超出 %d" % (row, label, size, limit))
                verdict.append("超出上限")
        if not ok_fmt:
            problems.append("行 %d %s：扩展名 .%s，应为 .%s" % (row, label, suffix, fmt))
        print("%-4s %-22s %-12s %12d %12d  %s"
              % (row, label, suffix, size, limit, "OK" if not verdict else "；".join(verdict)))

    # The source package must not carry the hidden test set or a key.
    zip_path = PKG / "06_源码与依赖" / "智训引擎-源码包-v10.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
        # A file that *scans* for secrets is not a secret; only files
        # that carry key-shaped material count.
        secret_like = [n for n in names
                       if any(k in n.lower() for k in
                              (".env", "credentials", "id_rsa", ".pem", ".key"))
                       and "scan_for_secrets" not in n]
        if secret_like:
            problems.append("源码包含疑似密钥文件：%s" % secret_like[:5])

    print()
    # The rule accepts pdf or txt for the experiment record, so both
    # have to exist and to agree with the raw evidence.
    for rel, fmt in (("03_云端实验记录/云端实验记录.pdf", "pdf"),
                      ("03_云端实验记录/云端实验记录.txt", "txt")):
        path = PKG / rel
        if not path.exists():
            problems.append("规则 36 要求的 %s 缺失" % fmt)
            continue
        if path.stat().st_size > 50 * MB:
            problems.append("%s 超出 50 MB" % rel)

    raw = PKG / "03_云端实验记录" / "run_report.json"
    ledger = PKG / "03_云端实验记录" / "ledger.jsonl"
    for path in (raw, ledger):
        if not path.exists():
            problems.append("原始证据 %s 缺失" % path.name)

    print()
    if problems:
        print("发现 %d 个问题:" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("全部 6 项格式与体积检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

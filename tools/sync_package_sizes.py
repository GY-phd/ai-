# -*- coding: utf-8 -*-
"""One-off: refresh the sizes quoted in the package README.

The README quotes the byte size of every deliverable, and those
numbers go stale as soon as a document is rebuilt.  This script
recomputes them from the files themselves so the README cannot drift
away from the package it describes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "提交材料_20261125"
README = PKG / "README.md"

BT = chr(96)  # backtick, kept out of the literal to avoid quoting hell


def kb(path):
    return path.stat().st_size / 1024.0


def mib(path):
    return path.stat().st_size / 1048576.0


def main():
    works = PKG / "01_模型配置" / "作品文档.txt"
    model_cfg = PKG / "01_模型配置" / "模型配置.txt"
    run_doc = PKG / "04_系统运行说明文档" / "系统运行说明文档.pdf"
    spec = PKG / "05_项目说明书" / "项目说明书.pdf"
    report = PKG / "03_云端实验记录" / "run_report.json"
    ledger = PKG / "03_云端实验记录" / "ledger.jsonl"
    record_pdf = PKG / "03_云端实验记录" / "云端实验记录.pdf"
    record_txt = PKG / "03_云端实验记录" / "云端实验记录.txt"
    swanlab = PKG / "03_云端实验记录" / "swanlab"
    zip_path = PKG / "06_源码与依赖" / "智训引擎-源码包-v10.zip"

    swan_bytes = sum(p.stat().st_size for p in swanlab.glob("*.jsonl"))

    subs = [
        ("作品文档.txt%s（51.4 KB）" % BT, "作品文档.txt%s（%.1f KB）" % (BT, kb(works))),
        ("作品文档.txt%s（62.2 KB）" % BT, "作品文档.txt%s（%.1f KB）" % (BT, kb(works))),
        ("| 34 | 模型配置（txt，≤50M） | %s01_模型配置/作品文档.txt%s | 51.4 KB |"
         % (BT, BT), "| 34 | 模型配置（txt，≤50M） | %s01_模型配置/作品文档.txt%s | %.1f KB |"
         % (BT, BT, kb(works))),
        ("| 34 | 模型配置（txt，≤50M） | %s01_模型配置/作品文档.txt%s | 62.2 KB |"
         % (BT, BT), "| 34 | 模型配置（txt，≤50M） | %s01_模型配置/作品文档.txt%s | %.1f KB |"
         % (BT, BT, kb(works))),
        ("平台唯一文件入口 (62.2 KB)", "平台唯一文件入口 (%.1f KB)" % kb(works)),
        ("| 37 | 系统运行说明文档（pdf，≤10M） | %s04_系统运行说明文档/系统运行说明文档.pdf%s | 236 KB |"
         % (BT, BT), "| 37 | 系统运行说明文档（pdf，≤10M） | %s04_系统运行说明文档/系统运行说明文档.pdf%s | %.0f KB |"
         % (BT, BT, kb(run_doc))),
        ("| 38 | 项目说明书（pdf，≤10M） | %s05_项目说明书/项目说明书.pdf%s | 278 KB |"
         % (BT, BT), "| 38 | 项目说明书（pdf，≤10M） | %s05_项目说明书/项目说明书.pdf%s | %.0f KB |"
         % (BT, BT, kb(spec))),
        ("44.6 KB + 26 KB + 36 KB", "%.1f KB + %.0f KB + %.0f KB"
         % (kb(report), kb(ledger), swan_bytes / 1024.0)),
        ("168 KB + 7 KB + 45 KB + 26 KB + 30 KB",
         "%.0f KB + %.0f KB + %.0f KB + %.0f KB + %.0f KB"
         % (kb(record_pdf), kb(record_txt), kb(report), kb(ledger),
            swan_bytes / 1024.0)),
        ("云端实验记录.pdf                    ★ 上传用 (168 KB)",
         "云端实验记录.pdf                    ★ 上传用 (%.0f KB)" % kb(record_pdf)),
        ("云端实验记录.txt                    ★ 上传用 (7 KB)",
         "云端实验记录.txt                    ★ 上传用 (%.0f KB)" % kb(record_txt)),
        ("≈4.59 MiB", "≈%.2f MiB" % mib(zip_path)),
        ("系统运行说明文档.pdf%s（236 KB, 4 页）" % BT,
         "系统运行说明文档.pdf%s（%.0f KB, 4 页）" % (BT, kb(run_doc))),
        ("模型配置.txt                        单独的模型配置 (8.1 KB)",
         "模型配置.txt                        单独的模型配置 (%.1f KB)" % kb(model_cfg)),
    ]

    text = README.read_text(encoding="utf-8")
    for old, new in subs:
        if old not in text:
            print("MISS: %s" % old[:60])
            continue
        text = text.replace(old, new)
    README.write_text(text, encoding="utf-8")
    print("README sizes refreshed")


if __name__ == "__main__":
    main()

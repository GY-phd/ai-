# -*- coding: utf-8 -*-
"""The net-disk link must be the same everywhere in the package.

The mirror is over the 50 MB limit, so the rules let a net-disk link
replace it.  That link is the only route by which a reviewer can get
the image, and the image is what the whole submission is judged on.

A stale share is the quiet failure mode here: the package keeps its
own copy of the tar, so nothing breaks locally, and the reviewer
simply downloads an older image that lacks the two root-cause fixes.
The check is therefore on *consistency* across the package rather than
on the presence of a link.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "提交材料_20261125"

LINK_RE = re.compile(r"https://pan\.baidu\.com/s/([A-Za-z0-9]+)")


def _text_files():
    for path in sorted(PKG.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        yield path


def _docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", "", xml)


class TestNetDiskLinkIsConsistent:
    def test_every_document_that_names_a_link_names_the_same_one(self):
        found = {}
        for path in _text_files():
            body = path.read_text(encoding="utf-8", errors="replace")
            for m in LINK_RE.finditer(body):
                found.setdefault(m.group(1), []).append(path.name)
        assert found, "没有任何文档给出网盘链接"
        assert len(found) == 1, (
            "网盘链接不一致，评审会下载到旧镜像: %s" % found)

    def test_the_docx_upload_carries_the_same_link(self):
        docx = PKG / "02_Harness系统镜像" / "harness网盘链接.docx"
        assert docx.exists(), "缺少镜像上传位要用的 docx"
        found = LINK_RE.findall(_docx_text(docx))
        assert found, "docx 里没有网盘链接"

        text_links = set()
        for path in _text_files():
            body = path.read_text(encoding="utf-8", errors="replace")
            text_links.update(LINK_RE.findall(body))
        assert set(found) & text_links, (
            "docx 的链接 %s 与文档里的 %s 不是同一个" % (found, text_links))

    def test_the_works_document_has_the_link(self):
        works = PKG / "01_模型配置" / "作品文档.txt"
        assert works.exists()
        body = works.read_text(encoding="utf-8")
        assert LINK_RE.search(body), "作品文档里没有网盘链接"

    def test_no_api_key_leaked_into_the_package(self):
        for path in _text_files():
            body = path.read_text(encoding="utf-8", errors="replace")
            assert "8Y2TYLhjvLG8PyAr25Il1" not in body, path.name
            assert "trio_4p7Nl7jB8p48twyFqagXQ1lsthaJrYzI2ImwH" not in body, path.name

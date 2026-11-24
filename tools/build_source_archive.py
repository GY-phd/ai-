# -*- coding: utf-8 -*-
"""Build the source-code archive for the submission.

Rules item 40 asks for the project source as a zip of at most 50 MB.  The
archive carries the *submission* and the *research* runtime, and it
excludes the development scratch that has no place in a deliverable:

* ``art/repo-inspect/`` and ``_tmp_rm/`` - read-only copies of a
  teammate's history kept for review; not part of this work.
* the offline rehearsal checkpoints (regenerable from the ledger).
* the Docker image tar (delivered separately, 174 MB).
* the working tree's stray scratch files (``_img*.txt`` and friends).

Usage::

    python tools/build_source_archive.py
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "提交材料_20261125" / "06_源码与依赖" / "智训引擎-源码包-v8.zip"

# Directories and files that are part of the deliverable.
INCLUDE_DIRS = [
    "contest_baselines",
    "contest_harness_sdk",
    "harness",
    "research",
    "train",
    "tools",
    "tests",
    "docs",
    "runs/harness-offline",
    "evidence",
    "artifacts/tracks",
]
INCLUDE_FILES = [
    "contest_submission.py",
    "harness.py",
    "Dockerfile",
    "Dockerfile.research",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "MANIFEST.json",
    "run_harness.ps1",
    "run_harness.sh",
    "run_all.ps1",
    ".gitattributes",
]

# Never shipped.
EXCLUDE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "node_modules", "site-packages", ".ipynb_checkpoints",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}

# The Docker image and the scratch copies of the teammate's history are
# large and are not source.
EXCLUDE_PATHS = {
    "art/repo-inspect",
    "_tmp_rm",
    "art",
    "_tmp_base_valid_pred.npy",
    "_tmp_valid_y.npy",
    "_tmp_calib_run.txt",
    "_tmp_rule_run.txt",
    "zhixun-engine-linux-amd64.tar",
    "zhixun-engine-linux-amd64.tar.sha256",
    "contest-harness-sdk-1.0.2.tar.gz",
}


def _skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in EXCLUDE_PATHS:
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def _walk(base: Path):
    """Yield the files under ``base``, relative to the repository root."""
    for dirpath, dirnames, filenames in os.walk(base):
        current = Path(dirpath)
        dirnames[:] = [d for d in sorted(dirnames)
                       if d not in EXCLUDE_DIR_NAMES
                       and not (current / d).name.startswith(".")]
        for name in sorted(filenames):
            path = current / name
            if _skip(path):
                continue
            yield path


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    members = []
    for rel in INCLUDE_DIRS:
        base = ROOT / rel
        if not base.is_dir():
            print("  skip (missing): %s" % rel)
            continue
        for path in _walk(base):
            members.append(path)

    for rel in INCLUDE_FILES:
        path = ROOT / rel
        if path.is_file() and not _skip(path):
            members.append(path)

    seen = set()
    unique = []
    for path in members:
        arc = path.relative_to(ROOT).as_posix()
        if arc not in seen:
            seen.add(arc)
            unique.append((path, arc))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, arc in unique:
            # A fixed timestamp keeps the archive byte-reproducible.
            info = zipfile.ZipInfo(arc, date_time=(2026, 11, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            data = path.read_bytes()
            info.file_size = len(data)
            zf.writestr(info, data)

    size = OUT.stat().st_size
    print("wrote %s" % OUT)
    print("  files: %d" % len(unique))
    print("  size : %d bytes (%.2f MB)" % (size, size / 1048576.0))
    limit = 50 * 1024 * 1024
    print("  limit: 50 MB -> %s" % ("OK" if size <= limit else "EXCEEDED"))
    if size > limit:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

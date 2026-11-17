# -*- coding: utf-8 -*-
"""Regenerate ``MANIFEST.json``.

The manifest is the inventory of what the project ships: every file's
size and digest, and the explicit statement that no model weights are
included.  It is generated rather than hand-written so that it cannot
drift away from the tree it describes.

Run:

    python tools/refresh_manifest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_canonical import canonical_digest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The directories and files the project ships.
TREES = ("harness", "rulkit", "tracks", "tools", "research", "tests",
         "evidence", "docs", "runs", "figureuse", "contest_baselines",
         "contest_harness_sdk")
FILES = ("requirements.txt", "run_harness.sh", "run_all.ps1", "run_harness.ps1",
         "README.md", "LICENSE", "MANIFEST.json", "harness.py",
         "contest_submission.py", "Dockerfile", ".dockerignore", ".gitignore",
         ".gitattributes", "GIT_PUSH.md",
         "智训引擎-系统总思路.md", "基岩智序-完整调试过程.md",
         ".github/workflows/ci.yml")

# Files that must never be listed: the manifest itself, the caches and
# the version control metadata.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules",
             ".pytest_tmp", "dist", "_tmp_rm"}
SKIP_NAMES = {"MANIFEST.json"}

# Developer-only helpers that are kept out of the shipped image (see
# ``EXCLUDED_FROM_IMAGE`` in ``tools/build_docker_image.py``).  They
# reach out to the model provider and read the developer's own
# credentials, so they are not part of what the project ships.
SKIP_FILES = {
    "tools/check_pytrio_live.py",
    "tools/check_pytrio_profile.py",
}

# The model weights are deliberately absent.  The manifest says so, and
# the check below makes sure the statement stays true.
WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite", ".model", ".gguf",
}


def collect():
    """Every shipped file, as ``(relative path, size, digest)``."""
    items = []

    def add(path: Path):
        relative = path.relative_to(ROOT).as_posix()
        if relative in SKIP_NAMES:
            return
        if relative in SKIP_FILES:
            return
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            return
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            return
        data = path.read_bytes()
        items.append((relative, len(data),
                      canonical_digest(data)))

    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                add(path)
    for name in FILES:
        path = ROOT / name
        if path.is_file():
            add(path)
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="MANIFEST.json")
    args = parser.parse_args(argv)

    items = collect()
    weights = []
    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
                weights.append(path.relative_to(ROOT).as_posix())
    for name in FILES:
        path = ROOT / name
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
            weights.append(name)

    manifest = {
        "name": "zhixun-engine-b",
        "contains_model_weights": bool(weights),
        "total_bytes": sum(item[1] for item in items),
        "file_count": len(items),
        "note": "Weights are intentionally absent; the harness regenerates them "
                "from the data package at evaluation time.",
        "files": {
            relative: {"bytes": size, "sha256": digest}
            for relative, size, digest in sorted(items)
        },
    }
    if weights:
        manifest["note"] = ("WARNING: weight files are present: %s"
                             % ", ".join(sorted(weights)))
    if "dist" not in SKIP_DIRS:
        manifest["note"] += " The built Docker archive lives in dist/ and is not listed."

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("wrote %s (%d files, %d bytes, weights=%s)"
          % (out, manifest["file_count"], manifest["total_bytes"],
             manifest["contains_model_weights"]))
    return 1 if weights else 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Verify a prepared repository folder: no weights, complete evidence, sane size.

    python tools/verify_repo.py --dir dist/github

Exits non-zero when the folder is not safe to publish.  This is the check the
push guide asks the user to run before ``git push``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_canonical import canonical_digest  # noqa: E402

WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".hdf5", ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite",
    ".model", ".weights", ".mlmodel", ".gguf",
}
MAX_TOTAL_BYTES = 4 * 1024 * 1024
REQUIRED = ("README.md", "LICENSE", "requirements.txt", "MANIFEST.json")

# The version control metadata is not part of the payload: it is often
# larger than the repository itself, and it is never published.
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules"}


def main() -> int:
    """Check the folder and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=None,
        help="the folder to check.  Defaults to the folder this script lives in "
             "when that folder is itself a prepared repository (i.e. it has a "
             "MANIFEST.json), and to dist/github otherwise.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="only check that no model weight is present.  The runtime image "
             "holds a subset of the repository (the Dockerfile copies the code, "
             "the evidence and the docs, but not the .git metadata, the "
             "workflows or the push guide), so the full manifest comparison "
             "does not apply inside the image - the weight check does, and "
             "that is the one the image build cares about.",
    )
    args = parser.parse_args()

    if args.dir:
        root = Path(args.dir).resolve()
    else:
        # The script ships inside the package it is checking, so prefer its own
        # folder.  When it is run from the source tree instead, fall back to
        # the conventional output path.
        here = Path(__file__).resolve().parent.parent
        root = here if (here / "MANIFEST.json").is_file() else here / "dist" / "github"
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    problems = []
    for name in REQUIRED:
        if not (root / name).is_file():
            problems.append(f"missing required file: {name}")

    weights = []
    total = 0
    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files += 1
        total += path.stat().st_size
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            weights.append(str(path.relative_to(root)))

    if weights:
        problems.append(f"weight files present: {weights}")
    if total > MAX_TOTAL_BYTES:
        problems.append(f"payload {total} bytes exceeds {MAX_TOTAL_BYTES}")

    # The manifest must agree with what is actually on disk.
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contains_model_weights") is not False:
            problems.append("manifest does not declare contains_model_weights=false")
        if not args.weights_only:
            for rel, entry in (manifest.get("files") or {}).items():
                target = root / rel
                if not target.is_file():
                    problems.append(f"manifest lists a missing file: {rel}")
                    continue
                actual = canonical_digest(target.read_bytes())
                if actual != entry.get("sha256"):
                    problems.append(f"digest mismatch: {rel}")

    # The evidence trail is the whole point of the repository; check it is present.
    for rel in ("evidence/ledger.jsonl", "evidence/run_report.json"):
        if not (root / rel).is_file():
            problems.append(f"missing evidence: {rel}")

    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        return 1

    # The console on Windows defaults to GBK, which cannot render the CJK
    # summary; fall back to ASCII rather than raising UnicodeEncodeError.
    try:
        summary = f"OK  无权重文件 / {files} 个文件 / 总大小 {total / 1024:.1f} KB"
        print(summary)
    except UnicodeEncodeError:
        print(f"OK  no weight files / {files} files / {total / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Verify a prepared repository folder: no weights, complete evidence, sane size.

    python tools/verify_repo.py --dir dist/github

Exits non-zero when the folder is not safe to publish.  This is the check the
push guide asks the user to run before ``git push``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".hdf5", ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite",
    ".model", ".weights", ".mlmodel", ".gguf",
}
MAX_TOTAL_BYTES = 4 * 1024 * 1024
REQUIRED = ("README.md", "LICENSE", "requirements.txt", "MANIFEST.json")


def main() -> int:
    """Check the folder and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="dist/github")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
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
        for rel, entry in (manifest.get("files") or {}).items():
            target = root / rel
            if not target.is_file():
                problems.append(f"manifest lists a missing file: {rel}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
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

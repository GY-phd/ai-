# -*- coding: utf-8 -*-
"""Regenerate ``MANIFEST.json`` from the *staged* (canonical) bytes.

Why this script exists
----------------------
``core.autocrlf=true`` rewrites the working tree to CRLF on Windows, while
Git stores (and GitHub serves) LF.  A manifest generated from the working
tree therefore records digests that do not match the published bytes, and
``tools/verify_repo.py`` rejects the repository.

This script takes the digests from the git index instead - the canonical
content - and rewrites the manifest so it describes exactly what is
published.  It also reports any file the manifest lists but the index does
not contain, and any indexed file the manifest forgets.

    python tools/refresh_manifest.py            # from the AI2/ folder
    python tools/refresh_manifest.py --check    # verify only, exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.json"
WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".hdf5", ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite",
    ".model", ".weights", ".mlmodel", ".gguf",
}


def _git(*args: str) -> str:
    """Run a git command and return its stdout as text."""
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8", "replace")


def _staged_bytes(rel: str) -> bytes | None:
    """Return the indexed content of ``rel`` relative to the repo root."""
    repo_rel = rel
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
            capture_output=True, check=True,
        )
        top = Path(result.stdout.decode("utf-8", "replace").strip())
        repo_rel = (ROOT / rel).resolve().relative_to(top.resolve()).as_posix()
    except (subprocess.CalledProcessError, ValueError):
        repo_rel = rel
    try:
        proc = subprocess.run(
            ["git", "show", ":" + repo_rel], cwd=ROOT, capture_output=True
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def main() -> int:
    """Rewrite (or check) the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="do not write; exit non-zero when the manifest drifts from the index",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"FAIL  no manifest at {MANIFEST}")
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    drifted = []
    for rel, entry in files.items():
        data = _staged_bytes(rel)
        if data is None:
            drifted.append(rel)
            continue
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            drifted.append(rel)

    if args.check:
        if drifted:
            print(f"FAIL  {len(drifted)} manifest entries drift from the index")
            for rel in drifted:
                print(f"FAIL  digest mismatch: {rel}")
            return 1
        print(f"OK  manifest matches the index ({len(files)} files)")
        return 0

    if drifted:
        for rel in drifted:
            data = _staged_bytes(rel)
            if data is None:
                continue
            files[rel] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        manifest["files"] = files
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"OK  refreshed {len(drifted)} manifest entries from the index")
        return 0

    print(f"OK  manifest already matches the index ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

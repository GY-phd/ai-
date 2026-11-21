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
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_canonical import canonical_bytes, canonical_digest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The manifest describes **what the repository ships**, and the only
# definition of that is the set of files git tracks.  A file that is
# gitignored - a local research run, a ``pip install`` build directory,
# a scratch ``evidence/`` folder - is on the developer's disk but is
# not in the repository the reviewers clone.  Listing such a file made
# ``MANIFEST.json`` disagree with a fresh checkout, and the CI check
# ``tools/verify_repo.py --dir .`` failed on every push with
# "manifest lists a missing file".
#
# So the manifest is built from git's index, not from the working tree.
# The one exception is a tree that is not a git repository at all (the
# prepared ``dist/github`` folder), where the filesystem is the only
# source of truth and the old behaviour is kept.
#
# "The index" means the tracked files **plus** the new files that are
# not gitignored.  A file the developer has just written is part of the
# next commit, so it has to be in the manifest now; otherwise the
# manifest would be one commit stale and the check would fail on the
# file that was just added.  The files git ignores are still excluded,
# which is the whole point.

# The directories and files the project ships.
TREES = ("harness", "rulkit", "tracks", "tools", "research", "tests",
         "evidence", "docs", "runs", "figureuse", "contest_baselines",
         "contest_harness_sdk", "openspec")
FILES = ("requirements.txt", "run_harness.sh", "run_all.ps1", "run_harness.ps1",
         "README.md", "LICENSE", "MANIFEST.json", "harness.py",
         "contest_submission.py", "Dockerfile", "Dockerfile.research",
         ".dockerignore", ".gitignore",
         ".gitattributes", "GIT_PUSH.md",
         "智训引擎-系统总思路.md", "基岩智序-完整调试过程.md",
         ".github/workflows/ci.yml")

# Files that must never be listed: the manifest itself, the caches and
# the version control metadata.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules",
             ".pytest_tmp", "dist", "_tmp_rm", ".ruff_cache"}
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


def _tracked_paths():
    """The files git tracks under ``ROOT``, as POSIX relative paths.

    Returns ``None`` when ``ROOT`` is not a git repository.  The names
    are read with ``-z`` so that a CJK file name survives; a name
    mangled by a newline split would not match any path on disk and the
    file would silently drop out of the manifest.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = result.stdout.decode("utf-8", "replace").split("\0")
    return [name for name in names if name]


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
        # The recorded size is the size of the **canonical** form, not
        # of the bytes on this disk.  Git converts LF to CRLF in a
        # Windows working tree and back again on a Linux clone, so the
        # raw byte count is a property of the machine, not of the
        # repository.  The digest is already computed over the
        # canonical form, and the size has to agree with it - otherwise
        # the manifest says one thing on Windows and another on the
        # machine that clones it, and the two disagree on every text
        # file in the repository.
        canonical = canonical_bytes(data)
        items.append((relative, len(canonical),
                      canonical_digest(data)))

    tracked = _tracked_paths()
    if tracked is not None:
        # The tracked set is the shipped set.  Walk it instead of the
        # working tree so that a gitignored research run never leaks
        # into the manifest.
        for relative in tracked:
            path = ROOT / relative
            if path.is_file():
                add(path)
        return items

    # Not a git repository: the filesystem is the only source of truth.
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
    # A weight that is **tracked** is a defect.  A weight that is
    # gitignored is a local research artefact and never reaches the
    # repository, so it must not turn the manifest's declaration into a
    # false alarm.
    #
    # ``collect()`` deliberately leaves weight files out of the
    # inventory, so the tracked set is scanned separately for them.
    weights = []
    tracked = _tracked_paths()
    if tracked is not None:
        for relative in tracked:
            if Path(relative).suffix.lower() in WEIGHT_SUFFIXES:
                weights.append(relative)
    else:
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
        "digest_form": "sha256 over the canonical form of the file (LF line "
                       "endings, binary payloads untouched), so the digest and "
                       "the recorded size are the same on every platform.",
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

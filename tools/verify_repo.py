# -*- coding: utf-8 -*-
"""Verify a prepared repository folder: no weights, complete evidence, sane size.

    python tools/verify_repo.py --dir dist/github

Exits non-zero when the folder is not safe to publish.  This is the check the
push guide asks the user to run before ``git push``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_canonical import canonical_digest  # noqa: E402

WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".hdf5", ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite",
    ".model", ".weights", ".mlmodel", ".gguf",
}
MAX_TOTAL_BYTES = 12 * 1024 * 1024
REQUIRED = ("README.md", "LICENSE", "requirements.txt", "MANIFEST.json")

# The version control metadata is not part of the payload: it is often
# larger than the repository itself, and it is never published.
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules"}


def _tracked_weights(root: Path):
    """The weight files git is tracking under ``root``.

    Returns ``None`` when the folder is not a git repository - the check
    is then simply not applicable, and saying so is better than
    pretending the tree is clean.

    The names are read with ``-z`` so that a CJK file name survives; a
    name mangled by a newline split would be checked against the wrong
    suffix and a real weight would slip through.

    The set is the tracked files **plus** the new files that are not
    gitignored, because a weight that is about to be committed is just
    as much of a defect as one that already is.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [name for name in result.stdout.decode("utf-8", "replace").split("\0") if name]
    return [name for name in names
            if Path(name).suffix.lower() in WEIGHT_SUFFIXES]


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
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="check the folder as a **working repository** rather than as a "
             "prepared package.  A working tree holds the local research runs "
             "and their weights - that is what the weights are for - so the "
             "size and the on-disk weight checks do not apply to it.  What "
             "does apply is the check on the files **git tracks**, because "
             "that is what the reviewers clone.  Without this flag the "
             "script judges the folder as a package, which is the right "
             "question to ask of dist/github and the wrong one to ask of the "
             "repository root.",
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

    # A **working tree** is a different thing from a **package**.  The
    # working tree holds the local research runs and the weights those
    # runs produced - a gigabyte and a half of it - and that is the
    # point of having them.  What must never happen is that one of those
    # files reaches a **commit**, because that is what the reviewers
    # clone.
    #
    # So for a working tree the on-disk weight check and the size check
    # are not the question; the question is what git is tracking.  The
    # check on the tracked set is made for **both** kinds of tree, and
    # for a package it is the only one that matters.
    if not args.working_tree:
        if weights:
            problems.append(f"weight files present: {weights}")
        if total > MAX_TOTAL_BYTES:
            problems.append(f"payload {total} bytes exceeds {MAX_TOTAL_BYTES}")

    # The **committed** set is what the reviewers clone, and it is a
    # different question from "what is on this disk".  A working tree
    # holds the local research runs and their weights; a commit must not.
    # Git is the only thing that knows the difference, so the check goes
    # through it.
    tracked_weights = _tracked_weights(root)
    if tracked_weights is not None and tracked_weights:
        problems.append(f"weight files are tracked by git: {tracked_weights}")
    elif tracked_weights is None and not args.working_tree and not args.weights_only:
        # ``--weights-only`` is the mode used inside the runtime image, where
        # ``.git`` is deliberately absent.  The on-disk weight check above is
        # still active; the tracked-set question only makes sense in a clone.
        problems.append("the folder is not a git repository, so the tracked "
                        "set could not be checked")

    # The manifest must agree with what is actually on disk.  It describes
    # the **package**, so it is checked for a package and not for a
    # working tree: the working tree has more files than the package,
    # and that is normal.
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contains_model_weights") is not False:
            problems.append("manifest does not declare contains_model_weights=false")
        if not args.weights_only and not args.working_tree:
            missing = []
            mismatched = []
            for rel, entry in (manifest.get("files") or {}).items():
                target = root / rel
                if not target.is_file():
                    missing.append(rel)
                    continue
                actual = canonical_digest(target.read_bytes())
                if actual != entry.get("sha256"):
                    mismatched.append(rel)
            # A manifest that names files the tree does not have is
            # **stale**, not broken: it was generated from a working tree
            # that held files the repository does not ship.  The remedy
            # is to regenerate it, and saying so is far more useful than
            # printing one line per missing file.
            if missing:
                problems.append(
                    "manifest lists %d file(s) the repository does not "
                    "ship; it is stale - regenerate it with "
                    "`python tools/refresh_manifest.py` (first %d: %s)"
                    % (len(missing), min(len(missing), 5),
                       ", ".join(sorted(missing)[:5])))
            for rel in sorted(mismatched):
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

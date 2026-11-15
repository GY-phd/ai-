# -*- coding: utf-8 -*-
"""Build the submission bundle: code + config + run evidence, never weights.

    python tools/package_submission.py --workspace runs/harness-b --out dist

The archive is rejected (non-zero exit) if any weight-like file slips in, if the
harness package exceeds the size budget, or if the evidence trail is missing.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHT_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5", ".onnx", ".pkl"}
CODE_DIRS = ("harness", "rulkit", "tools", "research", "tests")
TOP_FILES = ("README.md", "requirements.txt", "run_harness.ps1", "run_all.ps1")
DOC_DIRS = ("docs",)
EVIDENCE = ("ledger.jsonl", "run_report.json", "swanlab.jsonl")
MAX_CODE_BYTES = 2 * 1024 * 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(workspace):
    files = []
    for name in TOP_FILES:
        path = ROOT / name
        if path.is_file():
            files.append((path, Path(name)))
    for directory in CODE_DIRS + DOC_DIRS:
        base = ROOT / directory
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, path.relative_to(ROOT)))
    for name in EVIDENCE:
        path = workspace / name
        if path.is_file():
            files.append((path, Path("evidence") / name))
    registry = workspace / "registry"
    if registry.is_dir():
        for path in sorted(registry.glob("*.json*")):
            files.append((path, Path("evidence/registry") / path.name))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="runs/harness-b")
    parser.add_argument("--out", default="dist")
    parser.add_argument("--name", default="zhixun-engine-b-submission")
    args = parser.parse_args()

    workspace = (ROOT / args.workspace).resolve()
    if not (workspace / "run_report.json").is_file():
        raise SystemExit(f"no run evidence under {workspace}; run the harness first")

    files = collect(workspace)
    weights = [str(rel) for path, rel in files if path.suffix.lower() in WEIGHT_SUFFIXES]
    if weights:
        raise SystemExit(f"refusing to package weight files: {weights}")

    code_bytes = sum(path.stat().st_size for path, rel in files if rel.parts[0] != "evidence")
    if code_bytes > MAX_CODE_BYTES:
        raise SystemExit(f"code payload {code_bytes} bytes exceeds the {MAX_CODE_BYTES} byte budget")

    manifest = {
        "name": args.name,
        "contains_model_weights": False,
        "code_bytes": code_bytes,
        "file_count": len(files),
        "planner_model_id": "Qwen3.8-27B",
        "evidence_dir": str(workspace),
        "files": {str(rel).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": digest(path)}
                  for path, rel in files},
    }

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{args.name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, rel in files:
            archive.write(path, str(Path(args.name) / rel).replace("\\", "/"))
        archive.writestr(str(Path(args.name) / "MANIFEST.json"),
                         json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    size = archive_path.stat().st_size
    print(json.dumps({"archive": str(archive_path), "bytes": size, "code_bytes": code_bytes,
                      "files": len(files), "contains_model_weights": False}, ensure_ascii=False, indent=2))
    print(f"SUBMISSION OK  {archive_path}  ({size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

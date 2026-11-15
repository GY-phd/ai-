"""Build a relocatable ZIP from the explicitly included delivery files."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"task_a_engine", "examples", "tests", "docs", "reports", "history", "tools", "wheels"}
FILES = {"README.md", "requirements.txt", "pyproject.toml", "BUILD_INFO.json"}


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("use a new output path to preserve an existing delivery")
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(ROOT).parts)
                   and (path.relative_to(ROOT).parts[0] in FOLDERS or path.name in FILES)
                   and path.suffix not in {".pyc", ".sqlite3"})
    checksums = {path.relative_to(ROOT).as_posix(): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in paths}
    checksum_data = (json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (ROOT / "FILE_MANIFEST.json").write_bytes(checksum_data)
    entries = [(path.relative_to(ROOT).as_posix(), path.read_bytes()) for path in paths]
    entries.append(("FILE_MANIFEST.json", checksum_data))
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(ROOT.name + "/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return {"file": str(output), "bytes": output.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(), "entries": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))

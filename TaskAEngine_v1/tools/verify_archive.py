"""Extract the delivery elsewhere, verify hashes, and run its engine tests."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile


def verify(archive, workdir):
    start = perf_counter()
    with TemporaryDirectory(prefix=".task-a-relocation-", dir=workdir) as temporary:
        target = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as bundle:
            if any(not (target / name).resolve().is_relative_to(target) for name in bundle.namelist()):
                raise ValueError("archive contains an escaping path")
            bundle.extractall(target)
        package = target / "TaskAEngine_v1"
        manifest = json.loads((package / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
        for name, expected in manifest.items():
            actual = "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"checksum mismatch: {name}")
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                                cwd=package, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return {"archive_sha256": "sha256:" + hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                "verified_files": len(manifest), "relocated_engine_tests_passed": True,
                "test_log": result.stdout + result.stderr, "wall_seconds": perf_counter() - start}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.archive, args.workdir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "test_log"}))

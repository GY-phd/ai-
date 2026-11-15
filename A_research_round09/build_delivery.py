"""Package the unchanged Engine and optional recipe; smoke-test the actual ZIP."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAME = "TaskAReplayRecipe_v1"


def main():
    destination = ROOT / (NAME + ".zip")
    files = {}
    with zipfile.ZipFile(ROOT / "TaskAEngine_v1.zip") as old:
        for member in old.infolist():
            if not member.is_dir():
                relative = member.filename
                if not relative.startswith("TaskAEngine_v1/"):
                    relative = "TaskAEngine_v1/" + relative
                files[relative] = old.read(member)
    selected = list(HERE.glob("*.py")) + list(HERE.glob("*.md"))
    selected += [HERE / name for name in ("requirements.txt", "costs.json", "agent_decision.json", "mock_run/summary.json")]
    selected += [path for path in (HERE / "run_01").rglob("*") if path.is_file()]
    selected += [path for path in (HERE / "verification").iterdir() if path.is_file()]
    for path in selected:
        files["A_research_round09/" + path.relative_to(HERE).as_posix()] = path.read_bytes()
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in sorted(files.items()):
            bundle.writestr(f"{NAME}/{name}", content)
        bundle.writestr(f"{NAME}/delivery_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with TemporaryDirectory(prefix=".zip-check-", dir=HERE) as temporary:
        target = Path(temporary).resolve()
        assert target.is_relative_to(HERE.resolve())
        with zipfile.ZipFile(destination) as bundle:
            assert bundle.testzip() is None
            assert all((target / member.filename).resolve().is_relative_to(target) for member in bundle.infolist())
            bundle.extractall(target)
        delivery = target / NAME
        for name, expected in manifest.items():
            assert hashlib.sha256((delivery / name).read_bytes()).hexdigest() == expected
        completed = subprocess.run([sys.executable, "-X", "utf8", "-B", str(delivery / "A_research_round09/mock_demo.py"),
                                    "--output", str(target / "mock")], cwd=delivery, text=True, encoding="utf-8", capture_output=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
        mock = json.loads((target / "mock/summary.json").read_text())
        assert mock["final_active_model"] == "stable" and len(mock["stages"]) == 3
    result = {"archive": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "bytes": destination.stat().st_size, "files_verified": len(files), "zip_crc_passed": True,
              "extracted_zip_three_stage_mock_passed": True, "default_model": "stable", "real_activation_performed": False}
    (HERE / "bundle_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

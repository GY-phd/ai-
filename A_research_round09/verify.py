"""Verify fixed experiments, the unchanged v1 bundle and portable integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

from recipe import ROOT, np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((HERE / "run_01/results.json").read_text())
    replay = json.loads((HERE / "run_replay_verified/results.json").read_text())
    excluded = {"cost", "update_cost", "inference_cost", "code_sha256", "checkpoints", "checkpoint_sha256", "parent_sha256"}

    def scientific(value):
        if isinstance(value, dict):
            return {k: scientific(v) for k, v in value.items() if k not in excluded}
        if isinstance(value, list):
            return [scientific(v) for v in value]
        return value

    # All splits, class metrics, confusion matrices and attribution statistics.
    assert scientific(original) == scientific(replay)
    same_digest = 0
    for name, expected in original["checkpoints"].items():
        a, b = HERE / "run_01" / name, HERE / "run_replay_verified" / name
        assert "sha256:" + sha(a) == expected
        same_digest += sha(a) == sha(b)
        with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
            assert left.files == right.files
            for key in left.files:
                if left[key].dtype.kind in "fiu":
                    np.testing.assert_allclose(left[key], right[key], rtol=1e-9, atol=1e-10)
                else:
                    np.testing.assert_array_equal(left[key], right[key])
    for name, expected in original["code_sha256"].items():
        assert "sha256:" + sha(HERE / "run_01/source" / name) == expected
    stable = sha(ROOT / "my-second-run/checkpoint.npz")
    assert stable == "2c2050b0b0d738b633e9edd1e0bd869a1a89d1254f8e02013125068ad6638ed5"
    archive = sha(ROOT / "TaskAEngine_v1.zip")
    assert archive == "8ff9effc8db11578c8b764a835c62177725e4f1750e462b51563200fd4809e8c"
    checked = 0
    with zipfile.ZipFile(ROOT / "TaskAEngine_v1.zip") as bundle:
        for entry in bundle.infolist():
            if not entry.is_dir():
                path = ROOT / entry.filename
                if not path.is_file():
                    path = ROOT / "TaskAEngine_v1" / entry.filename
                assert path.read_bytes() == bundle.read(entry), str(path)
                checked += 1
    runs = []
    for name, folder, test in (("recipe", HERE, "test_recipe.py"), ("engine_v1", ROOT / "TaskAEngine_v1", "test_engine.py")):
        start = "." if name == "recipe" else "tests"
        command = [sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", start, "-p", test, "-v"]
        completed = subprocess.run(command, cwd=folder, text=True, encoding="utf-8", capture_output=True, timeout=120)
        (args.output / f"{name}_tests.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        assert completed.returncode == 0, completed.stderr
        runs.append({"suite": name, "exit_code": completed.returncode,
                     "summary": [line for line in completed.stderr.splitlines() if line.startswith("Ran ")]})
    # An isolated copy contains only the existing Engine and new runtime files;
    # no round-08 scripts, historical experiments, Batch1 or Batch2 are available.
    with TemporaryDirectory(prefix=".portable-", dir=HERE) as temporary:
        target = Path(temporary)
        shutil.copytree(ROOT / "TaskAEngine_v1", target / "TaskAEngine_v1", ignore=shutil.ignore_patterns("__pycache__", ".engine-test-*"))
        runtime = target / "A_research_round09"
        runtime.mkdir()
        for name in ("engine_recipe.py", "recipe.py", "structured_model.py", "mock_demo.py"):
            shutil.copyfile(HERE / name, runtime / name)
        completed = subprocess.run([sys.executable, "-X", "utf8", "-B", str(runtime / "mock_demo.py"), "--output", str(target / "mock")],
                                   cwd=target, text=True, encoding="utf-8", capture_output=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
        (args.output / "portable_mock_summary.json").write_bytes((target / "mock/summary.json").read_bytes())
    result = {"scientific_records_exactly_reproduced": True, "checkpoints_verified": len(original["checkpoints"]),
              "checkpoint_digests_exactly_reproduced": same_digest, "original_sources_hash_verified": True,
              "stable_checkpoint_sha256": stable, "v1_zip_sha256": archive, "unchanged_v1_files_checked": checked,
              "test_suites": runs, "portable_three_stage_mock_passed": True, "default_active_model": "stable",
              "independent_real_activation_tested": False,
              "current_source_sha256": {p.name: sha(p) for p in HERE.glob("*.py")}}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "current_source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Verify the built archive the way the platform will see it.

The archive is opened, the ``/app`` tree is extracted *from the tar itself*
(not from the working directory), and the extracted tree is exercised with
the SDK's fixed entry point.  That catches the class of defect where the
working directory works but the shipped image does not.

The offline host speaks the same JSONL protocol as the platform and is
backed by the real public data packages and the real parent checkpoints,
so a pass here means the code path in the image is the one the platform
will run.

    python tools/verify_image_roundtrip.py --archive zhixun-engine-linux-amd64.tar
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contest_baselines.offline_host import package_root, start_frame  # noqa: E402


def extract_app(archive_path: Path, destination: Path) -> Path:
    """Extract the ``/app`` tree out of the archive's project layer."""
    with tarfile.open(archive_path, "r") as archive:
        for layer in archive.getnames():
            if not layer.endswith("/layer.tar"):
                continue
            handle = archive.extractfile(layer)
            with tarfile.open(fileobj=handle, mode="r|*") as inner:
                for member in inner:
                    if not member.isfile() or not member.name.startswith("app/"):
                        continue
                    relative = member.name[len("app/"):]
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as out:
                        shutil.copyfileobj(inner.extractfile(member), out)
    return destination


def check_layout(app: Path) -> list:
    """The layout the platform's inspection depends on."""
    problems = []
    for required in ("harness.py", "contest_submission.py"):
        if not (app / required).is_file():
            problems.append("missing /app/%s" % required)
    # A package that shadows an entry file makes the submission ambiguous.
    for shadow in ("harness",):
        if (app / shadow).is_dir():
            problems.append("/app/%s/ shadows /app/%s.py" % (shadow, shadow))
    # The research runtime must not travel in the submission.
    for research in ("rulkit", "tracks", "research", "tools", "tests", "evo"):
        if (app / research).is_dir():
            problems.append("/app/%s/ is research code and must not ship "
                            "in the submission image" % research)
    return problems


def run_describe(app: Path) -> tuple:
    """Run the SDK's fixed entry point in ``describe`` mode."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(app)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_APP_ROOT"] = str(app)
    env.pop("CONTEST_ARTIFACT_PATH", None)
    env.pop("CONTEST_PREDICTION_CONTEXT", None)
    result = subprocess.run(
        [sys.executable, str(app / "harness.py"), "describe"],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(app))
    return result.returncode, result.stdout, result.stderr


def run_round(app: Path, dataset_key: str) -> tuple:
    """Run one full round against the offline host, from the extracted tree."""
    sys.path.insert(0, str(ROOT))
    from contest_baselines.offline_host import OfflineHost

    work = Path(tempfile.mkdtemp()) / "work"
    work.mkdir(parents=True)
    frame = start_frame(dataset_key, 1)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(app)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_APP_ROOT"] = str(app)
    env["CONTEST_WORK"] = str(work)
    env["CONTEST_PARENT_ROOT"] = str(package_root(dataset_key))
    env.pop("CONTEST_ARTIFACT_PATH", None)
    env.pop("CONTEST_PREDICTION_CONTEXT", None)
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        [sys.executable, str(app / "harness.py"), "train"],
        cwd=str(app), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8")
    host = OfflineHost(frame)
    thread = host.serve(process)
    thread.join(timeout=1800)
    stdout = "".join(host.stdout_lines)
    code = process.wait(timeout=60)
    stderr = process.stderr.read()
    process.stderr.close()
    return code, stdout, stderr, host, work


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="zhixun-engine-linux-amd64.tar")
    parser.add_argument("--datasets", default="A,B,C")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path

    problems = []
    temporary = Path(tempfile.mkdtemp())
    app = extract_app(archive_path, temporary / "app")
    print("extracted %d file(s) from the archive's /app" % (
        sum(1 for _ in app.rglob("*") if _.is_file())))

    problems.extend(check_layout(app))
    if problems:
        for problem in problems:
            print("FAIL  %s" % problem)
        shutil.rmtree(temporary, ignore_errors=True)
        return 1
    print("OK    layout: one entry point, no shadowing package, no research code")

    code, stdout, stderr = run_describe(app)
    if code != 0:
        print("FAIL  describe exited %d\n%s" % (code, stderr[-1500:]))
        shutil.rmtree(temporary, ignore_errors=True)
        return 1
    try:
        contract = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        print("FAIL  describe did not answer a JSON contract: %s" % exc)
        shutil.rmtree(temporary, ignore_errors=True)
        return 1
    if contract.get("protocol") != "contest.harness.workspace.v2":
        print("FAIL  describe reports protocol %r" % contract.get("protocol"))
        shutil.rmtree(temporary, ignore_errors=True)
        return 1
    if sorted(contract.get("tasks") or []) != ["classification", "regression"]:
        print("FAIL  describe reports tasks %r" % contract.get("tasks"))
        shutil.rmtree(temporary, ignore_errors=True)
        return 1
    print("OK    describe: %s" % json.dumps(contract, ensure_ascii=False))

    for dataset_key in [k.strip() for k in args.datasets.split(",") if k.strip()]:
        code, stdout, stderr, host, work = run_round(app, dataset_key)
        if code != 0:
            print("FAIL  %s train exited %d\n%s" % (dataset_key, code, stderr[-1500:]))
            shutil.rmtree(temporary, ignore_errors=True)
            return 1
        if not host.model_requests:
            print("FAIL  %s made no model call during training" % dataset_key)
            shutil.rmtree(temporary, ignore_errors=True)
            return 1
        if host.candidate is None:
            print("FAIL  %s submitted no candidate" % dataset_key)
            shutil.rmtree(temporary, ignore_errors=True)
            return 1
        if host.finished is None:
            print("FAIL  %s did not finish the round" % dataset_key)
            shutil.rmtree(temporary, ignore_errors=True)
            return 1
        measured = [line for line in stderr.splitlines() if line.startswith("[contest]")]
        summary = [line for line in measured if "candidate" in line and "=" in line]
        print("OK    %s train: %s | %d model call(s) | candidate submitted"
              % (dataset_key, summary[0].strip() if summary else "round finished",
                 len(host.model_requests)))

        # The prediction path, from the extracted tree as well.
        checkpoints = sorted(work.rglob("checkpoint.npz"))
        if not checkpoints:
            print("FAIL  %s produced no checkpoint to predict with" % dataset_key)
            shutil.rmtree(temporary, ignore_errors=True)
            return 1
        task = "classification" if dataset_key == "A" else "regression"
        row = json.loads((package_root(dataset_key) / "public" / "validation.jsonl")
                         .read_text(encoding="utf-8").splitlines()[0])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(app)
        env["CONTEST_APP_ROOT"] = str(app)
        env["CONTEST_ARTIFACT_PATH"] = str(checkpoints[0])
        env.pop("CONTEST_PREDICTION_CONTEXT", None)
        request = json.dumps({"task": task, "inputs": row["inputs"]})
        result = subprocess.run(
            [sys.executable, str(app / "harness.py"), "predict"],
            input=request + "\n", capture_output=True, text=True,
            encoding="utf-8", env=env, cwd=str(app))
        if result.returncode != 0:
            print("FAIL  %s predict exited %d\n%s"
                  % (dataset_key, result.returncode, result.stderr[-1500:]))
            shutil.rmtree(temporary, ignore_errors=True)
            return 1
        body = json.loads(result.stdout.strip().splitlines()[-1])
        if task == "classification":
            if not isinstance(body.get("outputs"), list) or not body["outputs"]:
                print("FAIL  %s predict did not return a label list" % dataset_key)
                shutil.rmtree(temporary, ignore_errors=True)
                return 1
            shape = "classification -> %d labels" % len(body["outputs"])
        else:
            if not isinstance(body.get("outputs"), dict) or \
                    "value" not in body["outputs"]:
                print("FAIL  %s predict did not return a value" % dataset_key)
                shutil.rmtree(temporary, ignore_errors=True)
                return 1
            shape = "regression -> value=%s" % round(float(body["outputs"]["value"]), 4)
        print("OK    %s predict: %s" % (dataset_key, shape))

    shutil.rmtree(temporary, ignore_errors=True)
    print()
    print("OK    the archive's /app answers describe, train and predict for %s"
          % args.datasets)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""The offline host stand-in: run a whole round without the platform.

The SDK's own tests prove that the SDK talks the protocol.  They do not
prove that *this* participant's round works: that the parent is read, that
the hypothesis is applied, that the candidate is measured, that the
decision is taken, and that the submission and the finish frame are
well-formed.

This module is a minimal host that speaks the same JSONL protocol, backed
by the **real** public data packages and the **real** parent checkpoints
that ship on this machine.  It runs the participant's own ``train`` and
``predict`` functions, so a round that passes here is a round whose code
path is the same one the platform will execute.

It does **not** call a real model: the stand-in answers the model request
with a canned review.  That is the same trade-off the SDK's own tests
make, and it is the reason the platform insists that passing the offline
tests does not mean the submission will score.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .adapters import get as get_adapter

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- data
def package_root(dataset_key):
    """Where the local copy of a data package lives."""
    configured = os.environ.get("CONTEST_PACKAGES")
    if configured:
        return Path(configured) / {"A": "cpu-a-guide-20261110-v1",
                                   "B": "cpu-b-rul-20261110-v1",
                                   "C": "cpu-c-guide-20261110-v1"}[dataset_key]
    return Path(r"D:\GY\cpu") / {"A": "cpu-a-guide-20261110-v1",
                                 "B": "cpu-b-rul-20261110-v1",
                                 "C": "cpu-c-guide-20261110-v1"}[dataset_key]


def start_frame(dataset_key, round_index=1, parent_root=None, parent_artifact_id=None,
                parent_checkpoint=None):
    """Build the ``start`` frame the platform would send.

    ``parent_root`` overrides where the parent artefact is mounted, and
    ``parent_checkpoint`` names the checkpoint inside it.  The offline
    harness uses both to **advance the parent** from round to round, the
    way the platform does when it adopts a candidate.

    The **public projection always comes from the data package**, never
    from the parent artefact: on the platform the two are separate mounts
    (``/data`` and ``/previous``), and the parent artefact holds only
    the checkpoint and the code.
    """
    package = package_root(dataset_key)
    train = package / "public" / "train.jsonl"
    validation = package / "public" / "validation.jsonl"
    checkpoint = Path(parent_checkpoint) if parent_checkpoint else (package / "checkpoint.npz")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # The task and the class list come from the profile's adapter, so the
    # offline host cannot drift away from the submission's own view of
    # what a profile is.
    adapter = get_adapter(dataset_key)
    classes = list(adapter.expected_classes()) if adapter.task == "classification" else []
    task = adapter.task
    parent = {
        "artifact_id": parent_artifact_id or ("baseline-%s" % dataset_key),
        "artifact_sha256": digest,
        "artifact": {
            "checkpoint_paths": ["checkpoint.npz"],
            "code_paths": [],
            "predict_argv": ["python3", "/app/harness.py", "predict"],
        },
        "files": [{"path": "checkpoint.npz", "sha256": digest,
                   "size_bytes": checkpoint.stat().st_size}],
    }
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "offline-%s" % dataset_key,
        "dataset_key": dataset_key,
        "round_index": round_index,
        "stage_id": "offline-test",
        "prompt": "OFFLINE TEST PROMPT",
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "parent_checkpoint_sha256": digest,
        "task_contract": {
            "dataset_key": dataset_key,
            "task": task,
            "metric": "macro_f1" if task == "classification" else "rmse",
            "metric_direction": "higher" if task == "classification" else "lower",
            # The frozen label list for a classification profile.  The
            # parent checkpoint's own list stays authoritative, but the
            # contract states it so the frame is self-describing.
            "classes": classes,
            "public_projection": {
                "train_path": str(train),
                "validation_path": str(validation),
                "row_fields": ["id", "inputs", "answer"],
                "source_metadata": False,
            },
        },
        "model_name": "base:offline-test-model",
        "parent_artifact": parent,
        "artifacts": [parent],
        "state_sha256": "",
        "state_path": None,
        "max_output_tokens": 512,
        "resource_limits": {"candidate_upload_bytes_per_round": 64 * 1024 * 1024},
        "current_validation": {},
        "wall_seconds": 600,
    }


class OfflineHost:
    """A minimal host: answers the protocol frames, records what it saw."""

    def __init__(self, start, model_reply=None):
        self.start = start
        self.model_reply = model_reply or (
            "The measurement is consistent with the hypothesis: the primary metric "
            "moved in the predicted direction.")
        self.records = []
        self.model_requests = []
        self.candidate = None
        self.state_files = []
        self.finished = None
        self.failure = None
        # The host thread is the *only* reader of the participant's stdout.
        # Anything else that wants the transcript reads it from here.
        self.stdout_lines = []

    def serve(self, process):
        thread = threading.Thread(target=self._serve, args=(process,), daemon=True)
        thread.start()
        return thread

    def _serve(self, process):
        try:
            self._loop(process)
        except Exception as exc:  # pragma: no cover - a test failure path
            self.failure = exc
            try:
                process.kill()
            except OSError:
                pass

    def _loop(self, process):
        process.stdin.write(json.dumps(self.start, ensure_ascii=False) + "\n")
        process.stdin.flush()
        sequence = 0
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("the participant closed stdout")
            self.stdout_lines.append(line)
            frame = json.loads(line)
            kind = frame.get("type")
            if kind == "round_failed":
                raise RuntimeError("the participant failed: %s" % frame.get("error"))
            if kind == "round_complete":
                self.finished = frame["record"]
                process.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                return
            if frame.get("id") != sequence:
                raise RuntimeError("rpc sequence mismatch: %s" % frame.get("id"))
            sequence += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": []}
            elif kind == "model_request":
                self.model_requests.append(frame.get("messages"))
                reply = {"type": "model_response", "id": frame["id"], "call_id": "offline-call-1",
                         "response": {"request_id": "offline-call-1",
                                      "id": "chatcmpl-offline",
                                      "choices": [{"message": {"role": "assistant",
                                                               "content": self.model_reply}}]}}
            elif kind == "state_request":
                self.state_files = frame.get("files") or []
                reply = {"type": "state_response", "id": frame["id"], "state_sha256": "sha256:" + "22" * 32}
            elif kind == "candidate_request":
                self.candidate = frame
                reply = {"type": "candidate_response", "id": frame["id"],
                         "artifact_id": "offline-artifact-1", "summary": {"metrics": {}}}
            else:
                raise RuntimeError("unexpected frame %r" % kind)
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()


def run_entrypoint(command, dataset_key, work, *, round_index=1, entry=None):
    """Run ``/app/harness.py <command>`` against the offline host.

    Returns ``(returncode, stdout, stderr, host)``.
    """
    entry = entry or (ROOT / "harness.py")
    frame = start_frame(dataset_key, round_index)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "contest_baselines")
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work)
    env["CONTEST_APP_ROOT"] = str(ROOT)
    # The platform mounts the parent artefact at ``/previous``.  The offline
    # harness points that mount at the local copy of the package.
    env["CONTEST_PARENT_ROOT"] = str(package_root(dataset_key))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("CONTEST_ARTIFACT_PATH", None)
    env.pop("CONTEST_PREDICTION_CONTEXT", None)
    work.mkdir(parents=True, exist_ok=True)
    # stderr goes to a file: a full stderr pipe would block the participant
    # while this function is busy reading stdout.
    stderr_path = work.parent / "stderr.txt"
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(entry), command],
        cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr_handle, text=True, encoding="utf-8",
    )
    host = OfflineHost(frame)
    thread = host.serve(process)
    thread.join(timeout=1800)
    stdout = "".join(host.stdout_lines)
    code = process.wait(timeout=60)
    stderr_handle.close()
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return code, stdout, stderr, host

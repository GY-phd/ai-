# -*- coding: utf-8 -*-
"""Run the whole Harness integration offline and write the record.

The competition asks for at least three complete
"read the history - run an experiment - decide from the result" records
per dataset.  This script produces them **offline**, against the real
public packages and the real parent checkpoints, so the evidence exists
before the first trial run on the platform.

It runs the participant's own ``train`` function through the SDK, once per
round, with the host stand-in carrying the previous round's ledger forward
as the history.  That is exactly the loop the platform runs; only the
model call is answered locally.

Run:

    python tools/run_harness_offline.py --rounds 3 --out runs/harness-offline
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines.offline_host import OfflineHost, package_root, start_frame  # noqa: E402

DATASETS = ("A", "B", "C")


class _HistoryHost(OfflineHost):
    """A host that carries the previous rounds' ledger forward.

    The platform hands the participant the finished checkpoints of this
    dataset.  The stand-in does the same: the previous rounds' ledger
    entries become the history, so the round's hypothesis choice really
    depends on what has already been tried.
    """

    def __init__(self, start, history, model_reply=None):
        super().__init__(start, model_reply=model_reply)
        self.history = list(history)

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
                raise RuntimeError("rpc sequence mismatch")
            sequence += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": self.history}
            elif kind == "model_request":
                self.model_requests.append(frame.get("messages"))
                reply = {"type": "model_response", "id": frame["id"], "call_id": "offline-call-%d" % len(self.model_requests),
                         "response": {"request_id": "offline-call-%d" % len(self.model_requests),
                                      "id": "chatcmpl-offline",
                                      "choices": [{"message": {"role": "assistant",
                                                               "content": self.model_reply}}]}}
            elif kind == "state_request":
                self.state_files = frame.get("files") or []
                reply = {"type": "state_response", "id": frame["id"],
                         "state_sha256": "sha256:" + "%02x" % (33 + len(self.history)) * 32}
            elif kind == "candidate_request":
                self.candidate = frame
                reply = {"type": "candidate_response", "id": frame["id"],
                         "artifact_id": "offline-artifact-%d" % (len(self.history) + 1),
                         "summary": {"metrics": {}}}
            else:
                raise RuntimeError("unexpected frame %r" % kind)
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()


def _run_round(dataset_key, round_index, history, work):
    """Run one round and return the host's view of it."""
    frame = start_frame(dataset_key, round_index)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work)
    env["CONTEST_APP_ROOT"] = str(ROOT)
    env["CONTEST_PARENT_ROOT"] = str(package_root(dataset_key))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    work.mkdir(parents=True, exist_ok=True)
    stderr_path = work.parent / ("stderr-round-%d.txt" % round_index)
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "harness.py"), "train"],
        cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr_handle, text=True, encoding="utf-8",
    )
    host = _HistoryHost(frame, history)
    thread = threading.Thread(target=host.serve, args=(process,), daemon=True)
    thread.start()
    thread.join(timeout=3600)
    code = process.wait(timeout=60)
    stderr_handle.close()
    host.exit_code = code
    host.stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return host


def _ledger_entry(host, dataset_key, round_index):
    """The history record this round leaves behind."""
    record = host.finished or {}
    configuration = record.get("configuration") or {}
    return {
        "id": "offline-%s-round-%d" % (dataset_key, round_index),
        "round_index": round_index,
        "dataset_key": dataset_key,
        "task": configuration.get("task"),
        "hypothesis": configuration.get("hypothesis"),
        "adopted": configuration.get("adopted"),
        "validation_before": configuration.get("validation_before"),
        "validation_after": configuration.get("validation_after"),
        "selected_artifact_id": record.get("selected_artifact_id"),
        "model_call_ids": record.get("model_call_ids"),
        "reason": record.get("reason"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3,
                        help="how many rounds to run per dataset")
    parser.add_argument("--datasets", default="A,B,C",
                        help="which datasets to run")
    parser.add_argument("--out", default="runs/harness-offline",
                        help="where to write the record")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    keys = [key.strip() for key in args.datasets.split(",") if key.strip()]

    report = {"format": "contest.harness-offline-report.v1",
              "rounds_per_dataset": args.rounds,
              "note": "This is an offline rehearsal of the platform's loop. The "
                      "model call is answered by a local stand-in, so it proves "
                      "the code path and not the score.",
              "datasets": {}}

    for key in keys:
        history = []
        rounds = []
        for round_index in range(1, args.rounds + 1):
            work = Path(tempfile.mkdtemp(prefix="harness-%s-r%d-" % (key, round_index)))
            started = time.monotonic()
            host = _run_round(key, round_index, history, work / "work")
            if host.failure is not None:
                raise SystemExit("round %d of %s failed: %s\n%s"
                                 % (round_index, key, host.failure, host.stderr_text))
            entry = _ledger_entry(host, key, round_index)
            history.append(entry)
            rounds.append({
                "round": round_index,
                "exit_code": host.exit_code,
                "hypothesis": entry["hypothesis"],
                "adopted": entry["adopted"],
                "validation_before": entry["validation_before"],
                "validation_after": entry["validation_after"],
                "selected_artifact_id": entry["selected_artifact_id"],
                "model_call_ids": entry["model_call_ids"],
                "seconds": round(time.monotonic() - started, 2),
                "submitted": host.candidate is not None,
            })
            print("%s round %d: %s adopted=%s %s"
                  % (key, round_index, entry["hypothesis"], entry["adopted"],
                     _delta(entry)))
            shutil.rmtree(work, ignore_errors=True)
        report["datasets"][key] = {"rounds": rounds}

    path = out / "run_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")
    print("wrote %s" % path)

    # The ledger, in the same shape the platform's state carries.
    with (out / "ledger.jsonl").open("w", encoding="utf-8") as stream:
        for key in keys:
            for record in report["datasets"][key]["rounds"]:
                stream.write(json.dumps({
                    "dataset_key": key,
                    "round": record["round"],
                    "hypothesis": record["hypothesis"],
                    "adopted": record["adopted"],
                    "validation_before": record["validation_before"],
                    "validation_after": record["validation_after"],
                    "selected_artifact_id": record["selected_artifact_id"],
                    "model_call_ids": record["model_call_ids"],
                }, ensure_ascii=False, allow_nan=False) + "\n")
    print("wrote %s" % (out / "ledger.jsonl"))
    return 0


def _delta(entry):
    """A short before/after summary of one round."""
    before = entry.get("validation_before") or {}
    after = entry.get("validation_after") or {}
    name = "macro_f1" if "macro_f1" in before else "rmse"
    if name not in before or name not in after:
        return ""
    return "%s %.4f -> %.4f" % (name, before[name], after[name])


if __name__ == "__main__":
    raise SystemExit(main())

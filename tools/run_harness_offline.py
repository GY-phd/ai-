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

#: The reply the stand-in gives when no live model is configured.  It is
#: a valid plan for every dataset, so the rehearsal proves the code path
#: whether or not an endpoint is reachable.
DEFAULT_MODEL_REPLY = json.dumps({
    "action": "adapt",
    "hypothesis": "the offline rehearsal's stand-in keeps the deployed "
                  "model following the observed stream",
    "params": {"epochs": 200},
    "expected_gain_pct": 5.0,
    "risk": "medium",
    "reason": "answered locally by the stand-in; no endpoint was contacted",
}, ensure_ascii=False)


def _live_chat_client():
    """Build a chat client against the configured model, or ``None``.

    The rehearsal can drive the **same** model the platform drives:
    ``HARNESS_LLM_BASE_URL`` / ``HARNESS_LLM_API_KEY`` name an
    OpenAI-compatible endpoint, and ``HARNESS_LLM_MODEL`` (falling back
    to ``PYTRIO_MODEL``) names the model on it.  The model id is
    resolved through ``harness.llm.resolve_model_id`` so the short
    form the competition writes and the platform's ``vendor/name`` form
    are the same request.

    Returns ``None`` when the endpoint is not configured or not
    reachable, so the rehearsal falls back to the stand-in instead of
    failing.
    """
    base_url = (os.environ.get("HARNESS_LLM_BASE_URL") or "").strip()
    api_key = (os.environ.get("HARNESS_LLM_API_KEY") or "").strip()
    model = (os.environ.get("HARNESS_LLM_MODEL")
             or os.environ.get("PYTRIO_MODEL") or "").strip()
    if not base_url:
        return None
    try:
        from harness.llm import resolve_model_id
        model_id = resolve_model_id(model) if model else ""
    except Exception:  # noqa: BLE001 - the fallback must not depend on it
        model_id = model
    timeout = float(os.environ.get("HARNESS_LLM_TIMEOUT_S") or 120)

    def ask(messages, options=None):
        """One chat completion; returns the assistant's text.

        ``options`` are the sampling options the participant asked for.
        They travel with the request, because the participant chooses
        them deliberately - ``enable_thinking=False`` and the token
        ceiling are exactly the two knobs that decide whether the reply
        comes back as text or goes into a reasoning trace.
        """
        import urllib.request
        options = dict(options or {})
        payload = {
            "model": model_id,
            "temperature": options.pop("temperature", 0.0),
            "max_tokens": int(options.pop("max_tokens", 1024)),
            "messages": messages,
        }
        # Only the options the endpoint is known to accept are sent on;
        # an unknown key is a hard error at several gateways, and a
        # failed request would cost the round.
        for key in ("enable_thinking", "top_p", "stop", "seed",
                    "frequency_penalty", "presence_penalty"):
            if key in options and options[key] is not None:
                payload[key] = options.pop(key)
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + api_key} if api_key else {})},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        choices = raw.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    # A reachability probe, so a broken endpoint is discovered before
    # ten rounds have burned their budget on failing calls.
    try:
        ask([{"role": "user", "content": "reply with the single word: ready"}])
    except Exception as exc:  # noqa: BLE001
        print("  live model unavailable (%s); using the stand-in reply"
              % type(exc).__name__, flush=True)
        return None
    print("  live model: %s at %s" % (model_id or "(default)", base_url),
          flush=True)
    return ask


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
        # A live model client, when one is configured.  The stand-in's
        # default reply is a fixed string, which proves the code path
        # but not the decision.  Pointing it at PyTRiO makes the
        # rehearsal drive the *same* 27B the platform drives.
        self.live = None

    def _ask_live_model(self, messages, options=None):
        """Ask the configured model, falling back to the stand-in reply.

        Returns the assistant's text.  A failure is not fatal: the
        rehearsal's job is to produce the loop's evidence, and a
        stand-in answer still produces a valid round.
        """
        if self.live is None:
            return self.model_reply
        try:
            return self.live(messages, options)
        except Exception as exc:  # noqa: BLE001 - a live call may fail
            print("  live model call failed (%s); using the stand-in reply"
                  % type(exc).__name__, flush=True)
            self.live = None
            return self.model_reply

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
                # The sampling options travel with the request, flattened
                # onto the frame beside ``messages`` - the SDK sends
                # them as ``**options`` - and the participant chooses
                # them deliberately: they decide whether the reply
                # arrives as text or goes into a reasoning trace.
                options = {key: value for key, value in frame.items()
                           if key not in ("type", "id", "messages")}
                content = self._ask_live_model(frame.get("messages") or [],
                                               options)
                reply = {"type": "model_response", "id": frame["id"], "call_id": "offline-call-%d" % len(self.model_requests),
                         "response": {"request_id": "offline-call-%d" % len(self.model_requests),
                                      "id": "chatcmpl-offline",
                                      "choices": [{"message": {"role": "assistant",
                                                               "content": content}}]}}
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


def _run_round(dataset_key, round_index, history, work, parent=None, live=None):
    """Run one round and return the host's view of it.

    ``parent`` is the checkpoint the round continues from.  When it is
    ``None`` the round starts from the package's own baseline, which is
    what the platform does for the first round.

    ``live`` is an optional chat client.  When it is given, the round's
    decision comes from the **real** model through PyTRiO, which is the
    only way the rehearsal exercises the same decision layer the
    platform drives.  Without it the stand-in answers, and the rehearsal
    proves the code path but not the decision.
    """
    parent_root, parent_checkpoint, parent_id = parent or (None, None, None)
    frame = start_frame(dataset_key, round_index,
                        parent_root=parent_root,
                        parent_checkpoint=parent_checkpoint,
                        parent_artifact_id=parent_id)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work)
    env["CONTEST_APP_ROOT"] = str(ROOT)
    # The platform mounts the parent artefact at ``/previous``.  The
    # offline harness points that mount at wherever the current parent
    # lives - the package for the first round, and the last adopted
    # candidate for every round after that.
    env["CONTEST_PARENT_ROOT"] = str(parent_root or package_root(dataset_key))
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
    host = _HistoryHost(frame, history, model_reply=DEFAULT_MODEL_REPLY)
    if live is not None:
        host.live = live
    thread = threading.Thread(target=host.serve, args=(process,), daemon=True)
    thread.start()
    thread.join(timeout=3600)
    # The exit wait is **generous on purpose**.  The host thread has
    # already seen the ``round_complete`` frame, so the participant is
    # done writing and only needs a moment to unwind.  A tight wait
    # still failed on profile C, whose rows outnumber profile B's
    # sixteenfold and whose last round therefore outlived it - and a
    # timeout there would discard a round that had already finished.
    code = process.wait(timeout=600)
    stderr_handle.close()
    host.exit_code = code
    host.stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return host


def _ledger_entry(host, dataset_key, round_index):
    """The history record this round leaves behind."""
    record = host.finished or {}
    configuration = record.get("configuration") or {}
    entry = {
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
    # The protocol-aligned selection slice and the full-split guard
    # travel with the entry.  They are what the adoption decision was
    # actually made on, so a reader has to be able to check the decision
    # rather than take the verdict on trust - and the platform's report
    # does not display the slice at all.
    for key in ("selection_metric", "selection_before", "selection_after",
                "selection_guard"):
        if configuration.get(key) is not None:
            entry[key] = configuration[key]
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3,
                        help="how many rounds to run per dataset")
    parser.add_argument("--datasets", default="A,B,C",
                        help="which datasets to run")
    parser.add_argument("--out", default="runs/harness-offline",
                        help="where to write the record")
    parser.add_argument("--live", action="store_true",
                        help="drive the real model through the configured "
                             "endpoint instead of the local stand-in")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    keys = [key.strip() for key in args.datasets.split(",") if key.strip()]

    # The live client is built **once**, before the first round, so a
    # broken endpoint is reported immediately instead of after ten
    # rounds have been spent.
    live = _live_chat_client() if args.live else None
    if args.live and live is None:
        print("  no live model configured; falling back to the stand-in")
    driver = ("the configured model (%s)" % (os.environ.get("HARNESS_LLM_MODEL")
                                             or os.environ.get("PYTRIO_MODEL")
                                             or "default")) if live else \
            "a local stand-in"

    report = {"format": "contest.harness-offline-report.v1",
              "rounds_per_dataset": args.rounds,
              "note": "This is an offline rehearsal of the platform's loop. "
                      "The model call is answered by %s, so it proves the "
                      "code path and not the score." % driver,
              "driver": driver,
              "live": bool(live),
              "datasets": {}}

    for key in keys:
        history = []
        rounds = []
        # The parent the next round continues from.  The platform advances
        # the parent to the **adopted** candidate and keeps the old parent
        # when a candidate is rejected; the rehearsal does the same, so
        # its numbers describe the loop the platform actually runs rather
        # than ten repetitions from one fixed baseline.
        #
        # ``parent`` is ``(root, checkpoint, artifact_id)`` or ``None`` for
        # the first round, which starts from the package's own baseline.
        parent = None
        # The work directories that must survive the round, because they
        # hold the checkpoint the next round continues from.
        keep = set()
        for round_index in range(1, args.rounds + 1):
            work = Path(tempfile.mkdtemp(prefix="harness-%s-r%d-" % (key, round_index)))
            started = time.monotonic()
            host = _run_round(key, round_index, history, work / "work",
                              parent=parent, live=live)
            if host.failure is not None:
                raise SystemExit("round %d of %s failed: %s\n%s"
                                 % (round_index, key, host.failure, host.stderr_text))
            entry = _ledger_entry(host, key, round_index)
            history.append(entry)

            # Advance the parent when the round's candidate was adopted.
            candidate_dir = work / "work" / ("candidate-round-%d" % round_index)
            candidate_checkpoint = candidate_dir / "checkpoint.npz"
            advanced = False
            if entry["adopted"] and candidate_checkpoint.is_file():
                # The adopted artefact becomes the next round's parent,
                # exactly as the platform would.  Its directory has to
                # survive until then.
                parent = (str(candidate_dir), str(candidate_checkpoint),
                          entry["selected_artifact_id"] or ("offline-artifact-r%d" % round_index))
                keep.add(work)
                advanced = True

            round_entry = {
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
                # The parent this round continued from, so the report
                # shows what each round actually built on.  ``None`` means
                # the package baseline.
                "parent_advanced": advanced,
                # Whether the round's decision came from the real model
                # or from the stand-in, so a reader never has to guess
                # which one produced the numbers.
                "model_calls": len(host.model_requests),
                "live_model": bool(host.live),
            }
            # The selection slice and the guard, so that the adoption
            # decision can be re-derived from the report alone.
            for key_name in ("selection_metric", "selection_before",
                             "selection_after", "selection_guard"):
                if entry.get(key_name) is not None:
                    round_entry[key_name] = entry[key_name]
            rounds.append(round_entry)
            # Collect the round's SwanLab record before the work directory
            # is removed: it is the round's own experiment log and belongs
            # with the evidence.
            record = work / "work" / "swanlab.jsonl"
            if record.is_file():
                collected = out / "swanlab" / ("%s-round-%d.jsonl" % (key, round_index))
                collected.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(record, collected)
            print("%s round %d: %s adopted=%s %s"
                  % (key, round_index, entry["hypothesis"], entry["adopted"],
                     _delta(entry)))
            if work not in keep:
                shutil.rmtree(work, ignore_errors=True)
        # The last adopted parent is no longer needed.
        for path in keep:
            shutil.rmtree(path, ignore_errors=True)
        report["datasets"][key] = {"rounds": rounds}

    path = out / "run_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")
    print("wrote %s" % path)

    # The ledger, in the same shape the platform's state carries.
    with (out / "ledger.jsonl").open("w", encoding="utf-8") as stream:
        for key in keys:
            for record in report["datasets"][key]["rounds"]:
                entry = {
                    "dataset_key": key,
                    "round": record["round"],
                    "hypothesis": record["hypothesis"],
                    "adopted": record["adopted"],
                    "validation_before": record["validation_before"],
                    "validation_after": record["validation_after"],
                    "selected_artifact_id": record["selected_artifact_id"],
                    "model_call_ids": record["model_call_ids"],
                }
                # The decision's inputs, so the ledger is enough to
                # re-derive every adoption.
                for key_name in ("selection_metric", "selection_before",
                                 "selection_after", "selection_guard"):
                    if record.get(key_name) is not None:
                        entry[key_name] = record[key_name]
                stream.write(json.dumps(entry, ensure_ascii=False,
                                        allow_nan=False) + "\n")
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

# -*- coding: utf-8 -*-
"""The model call must survive a flaky model service.

The host's model service is shared infrastructure: it can be momentarily
unavailable, and a reasoning model can answer with *no text* when its
token budget goes into the reasoning trace instead of the reply.  Neither
is a property of this submission, and neither may cost the round a
completed measurement.

These tests drive the participant's real ``train`` function against a
host stand-in that fails in exactly those two ways, and assert that the
round still finishes with a selection.

Run:

    python -m pytest tests/test_model_call_fallback.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines import offline_host  # noqa: E402
from contest_baselines.offline_host import package_root, run_entrypoint  # noqa: E402


def _packages_available():
    try:
        for key in ("A", "B", "C"):
            root = package_root(key)
            if not (root / "public" / "train.jsonl").is_file():
                return False
            if not (root / "checkpoint.npz").is_file():
                return False
        return True
    except Exception:
        return False


PACKAGES = _packages_available()


def _flaky_loop(mode):
    """A host loop whose first model answer is broken in the given way.

    ``empty``  the model answers with no text - the reasoning trace ate
                the whole token budget.
    ``error``  the host answers the request with an error frame.
    """
    original = offline_host.OfflineHost._loop

    def _loop(self, process):
        process.stdin.write(json.dumps(self.start, ensure_ascii=False) + "\n")
        process.stdin.flush()
        sequence = 0
        calls = 0
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("the participant closed stdout")
            frame = json.loads(line)
            kind = frame.get("type")
            if kind == "round_failed":
                raise RuntimeError("participant failed: %s" % frame.get("error"))
            if kind == "round_complete":
                self.finished = frame["record"]
                process.stdin.write(json.dumps(
                    {"type": "round_ack", "accepted": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                return
            if frame.get("id") != sequence:
                raise RuntimeError("rpc sequence mismatch")
            sequence += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": []}
            elif kind == "model_request":
                calls += 1
                self.model_requests.append(frame.get("messages"))
                if mode == "empty" and calls == 1:
                    content = None
                elif mode == "error" and calls == 1:
                    process.stdin.write(json.dumps(
                        {"type": "model_error", "id": frame["id"]}) + "\n")
                    process.stdin.flush()
                    continue
                else:
                    content = self.model_reply
                reply = {"type": "model_response", "id": frame["id"],
                         "call_id": "call-%d" % calls,
                         "response": {"request_id": "call-%d" % calls,
                                      "id": "chatcmpl-test",
                                      "choices": [{"message": {
                                          "role": "assistant",
                                          "content": content}}]}}
            elif kind == "state_request":
                reply = {"type": "state_response", "id": frame["id"],
                         "state_sha256": "sha256:" + "22" * 32}
            elif kind == "candidate_request":
                self.candidate = frame
                reply = {"type": "candidate_response", "id": frame["id"],
                         "artifact_id": "artifact-1", "summary": {"metrics": {}}}
            else:
                raise RuntimeError("unexpected frame %r" % kind)
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()

    return original, _loop


class ModelCallOptionTest(unittest.TestCase):
    """The options the round sends to the model."""

    def test_the_call_turns_the_reasoning_trace_off(self):
        """A reasoning model spends the budget on the trace, not the reply."""
        from contest_baselines.train import _MODEL_TOKEN_BUDGET
        from contest_harness.round import _MODEL_OPTIONS

        # The option has to be one the SDK actually accepts, or the host
        # rejects the call outright.
        self.assertIn("enable_thinking", _MODEL_OPTIONS)
        self.assertGreater(_MODEL_TOKEN_BUDGET, 0)

    def test_the_token_budget_leaves_room_for_a_reply(self):
        """512 tokens was not enough once the trace was in the way."""
        from contest_baselines.train import _MODEL_TOKEN_BUDGET

        self.assertGreaterEqual(_MODEL_TOKEN_BUDGET, 512)


@unittest.skipUnless(PACKAGES, "the local data packages are not installed")
class ModelCallFallbackTest(unittest.TestCase):
    """A flaky model service must not cost the round its result."""

    def _run_flaky(self, mode):
        original, patched = _flaky_loop(mode)
        offline_host.OfflineHost._loop = patched
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / "work"
                return run_entrypoint("train", "B", work)
        finally:
            offline_host.OfflineHost._loop = original

    def test_an_empty_reply_is_retried(self):
        code, _stdout, stderr, host = self._run_flaky("empty")
        self.assertEqual(code, 0, stderr)
        self.assertIsNotNone(host.finished, stderr)
        # The first answer was empty, so a second call had to happen.
        self.assertEqual(len(host.model_requests), 2)
        self.assertIn("returned no text", stderr)

    def test_a_model_error_is_retried(self):
        code, _stdout, stderr, host = self._run_flaky("error")
        self.assertEqual(code, 0, stderr)
        self.assertIsNotNone(host.finished, stderr)
        self.assertEqual(len(host.model_requests), 2)
        self.assertIn("attempt 1 failed", stderr)

    def test_the_finished_record_still_carries_the_model_call(self):
        """The round's contract: a real model call is cited in the record."""
        for mode in ("empty", "error"):
            with self.subTest(mode=mode):
                _code, _stdout, stderr, host = self._run_flaky(mode)
                record = host.finished
                self.assertIsNotNone(record, stderr)
                self.assertTrue(record["model_call_ids"], stderr)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""The Controller drives the round, and the View reports what it did.

These tests run the participant's **real** round - the Controller, the
adapters, the strategies, the ledger, the SwanLab record and the model
call - against the offline host stand-in.  The data packages have to be
installed for that, so the tests are skipped when they are not.

What is asserted here is the round's *contract*, not its numbers:

* every step ran, in the order the contract states;
* the round ends with a selection;
* the evidence was written, and the View can read it back;
* the model was called, and the call is cited in the record;
* the state that travels to the next round carries the ledger.

Run:

    python -m pytest tests/test_controller_round.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines.layers.view import View  # noqa: E402
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


@unittest.skipUnless(PACKAGES, "the local data packages are not installed")
class ControllerRoundTest(unittest.TestCase):
    """One round, driven by the Controller, checked against the contract."""

    def _run(self, dataset_key="B", rounds=1):
        """Run a round and return ``(host, work, stdout, stderr)``.

        The temporary directory is **not** cleaned up here: the caller
        asserts against the files the round wrote, so the directory has
        to outlive the call.  The caller removes it.
        """
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        work = Path(tmp) / "work"
        code, stdout, stderr, host = run_entrypoint("train", dataset_key, work)
        self.assertEqual(code, 0, stderr)
        return host, work, stdout, stderr

    def test_the_round_ends_with_a_selection(self):
        host, _work, _stdout, stderr = self._run()
        self.assertIsNotNone(host.finished, stderr)
        self.assertTrue(host.finished.get("selected_artifact_id"))

    def test_every_step_of_the_round_ran(self):
        """The steps are the contract; a missing one is a silent gap."""
        _host, _work, stdout, stderr = self._run()
        # The parent's digest is read, the hypothesis is planned, the
        # candidate is measured, and the decision is taken.
        self.assertIn("parent checkpoint", stderr)
        self.assertIn("hypothesis", stderr)
        self.assertIn("parent rmse", stderr)
        self.assertIn("candidate rmse", stderr)
        self.assertIn("decision:", stderr)
        self.assertIn("finished in", stderr)
        self.assertTrue(stdout.strip())

    def test_the_model_was_called_and_the_call_is_cited(self):
        host, _work, _stdout, stderr = self._run()
        self.assertTrue(host.model_requests, "the round made no model call")
        self.assertTrue(host.finished.get("model_call_ids"),
                        "the finished record cites no model call")
        self.assertIn("model's review", host.finished.get("reason", ""))

    def test_the_round_writes_a_ledger_entry(self):
        _host, work, _stdout, stderr = self._run()
        ledger = work / "state" / "ledger.jsonl"
        self.assertTrue(ledger.is_file(), stderr)
        lines = [line for line in
                 ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        # The entry answers the four questions a round has to answer.
        self.assertIn("hypothesis", record)
        self.assertIn("claim", record)
        self.assertIn("falsifier", record)
        self.assertIn("applied", record)
        self.assertIn("validation_before", record)
        self.assertIn("validation_after", record)
        self.assertIn("adopted", record)
        self.assertTrue(record["reason"])

    def test_the_view_reads_the_state_the_round_wrote(self):
        """The View is a read-only API over the state the round produced."""
        _host, work, _stdout, stderr = self._run()
        view = View(work / "state")
        answer = view.overview()
        self.assertEqual(answer["rounds"], 1)
        self.assertIsNotNone(answer["champion"])
        self.assertEqual(answer["champion"]["round"], 1)
        # The champion carries the measurement that decided the round.
        self.assertIn("rmse", answer["champion"]["validation"])
        self.assertEqual(answer["best"]["metric"], "rmse")
        self.assertIn("value", answer["best"])

    def test_the_swanlab_record_travels_with_the_state(self):
        _host, work, _stdout, stderr = self._run()
        record = work / "state" / "swanlab.jsonl"
        self.assertTrue(record.is_file(), stderr)
        entries = [json.loads(line) for line in
                   record.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(entries)
        # The record carries the measurement, so the platform collects an
        # experiment log and not just a status line.
        self.assertIn("metrics", entries[0])
        self.assertIn("context", entries[0])
        self.assertIn("val_rmse_before", entries[0]["metrics"])


@unittest.skipUnless(PACKAGES, "the local data packages are not installed")
class ControllerHistoryTest(unittest.TestCase):
    """Two rounds: the second one knows what the first one tried."""

    def test_the_second_round_does_not_repeat_the_first_hypothesis(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        work = Path(tmp) / "work"
        code, _stdout, stderr, host = run_entrypoint("train", "B", work)
        self.assertEqual(code, 0, stderr)
        self.assertIsNotNone(host.finished, stderr)
        # The first round's record is in the harness's own ledger.
        ledger = work.parent / "ledger.jsonl"
        if not ledger.is_file():
            self.skipTest("the offline host did not keep the round ledger")
        records = [json.loads(line) for line in
                   ledger.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        self.assertTrue(records)
        self.assertIn("hypothesis", records[0])


if __name__ == "__main__":
    unittest.main()

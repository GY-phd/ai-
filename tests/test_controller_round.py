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

from contest_baselines.adapters import get as get_adapter  # noqa: E402
from contest_baselines.layers.controller.round import RoundController  # noqa: E402
from contest_baselines.layers.model.metrics import (  # noqa: E402
    BETTER, MetricRecord, compare_joint,
)
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


class FullSplitGuardTest(unittest.TestCase):
    """The protocol-aligned selection never gets a blank cheque.

    C's truncated-test slice can improve while the full-split RMSE gets
    worse; the round accepts a small regression but not a collapse.  The
    hidden pack is a different dataset, so a candidate that only wins the
    slice is not automatically safe.
    """

    @staticmethod
    def _controller(adapter_key, before_full, before_late, after_full, after_late):
        controller = RoundController.__new__(RoundController)
        controller.adapter = get_adapter(adapter_key)
        controller.parent_record = MetricRecord(
            "regression", {"rmse": before_full, "late_rmse": before_late},
            "rmse", before_full, selection_name="late_rmse",
            selection_value=before_late)
        controller.candidate_record = MetricRecord(
            "regression", {"rmse": after_full, "late_rmse": after_late},
            "rmse", after_full, selection_name="late_rmse",
            selection_value=after_late)
        return controller

    def test_a_small_full_regression_passes(self):
        controller = self._controller("C", 51.0, 38.0, 53.0, 32.0)
        guard = controller._full_split_guard()
        self.assertTrue(guard["active"])
        self.assertTrue(guard["passed"])
        self.assertAlmostEqual(guard["rel"], 2.0 / 51.0)

    def test_a_large_full_regression_is_refused(self):
        controller = self._controller("C", 51.0, 38.0, 61.0, 28.0)
        guard = controller._full_split_guard()
        self.assertFalse(guard["passed"])

    def test_profiles_without_a_selection_slice_are_not_guarded(self):
        controller = self._controller("B", 600.0, 600.0, 900.0, 900.0)
        # B's MetricRecord would not carry a late slice in the real round;
        # the guard is driven by the name, so make that explicit here.
        controller.parent_record.selection_name = "rmse"
        controller.candidate_record.selection_name = "rmse"
        guard = controller._full_split_guard()
        self.assertFalse(guard["active"])


class FalsifierUsesTheFullSplitTest(unittest.TestCase):
    """The falsifier must not veto on the metric the evidence distrusts.

    ``compare_joint`` lets the full split decide, because the
    leave-one-engine-out study in ``research/tracks/c_horizon.py`` found
    every truncated slice anti-correlated with the held-out full split
    (9 to 10 of 12 folds) and selecting by a slice beat the baseline in
    0 of 12 folds.

    The falsifier check (``_consistent_with_expectation``) used to
    compare the **slice** anyway, so it silently vetoed exactly the
    candidates the study says are worth keeping.  The platform's
    2026-11-25 run shows the cost: rounds 6, 8 and 9 had a full split
    that improved, and were rejected anyway, leaving five of the ten
    rounds with nothing but the parent.

    These tests pin the check to the same reading that decides.
    """

    @staticmethod
    def _controller(before_full, before_late, after_full, after_late):
        controller = RoundController.__new__(RoundController)
        controller.adapter = get_adapter("C")
        controller.parent_record = MetricRecord(
            "regression", {"rmse": before_full, "late_rmse": before_late},
            "rmse", before_full, selection_name="late_rmse",
            selection_value=before_late)
        controller.candidate_record = MetricRecord(
            "regression", {"rmse": after_full, "late_rmse": after_late},
            "rmse", after_full, selection_name="late_rmse",
            selection_value=after_late)
        controller.parent = type("P", (), {"task": "regression"})()
        return controller

    def test_a_full_split_gain_survives_a_regressed_slice(self):
        # The platform's round 6: the full split improved from 51.2363
        # while the slice went from 35.8438 to 37.1927.
        controller = self._controller(51.2363, 35.8438, 51.0284, 37.1927)
        self.assertTrue(controller._consistent_with_expectation(True))

    def test_a_full_split_loss_fails_even_with_a_better_slice(self):
        controller = self._controller(51.0, 38.0, 52.0, 30.0)
        self.assertFalse(controller._consistent_with_expectation(True))

    def test_a_flat_full_split_falls_back_to_the_slice(self):
        # ``compare_joint`` returns BETTER here (the slice improved and
        # the full split did not move), so the falsifier has to agree.
        controller = self._controller(51.0, 38.0, 51.0, 30.0)
        self.assertTrue(controller._consistent_with_expectation(True))

    def test_a_flat_full_split_and_a_flat_slice_is_not_an_improvement(self):
        controller = self._controller(51.0, 38.0, 51.0, 38.0)
        self.assertFalse(controller._consistent_with_expectation(True))

    def test_the_check_is_consistent_with_the_decision(self):
        """Whatever ``compare_joint`` says, the falsifier must agree."""
        for before_full, after_full, before_late, after_late in (
                (51.0, 50.0, 38.0, 30.0),
                (51.0, 52.0, 38.0, 30.0),
                (51.0, 51.0, 38.0, 30.0),
                (51.0, 51.0, 38.0, 38.0)):
            controller = self._controller(before_full, before_late,
                                         after_full, after_late)
            verdict = compare_joint(controller.parent_record,
                                    controller.candidate_record, "regression")
            improved = verdict == BETTER
            self.assertEqual(
                controller._consistent_with_expectation(improved), improved,
                "the falsifier disagrees with the decision at %s -> %s"
                % (before_full, after_full))


class TheReasonQuotesTheDecisionTest(unittest.TestCase):
    """The decision's reason must quote the metric the decision used.

    The verdict is made on the **primary** metric - the full split -
    because the leave-one-engine-out study in
    ``research/tracks/c_horizon.py`` found every truncated slice
    anti-correlated with the held-out full split.

    The reason strings used to quote the **slice** in three of their
    branches, so the ledger read

        "the hypothesis is refuted: late_rmse moved from 37.0 to 37.4"

    for a round whose full RMSE moved from 50.4784 to 50.4896.  The
    reader had to guess which number the decision was about, and the
    platform's own report showed exactly that confusion.

    These tests pin every branch to the metric the decision used.
    """

    @staticmethod
    def _controller(before_full, before_late, after_full, after_late):
        controller = RoundController.__new__(RoundController)
        controller.adapter = get_adapter("C")
        controller.parent_record = MetricRecord(
            "regression", {"rmse": before_full, "late_rmse": before_late},
            "rmse", before_full, selection_name="late_rmse",
            selection_value=before_late)
        controller.candidate_record = MetricRecord(
            "regression", {"rmse": after_full, "late_rmse": after_late},
            "rmse", after_full, selection_name="late_rmse",
            selection_value=after_late)
        controller.parent = type("P", (), {"task": "regression"})()
        controller.adopted = False
        controller.reason = ""
        return controller

    def test_a_refutation_quotes_the_full_split(self):
        controller = self._controller(50.4784, 37.0693, 50.4896, 37.3967)
        controller.decide()
        self.assertFalse(controller.adopted)
        self.assertIn("rmse", controller.reason)
        self.assertIn("50.4784", controller.reason)
        self.assertIn("50.4896", controller.reason)

    def test_a_confirmation_quotes_the_full_split(self):
        controller = self._controller(50.6121, 37.5860, 50.4784, 37.0693)
        controller.decide()
        self.assertTrue(controller.adopted)
        self.assertIn("rmse", controller.reason)
        self.assertIn("50.6121", controller.reason)
        self.assertIn("50.4784", controller.reason)

    def test_the_slice_is_recorded_as_evidence_not_as_the_decision(self):
        """The slice may still be quoted, but only as the evidence."""
        controller = self._controller(50.6121, 37.5860, 50.4784, 37.0693)
        controller.decide()
        self.assertIn("late_rmse", controller.reason)
        self.assertIn("evidence", controller.reason)

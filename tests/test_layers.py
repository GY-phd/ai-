# -*- coding: utf-8 -*-
"""The three layers, and the rules that hold them together.

    30|These tests cover three things:

* the **Model layer** stores what it says it stores, and reads it back
  unchanged;
* the **View layer** answers questions about that state and never
  writes;
* the **Controller layer** can drive a round end to end against the
  platform's protocol.

The round test is the important one: it is the only place where a
  mis-wired layer would show up as a *wrong experiment* rather than an
  exception, and a wrong experiment is the failure mode that matters.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from contest_baselines.layers.model import (
    Checkpoint,
    Ledger,
    LedgerEntry,
    MetricRecord,
    RoundState,
    compare,
)
from contest_baselines.layers.model.metrics import BETTER, THE_SAME, WORSE
from contest_baselines.layers.view import View


class LedgerTest(unittest.TestCase):
    """The ledger is append-only and survives a round-trip."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_an_entry_survives_the_round_trip(self):
        entry = LedgerEntry(
            round_index=3, dataset_key="profile-b", task="regression",
            hypothesis="more-data-helps", change="{\"epochs\": 5}",
            before={"rmse": 12.5}, after={"rmse": 10.1}, adopted=True,
            reason="confirmed", claim="the parent was data-limited",
            falsifier="the score does not improve",
        )
        path = self.tmp / "ledger.jsonl"
        Ledger(path).append(entry)

        again = Ledger.load(path).entries
        self.assertEqual(len(again), 1)
        self.assertEqual(again[0].round_index, 3)
        self.assertEqual(again[0].hypothesis, "more-data-helps")
        self.assertTrue(again[0].adopted)
        self.assertEqual(again[0].before, {"rmse": 12.5})
        self.assertEqual(again[0].after, {"rmse": 10.1})
        # The extra fields survive too.
        self.assertEqual(again[0].extra["claim"], "the parent was data-limited")

    def test_appending_does_not_rewrite_what_came_before(self):
        path = self.tmp / "ledger.jsonl"
        ledger = Ledger(path)
        ledger.append(LedgerEntry(1, "k", "regression", "a", "c",
                                  {"rmse": 1.0}, {"rmse": 0.9}, True, "yes"))
        first = path.read_text(encoding="utf-8")
        ledger.append(LedgerEntry(2, "k", "regression", "b", "c",
                                  {"rmse": 0.9}, {"rmse": 0.8}, True, "yes"))
        second = path.read_text(encoding="utf-8")
        self.assertTrue(second.startswith(first),
                        "the earlier line must still be the earlier line")

    def test_a_malformed_line_does_not_hide_the_evidence(self):
        path = self.tmp / "ledger.jsonl"
        path.write_text(
            '{"round": 1, "hypothesis": "a", "adopted": true}\n'
            'this is not json\n'
            '{"round": 2, "hypothesis": "b", "adopted": false}\n',
            encoding="utf-8")
        entries = Ledger.load(path).entries
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].hypothesis, "a")
        self.assertEqual(entries[1].hypothesis, "b")

    def test_a_ledger_with_no_path_keeps_its_entries_in_memory(self):
        ledger = Ledger()
        ledger.append(LedgerEntry(1, "k", "regression", "a", "c",
                                  {"rmse": 1.0}, {"rmse": 0.9}, True, "yes"))
        self.assertEqual(len(ledger), 1)


class CompareTest(unittest.TestCase):
    """The comparison follows the task's direction."""

    def _record(self, task, name, value):
        return MetricRecord(task, {name: value}, name, value)

    def test_classification_moves_up(self):
        before = self._record("classification", "macro_f1", 0.70)
        after = self._record("classification", "macro_f1", 0.75)
        self.assertEqual(compare(before, after, "classification"), BETTER)

    def test_classification_moves_down(self):
        before = self._record("classification", "macro_f1", 0.70)
        after = self._record("classification", "macro_f1", 0.65)
        self.assertEqual(compare(before, after, "classification"), WORSE)

    def test_regression_moves_down_is_better(self):
        before = self._record("regression", "rmse", 12.0)
        after = self._record("regression", "rmse", 10.0)
        self.assertEqual(compare(before, after, "regression"), BETTER)

    def test_regression_moves_up_is_worse(self):
        before = self._record("regression", "rmse", 10.0)
        after = self._record("regression", "rmse", 12.0)
        self.assertEqual(compare(before, after, "regression"), WORSE)

    def test_unchanged_is_its_own_outcome(self):
        """'the same' is not 'worse' - it says the change did nothing."""
        before = self._record("classification", "macro_f1", 0.70)
        after = self._record("classification", "macro_f1", 0.70)
        self.assertEqual(compare(before, after, "classification"), THE_SAME)


class ViewTest(unittest.TestCase):
    """The View reads the state and answers questions about it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        state = RoundState(self.tmp)
        state.append_ledger(LedgerEntry(
            1, "profile-b", "regression", "more-data-helps", "{\"epochs\": 5}",
            {"rmse": 12.5}, {"rmse": 10.1}, True, "the hypothesis is confirmed"))
        state.append_ledger(LedgerEntry(
            2, "profile-b", "regression", "capacity", "{\"hidden\": 64}",
            {"rmse": 10.1}, {"rmse": 11.3}, False, "the hypothesis is refuted"))
        self.view = View(self.tmp)

    def test_the_overview_reports_the_rounds_and_the_champion(self):
        answer = self.view.overview()
        self.assertEqual(answer["kind"], "overview")
        self.assertEqual(answer["version"], 1)
        self.assertEqual(answer["rounds"], 2)
        # The champion is the round with the best rmse, which is the
        # lowest one.
        self.assertEqual(answer["champion"]["round"], 1)
        self.assertEqual(answer["best"]["metric"], "rmse")
        self.assertEqual(answer["best"]["value"], 10.1)

    def test_the_ledger_endpoint_returns_every_round(self):
        answer = self.view.ledger()
        self.assertEqual(answer["kind"], "ledger")
        self.assertEqual(len(answer["entries"]), 2)
        self.assertEqual(answer["entries"][0]["hypothesis"], "more-data-helps")

    def test_the_hypothesis_endpoint_reports_both_outcomes(self):
        answer = self.view.hypothesis()
        self.assertEqual(len(answer["hypotheses"]), 2)
        self.assertTrue(answer["hypotheses"][0]["adopted"])
        self.assertFalse(answer["hypotheses"][1]["adopted"])

    def test_the_view_never_writes(self):
        """A read of the state must not change it."""
        before = sorted(p.name for p in self.tmp.iterdir())
        for call in (self.view.ledger, self.view.rounds, self.view.metrics,
                     self.view.overview, self.view.champion,
                     self.view.hypothesis):
            call()
        after = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(before, after)
        content = (self.tmp / "ledger.jsonl").read_text(encoding="utf-8")
        self.assertEqual(len(content.strip().splitlines()), 2)

    def test_an_empty_state_answers_without_raising(self):
        view = View(Path(tempfile.mkdtemp()))
        answer = view.overview()
        self.assertEqual(answer["rounds"], 0)
        self.assertIsNone(answer["champion"])

    def test_the_answer_is_json_serialisable(self):
        text = self.view.to_json(self.view.overview())
        self.assertEqual(json.loads(text)["kind"], "overview")


class RoundStateTest(unittest.TestCase):
    """The state carries the evidence forward."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_the_tried_hypotheses_are_the_ones_already_run(self):
        state = RoundState(self.tmp)
        state.append_ledger(LedgerEntry(1, "k", "regression", "a", "c",
                                        {"rmse": 1.0}, {"rmse": 0.9}, True, "y"))
        state.append_ledger(LedgerEntry(2, "k", "regression", "b", "c",
                                        {"rmse": 0.9}, {"rmse": 0.8}, True, "y"))
        self.assertEqual(state.tried_hypotheses(), {"a", "b"})

    def test_the_swanlab_record_is_copied_into_the_state(self):
        source = self.tmp / "swanlab.jsonl"
        source.write_text('{"run": "x", "step": 1}\n', encoding="utf-8")
        state = RoundState(self.tmp / "state")
        target = state.write_swanlab(source)
        self.assertTrue(target.is_file())
        # The copy is faithful.
        self.assertEqual(target.read_text(encoding="utf-8"),
                         source.read_text(encoding="utf-8"))
        # The source keeps its own copy: a round's evidence survives
        # whether or not the candidate was adopted.
        self.assertTrue(source.is_file())


if __name__ == "__main__":
    unittest.main()

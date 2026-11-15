# -*- coding: utf-8 -*-
"""The safety gate: fleet-level gains must not hide a local regression."""
import unittest

from harness.experiments import _passes_screen
from harness.gate import evaluate as gate_evaluate

GATE = {"min_gain_pct": 1.0, "max_local_regression_pct_of_global": 20.0}
BUDGET = {"train_seconds_left": 600}


def metrics(rmse, per_bearing):
    return {"rmse": rmse, "mae": rmse * 0.8,
            "per_bearing": {k: {"rmse": v, "rows": 10} for k, v in per_bearing.items()}}


class GateTest(unittest.TestCase):
    def setUp(self):
        self.champion = metrics(500.0, {"a": 100.0, "b": 900.0})

    def test_promotes_a_clear_fleet_level_win(self):
        candidate = metrics(450.0, {"a": 120.0, "b": 850.0})
        decision = gate_evaluate(self.champion, candidate, GATE, BUDGET)
        self.assertEqual(decision.decision, "promote")
        self.assertAlmostEqual(decision.gain_pct, 10.0, places=6)

    def test_rejects_a_gain_below_threshold(self):
        candidate = metrics(499.0, {"a": 100.0, "b": 899.0})
        decision = gate_evaluate(self.champion, candidate, GATE, BUDGET)
        self.assertEqual(decision.decision, "reject")
        self.assertIn("below the 1.0% promotion threshold", decision.reason)

    def test_rejects_when_one_bearing_regresses_too_much(self):
        # global gain is large, but bearing a triples its error -> veto
        candidate = metrics(400.0, {"a": 300.0, "b": 410.0})
        decision = gate_evaluate(self.champion, candidate, GATE, BUDGET)
        self.assertEqual(decision.decision, "reject")
        self.assertFalse(decision.checks["no_local_regression"])
        self.assertEqual(decision.checks["worst_bearing"], "a")

    def test_local_regression_is_measured_against_the_global_scale(self):
        # +60 minutes is 12% of the 500-minute global RMSE -> tolerated even though
        # bearing b is only 40 minutes wide
        candidate = metrics(480.0, {"a": 100.0, "b": 960.0})
        decision = gate_evaluate(self.champion, candidate, GATE, BUDGET)
        self.assertEqual(decision.decision, "promote")

    def test_unseen_bearing_does_not_block_promotion(self):
        candidate = metrics(450.0, {"a": 100.0, "c": 900.0})
        decision = gate_evaluate(self.champion, candidate, GATE, BUDGET)
        self.assertEqual(decision.decision, "promote")

    def test_screen_matches_the_gate_for_a_single_row(self):
        row = {"rmse": 400.0, "per_bearing": {"a": {"rmse": 300.0}, "b": {"rmse": 410.0}}}
        self.assertFalse(_passes_screen(self.champion, row, GATE))
        row_ok = {"rmse": 450.0, "per_bearing": {"a": {"rmse": 120.0}, "b": {"rmse": 850.0}}}
        self.assertTrue(_passes_screen(self.champion, row_ok, GATE))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Deterministic policy: rule mapping, budget downgrades, and the cost ladder."""
import unittest

from harness.policy import PolicyPlanner

CHAMPION = {"rmse": 500.0, "per_bearing": {}}


def budget(**kwargs):
    base = {"experiments_left": 5, "train_seconds_left": 600, "sweep_exhausted": False}
    base.update(kwargs)
    return base


def plan(drift_type, history=None, **budget_kwargs):
    card = {"drift_type": drift_type, "confidence": 0.8, "needs_labels": drift_type == "concept"}
    return PolicyPlanner().plan(card, CHAMPION, budget(**budget_kwargs), history or [])


class PolicyTest(unittest.TestCase):
    def test_cold_start_trains_a_candidate(self):
        self.assertEqual(plan("cold_start")["action"], "hpo")

    def test_prior_bias_prefers_the_cheap_head_refit(self):
        self.assertEqual(plan("prior")["action"], "retrain_head")

    def test_covariate_shift_recalibrates_before_spending_labels(self):
        self.assertEqual(plan("covariate")["action"], "recalibrate_inputs")

    def test_concept_drift_escalates_to_a_sweep(self):
        self.assertEqual(plan("concept")["action"], "hpo")

    def test_no_action_when_evidence_is_weak(self):
        self.assertEqual(plan("none")["action"], "do_nothing")
        self.assertEqual(plan("data")["action"], "do_nothing")

    def test_exhausted_budget_forces_do_nothing(self):
        self.assertEqual(plan("concept", experiments_left=0)["action"], "do_nothing")
        self.assertEqual(plan("prior", train_seconds_left=0)["action"], "do_nothing")

    def test_exhausted_search_space_falls_back_to_the_head_refit(self):
        self.assertEqual(plan("concept", sweep_exhausted=True)["action"], "retrain_head")

    def test_rejected_sweep_climbs_down_the_cost_ladder(self):
        history = [{"action": "hpo", "decision": "reject"}]
        self.assertEqual(plan("concept", history=history)["action"], "retrain_head")

    def test_plan_payload_is_schema_valid(self):
        result = plan("concept")
        for key in ("action", "params", "hypothesis", "expected_gain_pct", "risk", "reason"):
            self.assertIn(key, result)
        self.assertIn(result["risk"], ("low", "medium", "high"))
        self.assertTrue(result["hypothesis"])


if __name__ == "__main__":
    unittest.main()

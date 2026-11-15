from pathlib import Path
import unittest

from harness.config import from_mapping, load_config
from harness.drift import DriftFinding, DriftReport
from harness.policy import (best_passing_candidate, diagnose, gate_candidate,
                            plan_experiments, rollback_decision)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "harness" / "configs" / "default.json")


def metric(value):
    return {"rmse": value, "mae": value}


def pair(batch, guard):
    return {"batch": metric(batch), "guard": metric(guard)}


def report(severity="stable", findings=None):
    return DriftReport(severity=severity, drifted=severity != "stable", signals={},
                       findings=findings or [], summary="test")


class GateTests(unittest.TestCase):
    def gate(self, candidate, champion=(100, 100), static=(100, 100)):
        return gate_candidate(candidate=pair(*candidate), champion=pair(*champion),
                              static=pair(*static), config=CONFIG.gate)

    def test_improving_candidate_passes(self):
        result = self.gate((98, 100))
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.improvement, 0.02)
        self.assertEqual(result.failed_checks, [])

    def test_tiny_improvement_is_rejected(self):
        result = self.gate((99.5, 100))
        self.assertFalse(result.passed)
        self.assertIn("beats_champion_on_batch", result.failed_checks)

    def test_guard_degradation_is_rejected(self):
        result = self.gate((98, 105))
        self.assertFalse(result.passed)
        self.assertIn("guard_not_degraded", result.failed_checks)
        self.assertIn("guard_not_worse_than_static", result.failed_checks)

    def test_candidate_must_not_lose_to_static_on_batch(self):
        result = self.gate((101, 100), champion=(120, 100), static=(100, 100))
        self.assertFalse(result.passed)
        self.assertIn("not_worse_than_static_on_batch", result.failed_checks)

    def test_best_passing_candidate_prefers_lowest_batch_rmse(self):
        first = self.gate((98, 100))
        second = self.gate((97, 99))
        third = self.gate((96, 105))
        results = [
            {"stage_id": "a", "metrics": pair(98, 100), "gate": first},
            {"stage_id": "b", "metrics": pair(97, 99), "gate": second},
            {"stage_id": "c", "metrics": pair(96, 105), "gate": third},
        ]
        best = best_passing_candidate(results)
        self.assertEqual(best["stage_id"], "b")
        self.assertIsNone(best_passing_candidate([results[2]]))


class PlanningTests(unittest.TestCase):
    champion_meta = {"learning_rate": 0.002, "batch_size": 256}

    def plan(self, drift, attempts, batch_index):
        hypotheses = diagnose(drift, attempts=attempts, batch_index=batch_index,
                              config=CONFIG.experiments)
        specs = plan_experiments(drift, hypotheses, champion_meta=self.champion_meta,
                                 attempts=attempts, batch_index=batch_index,
                                 config=CONFIG.experiments)
        return hypotheses, specs

    def test_initial_stable_batch_completes_attempt_quota(self):
        hypotheses, specs = self.plan(report(), attempts=0, batch_index=1)
        self.assertEqual(hypotheses[0].code, "exploration")
        self.assertEqual(len(specs), CONFIG.experiments.min_attempts)
        self.assertEqual([item.learning_rate_multiplier for item in specs], [1.0, 0.5, 2.0])

    def test_quota_met_and_no_periodic_exploration_holds(self):
        _, specs = self.plan(report(), attempts=3, batch_index=1)
        self.assertEqual(specs, [])

    def test_periodic_exploration_runs_one_candidate(self):
        _, specs = self.plan(report(), attempts=3, batch_index=3)
        self.assertEqual(len(specs), 1)

    def test_critical_drift_uses_full_candidate_budget(self):
        drift = report("critical", [DriftFinding("performance_degradation", "critical", "bad")])
        _, specs = self.plan(drift, attempts=100, batch_index=1)
        self.assertEqual(len(specs), CONFIG.experiments.max_candidates)

    def test_short_multiplier_list_is_extended_with_epoch_variants(self):
        config = from_mapping({"experiments": {
            "min_attempts": 5, "max_candidates": 5, "epochs": 2,
            "learning_rate_multipliers": [1.0],
        }})
        drift = report()
        hypotheses = diagnose(drift, attempts=0, batch_index=1, config=config.experiments)
        specs = plan_experiments(drift, hypotheses, champion_meta=self.champion_meta,
                                 attempts=0, batch_index=1, config=config.experiments)
        self.assertEqual(len(specs), 5)
        self.assertEqual([item.epochs for item in specs], [2, 3, 4, 5, 6])
        self.assertEqual(len({(item.learning_rate, item.epochs) for item in specs}), 5)

    def test_warning_drift_also_completes_the_attempt_quota(self):
        drift = report("warning", [DriftFinding("performance_degradation", "warning", "bad")])
        _, specs = self.plan(drift, attempts=0, batch_index=1)
        self.assertEqual(len(specs), CONFIG.experiments.min_attempts)

    def test_concept_drift_prefers_higher_learning_rate(self):
        drift = report("critical", [
            DriftFinding("target_distribution_shift", "critical", "target moved"),
            DriftFinding("performance_degradation", "critical", "bad"),
        ])
        hypotheses, specs = self.plan(drift, attempts=100, batch_index=1)
        self.assertEqual(hypotheses[0].code, "concept_drift")
        self.assertEqual(specs[0].learning_rate_multiplier, 2.0)
        self.assertGreater(specs[0].epochs, CONFIG.experiments.epochs)


class RollbackTests(unittest.TestCase):
    champion = {"stage_id": "candidate-1", "is_static": False}

    def decide(self, champion_metrics, static=(100, 100), previous=(90, 95), previous_id="old"):
        return rollback_decision(
            champion=self.champion, champion_metrics=pair(*champion_metrics),
            static_metrics=pair(*static), previous_metrics=pair(*previous) if previous else None,
            previous_stage_id=previous_id if previous else None, config=CONFIG.gate)

    def test_previous_checkpoint_is_preferred_when_better(self):
        decision = self.decide((120, 120))
        self.assertTrue(decision.should_rollback)
        self.assertEqual(decision.target, "previous")

    def test_static_is_used_without_a_safe_previous_checkpoint(self):
        decision = self.decide((120, 120), previous=None)
        self.assertTrue(decision.should_rollback)
        self.assertEqual(decision.target, "static")

    def test_previous_checkpoint_below_static_floor_is_not_used(self):
        decision = self.decide((120, 120), static=(100, 100), previous=(110, 95))
        self.assertTrue(decision.should_rollback)
        self.assertEqual(decision.target, "static")
        self.assertFalse(decision.checks["previous_batch_not_worse_than_static"])

    def test_no_rollback_below_margin(self):
        decision = self.decide((100.5, 100.5))
        self.assertFalse(decision.should_rollback)

    def test_static_champion_cannot_roll_back(self):
        decision = rollback_decision(
            champion={"stage_id": "official-baseline", "is_static": True},
            champion_metrics=pair(120, 120), static_metrics=pair(100, 100),
            previous_metrics=None, previous_stage_id=None, config=CONFIG.gate)
        self.assertFalse(decision.should_rollback)


if __name__ == "__main__":
    unittest.main()

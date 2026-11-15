import json
import unittest

from harness.telemetry import (batch_metrics, candidate_metrics, epoch_metrics,
                               final_metrics, text_records)


def metric(value):
    return {"rmse": value, "mae": value * 0.8, "nasa_score": value * 2, "n": 10}


def record():
    return {
        "format": "harness.decision.v1",
        "workdir": "/tmp/work",
        "batch": {"id": "batch-01", "index": 2, "validation_sha256": "sha256:x"},
        "decision": "promote",
        "reason": "candidate improved",
        "champion_before": {"stage_id": "old"},
        "selected_model": {"stage_id": "new", "generation": 1,
                           "metrics": {"batch": metric(90), "guard": metric(95)}},
        "metrics": {
            "champion_before": {"batch": metric(100), "guard": metric(101)},
            "static": {"batch": metric(110), "guard": metric(111)},
        },
        "drift": {
            "severity": "critical", "summary": "input shift",
            "signals": {"performance_delta": 0.25, "feature_max_abs_mean_z": 4.2,
                        "top_features": [{"feature": "sensors[0]"}]},
            "findings": [{"code": "input_distribution_shift", "severity": "critical",
                          "statement": "shifted", "evidence": {"max_abs_mean_z": 4.2}}],
        },
        "hypotheses": [{"code": "covariate_shift", "statement": "adapt weights"}],
        "experiments_planned": [{"epochs": 3}],
        "experiments": [{
            "spec": {"epochs": 3, "learning_rate": 0.001, "batch_size": 256},
            "stage_id": "batch-01-c01-lr0.001-e3",
            "metrics": {"batch": metric(90), "guard": metric(95)},
            "gate": {"passed": True, "failed_checks": [], "improvement": 0.1},
            "training": {"training_seconds": 2.5, "duration_seconds": 2.6,
                         "initial_steps": 10, "final_steps": 20},
            "model_meta": {"parameter_count": 833},
            "history": [
                {"epoch": 1, "step": 11, "loss": 0.5,
                 "validation": {"rmse": 98, "mae": 77}},
                {"epoch": 2, "step": 12, "loss": 0.4,
                 "validation": {"rmse": 95, "mae": 75}},
            ],
            "report_mismatch": 0.0,
        }],
        "training_data": {"rows": 100, "excluded_ids": 5},
        "budget": {"time_budget_seconds": 180, "elapsed_seconds": 3.2},
        "counters": {"attempts": 3, "successful_experiments": 3, "failed_experiments": 0,
                     "promotions": 1, "rollbacks": 0, "holds": 1, "errors": 0},
        "dry_run": False,
    }


class TelemetryTests(unittest.TestCase):
    def test_batch_metrics_are_numeric_and_complete(self):
        metrics = batch_metrics(record())
        self.assertEqual(metrics["batch/index"], 2)
        self.assertEqual(metrics["batch/decision_promote"], 1)
        self.assertEqual(metrics["batch/decision_hold"], 0)
        self.assertEqual(metrics["batch/drift_severity"], 2)
        self.assertEqual(metrics["drift/performance_delta"], 0.25)
        self.assertEqual(metrics["batch/champion_batch_rmse"], 100)
        self.assertEqual(metrics["batch/static_batch_rmse"], 110)
        self.assertEqual(metrics["batch/selected_batch_rmse"], 90)
        self.assertEqual(metrics["batch/rows"], 10)
        self.assertEqual(metrics["batch/excluded_ids"], 5)
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float), key)
            self.assertNotIn(key, ("drift/top_features",))

    def test_candidate_metrics_include_gate_and_training(self):
        metrics = candidate_metrics(record())
        self.assertEqual(metrics["candidates/c01/batch_rmse"], 90)
        self.assertEqual(metrics["candidates/c01/guard_rmse"], 95)
        self.assertEqual(metrics["candidates/c01/gate_passed"], 1)
        self.assertEqual(metrics["candidates/c01/gate_failed_checks"], 0)
        self.assertEqual(metrics["candidates/c01/epochs"], 3)
        self.assertAlmostEqual(metrics["candidates/c01/improvement"], 0.1)
        self.assertEqual(metrics["candidates/c01/parameter_count"], 833)

    def test_epoch_curves_are_batch_and_candidate_scoped(self):
        curves = epoch_metrics(record())
        self.assertEqual(sorted(curves), [1, 2])
        self.assertEqual(curves[1]["curves/b0002/c01/train_loss"], 0.5)
        self.assertEqual(curves[1]["curves/b0002/c01/val_rmse"], 98)
        self.assertEqual(curves[2]["curves/b0002/c01/val_mae"], 75)

    def test_text_records_are_valid_json_and_contain_reason(self):
        texts = dict(text_records(record()))
        self.assertIn("drift/batch-01", texts)
        self.assertIn("decision/batch-01", texts)
        self.assertIn("candidate improved", texts["decision/batch-01"])
        json.loads(texts["drift/batch-01"].split("\n", 1)[1])

    def test_final_metrics_include_counters(self):
        status = {
            "champion": {"generation": 2, "is_static": False,
                         "metrics": {"batch": metric(80), "guard": metric(81)}},
            "counters": {"attempts": 4, "promotions": 2, "rollbacks": 0},
            "batches": [{"batch_id": "batch-01"}],
        }
        metrics = final_metrics(status)
        self.assertEqual(metrics["final/champion_generation"], 2)
        self.assertEqual(metrics["final/champion_batch_rmse"], 80)
        self.assertEqual(metrics["final/attempts"], 4)
        self.assertEqual(metrics["final/batches"], 1)


if __name__ == "__main__":
    unittest.main()

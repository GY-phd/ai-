from pathlib import Path
import unittest

from harness.benchmark import batch_result, summarize


def record(batch_id, index, static, selected, *, decision="promote", error=None,
           drift="stable", attempts=1):
    return {
        "batch": {"id": batch_id, "index": index,
                  "metadata": {"drift_type": "domain-shift", "source_subset": "FD002"}},
        "decision": decision,
        "error": error,
        "reason": "test",
        "drift": {"severity": drift},
        "metrics": {
            "static": {"batch": {"rmse": static, "mae": static * 0.8},
                       "guard": {"rmse": static + 1}},
            "champion_before": {"batch": {"rmse": static + 1}, "guard": {"rmse": static + 2}},
        },
        "selected_model": {"metrics": {
            "batch": {"rmse": selected, "mae": selected * 0.8},
            "guard": {"rmse": selected + 1},
        }},
        "experiments": [{"training": {"training_seconds": 1.5},
                         "gate": {"passed": True}}] * attempts,
    }


class BenchmarkTests(unittest.TestCase):
    def test_batch_result_computes_gain_and_metadata(self):
        row = batch_result(record("b01", 1, 100.0, 90.0))
        self.assertEqual(row["batch_id"], "b01")
        self.assertEqual(row["drift_type"], "domain-shift")
        self.assertAlmostEqual(row["gain_abs"], 10.0)
        self.assertAlmostEqual(row["gain_rel"], 0.1)
        self.assertAlmostEqual(row["guard_gain_abs"], 10.0)
        self.assertEqual(row["experiments"], 1)
        self.assertAlmostEqual(row["training_seconds"], 1.5)

    def test_error_batch_is_not_compared(self):
        row = batch_result(record("b01", 1, 100.0, 90.0, error="boom"))
        self.assertIsNone(row["selected_rmse"])
        self.assertIsNone(row["gain_rel"])

    def test_summary_counts_gain_regret_and_floor_violations(self):
        rows = [
            batch_result(record("b01", 1, 100.0, 90.0, attempts=2)),
            batch_result(record("b02", 2, 100.0, 102.0, decision="rollback")),
            batch_result(record("b03", 3, 100.0, 100.0, decision="hold")),
        ]
        summary = summarize(rows, wall_seconds=12.0)
        self.assertEqual(summary["batches"], 3)
        self.assertEqual(summary["attempts"], 4)
        self.assertEqual(summary["improved_batches"], 1)
        self.assertEqual(summary["equal_batches"], 1)
        self.assertEqual(summary["worse_batches"], 1)
        self.assertEqual(summary["static_floor_violations"], 1)
        self.assertAlmostEqual(summary["net_rmse_gain"], 8.0)
        self.assertAlmostEqual(summary["cumulative_gain"], 10.0)
        self.assertAlmostEqual(summary["cumulative_regret"], 2.0)
        self.assertAlmostEqual(summary["mean_gain_rel"], (0.1 - 0.02 + 0.0) / 3)
        self.assertAlmostEqual(summary["mean_static_rmse"], 100.0)
        self.assertAlmostEqual(summary["mean_selected_rmse"], 292.0 / 3)
        self.assertEqual(summary["decisions"], {"promote": 1, "rollback": 1, "hold": 1})
        self.assertEqual(summary["training_seconds"], 6.0)

    def test_empty_or_noncomparable_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize([], wall_seconds=0.0)
        with self.assertRaises(ValueError):
            summarize([batch_result(record("b01", 1, 100.0, 90.0, error="boom"))],
                      wall_seconds=0.0)


if __name__ == "__main__":
    unittest.main()

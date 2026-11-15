import json
from pathlib import Path
import unittest

import numpy as np

from harness.config import load_config
from harness.drift import analyze_drift, describe_feature

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "harness" / "configs" / "default.json").drift


def metrics(rmse=10.0):
    return {"rmse": rmse, "mae": rmse * 0.8, "n": 40}


class DriftTests(unittest.TestCase):
    def report(self, *, x=None, y=None, champion=10.0, reference=10.0, static=10.0):
        if x is None:
            x = np.zeros((40, 3))
        if y is None:
            y = np.full(40, 100.0)
        return analyze_drift(
            x=np.asarray(x, dtype=np.float64), y=np.asarray(y, dtype=np.float64),
            feature_mean=np.zeros(3), feature_scale=np.ones(3),
            feature_paths=['[["key","sensors"],["index",0]]',
                           '[["key","sensors"],["index",1]]',
                           '[["key","sensors"],["index",2]]'],
            target_mean=100.0, target_scale=50.0,
            champion_metrics=metrics(champion), reference_metrics=metrics(reference),
            static_metrics=metrics(static), config=CONFIG,
        )

    def test_stable_batch_has_no_findings(self):
        report = self.report()
        self.assertEqual(report.severity, "stable")
        self.assertFalse(report.drifted)
        self.assertEqual(report.findings, [])
        json.dumps(report.to_dict(), allow_nan=False)

    def test_feature_shift_is_critical(self):
        x = np.zeros((40, 3))
        x[:, 0] = 5.0
        report = self.report(x=x)
        self.assertEqual(report.severity, "critical")
        codes = {finding.code for finding in report.findings}
        self.assertIn("input_distribution_shift", codes)
        self.assertTrue(any(item["feature"] == "sensors[0]" for item in report.signals["top_features"]))

    def test_target_shift_is_detected(self):
        y = np.full(40, 160.0)
        report = self.report(y=y)
        self.assertIn("target_distribution_shift", {finding.code for finding in report.findings})
        self.assertGreaterEqual(report.signals["target_mean_shift"], 1.0)

    def test_performance_degradation_is_detected(self):
        report = self.report(champion=14.0, reference=10.0, static=10.0)
        findings = {finding.code for finding in report.findings}
        self.assertIn("performance_degradation", findings)
        self.assertAlmostEqual(report.signals["performance_delta"], 0.4)

    def test_champion_below_static_is_detected(self):
        report = self.report(champion=12.0, reference=12.0, static=10.0)
        findings = {finding.code for finding in report.findings}
        self.assertIn("champion_below_static", findings)

    def test_feature_path_rendering(self):
        self.assertEqual(describe_feature('[["key","settings"],["index",2]]'), "settings[2]")
        self.assertEqual(describe_feature("not-json"), "not-json")

    def test_schema_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_drift(
                x=np.zeros((4, 3)), y=np.zeros(4),
                feature_mean=np.zeros(2), feature_scale=np.ones(2), feature_paths=["a", "b"],
                target_mean=0.0, target_scale=1.0,
                champion_metrics=metrics(), reference_metrics=metrics(), static_metrics=metrics(),
                config=CONFIG,
            )


if __name__ == "__main__":
    unittest.main()

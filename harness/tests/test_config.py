import json
from pathlib import Path
import unittest

from harness.config import DEFAULTS, config_to_dict, from_mapping, load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "harness" / "configs" / "default.json"


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_and_round_trips(self):
        config = load_config(DEFAULT)
        self.assertEqual(config.profile, "C")
        self.assertEqual(config.task, "regression")
        self.assertGreaterEqual(config.experiments.min_attempts, 1)
        rebuilt = from_mapping(config_to_dict(config))
        self.assertEqual(rebuilt, config)
        self.assertEqual(config, from_mapping(DEFAULTS))

    def test_partial_override_merges_with_defaults(self):
        config = from_mapping({"gate": {"min_improvement": 0.02}})
        self.assertAlmostEqual(config.gate.min_improvement, 0.02)
        self.assertAlmostEqual(config.drift.performance_warn, 0.05)
        self.assertEqual(config.experiments.max_candidates, 3)

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown configuration"):
            from_mapping({"gate": {"min_improvements": 0.02}})

    def test_inconsistent_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "critical must be"):
            from_mapping({"drift": {"performance_warn": 0.2, "performance_critical": 0.1}})
        with self.assertRaisesRegex(ValueError, "critical must be"):
            from_mapping({"drift": {"static_warn": 0.2, "static_critical": 0.1}})

    def test_only_regression_rmse_is_supported(self):
        with self.assertRaisesRegex(ValueError, "supports the CPU-C"):
            from_mapping({"task": "classification"})
        with self.assertRaisesRegex(ValueError, "profile C"):
            from_mapping({"profile": "A"})

    def test_invalid_list_and_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            from_mapping({"experiments": {"learning_rate_multipliers": []}})
        with self.assertRaises(ValueError):
            from_mapping({"logging": {"swanlab": "sometimes"}})

    def test_json_config_file_is_read(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"guard": {"rows": 17}}), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.guard.rows, 17)


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from harness.config import load_config
from harness.doctor import collect_environment

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "cpu-c-guide-20261110-v1"
DEFAULT = ROOT / "harness" / "configs" / "default.json"


class DoctorTests(unittest.TestCase):
    def test_offline_environment_report(self):
        config = load_config(DEFAULT)
        with tempfile.TemporaryDirectory() as temp:
            report = collect_environment(baseline_root=BASELINE, workdir=temp, config=config)
            self.assertEqual(report["format"], "harness.doctor.v1")
            self.assertFalse(report["gpu_required"])
            self.assertFalse(report["network_required"])
            self.assertTrue(report["offline_capable"])
            self.assertTrue(report["checks"]["numpy_available"])
            self.assertTrue(report["checks"]["baseline_integrity"])
            self.assertTrue(report["checks"]["workdir_writable"])
            self.assertTrue(report["baseline"]["checkpoint_sha256"].startswith("sha256:"))
            self.assertEqual(report["errors"], [])

    def test_cloud_mode_requires_network(self):
        config = load_config(DEFAULT)
        config = replace(config, logging=replace(config.logging, swanlab="cloud"))
        with tempfile.TemporaryDirectory() as temp:
            report = collect_environment(baseline_root=BASELINE, workdir=temp, config=config)
            self.assertTrue(report["network_required"])
            self.assertFalse(report["offline_capable"])


if __name__ == "__main__":
    unittest.main()

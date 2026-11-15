from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from harness.config import load_config
from harness.swanlab_logger import SwanLabLogger

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "harness" / "configs" / "default.json").logging


class FakeRun:
    id = "fake-run"
    dir = "fake-run-dir"

    @property
    def url(self):
        raise ValueError("url unavailable offline")


class FakeSwanLab:
    def __init__(self, fail_log=False, fail_text=False, legacy_only=False):
        self.calls = {"init": [], "log": [], "text": [], "finish": 0}
        self.fail_log = fail_log
        self.fail_text = fail_text
        self.legacy_only = legacy_only

    def init(self, **kwargs):
        if self.legacy_only and "name" in kwargs:
            raise TypeError("unexpected keyword argument 'name'")
        self.calls["init"].append(kwargs)
        return FakeRun()

    def log(self, data, step=None):
        if self.fail_log:
            raise RuntimeError("log backend down")
        self.calls["log"].append((data, step))

    def log_text(self, key, data, step=None):
        if self.fail_text:
            raise RuntimeError("text backend down")
        self.calls["text"].append((key, data, step))

    def finish(self):
        self.calls["finish"] += 1


def offline_config(logdir="logs/swanlab"):
    return replace(CONFIG, swanlab="offline", logdir=logdir)


class SwanLabLoggerTests(unittest.TestCase):
    def test_offline_init_logs_text_and_finish(self):
        fake = FakeSwanLab()
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"swanlab": fake}):
            logger = SwanLabLogger(offline_config(), workdir=temp, run_config={"task": "C"})
            self.assertTrue(logger.enabled)
            self.assertEqual(fake.calls["init"][0]["mode"], "offline")
            self.assertEqual(fake.calls["init"][0]["project"], CONFIG.project)
            self.assertEqual(fake.calls["init"][0]["name"], CONFIG.experiment)
            self.assertEqual(fake.calls["init"][0]["config"], {"task": "C"})
            self.assertTrue(fake.calls["init"][0]["log_dir"].startswith(str(Path(temp).resolve())))
            self.assertEqual(logger.run_id, "fake-run")
            self.assertIsNone(logger.url)
            logger.log({"rmse": 1.0}, step=2)
            logger.log_text("decision/batch-01", "promote", step=2)
            logger.finish()
            self.assertEqual(fake.calls["log"], [({"rmse": 1.0}, 2)])
            self.assertEqual(fake.calls["text"], [("decision/batch-01", "promote", 2)])
            self.assertEqual(fake.calls["finish"], 1)

    def test_legacy_init_signature_is_supported(self):
        fake = FakeSwanLab(legacy_only=True)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"swanlab": fake}):
            logger = SwanLabLogger(offline_config(), workdir=temp)
            self.assertTrue(logger.enabled)
            self.assertEqual(len(fake.calls["init"]), 1)
            self.assertIn("experiment_name", fake.calls["init"][0])

    def test_missing_package_disables_offline_but_raises_when_required(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"swanlab": None}):
            logger = SwanLabLogger(offline_config(), workdir=temp)
            self.assertFalse(logger.enabled)
            self.assertIn("unavailable", logger.disabled_reason)
            with self.assertRaises(RuntimeError):
                SwanLabLogger(replace(offline_config(), swanlab="required"), workdir=temp)

    def test_log_failure_disables_backend_but_text_failure_does_not(self):
        fake = FakeSwanLab(fail_log=True)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"swanlab": fake}):
            logger = SwanLabLogger(offline_config(), workdir=temp)
            logger.log({"rmse": 1.0}, step=1)
            self.assertFalse(logger.enabled)
        fake = FakeSwanLab(fail_text=True)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"swanlab": fake}):
            logger = SwanLabLogger(offline_config(), workdir=temp)
            logger.log_text("key", "text", step=1)
            self.assertTrue(logger.enabled)

    def test_off_mode_never_imports(self):
        with patch.dict(sys.modules, {"swanlab": None}):
            logger = SwanLabLogger(replace(CONFIG, swanlab="off"))
            self.assertFalse(logger.enabled)
            self.assertEqual(logger.mode, "off")


if __name__ == "__main__":
    unittest.main()

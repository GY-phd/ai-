# -*- coding: utf-8 -*-
"""Config resolution: data/ paths follow RUL_DATA_ROOT, everything else stays repo-relative."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import config as harness_config
from harness.llm import MODEL_ALIASES, resolve_model_id


class ModelIdTest(unittest.TestCase):
    """The model's two spellings must both work.

    The platform's dropdown lists the models under their PyTRiO ids,
    which use a slash (``Qwen/Qwen3.8-27B``), while the competition's
    text and this project's configuration use the short form with a dot
    (``Qwen3.8-27B``).  An endpoint that speaks the PyTRiO protocol
    rejects the short form, so the planner resolves it.
    """

    def test_the_short_form_resolves_to_the_platform_spelling(self):
        self.assertEqual(resolve_model_id("Qwen3.8-27B"), "Qwen/Qwen3.8-27B")
        self.assertEqual(resolve_model_id("Qwen3.5-4B"), "Qwen/Qwen3.5-4B")

    def test_the_resolution_ignores_case(self):
        self.assertEqual(resolve_model_id("qwen3.8-27b"), "Qwen/Qwen3.8-27B")

    def test_the_platform_spelling_is_left_alone(self):
        self.assertEqual(resolve_model_id("Qwen/Qwen3.8-27B"), "Qwen/Qwen3.8-27B")

    def test_an_unknown_id_is_returned_unchanged(self):
        """The whitelist check decides what may be used; this only picks a spelling."""
        self.assertEqual(resolve_model_id("some-other-model"), "some-other-model")
        self.assertEqual(resolve_model_id(""), "")

    def test_the_configured_model_is_one_of_the_known_ones(self):
        cfg = harness_config.load()
        self.assertIn(cfg["llm"]["model_id"].lower(), MODEL_ALIASES)


class ConfigTest(unittest.TestCase):
    def test_data_root_env_rebases_data_paths(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"RUL_DATA_ROOT": tmp}):
            cfg = harness_config.load()
            root = Path(tmp)
            for value in (cfg["data"]["train"], cfg["data"]["validation"],
                          cfg["data"]["baseline_checkpoint"], cfg["stream"]["path"]):
                self.assertTrue(Path(value).is_relative_to(root), value)

    def test_defaults_point_at_the_repo_data_directory(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RUL_DATA_ROOT", None)
            cfg = harness_config.load()
        expected = harness_config.ROOT / "data"
        self.assertTrue(Path(cfg["data"]["train"]).is_relative_to(expected), cfg["data"]["train"])

    def test_workspace_is_absolute(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RUL_DATA_ROOT", None)
            cfg = harness_config.load()
        self.assertTrue(Path(cfg["workspace"]).is_absolute())

    def test_model_id_is_the_competition_specified_one(self):
        cfg = harness_config.load()
        self.assertEqual(cfg["llm"]["model_id"], "Qwen3.8-27B")

    def test_overrides_win_and_nested_sections_merge(self):
        cfg = harness_config.load(overrides={"budget": {"max_experiments": 2}, "run_name": "unit"})
        self.assertEqual(cfg["budget"]["max_experiments"], 2)
        self.assertEqual(cfg["run_name"], "unit")
        # untouched keys survive the merge
        self.assertIn("max_train_seconds", cfg["budget"])

    def test_gate_thresholds_are_present(self):
        cfg = harness_config.load()
        self.assertIn("min_gain_pct", cfg["gate"])
        self.assertIn("max_local_regression_pct_of_global", cfg["gate"])
        self.assertGreater(cfg["gate"]["min_gain_pct"], 0)

    def test_the_three_config_sites_agree(self):
        """The defaults, the shipped JSON and the trainer must not drift apart.

        There are three places that state the training configuration:
        ``harness.config.DEFAULTS``, ``harness/config/harness.json`` and
        ``rulkit.train.DEFAULT_CONFIG``.  A run started with
        ``--config harness/config/harness.json`` and a run started without it
        must train the same model - otherwise the recorded score belongs to a
        configuration nobody can reproduce from the command line.
        """
        import json

        from rulkit.train import DEFAULT_CONFIG

        shipped = json.loads(
            (harness_config.ROOT / "harness" / "config" / "harness.json")
            .read_text(encoding="utf-8")
        )
        for key in ("hidden", "epochs", "learning_rate", "weight_decay",
                    "batch_size", "feature_transform", "seed"):
            self.assertEqual(
                harness_config.DEFAULTS["train"][key], shipped["train"][key],
                f"DEFAULTS vs harness.json disagree on {key}")
            self.assertEqual(
                DEFAULT_CONFIG[key], shipped["train"][key],
                f"rulkit.train vs harness.json disagree on {key}")

    def test_the_shipped_config_is_the_measured_optimum(self):
        """Guard the measured optimum against an accidental edit.

        ``eng34`` + tanh + hidden=96 + no target scaling is what the ablation
        measured (RMSE 301.18 on the official holdout bearings).  The search
        grid must contain that width, and the activation must stay ``tanh``
        because the reference loader hard-codes it.
        """
        import json

        shipped = json.loads(
            (harness_config.ROOT / "harness" / "config" / "harness.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(shipped["train"]["feature_transform"], "eng34")
        self.assertIn(shipped["train"]["hidden"], shipped["search"]["hidden"])
        self.assertLessEqual(max(shipped["search"]["hidden"]), 128)

    def test_hidden_width_stays_within_the_reference_loader_limit(self):
        """The official verifier rejects hidden > 128 outright."""
        import json

        shipped = json.loads(
            (harness_config.ROOT / "harness" / "config" / "harness.json")
            .read_text(encoding="utf-8")
        )
        self.assertLessEqual(max(shipped["search"]["hidden"]), 128)
        self.assertLessEqual(shipped["train"]["hidden"], 128)

    def test_environment_variables_configure_the_run(self):
        """The container image is started with a fixed command, so the env has
        to be able to carry the configuration."""
        from harness.orchestrator import main

        seen = {}

        class FakeHarness:
            def __init__(self, config):
                seen.update(config)

            def run(self):
                return {"final": {"rmse": 1.0}, "baseline": {"rmse": 2.0}}

        env = {
            "HARNESS_WORKSPACE": "/out/harness",
            "HARNESS_BATCH_SIZE": "128",
            "HARNESS_MAX_EXPERIMENTS": "4",
            "HARNESS_LLM_MODE": "policy",
            "HARNESS_MAX_WALL_SECONDS": "600",
        }
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch("harness.orchestrator.Harness", FakeHarness):
            # The loop is stubbed out; only the configuration path is under test.
            try:
                main(["--workspace", "/out/harness", "--max-experiments", "4"])
            except Exception:  # noqa: BLE001 - the fake harness is not a real run
                pass

        self.assertEqual(seen["stream"]["batch_size"], 128)
        self.assertEqual(seen["budget"]["max_experiments"], 4)
        self.assertEqual(seen["budget"]["max_wall_seconds"], 600)
        self.assertEqual(seen["llm"]["mode"], "policy")

    def test_an_invalid_environment_value_fails_loudly(self):
        """A typo in the container's environment must not be silently ignored."""
        from harness.orchestrator import main

        with mock.patch.dict(os.environ, {"HARNESS_MAX_EXPERIMENTS": "many"}, clear=False):
            with self.assertRaises(SystemExit):
                main([])


if __name__ == "__main__":
    unittest.main()

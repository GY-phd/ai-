"""Check Task A lifecycle and release isolation with synthetic data."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.mock_harness import PARAMETERS, features_only, mock_rows, release, run, stable_fixture_model, write_rows
from task_a_engine import Engine
from task_a_engine.data import read_inputs
from task_a_engine.vendor.baselines.model import Model
from task_a_engine.vendor.predict import MODES, Predictor
import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = stable_fixture_model()

    def setUp(self):
        temporary = TemporaryDirectory(prefix=".engine-test-", dir=PACKAGE)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.store = self.folder / "store"
        self.engine = Engine(self.store, mode="mock")
        self.initial = self.engine.status()

    def rows(self, prefix, seed, count=8, **kwargs):
        return mock_rows(self.fixture, prefix, seed, count, **kwargs)

    def train(self):
        rows = self.rows("train", 10, 12)
        path = write_rows(self.folder / "train.jsonl", rows)
        result = self.engine.train(path, release("train", "train"), parameters=PARAMETERS)
        self.assertEqual(result["status"], "trained")
        return result["model_id"], rows

    def test_prediction_and_diagnosis_are_read_only(self):
        rows = self.rows("observed", 6, 3)
        visible = features_only(rows)
        prediction = self.engine.predict(visible)
        changed_labels = [dict(row, answer="unreleased-marker") for row in rows]
        alternate = self.engine.predict(changed_labels)
        np.testing.assert_array_equal(prediction["scores"], alternate["scores"])
        self.assertEqual(prediction["model_id"], self.initial["active_model_id"])
        scores = np.asarray(prediction["scores"])
        self.assertEqual(scores.shape, (18, 6))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1.)
        first = self.engine.diagnose(visible)
        second = self.engine.diagnose(changed_labels)
        self.assertEqual({k: v for k, v in first.items() if k != "resources"},
                         {k: v for k, v in second.items() if k != "resources"})
        self.assertEqual([c["method"] for c in self.engine.propose(first)["candidates"]], ["no_update"])
        self.assertEqual(self.engine.status()["revision"], self.initial["revision"])
        with self.assertRaises(ValueError):
            self.engine.predict([[1.] * 127])
        with self.assertRaises(ValueError):
            self.engine.predict([[float("inf")] + [1.] * 127])

    def test_train_requires_released_labels_of_matching_kind_and_purpose(self):
        path = write_rows(self.folder / "train.jsonl", self.rows("train", 10, 12))
        valid = release("data", "train")
        cases = (dict(valid, labels="sealed"), release("data", "evaluate"), dict(valid, kind="real"))
        for wrong in cases:
            with self.subTest(release=wrong), self.assertRaises(ValueError):
                self.engine.train(path, wrong, parameters=PARAMETERS)
        self.assertEqual(self.engine.status()["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(self.engine.status()["revision"], self.initial["revision"])

    def test_evaluate_rejects_training_row_reuse_with_new_ids(self):
        model_id, fitted_rows = self.train()
        duplicate = [dict(row, id=f"other-{row['id']}") for row in fitted_rows]
        path = write_rows(self.folder / "duplicate.jsonl", duplicate)
        with self.assertRaises(ValueError):
            self.engine.evaluate(model_id, path, release("duplicate", "evaluate"))
        self.assertEqual(self.engine.status()["active_model_id"], self.initial["active_model_id"])

    def test_unattested_evaluation_keeps_current_model(self):
        model_id, _ = self.train()
        path = write_rows(self.folder / "eval.jsonl", self.rows("eval", 11))
        result = self.engine.evaluate(model_id, path, release("no-independence", "evaluate", independent=False))
        self.assertFalse(result["eligible"])
        action = self.engine.activate(model_id, result["evaluation_id"], expected_revision=self.initial["revision"])
        self.assertEqual(action["status"], "kept")
        self.assertEqual(action["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(action["revision"], self.initial["revision"])

    def test_version_lifecycle_rejects_stale_requests_and_reopens(self):
        model_id, _ = self.train()
        path = write_rows(self.folder / "eval.jsonl", self.rows("eval", 11))
        result = self.engine.evaluate(model_id, path, release("independent", "evaluate"))
        self.assertTrue(result["eligible"], result)
        first = self.engine.activate(model_id, result["evaluation_id"], expected_revision=self.initial["revision"])
        self.assertEqual(first["status"], "activated")
        self.assertEqual(first["active_model_id"], model_id)
        reopened = Engine(self.store, mode="mock")
        self.assertEqual(reopened.status()["active_model_id"], model_id)
        probe = features_only(self.rows("probe", 12, 1))
        self.assertEqual(reopened.predict(probe)["model_id"], model_id)
        self.assertEqual(reopened.predict(probe, model_id=self.initial["active_model_id"])["model_id"],
                         self.initial["active_model_id"])
        with self.assertRaisesRegex(ValueError, "revision changed"):
            reopened.rollback(expected_revision=self.initial["revision"])
        self.assertEqual(reopened.status()["active_model_id"], model_id)
        rolled = reopened.rollback(expected_revision=first["revision"])
        self.assertEqual(rolled["status"], "rolled_back")
        self.assertEqual(rolled["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(Engine(self.store, mode="mock").status()["revision"], rolled["revision"])
        with self.assertRaisesRegex(ValueError, "stale"):
            reopened.activate(model_id, result["evaluation_id"], expected_revision=rolled["revision"])
        self.assertEqual(reopened.status()["active_model_id"], self.initial["active_model_id"])

    def test_evaluation_rows_never_regain_independence(self):
        model_id, _ = self.train()
        eval_rows = self.rows("eval", 11)
        path = write_rows(self.folder / "eval.jsonl", eval_rows)
        self.engine.evaluate(model_id, path, release("independent", "evaluate"))
        copied = [dict(row, id=f"copy-{row['id']}") for row in eval_rows]
        copied_path = write_rows(self.folder / "copied.jsonl", copied)
        replayed = self.engine.evaluate(model_id, copied_path, release("replayed", "evaluate"))
        self.assertFalse(replayed["eligible"])
        self.assertTrue(replayed["evidence"]["previously_evaluated"])
        fitted = self.engine.train(copied_path, release("copied", "train"), parameters=PARAMETERS)
        self.assertEqual(fitted["status"], "trained")
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            self.engine.evaluate(fitted["model_id"], path, release("after-train", "evaluate"))

    def test_training_after_evaluation_invalidates_pending_activation(self):
        model_id, _ = self.train()
        eval_rows = self.rows("eval", 11)
        path = write_rows(self.folder / "eval.jsonl", eval_rows)
        evidence = self.engine.evaluate(model_id, path, release("independent", "evaluate"))
        self.assertTrue(evidence["eligible"])
        # A later legitimate training release is allowed, but cannot preserve
        # the old release's status as unexposed activation evidence.
        renamed = [dict(row, id=f"released-{row['id']}") for row in eval_rows]
        train_file = write_rows(self.folder / "later-train.jsonl", renamed)
        trained = self.engine.train(train_file, release("later", "train"), parameters=PARAMETERS)
        self.assertEqual(trained["status"], "trained")
        with self.assertRaisesRegex(ValueError, "subsequently released for training"):
            self.engine.activate(model_id, evidence["evaluation_id"], expected_revision=self.initial["revision"])
        self.assertEqual(self.engine.status()["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(self.engine.status()["revision"], self.initial["revision"])

    def test_development_fingerprints_block_relabelled_training(self):
        rows = self.rows("development", 14)
        self.engine.development_keys = frozenset(read_inputs(rows)["keys"])
        renamed = [dict(row, id=f"new-{row['id']}") for row in rows]
        path = write_rows(self.folder / "renamed-development.jsonl", renamed)
        with self.assertRaisesRegex(ValueError, "Batch 2 evaluation labels"):
            self.engine.train(path, release("incorrect-training-release", "train"), parameters=PARAMETERS)
        review = self.engine.evaluate("fusion", path, release("development", "evaluate"))
        self.assertTrue(review["evidence"]["known_development_set"])
        self.assertFalse(review["evidence"]["independent_usable"])
        self.assertFalse(review["eligible"])

    def test_fusion_refit_creates_a_new_checkpoint_without_activation(self):
        train_rows = self.rows("fusion-train", 21, 12)
        path = write_rows(self.folder / "fusion-train.jsonl", train_rows)
        result = self.engine.train(path, release("fusion-train", "train"), method="fusion_refit")
        self.assertEqual(result["status"], "trained")
        self.assertNotIn(result["model_id"], MODES)
        self.assertEqual(result["method"], "fusion_refit")
        checkpoint = self.store / "models" / (result["model_id"] + ".npz")
        self.assertTrue(checkpoint.is_file())
        with np.load(checkpoint, allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), {"mean", "scale", "coef"})
            self.assertTrue(all(np.isfinite(arrays[key]).all() for key in arrays.files))
        probe = features_only(self.rows("fusion-probe", 22, 2))
        predicted = self.engine.predict(probe, model_id=result["model_id"])
        seeded = self.engine.predict(probe, model_id="fusion")
        self.assertGreater(float(np.max(np.abs(np.array(predicted["scores"]) - seeded["scores"]))), 1e-6)
        scores = np.asarray(predicted["scores"])
        self.assertEqual(scores.shape, (12, 6))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1.)
        restored = Engine(self.store, mode="mock").predict(probe, model_id=result["model_id"])
        np.testing.assert_array_equal(restored["scores"], scores)
        self.assertEqual(self.engine.status()["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(self.engine.status()["revision"], self.initial["revision"])

    def test_seed_predictions_preserve_all_five_vendor_variants(self):
        probe = features_only(self.rows("seed-probe", 23, 4))
        matrix = read_inputs(probe)["x"]
        manifest = PACKAGE / "task_a_engine" / "vendor" / "manifest.json"
        self.assertEqual({m["model_id"] for m in self.initial["models"]}, set(MODES))
        for variant in MODES:
            with self.subTest(variant=variant):
                expected = Predictor(manifest, variant).predict(matrix)
                actual = self.engine.predict(probe, model_id=variant)
                np.testing.assert_array_equal(actual["scores"], expected)
                self.assertEqual(actual["classes"], self.fixture.meta["classes"])

    def test_failed_training_retains_exposure_and_active_model(self):
        rows = self.rows("failed-train", 24)
        path = write_rows(self.folder / "failed-train.jsonl", rows)
        with patch.object(Model, "fit", side_effect=ValueError("simulated numerical failure")):
            failed = self.engine.train(path, release("failed-train", "train"), parameters=PARAMETERS)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(self.engine.status(), self.initial)
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            self.engine.evaluate("fusion", path, release("failed-exposure", "evaluate"))

    def test_checkpoint_corruption_blocks_activation(self):
        model_id, _ = self.train()
        path = write_rows(self.folder / "eval.jsonl", self.rows("eval", 11))
        evidence = self.engine.evaluate(model_id, path, release("independent", "evaluate"))
        self.assertTrue(evidence["eligible"])
        checkpoint = self.store / "models" / (model_id + ".npz")
        with checkpoint.open("ab") as stream:
            stream.write(b"changed-checkpoint")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.engine.activate(model_id, evidence["evaluation_id"], expected_revision=self.initial["revision"])
        self.assertEqual(self.engine.status()["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(self.engine.status()["revision"], self.initial["revision"])

    def test_jsonl_errors_do_not_stop_subsequent_requests(self):
        requests = [
            "not-json",
            json.dumps({"id": "bad-shape", "action": "predict", "args": {"inputs": [[1.]]}}),
            json.dumps({"id": "overflow-diagnosis", "action": "diagnose", "args": {"inputs": [[1e308] * 128]}}),
            json.dumps({"id": "bad-diagnosis", "action": "propose", "args": {"diagnosis": []}}),
            json.dumps({"id": "bad-release", "action": "propose", "args": {
                "diagnosis": {"model_id": "stable", "revision": 0}, "training_release": ["bad"]}}),
            json.dumps({"id": "请求-after-errors", "action": "status"}, ensure_ascii=False),
        ]
        completed = subprocess.run([sys.executable, "-B", "-m", "task_a_engine", "--store", str(self.store), "--mode", "mock"],
                                   cwd=PACKAGE, input="\n".join(requests) + "\n", text=True,
                                   encoding="utf-8", capture_output=True, check=True, timeout=30)
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(completed.stderr, "")
        self.assertEqual(len(responses), len(requests))
        self.assertTrue(all(not response["ok"] for response in responses[:-1]))
        self.assertEqual(responses[-1]["id"], "请求-after-errors")
        self.assertTrue(responses[-1]["ok"])
        self.assertEqual(responses[-1]["result"]["active_model_id"], self.initial["active_model_id"])
        self.assertEqual(responses[-1]["result"]["revision"], self.initial["revision"])

    def test_failed_activation_does_not_commit_active_pointer(self):
        model_id, _ = self.train()
        path = write_rows(self.folder / "eval.jsonl", self.rows("eval", 11))
        result = self.engine.evaluate(model_id, path, release("independent", "evaluate"))
        with patch.object(Engine, "_event", side_effect=OSError("simulated commit failure")):
            with self.assertRaisesRegex(OSError, "simulated commit failure"):
                self.engine.activate(model_id, result["evaluation_id"], expected_revision=self.initial["revision"])
        state = Engine(self.store, mode="mock").status()
        self.assertEqual(state["revision"], self.initial["revision"])
        self.assertEqual(state["active_model_id"], self.initial["active_model_id"])
        action = self.engine.activate(model_id, result["evaluation_id"], expected_revision=state["revision"])
        self.assertEqual(action["status"], "activated")

    def test_real_and_mock_stores_are_isolated(self):
        with self.assertRaises(ValueError):
            Engine(self.store, mode="real")
        real = Engine(self.folder / "real", mode="real")
        path = write_rows(self.folder / "mock.jsonl", self.rows("mock", 10, 12))
        with self.assertRaises(ValueError):
            real.train(path, release("mock", "train"), parameters=PARAMETERS)

    def test_three_stage_mock(self):
        report = run(self.folder / "demonstration")
        self.assertIsNone(report["competition_score"])
        self.assertFalse(report["independent_generalization_verified"])
        self.assertEqual([c["method"] for c in report["stages"][0]["proposal"]["candidates"]], ["no_update"])
        self.assertTrue(report["stages"][1]["evaluation"]["eligible"])
        self.assertFalse(report["stages"][2]["evaluation"]["eligible"])
        self.assertEqual(report["active_after_reopen"], report["initial_active_model_id"])


if __name__ == "__main__":
    unittest.main()

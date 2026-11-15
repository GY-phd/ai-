"""Regression tests of update isolation and optional Engine integration."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from recipe import ROOT, StructuredModel, anchor_update, np
from engine_recipe import RecipeEngine
from task_a_engine.engine import CLASSES


def release(name, purpose="train", independent=False):
    return {"release_id": name, "purpose": purpose, "labels": "released", "kind": "mock", "independent": independent}


class RecipeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix=".test-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.engine = RecipeEngine(self.folder / "store", mode="mock")
        self.source, self.x, self.y = self.dataset("source", 8, 1)
        self.cal, self.cx, self.cy = self.dataset("cal", 2, 2)
        self.evaluation, _, _ = self.dataset("eval", 3, 3)
        self.bootstrap = self.engine.train(self.source, release("source"), method="structured_bootstrap")
        self.assertEqual(self.bootstrap["status"], "trained")

    def dataset(self, name, per_class, seed):
        y = np.repeat(np.arange(6), per_class)
        rng = np.random.default_rng(seed)
        x = np.exp(rng.normal(2 + y[:, None] / 4, .2, (len(y), 128)))
        x.reshape(-1, 16, 8)[:, :, 5:] *= -1
        path = self.folder / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps({"id": f"{name}-{i}", "inputs": {"features": row.tolist()},
                                            "answer": CLASSES[label]}) for i, (row, label) in enumerate(zip(x, y))), encoding="utf-8")
        return path, x, y

    def update(self, parent=None, **kwargs):
        arguments = {"parent_id": parent or self.bootstrap["model_id"], "method": "C_replay_anchor",
                     "replay_path": self.source, "replay_release": release("source")}
        arguments.update(kwargs)
        return self.engine.train(self.cal, release("cal"), **arguments)

    def test_zero_and_missing_labels_are_finite_and_scaler_frozen(self):
        model = StructuredModel.fit(self.x, self.y)
        empty = anchor_update(model, self.cx[:0], self.cy[:0], self.x, self.y)
        np.testing.assert_array_equal(empty.coef, model.coef)
        subset = self.cy != 3
        updated = anchor_update(model, self.cx[subset], self.cy[subset], self.x, self.y)
        for key in ("center", "scale", "amplitude_scale"):
            np.testing.assert_array_equal(getattr(updated, key), getattr(model, key))
        self.assertTrue(np.isfinite(updated.predict(self.cx)).all())
        self.assertEqual(self.engine.train(method="no_update")["status"], "kept")

    def test_three_stage_lifecycle_and_mock_cannot_activate(self):
        model_id = self.update()["model_id"]
        self.assertEqual(self.engine.status()["active_model_id"], "stable")
        prediction = self.engine.predict(self.cx, model_id=model_id)
        diagnosis = self.engine.diagnose(self.cx, model_id=model_id)
        self.assertEqual(diagnosis["rows"], len(self.cx))
        self.assertEqual(prediction["resources"]["llm_calls"], 0)
        result = self.engine.evaluate(model_id, self.evaluation, release("eval", "evaluate", True))
        self.assertFalse(result["eligible"])
        self.assertEqual(self.engine.activate(model_id, result["evaluation_id"])["status"], "kept")
        with self.assertRaises(ValueError):
            self.engine.activate(model_id, result["evaluation_id"], expected_revision=999)
        second_path, _, _ = self.dataset("cal2", 1, 4)
        result2 = self.engine.train(second_path, release("cal2"), method="C_replay_anchor", parent_id=model_id,
                                   replay_path=self.source, replay_release=release("source"))
        self.assertEqual(result2["status"], "trained")
        reopened = RecipeEngine(self.folder / "store", mode="mock")
        self.assertEqual(reopened.predict(self.cx, model_id=result2["model_id"])["model_id"], result2["model_id"])
        self.assertEqual(reopened.rollback()["status"], "kept")
        proposed = reopened.propose(reopened.diagnose(self.cx), training_release=release("next"))
        self.assertIn("C_replay_anchor", [row["method"] for row in proposed["candidates"]])

    def test_training_and_evaluation_overlap_by_features_is_rejected(self):
        model_id = self.update()["model_id"]
        rows = [json.loads(line) for line in self.cal.read_text().splitlines()]
        for i, row in enumerate(rows):
            row["id"] = f"renamed-{i}"
        path = self.folder / "renamed.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError):
            self.engine.evaluate(model_id, path, release("renamed", "evaluate", True))
        self.engine.evaluate(model_id, self.evaluation, release("eval", "evaluate", False))
        with self.assertRaises(ValueError):
            self.engine.train(self.evaluation, release("eval-now-train"), parent_id=model_id, method="C_replay_anchor",
                              replay_path=self.source, replay_release=release("source"))

    def test_replay_and_lineage_consistency(self):
        with self.assertRaises(ValueError):
            self.update(replay_path=self.cal)
        changed_path, _, _ = self.dataset("changed-replay", 8, 7)
        with self.assertRaises(ValueError):
            self.update(replay_path=changed_path)
        model_id = self.update()["model_id"]
        with self.assertRaises(ValueError):
            self.update(parent=model_id)

    def test_unreleased_labels_and_development_identity_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.train(self.cal, {**release("sealed"), "labels": "sealed"}, method="structured_bootstrap")
        rows = [json.loads(line) for line in self.cal.read_text().splitlines()]
        rows[0]["id"] = "gas_b02_test-fixture"
        blocked = self.folder / "blocked.jsonl"
        blocked.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError):
            self.engine.train(blocked, release("blocked"), method="structured_bootstrap")

    def test_failed_fit_keeps_active_and_records_exposure(self):
        before = self.engine.status()
        with patch("engine_recipe.anchor_update", side_effect=np.linalg.LinAlgError("test injected failure")):
            result = self.update()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.engine.status(), before)
        self.assertEqual(len(list((self.engine.store / "models").glob("*.npz"))), 1)
        with self.assertRaises(ValueError):
            self.engine.evaluate(self.bootstrap["model_id"], self.cal, release("used", "evaluate", True))

    def test_tampered_checkpoint_is_rejected(self):
        model_id = self.update()["model_id"]
        with self.engine._db() as db:
            path = self.engine.store / self.engine._spec(db, model_id)["checkpoint"]
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(ValueError):
            self.engine.predict(self.cx, model_id=model_id)

    def test_recipe_matches_previous_balanced_twelve_label_formula(self):
        model = StructuredModel.fit(self.x, self.y)
        d, source = model.design(self.cx), model.design(self.x)
        w = len(self.y) / (6 * np.bincount(self.y)[self.y])
        gram = d.T @ d + np.eye(d.shape[1]) + len(self.cy) / len(self.y) * source.T @ (source * w[:, None])
        expected = model.coef + np.linalg.solve(gram, d.T @ (np.eye(6)[self.cy] - d @ model.coef))
        np.testing.assert_allclose(anchor_update(model, self.cx, self.cy, self.x, self.y).coef, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

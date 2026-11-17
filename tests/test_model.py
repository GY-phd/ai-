# -*- coding: utf-8 -*-
"""Checkpoint round-trip, reference compatibility window, and function-preserving widening."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rulkit.features import CHANNEL_FIELDS, flatten
from rulkit.model import Model


def fake_paths():
    inputs = {"features": {ch: {name: 1.0 for name in CHANNEL_FIELDS} for ch in ("horizontal", "vertical")}}
    paths, _ = flatten(inputs)
    return paths


def make_data(n=64, d=34, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.01, 20.0, size=(n, d))
    y = np.abs(rng.normal(0, 500, size=n))
    return x, y


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.paths = fake_paths()
        self.x, self.y = make_data()

    def test_fit_predict_shapes_and_non_negative_output(self):
        model = Model("regression", self.paths, [], self.x, self.y, hidden=16, seed=1, feature_transform="eng34")
        model.fit(self.x, self.y, 2)
        pred = model.predict(self.x)
        self.assertEqual(pred.shape, (len(self.x),))
        self.assertTrue(np.all(pred >= 0))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_save_load_round_trip_is_exact(self):
        model = Model("regression", self.paths, [], self.x, self.y, hidden=12, seed=2, feature_transform="eng34")
        model.fit(self.x, self.y, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            model.save(path)
            restored = Model.load(path)
        np.testing.assert_array_equal(model.predict(self.x), restored.predict(self.x))
        self.assertEqual(restored.meta["parameter_count"], model.meta["parameter_count"])
        self.assertEqual(restored.meta["feature_transform"], "eng34")

    def test_save_refuses_to_overwrite(self):
        model = Model("regression", self.paths, [], self.x, self.y, hidden=8, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            model.save(path)
            with self.assertRaises(ValueError):
                model.save(path)

    def test_expand_keeps_the_learned_function(self):
        model = Model("regression", self.paths, [], self.x, self.y, hidden=8, seed=4, feature_transform="eng34")
        model.fit(self.x, self.y, 3)
        before = model.predict(self.x)
        info = model.expand(24)
        after = model.predict(self.x)
        self.assertTrue(info["migrated"])
        self.assertEqual(model.meta["hidden"], 24)
        # new hidden units start with zero outgoing weight, so the output is unchanged
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-9)

    def test_expand_rejects_narrowing(self):
        model = Model("regression", self.paths, [], self.x, self.y, hidden=16, seed=5)
        with self.assertRaises(ValueError):
            model.expand(8)

    def test_parameter_budget_is_enforced(self):
        # 34x128 stays inside the reference window (4481 parameters)
        ok = Model("regression", self.paths, [], self.x, self.y, hidden=128, seed=6)
        self.assertEqual(ok.meta["parameter_count"], 34 * 128 + 128 + 128 + 1)
        # a wide schema blows past the 100k cap and must be refused
        big_paths = [f"p{i}" for i in range(4096)]
        with self.assertRaises(ValueError):
            Model("regression", big_paths, [], np.zeros((2, 4096)), np.zeros(2), hidden=128, seed=7)

    def test_mismatched_feature_width_is_rejected(self):
        model = Model("regression", self.paths, [], self.x, self.y, hidden=8, seed=8)
        with self.assertRaises(ValueError):
            model.predict(np.zeros((3, 33)))


if __name__ == "__main__":
    unittest.main()

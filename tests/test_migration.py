# -*- coding: utf-8 -*-
"""The tests for the continuation model's migration and optimiser semantics.

These guard the two properties that make "continue the parent checkpoint"
true rather than nominal:

* a width migration produces a **valid** checkpoint - every tensor has the
  right shape and the Adam moments have the new shape too, so the
  migrated model still validates and still trains; and
* the migration is a **continuation**: the epoch and step counters, the
  surviving weights and the carried Adam moments are kept, not reset.

The second property matters because the epoch counter drives the
shuffling permutation and the step counter drives Adam's bias
correction.  Resetting either would silently replay the parent's own
schedule against the wrong denominator.

Run:

    python -m pytest tests/test_migration.py -v
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from contest_baselines.model import Model  # noqa: E402


def _toy_model(hidden=32, seed=7):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (64, 5))
    y = rng.integers(0, 3, 64)
    return Model("classification", ["f%d" % i for i in range(5)],
                 ["c0", "c1", "c2"], x, y, hidden=hidden, seed=seed)


class ResizeTest(unittest.TestCase):
    """``resize()`` must produce a valid, trainable checkpoint."""

    def test_narrowing_keeps_the_checkpoint_valid(self):
        model = _toy_model(hidden=32)
        model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                  np.random.default_rng(2).integers(0, 3, 64), 2)
        model.resize(16)
        # A valid checkpoint means ``validate()`` passes, which is what the
        # platform's own reference loader checks.
        model.validate()
        self.assertEqual(model.meta["hidden"], 16)
        self.assertEqual(model.params["w1"].shape, (5, 16))
        self.assertEqual(model.params["w2"].shape, (16, 3))

    def test_widening_keeps_the_checkpoint_valid(self):
        model = _toy_model(hidden=16)
        model.resize(32)
        model.validate()
        self.assertEqual(model.meta["hidden"], 32)
        self.assertEqual(model.params["w1"].shape, (5, 32))
        self.assertEqual(model.params["w2"].shape, (32, 3))

    def test_the_adam_moments_take_the_new_shape(self):
        """The bug this guards: the moments were allocated with the old shape."""
        model = _toy_model(hidden=32)
        model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                  np.random.default_rng(2).integers(0, 3, 64), 2)
        self.assertTrue(np.any(model.second["w1"] != 0))
        model.resize(16)
        for name in ("w1", "b1", "w2", "b2"):
            self.assertEqual(model.first[name].shape, model.params[name].shape,
                             "the first moment of %s has the wrong shape" % name)
            self.assertEqual(model.second[name].shape, model.params[name].shape,
                             "the second moment of %s has the wrong shape" % name)

    def test_the_surviving_units_keep_their_weights(self):
        model = _toy_model(hidden=32)
        before = model.params["w1"].copy()
        model.resize(16)
        np.testing.assert_allclose(model.params["w1"][:, :16], before[:, :16])

    def test_the_surviving_units_keep_their_adam_state(self):
        model = _toy_model(hidden=32)
        model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                  np.random.default_rng(2).integers(0, 3, 64), 3)
        before_first = model.first["w1"].copy()
        before_second = model.second["w1"].copy()
        model.resize(16)
        np.testing.assert_allclose(model.first["w1"][:, :16], before_first[:, :16])
        np.testing.assert_allclose(model.second["w1"][:, :16], before_second[:, :16])

    def test_the_counters_are_carried_over(self):
        """The epoch counter drives the shuffle, the step counter the bias."""
        model = _toy_model(hidden=32)
        model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                  np.random.default_rng(2).integers(0, 3, 64), 3)
        epochs, steps = model.meta["epochs"], model.meta["steps"]
        model.resize(16)
        self.assertEqual(model.meta["epochs"], epochs)
        self.assertEqual(model.meta["steps"], steps)

    def test_a_migrated_model_still_trains(self):
        model = _toy_model(hidden=32)
        model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                  np.random.default_rng(2).integers(0, 3, 64), 2)
        model.resize(16)
        losses = model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                           np.random.default_rng(2).integers(0, 3, 64), 2)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_the_migrated_checkpoint_round_trips(self):
        model = _toy_model(hidden=32)
        model.fit(np.random.default_rng(1).normal(0, 1, (64, 5)),
                  np.random.default_rng(2).integers(0, 3, 64), 2)
        model.resize(16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            model.save(path)
            again = Model.load(path)
        self.assertEqual(again.meta["hidden"], 16)
        self.assertEqual(again.meta["epochs"], model.meta["epochs"])


class ContinuationTest(unittest.TestCase):
    """Continuing the parent must reproduce one continuous run."""

    def test_the_epoch_index_drives_the_shuffle(self):
        """Two runs of the same total epochs shuffle identically."""
        x = np.random.default_rng(1).normal(0, 1, (64, 5))
        y = np.random.default_rng(2).integers(0, 3, 64)
        straight = _toy_model(hidden=8)
        straight.fit(x, y, 4)
        split = _toy_model(hidden=8)
        split.fit(x, y, 2)
        split.fit(x, y, 2)
        for name in ("w1", "b1", "w2", "b2"):
            np.testing.assert_allclose(straight.params[name], split.params[name],
                                       err_msg="%s diverged" % name)


if __name__ == "__main__":
    unittest.main()

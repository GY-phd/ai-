# -*- coding: utf-8 -*-
"""The tests for the round's strategy layer.

Two invariants are guarded here, and both exist because breaking either
one would produce a candidate that the platform cannot score - which the
platform itself warns about: passing the offline tests does not mean the
training, the prediction or the competition requirements have passed.

**The feature schema never changes.**  The platform's prediction request
carries only ``{task, inputs}``: there is no sample id.  A column that
is derived from the id therefore cannot be reproduced when the hidden
rows are scored, and the reference loader refuses a checkpoint whose
schema differs from the data's.  Every strategy must therefore change
the *training procedure* only.

**Every strategy is measurable.**  A strategy that leaves the training
matrix or the model in a state that cannot be fitted, measured or
predicted is not a strategy.

Run:

    python -m pytest tests/test_strategies.py -v
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from contest_baselines.model import Model  # noqa: E402
from contest_baselines.strategies import (  # noqa: E402
    BANK, build_plan, epoch_budget, for_profile, prepare)

CLASSES = ["c0", "c1", "c2"]


def _model(task="classification", hidden=32, features=5):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (48, features))
    if task == "classification":
        y = rng.integers(0, len(CLASSES), 48).astype(np.float64)
    else:
        y = rng.uniform(10, 200, 48)
    return Model(task, ["f%d" % i for i in range(features)],
                 CLASSES if task == "classification" else [], x, y, hidden=hidden)


def _train(task="classification", features=5, units=3, per_unit=16):
    """A training set whose ids carry a unit, in the profiles' own format."""
    ids, x, y = [], [], []
    rng = np.random.default_rng(1)
    for unit in range(units):
        for index in range(per_unit):
            ids.append("fd001_train_u%04d_c%05d" % (unit, index))
            x.append(rng.normal(0, 1, features))
            y.append(unit % len(CLASSES) if task == "classification"
                     else float(100 + unit * 10 + index))
    return {
        "ids": ids,
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.int64 if task == "classification" else np.float64),
        "paths": ["f%d" % i for i in range(features)],
    }


class SchemaGuardTest(unittest.TestCase):
    """No strategy may change the feature schema."""

    def test_every_strategy_keeps_the_schema(self):
        for hypothesis in BANK:
            with self.subTest(hypothesis=hypothesis.key):
                model = _model()
                before = list(model.meta["paths"])
                train = _train()
                prepare(hypothesis, model, train, "A", 42)
                self.assertEqual(list(model.meta["paths"]), before,
                                 "%s changed the feature schema" % hypothesis.key)

    def test_every_strategy_keeps_the_matrix_width(self):
        for hypothesis in BANK:
            with self.subTest(hypothesis=hypothesis.key):
                model = _model()
                train = _train()
                x, y, _changed = prepare(hypothesis, model, train, "A", 42)
                self.assertEqual(x.shape[1], len(model.meta["paths"]))
                self.assertEqual(len(x), len(y))

    def test_the_migration_keeps_the_schema(self):
        """A width migration changes the hidden layer, never the inputs."""
        model = _model()
        before = list(model.meta["paths"])
        train = _train()
        prepare(for_profile("A")[4], model, train, "A", 42)
        self.assertEqual(list(model.meta["paths"]), before)


class MeasurableTest(unittest.TestCase):
    """Every strategy must leave a model that can be fitted and predicted."""

    def test_every_strategy_leaves_a_fittable_model(self):
        for hypothesis in BANK:
            with self.subTest(hypothesis=hypothesis.key):
                model = _model()
                train = _train()
                x, y, _changed = prepare(hypothesis, model, train, "A", 42)
                losses = model.fit(x, y, 1)
                self.assertEqual(len(losses), 1)
                self.assertTrue(np.isfinite(losses[0]))
                prediction = model.predict(x[:4])
                self.assertEqual(len(prediction), 4)
                self.assertTrue(np.all(np.isfinite(prediction)))

    def test_every_strategy_leaves_a_valid_checkpoint(self):
        import tempfile

        for hypothesis in BANK:
            with self.subTest(hypothesis=hypothesis.key):
                model = _model()
                train = _train()
                x, y, _changed = prepare(hypothesis, model, train, "A", 42)
                model.fit(x, y, 1)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "checkpoint.npz"
                    model.save(path)
                    again = Model.load(path)
                self.assertEqual(list(again.meta["paths"]),
                                 list(model.meta["paths"]))


class EpochBudgetTest(unittest.TestCase):
    """A repeated hypothesis must still be a different experiment."""

    def test_the_budget_grows_with_the_round(self):
        for hypothesis in BANK:
            with self.subTest(hypothesis=hypothesis.key):
                budgets = [epoch_budget(hypothesis, index)
                           for index in range(1, 11)]
                self.assertEqual(budgets, sorted(budgets))
                self.assertEqual(len(set(budgets)), len(budgets),
                                 "two rounds share an epoch budget")

    def test_the_budget_stays_bounded(self):
        for hypothesis in BANK:
            with self.subTest(hypothesis=hypothesis.key):
                self.assertLessEqual(epoch_budget(hypothesis, 100), 150)


class PlanTest(unittest.TestCase):
    """The plan is a function of what has already been tried."""

    def test_the_first_plan_is_the_cheapest_hypothesis(self):
        history = []
        first = build_plan("A", 1, history)
        self.assertEqual(first.key, "more-data-helps")

    def test_a_tried_hypothesis_is_not_repeated(self):
        first = build_plan("A", 1, [])
        history = [{"hypothesis": first.key}]
        second = build_plan("A", 2, history)
        self.assertNotEqual(second.key, first.key)

    def test_every_hypothesis_is_tried_before_any_is_repeated(self):
        history = []
        seen = []
        for round_index in range(1, len(for_profile("B")) + 1):
            hypothesis = build_plan("B", round_index, history)
            seen.append(hypothesis.key)
            history.append({"hypothesis": hypothesis.key})
        self.assertEqual(len(set(seen)), len(seen),
                         "a hypothesis was tried twice before the bank ran out")

    def test_a_hypothesis_for_another_profile_is_not_used(self):
        keys = {hypothesis.key for hypothesis in for_profile("A")}
        self.assertNotIn("the-tail-is-underweighted", keys)
        self.assertIn("rare-class-is-underweighted", keys)


if __name__ == "__main__":
    unittest.main()

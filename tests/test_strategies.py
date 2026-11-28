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
        # Look the hypothesis up by name rather than by position: the
        # adapter owns the order, and the order is not part of the
        # contract this test protects.
        migration = next(h for h in for_profile("A")
                         if h.key == "capacity-is-overfitting")
        prepare(migration, model, train, "A", 42)
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


class DegradationWeightTest(unittest.TestCase):
    """The weighted-loss hypothesis must stay deployable and directional."""

    def test_unweighted_fit_is_bit_identical_with_unit_weights(self):
        model_a = _model("regression")
        model_b = _model("regression")
        train = _train("regression")
        losses_a = model_a.fit(train["x"], train["y"], 3)
        losses_b = model_b.fit(train["x"], train["y"], 3,
                               sample_weight=np.ones(len(train["y"])))
        self.assertEqual(losses_a, losses_b)
        for name in model_a.params:
            self.assertTrue(np.array_equal(model_a.params[name], model_b.params[name]))

    def test_tail_weighted_sets_late_life_weights(self):
        hypothesis = next(h for h in BANK
                          if h.key == "the-tail-is-underweighted-by-weight")
        model = _model("regression")
        train = _train("regression")
        # RUL decreases within a unit; the last row of each unit is the
        # one the asymmetric penalty punishes hardest.
        train["y"] = np.asarray([200.0 - (i % 16) for i in range(len(train["y"]))])
        x, y, changed = prepare(hypothesis, model, train, "C", 42)
        weights = model.sample_weight
        self.assertEqual(len(weights), len(y))
        self.assertGreater(weights[-1], weights[0])
        self.assertIn("sample_weight", changed)
        losses = model.fit(x, y, 1, sample_weight=weights)
        self.assertTrue(np.isfinite(losses[0]))


class RelativeDegradationWeightTest(unittest.TestCase):
    """The C-only relative-degradation hypothesis changes loss mass, nothing else."""

    @staticmethod
    def _sequence_train(units=4, length=500, features=5):
        ids, x, y = [], [], []
        rng = np.random.default_rng(3)
        for unit in range(units):
            for remaining in range(length, 0, -1):
                ids.append("fd001_train_u%04d_c%05d" % (unit, length - remaining))
                x.append(rng.normal(0, 1, features))
                y.append(float(remaining))
        return {"ids": ids, "x": np.asarray(x, dtype=np.float64),
                "y": np.asarray(y, dtype=np.float64),
                "paths": ["f%d" % i for i in range(features)]}

    @staticmethod
    def _mixed_life_train(features=5):
        """Units with very different lifetimes.

        The absolute-threshold family mislabels these: a cut at
        ``RUL <= 150`` calls the middle of a 400-cycle engine
        end-of-life while calling the start of a 500-cycle engine fresh.
        The relative form does not, because it never uses an absolute
        threshold.
        """
        ids, x, y = [], [], []
        rng = np.random.default_rng(7)
        for unit, length in enumerate((120, 400, 500)):
            for remaining in range(length, 0, -1):
                ids.append("fd001_train_u%04d_c%05d" % (unit, length - remaining))
                x.append(rng.normal(0, 1, features))
                y.append(float(remaining))
        return {"ids": ids, "x": np.asarray(x, dtype=np.float64),
                "y": np.asarray(y, dtype=np.float64),
                "paths": ["f%d" % i for i in range(features)]}

    def test_the_weight_is_relative_to_each_unit(self):
        hypothesis = next(h for h in BANK if h.key == "the-degradation-is-relative")
        model = _model("regression")
        train = self._sequence_train()
        x, y, changed = prepare(hypothesis, model, train, "C", 42)
        weights = model.sample_weight
        self.assertIn("sample_weight", changed)
        self.assertEqual(changed["weight_alpha"], 0.5)
        self.assertEqual(len(weights), len(y))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 1.0))
        # 1 at a unit's first snapshot, 1 + alpha at its last.
        # (The last row has RUL 1, not 0, so the extreme is just under
        # 1 + alpha - the weight is a smooth function of the position in
        # the unit's life, not a step at the boundary.)
        self.assertAlmostEqual(float(weights.min()), 1.0, places=9)
        self.assertAlmostEqual(float(weights.max()), 1.5, places=2)
        losses = model.fit(x, y, 1, sample_weight=weights)
        self.assertTrue(np.isfinite(losses[0]))

    def test_the_weight_does_not_use_an_absolute_threshold(self):
        """The failure mode of the absolute family, fixed.

        With units of 120, 400 and 500 cycles, the absolute cut at 150
        would give the 120-cycle unit's *whole* life the tail weight and
        the 500-cycle unit's first 350 cycles none.  The relative weight
        gives every unit the same profile.
        """
        hypothesis = next(h for h in BANK if h.key == "the-degradation-is-relative")
        model = _model("regression")
        train = self._mixed_life_train()
        x, y, _changed = prepare(hypothesis, model, train, "C", 42)
        weights = model.sample_weight
        units = np.asarray([i.rsplit("_c", 1)[0] for i in train["ids"]])
        starts = []
        ends = []
        for name in np.unique(units):
            rows = units == name
            # The row with the largest RUL is the unit's first snapshot.
            first = rows & (y == y[rows].max())
            last = rows & (y == y[rows].min())
            starts.append(float(weights[first].mean()))
            ends.append(float(weights[last].mean()))
        # Every unit has the same start weight, whatever its lifetime.
        self.assertAlmostEqual(max(starts) - min(starts), 0.0, places=9)
        self.assertAlmostEqual(starts[0], 1.0, places=9)
        # The end weights differ by a hair, and only because the last
        # row's remaining life is 1 rather than 0: the weight is
        # `1 + alpha * (1 - RUL / lifetime)`, so a unit's last row
        # sits at `1 + alpha * (1 - 1/lifetime)`.  What matters is
        # that the spread is a rounding artefact of that offset and not
        # a difference of scale - compare with the absolute family, where
        # a 120-cycle unit's *whole* life would carry the tail weight.
        self.assertLess(max(ends) - min(ends), 0.01)
        for weight in ends:
            self.assertGreater(weight, 1.4)

    def test_a_classification_profile_skips_instead_of_weighting(self):
        hypothesis = next(h for h in BANK if h.key == "the-degradation-is-relative")
        model = _model("classification")
        train = _train("classification")
        _x, _y, changed = prepare(hypothesis, model, train, "A", 42)
        self.assertIn("skipped", changed)
        self.assertFalse(hasattr(model, "sample_weight"))


class OutputCalibrationTest(unittest.TestCase):
    """The affine output map is fitted on the training rows and folded in.

    The invariants guarded here are the ones that make the calibration
    honest and deployable:

    * it is fitted on the **training** rows, never on the validation
      split;
    * it is folded into the output layer, so the calibrated model *is*
      the checkpoint and the platform's own forward pass applies it;
    * the fold is **exact** - the checkpoint's output after the fold is
      the affine map applied to the output before it;
    * it is refused when the fit is not a calibration (a non-positive
      slope), rather than applied as a sign flip.

    The synthetic rows are built so the target is a **positive** affine
    function of a feature - a real regression target has to correlate
    with the model's output, or the fit is a sign flip and the
    calibration is (correctly) refused.
    """

    @staticmethod
    def _hypothesis():
        return next(h for h in BANK if h.key == "the-output-is-biased")

    @staticmethod
    def _correlated_train(units=4, per_unit=32, features=5, seed=13):
        """Training rows whose target rises with a feature."""
        ids, x, y = [], [], []
        rng = np.random.default_rng(seed)
        for unit in range(units):
            for index in range(per_unit):
                ids.append("fd001_train_u%04d_c%05d" % (unit, index))
                row = rng.normal(0, 1, features)
                row[0] += unit
                x.append(row)
                y.append(80.0 + 12.0 * row[0] + 3.0 * row[1]
                         + 0.5 * index)
        return {"ids": ids,
                "x": np.asarray(x, dtype=np.float64),
                "y": np.asarray(y, dtype=np.float64),
                "paths": ["f%d" % i for i in range(features)]}

    @staticmethod
    def _regression_model(train, hidden=8):
        """A model fitted on the correlated rows, so the map has a slope."""
        return Model("regression", list(train["paths"]), [], train["x"],
                     train["y"], hidden=hidden)

    def test_the_fold_is_exact(self):
        train = self._correlated_train()
        model = self._regression_model(train)
        model.fit(train["x"], train["y"], 60)
        raw_before = np.asarray(model._internal_output(train["x"]), dtype=np.float64)
        record = model.calibrate(train["x"], train["y"])
        raw_after = np.asarray(model._internal_output(train["x"]), dtype=np.float64)
        slope, offset = record["slope"], record["offset"]
        self.assertGreater(slope, 0.0)
        self.assertLess(float(np.max(np.abs(raw_after - (slope * raw_before + offset)))),
                        1e-9)

    def test_the_calibration_removes_a_constant_offset(self):
        """A model with a known offset is corrected by the map."""
        rng = np.random.default_rng(11)
        features = 5
        x = rng.normal(0, 1, (400, features))
        y = 100.0 + 2.0 * x[:, 0] + 25.0          # a known 25-unit offset
        model = Model("regression", ["f%d" % i for i in range(features)], [],
                      x, y, hidden=8)
        model.fit(x, y, 60)
        before = np.asarray(model.predict(x), dtype=np.float64)
        offset_before = float(np.mean(before - y))
        model.calibrate(x, y)
        after = np.asarray(model.predict(x), dtype=np.float64)
        offset_after = float(np.mean(after - y))
        self.assertLess(abs(offset_after), abs(offset_before))
        self.assertAlmostEqual(offset_after, 0.0, delta=1.0)

    def test_the_calibration_does_not_change_the_schema(self):
        train = self._correlated_train()
        model = self._regression_model(train)
        model.fit(train["x"], train["y"], 60)
        before = list(model.meta["paths"])
        model.calibrate(train["x"], train["y"])
        self.assertEqual(list(model.meta["paths"]), before)
        self.assertEqual(model.meta["parameter_count"],
                         sum(v.size for v in model.params.values()))

    def test_a_non_positive_slope_is_refused(self):
        train = self._correlated_train()
        model = self._regression_model(train)
        model.fit(train["x"], train["y"], 60)
        raw = np.asarray(model._internal_output(train["x"]), dtype=np.float64)
        with self.assertRaises(ValueError):
            model.calibrate(train["x"], -raw)

    def test_a_classification_model_refuses_the_calibration(self):
        model = _model("classification")
        train = _train("classification")
        with self.assertRaises(ValueError):
            model.calibrate(train["x"], train["y"])

    def test_the_strategy_records_the_intent_and_the_controller_folds_it(self):
        from contest_baselines.layers.controller.round import RoundController

        hypothesis = self._hypothesis()
        model = _model("regression")
        train = self._correlated_train()
        _x, _y, changed = prepare(hypothesis, model, train, "C", 42)
        # ``prepare`` runs before the training, so it may only record the
        # intent - the map has to be fitted on the *trained* model.
        self.assertTrue(changed.get("calibration_requested"))
        self.assertNotIn("calibration_slope", changed)

        controller = RoundController.__new__(RoundController)
        controller.changed = changed
        controller.parent = type("P", (), {"task": "regression"})()
        trained = self._regression_model(train)
        trained.fit(train["x"], train["y"], 60)
        record = controller._apply_calibration(trained, train["x"], train["y"])
        self.assertIsNotNone(record)
        self.assertIn("calibration_slope", controller.changed)
        self.assertFalse(controller.changed.get("calibration_requested"))

    def test_the_controller_leaves_the_model_alone_when_not_asked(self):
        from contest_baselines.layers.controller.round import RoundController

        controller = RoundController.__new__(RoundController)
        controller.changed = {"sample_weight": "none"}
        controller.parent = type("P", (), {"task": "regression"})()
        model = _model("regression")
        before = {k: v.copy() for k, v in model.params.items()}
        self.assertIsNone(controller._apply_calibration(
            model, np.zeros((1, 5)), np.zeros(1)))
        for name, value in before.items():
            self.assertTrue(np.array_equal(value, model.params[name]))


class BoundedPriorShapeTest(unittest.TestCase):
    """The prior shape is a knob, and every shape is a valid loss mass."""

    def test_every_shape_is_finite_and_normalised(self):
        from contest_baselines.strategies import _bounded_prior_weights

        rng = np.random.default_rng(5)
        y = np.concatenate([np.arange(400, 0, -1, dtype=np.float64) for _ in range(3)])
        for shape in ("uniform", "tri", "exp"):
            with self.subTest(shape=shape):
                weights = _bounded_prior_weights(y, 200.0, shape=shape, floor=0.20)
                self.assertEqual(len(weights), len(y))
                self.assertTrue(np.all(np.isfinite(weights)))
                self.assertTrue(np.all(weights > 0))
                self.assertAlmostEqual(float(weights.mean()), 1.0, places=9)
                inside = (y > 0) & (y <= 200)
                outside = y > 300
                self.assertLess(float(weights[outside].mean()),
                                float(weights[inside].mean()))

    def test_an_unknown_shape_is_refused(self):
        from contest_baselines.strategies import _bounded_prior_weights

        with self.assertRaises(ValueError):
            _bounded_prior_weights(np.arange(1.0, 50.0), 200.0, shape="nope")

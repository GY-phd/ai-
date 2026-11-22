# -*- coding: utf-8 -*-
"""The tests for the adapter plugin layer and the round's fixes.

Three invariants are guarded here, and each one exists because breaking
it would produce a round that **lies about itself** - which is worse
than a round that fails, because the lie reaches the platform's record.

**A hypothesis that cannot be tested is not run.**  Profile A's public
    10|projection holds a single measurement batch, so
``the-batch-is-mixing-scales`` has nothing to oversample.  The round must
say so in its change summary instead of resampling every row uniformly
by a factor of eight and calling it "oversampling the rare batch".

**Every round leaves an experiment record.**  The rules ask for every
training / adjustment / evaluation event to be kept on SwanLab.  The
record is written locally in SwanLab's shape and travels with the
candidate and with the state the next round inherits.

**The profile's specifics live in its adapter.**  The loop must not
    20|reintroduce a ``if profile == "A"`` branch, and adding a profile
must be a matter of writing an adapter, not of editing the loop.

Run:

    python -m pytest tests/test_adapters.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from contest_baselines import adapters  # noqa: E402
from contest_baselines.adapters import Adapter  # noqa: E402
from contest_baselines.adapters import registry  # noqa: E402
from contest_baselines.model import Model  # noqa: E402
from contest_baselines.strategies import build_plan, for_profile, prepare  # noqa: E402
from contest_baselines.swanlab_log import SubmissionSwanLab, SWANLAB_FILE  # noqa: E402

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


def _train(ids, features=5, task="classification"):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (len(ids), features))
    y = (rng.integers(0, len(CLASSES), len(ids)).astype(np.float64)
         if task == "classification"
         else rng.uniform(10, 200, len(ids)))
    return {
        "ids": list(ids),
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.int64 if task == "classification" else np.float64),
        "paths": ["f%d" % i for i in range(features)],
    }


class RegistryTest(unittest.TestCase):
    """The registry is the only file that changes when a profile is added."""

    def test_every_known_profile_has_an_adapter(self):
        for key in ("A", "B", "C"):
            self.assertIn(key, registry.KNOWN_KEYS)

    def test_an_unknown_key_is_refused_rather_than_guessed(self):
        with self.assertRaises(KeyError):
            registry.get("Z")

    def test_a_plugin_can_be_registered_at_runtime(self):
        class Dummy(Adapter):
            key = "TEST"
            task = "regression"

        adapter = Dummy()
        registry.register(adapter)
        try:
            self.assertIs(registry.get("TEST"), adapter)
        finally:
            registry.unregister("TEST")
        self.assertNotIn("TEST", registry.KNOWN_KEYS)

    def test_registering_a_non_adapter_is_refused(self):
        with self.assertRaises(TypeError):
            registry.register(object())


class AdapterContractTest(unittest.TestCase):
    """Each adapter answers the loop's questions about its own profile."""

    def test_a_is_classification_and_b_c_are_regression(self):
        self.assertEqual(adapters.get("A").task, "classification")
        self.assertEqual(adapters.get("B").task, "regression")
        self.assertEqual(adapters.get("C").task, "regression")

    def test_a_declares_the_platform_class_list(self):
        self.assertEqual(list(adapters.get("A").expected_classes()),
                         ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde",
                          "Acetone", "Toluene"])

    def test_b_groups_by_bearing_and_c_by_engine(self):
        b = adapters.get("B")
        # The trailing ``_s<digits>`` is the snapshot index; everything
        # before it - the dataset prefix included - is the bearing.
        self.assertEqual(b.group_of("xjtu_Bearing1_1_s000001"), "xjtu_Bearing1_1")
        self.assertEqual(b.group_of("xjtu_Bearing2_7_s000042"), "xjtu_Bearing2_7")
        c = adapters.get("C")
        # The trailing ``_c<digits>`` is the cycle index; the subset
        # stays in the unit because FD001-FD004 are different conditions.
        self.assertEqual(c.group_of("fd001_train_u0001_c00001"), "fd001_train_u0001")
        self.assertEqual(c.group_of("fd004_test_u0012_c00099"), "fd004_test_u0012")

    def test_a_rows_are_their_own_unit(self):
        a = adapters.get("A")
        self.assertEqual(a.group_of("gas_b01_000001"), "gas_b01_000001")

    def test_a_recovers_the_measurement_batch(self):
        a = adapters.get("A")
        self.assertEqual(a.label_of("gas_b01_000001"), "b01")
        self.assertEqual(a.label_of("gas_b02_001244"), "b02")
        self.assertIsNone(a.label_of("xjtu_Bearing1_1_s000001"))

    def test_the_epoch_budget_is_scaled_by_the_adapter(self):
        self.assertEqual(adapters.get("B").epoch_scale(), 1.0)
        self.assertLess(adapters.get("C").epoch_scale(), 1.0)

    def test_a_migration_target_stays_inside_the_reference_limit(self):
        from contest_baselines.model import MAX_HIDDEN
        for key in ("A", "B", "C"):
            width = adapters.get(key).resize_hidden("regression")
            self.assertLessEqual(width, MAX_HIDDEN)
            self.assertGreaterEqual(width, 1)

    def test_only_c_declares_a_truncated_test_horizon(self):
        # C's hidden protocol is a truncated sequence; A and B keep the
        # full validation split as their protocol, so their selection
        # view is the primary metric.
        self.assertIsNone(adapters.get("A").test_horizon())
        self.assertIsNone(adapters.get("B").test_horizon())
        self.assertEqual(adapters.get("C").test_horizon(), 200.0)

    def test_c_selection_view_uses_only_rows_inside_the_horizon(self):
        c = adapters.get("C")
        data = {"y": np.asarray([0.0, 50.0, 100.0, 200.0, 201.0, 500.0])}
        predicted = np.asarray([10.0, 40.0, 120.0, 180.0, 220.0, 400.0])
        view = c.selection_view(data, predicted)
        self.assertEqual(view["name"], "late_rmse")
        self.assertEqual(view["n"], 4)
        expected = float(np.sqrt(np.mean((predicted[:4] - data["y"][:4]) ** 2)))
        self.assertAlmostEqual(view["value"], expected)
        self.assertEqual(view["horizon"], 200.0)
        # The full split is a safety reading, not a blank cheque: C allows
        # at most a 5% regression there.
        self.assertEqual(c.selection_guard(), 0.05)

    def test_a_and_b_have_no_selection_view(self):
        for key in ("A", "B"):
            data = {"y": np.asarray([0.0, 1.0])}
            self.assertIsNone(adapters.get(key).selection_view(data, data["y"]))

    def test_each_profile_has_its_own_bank(self):
        keys = {key: [h.key for h in for_profile(key)] for key in ("A", "B", "C")}
        self.assertIn("rare-class-is-underweighted", keys["A"])
        self.assertNotIn("rare-class-is-underweighted", keys["B"])
        self.assertIn("the-unit-is-the-confounder", keys["B"])
        self.assertIn("the-tail-is-underweighted", keys["C"])

    def test_the_plan_comes_from_the_adapter(self):
        adapter = adapters.get("C")
        plan = build_plan("C", 1, [], adapter=adapter)
        self.assertEqual(plan.key, adapter.hypotheses()[0].key)


class BatchVetoTest(unittest.TestCase):
    """A hypothesis the data cannot support must say so, not fake it.

    This is the regression for the defect where profile A's
    ``the-batch-is-mixing-scales`` resampled every row by a factor of
    eight - because the public training projection holds exactly one
    batch, so "the rarest batch" was every row - and then wrote
    ``"rare_batch": "b01"`` into the round's change summary as if a
    minority batch had been oversampled.
    """

    def _batch_hypothesis(self):
        return next(h for h in for_profile("A")
                    if h.key == "the-batch-is-mixing-scales")

    def test_a_single_batch_is_reported_as_skipped(self):
        model = _model()
        train = _train(["gas_b01_%06d" % i for i in range(20)])
        _x, _y, changed = prepare(self._batch_hypothesis(), model, train, "A", 42)
        self.assertIn("skipped", changed)
        self.assertIn("no minority batch", changed["skipped"])
        # The false claim must be gone.
        self.assertNotIn("rare_batch", changed)
        self.assertNotIn("resampled_rows", changed)

    def test_two_batches_do_produce_a_resampling(self):
        model = _model()
        ids = ["gas_b01_%06d" % i for i in range(20)]
        ids += ["gas_b02_%06d" % i for i in range(4)]
        train = _train(ids)
        _x, _y, changed = prepare(self._batch_hypothesis(), model, train, "A", 42)
        self.assertNotIn("skipped", changed)
        self.assertEqual(changed["rare_batch"], "b02")
        self.assertEqual(changed["batches"], 2)
        self.assertEqual(changed["resampled_rows"], len(ids))

    def test_ids_without_a_batch_label_are_counted_separately(self):
        model = _model()
        ids = ["gas_b01_%06d" % i for i in range(5)] + ["unlabelled_row_%d" % i
                                                        for i in range(3)]
        train = _train(ids)
        _x, _y, changed = prepare(self._batch_hypothesis(), model, train, "A", 42)
        self.assertIn("skipped", changed)
        self.assertEqual(changed["unlabelled_rows"], 3)


class SwanLabRecordTest(unittest.TestCase):
    """Every round leaves an experiment record in SwanLab's shape."""

    def setUp(self):
        # The cloud mirror is a **best effort**, and a developer machine
        # usually has a real ``SWANLAB_API_KEY`` in its environment.  A
        # unit test must never push its fixtures to somebody's account,
        # so the key is taken away for the duration of the test.
        self._saved = os.environ.pop("SWANLAB_API_KEY", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is not None:
            os.environ["SWANLAB_API_KEY"] = self._saved

    def test_the_record_is_written_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SubmissionSwanLab(tmp, "run-1", {"dataset_key": "A"})
            tracker.log({"val_macro_f1_before": 0.71, "val_macro_f1_after": 0.72},
                        context={"stage": "round", "hypothesis": "more-data-helps"})
            path = Path(tmp) / SWANLAB_FILE
            self.assertTrue(path.is_file())
            records = [json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["run"], "run-1")
            self.assertEqual(record["step"], 1)
            self.assertEqual(record["metrics"]["val_macro_f1_after"], 0.72)
            self.assertEqual(record["context"]["hypothesis"], "more-data-helps")
            self.assertIn("ts", record)

    def test_a_non_finite_metric_does_not_poison_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SubmissionSwanLab(tmp, "run-1")
            tracker.log({"rmse": float("nan"), "ok": 1.5})
            path = Path(tmp) / SWANLAB_FILE
            record = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertIsNone(record["metrics"]["rmse"])
            self.assertEqual(record["metrics"]["ok"], 1.5)

    def test_the_step_counter_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SubmissionSwanLab(tmp, "run-1")
            tracker.log({"a": 1})
            tracker.log({"a": 2})
            tracker.log({"a": 3}, step=10)
            records = [json.loads(line) for line in
                       (Path(tmp) / SWANLAB_FILE).read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            self.assertEqual([r["step"] for r in records], [1, 2, 10])


class PredictFallbackTest(unittest.TestCase):
    """The classification fallback must not invent a label.

    The platform requires every label to be one of the profile's own
    classes.  The old constant ``"normal"`` was not one of the six
    official gas classes, so a fallback would have produced a
    prediction the platform could not score.
    """

    def test_regression_falls_back_to_zero(self):
        from contest_baselines.predict import predict_sample

        class Sample:
            task = "regression"
            inputs = {"f": 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            result = predict_sample(Sample(), Path(tmp))
        self.assertEqual(result, {"outputs": {"value": 0.0}})

    def test_classification_refuses_to_invent_a_label(self):
        from contest_baselines.predict import predict_sample

        class Sample:
            task = "classification"
            inputs = {"f": 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                predict_sample(Sample(), Path(tmp))


if __name__ == "__main__":
    unittest.main()

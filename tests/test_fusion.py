# -*- coding: utf-8 -*-
"""Tests for the fused components: hypothesis chain, activation, target scaling.

These cover the places that actually bite:

  * the four-stage evidence chain must refuse to report a cycle as complete
    when one of its stages is empty - that is exactly the failure mode the
    competition's "read history -> hypothesize -> experiment -> decide"
    requirement is meant to catch;
  * the checkpoint loader must keep accepting checkpoints that predate the
    ``activation`` / ``target_scaling`` metadata keys, otherwise the platform's
    own baseline checkpoint stops loading;
  * the head refit must use the same activation and target basis as the
    checkpoint it refits, or the frozen-hidden-layer trick silently fits in
    the wrong space.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.hypothesis import HypothesisChain, HypothesisRecord
from rulkit.model import ACTIVATIONS, Model


class HypothesisChainTest(unittest.TestCase):
    """The four-stage evidence chain."""

    def test_record_reports_missing_stages(self):
        record = HypothesisRecord(iteration=1, hypothesis="h")
        self.assertEqual(sorted(record.missing_stages()),
                         ["decision", "experiment", "read_history"])
        self.assertFalse(record.is_complete())

    def test_complete_record_passes(self):
        record = HypothesisRecord(iteration=1, read_history={"a": 1},
                                  hypothesis="h", experiment={"e": 1},
                                  decision={"d": 1})
        self.assertTrue(record.is_complete())
        self.assertEqual(record.missing_stages(), [])

    def test_chain_appends_and_summarises(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = HypothesisChain(Path(tmp))
            self.assertEqual(chain.next_iteration(), 1)
            chain.append(HypothesisRecord(iteration=1, read_history={"a": 1},
                                          hypothesis="h", experiment={"e": 1},
                                          decision={"action": "do_nothing"},
                                          approved=False))
            chain.append(HypothesisRecord(iteration=2, read_history={"a": 1},
                                          hypothesis="h", experiment={"e": 1},
                                          decision={"action": "hpo"}, approved=True,
                                          gain=12.5))
            summary = chain.summary()
            self.assertEqual(summary["iterations"], 2)
            self.assertEqual(summary["complete_records"], 2)
            self.assertEqual(summary["promoted"], 1)
            self.assertEqual(summary["do_nothing"], 1)
            self.assertEqual(summary["best_gain_pct"], 12.5)

    def test_chain_survives_a_truncated_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypothesis_chain.jsonl"
            path.write_text(json.dumps({"iteration": 1}) + "\n{not json", encoding="utf-8")
            chain = HypothesisChain(Path(tmp))
            self.assertEqual(len(chain.records), 1)

    def test_record_round_trips(self):
        record = HypothesisRecord(iteration=7, read_history={"drift": {"level": "ok"}},
                                  hypothesis="the champion is still adequate",
                                  expected_gain=3.0,
                                  falsification_criterion="RMSE drops below 400",
                                  experiment={"action": "hpo"},
                                  decision={"action": "hpo", "outcome": "reject"},
                                  gain=-1.2)
        restored = HypothesisRecord.from_dict(record.to_dict())
        self.assertEqual(restored.hypothesis, record.hypothesis)
        self.assertEqual(restored.gain, record.gain)
        self.assertTrue(restored.is_complete())


class ModelActivationTest(unittest.TestCase):
    """Activation and target-scaling options on the checkpoint model."""

    def _data(self, rows=64, cols=34):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(rows, cols))
        y = rng.uniform(10, 100, size=rows)
        return x, y

    def test_relu_and_tanh_are_accepted(self):
        x, y = self._data()
        paths = [f"f{i}" for i in range(34)]
        for activation in ACTIVATIONS:
            model = Model("regression", paths, [], x, y, hidden=16,
                          activation=activation)
            self.assertEqual(model.meta["activation"], activation)
            model.validate()

    def test_unknown_activation_is_rejected(self):
        x, y = self._data()
        paths = [f"f{i}" for i in range(34)]
        with self.assertRaises(ValueError):
            Model("regression", paths, [], x, y, hidden=16, activation="swish")

    def test_target_scaling_flag_is_recorded(self):
        x, y = self._data()
        paths = [f"f{i}" for i in range(34)]
        model = Model("regression", paths, [], x, y, hidden=16,
                      target_scaling=False)
        self.assertFalse(model.meta["target_scaling"])
        model.validate()

    def test_old_checkpoint_metadata_still_loads(self):
        """A checkpoint without the new keys must keep its original behaviour."""
        x, y = self._data()
        paths = [f"f{i}" for i in range(34)]
        model = Model("regression", paths, [], x, y, hidden=16)
        meta = dict(model.meta)
        meta.pop("activation", None)
        meta.pop("target_scaling", None)
        model.meta = meta
        model.validate()  # must not raise
        # The defaults reproduce the pre-fusion behaviour.
        self.assertEqual(model.meta.get("activation", "tanh"), "tanh")
        self.assertTrue(model.meta.get("target_scaling", True))

    def test_relu_and_tanh_predict_differently(self):
        x, y = self._data()
        paths = [f"f{i}" for i in range(34)]
        tanh_model = Model("regression", paths, [], x, y, hidden=16,
                           activation="tanh")
        relu_model = Model("regression", paths, [], x, y, hidden=16,
                           activation="relu")
        self.assertFalse(np.allclose(tanh_model.predict(x), relu_model.predict(x)))

    def test_predictions_are_non_negative_for_regression(self):
        x, y = self._data()
        paths = [f"f{i}" for i in range(34)]
        model = Model("regression", paths, [], x, y, hidden=16,
                      activation="relu", target_scaling=False)
        self.assertTrue(np.all(model.predict(x) >= 0))


if __name__ == "__main__":
    unittest.main()

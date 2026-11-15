import json
from pathlib import Path
import tempfile
import unittest

from harness.config import load_config
from harness.state import (StateError, add_event, load_state, new_state, processed_batches,
                           record_batch, save_state)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "harness" / "configs" / "default.json")
METRICS = {"rmse": 10.0, "mae": 8.0, "n": 10, "nasa_score": 1.0, "bias": 0.0}


def make_state():
    return new_state(
        baseline_root="C:/baseline", profile="C",
        baseline_checkpoint_sha256="sha256:official",
        champion_checkpoint="models/official-baseline/checkpoint.npz",
        champion_sha256="sha256:official", champion_meta={"learning_rate": 0.002, "batch_size": 256},
        public_validation_metrics=METRICS, guard_metrics=METRICS,
        guard_info={"path": "data/guard.jsonl", "sha256": "sha256:guard"},
        replay_info=None, config=CONFIG,
    )


class StateTests(unittest.TestCase):
    def test_state_round_trip_and_ledgers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = make_state()
            record_batch(state, {"batch_id": "batch-01", "decision": "hold"})
            record_batch(state, {"batch_id": "batch-01", "decision": "promote"}, replace=True)
            record_batch(state, {"batch_id": "batch-02", "decision": "rollback"})
            add_event(state, "promote", generation=1)
            save_state(path, state)
            loaded = load_state(path)
            self.assertEqual(processed_batches(loaded), {"batch-01", "batch-02"})
            self.assertEqual(len(loaded["batches"]), 2)
            self.assertEqual(loaded["batches"][0]["decision"], "promote")
            self.assertEqual(loaded["events"][-1]["event"], "promote")
            self.assertEqual(loaded["champion"]["stage_id"], "official-baseline")

    def test_invalid_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            path.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
            with self.assertRaises(StateError):
                load_state(path)

    def test_counters_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = make_state()
            state["counters"]["attempts"] = -1
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaises(StateError):
                load_state(path)


if __name__ == "__main__":
    unittest.main()

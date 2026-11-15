import json
from pathlib import Path
import tempfile
import unittest

from harness.data import guard_dataset, merge_training_sources, replay_dataset
from harness.util import (parse_json, resolve_workdir_path, sample_jsonl_lines,
                          unique_path, write_json_atomic)


def row(identity):
    return json.dumps({"id": identity, "inputs": {"x": [1.0]}, "answer": 2})


class DataTests(unittest.TestCase):
    def write(self, root, name, ids):
        path = Path(root) / name
        path.write_text("\n".join(row(identity) for identity in ids) + "\n", encoding="utf-8")
        return path

    def test_merge_deduplicates_and_drops_evaluation_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.write(root, "first.jsonl", ["a", "b"])
            second = self.write(root, "second.jsonl", ["b", "c"])
            destination = root / "merged.jsonl"
            info = merge_training_sources([("first", first), ("second", second)], destination,
                                          excluded_ids={"c"})
            self.assertEqual(info["rows"], 2)
            ids = [json.loads(line)["id"] for line in destination.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(ids, ["a", "b"])
            self.assertEqual(info["sources"][1]["rows_dropped_duplicate"], 1)
            self.assertEqual(info["sources"][1]["rows_dropped_excluded"], 1)

    def test_merge_rejects_fully_excluded_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.write(root, "source.jsonl", ["a"])
            with self.assertRaisesRegex(ValueError, "no training rows"):
                merge_training_sources([("source", source)], root / "merged.jsonl",
                                       excluded_ids={"a"})

    def test_guard_and_replay_sampling_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.write(root, "source.jsonl", [str(index) for index in range(20)])
            first = guard_dataset(source, root / "guard-a.jsonl", rows=5, seed=3)
            second = guard_dataset(source, root / "guard-b.jsonl", rows=5, seed=3)
            third = replay_dataset(source, root / "replay.jsonl", rows=5, seed=3)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["sha256"], third["sha256"])
            self.assertEqual(first["rows"], 5)
            self.assertEqual(len(sample_jsonl_lines(source, 5, 3)), 5)

    def test_unique_path_preserves_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "decision"
            first = unique_path(base, suffix=".json")
            self.assertEqual(first.name, "decision.json")
            first.write_text("{}", encoding="utf-8")
            second = unique_path(base, suffix=".json")
            self.assertEqual(second.name, "decision-r2.json")

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_json('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_json('{"a": NaN}')

    def test_workdir_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(resolve_workdir_path(root, "models/x.npz"),
                             (root / "models" / "x.npz").resolve())
            with self.assertRaisesRegex(ValueError, "escapes the workdir"):
                resolve_workdir_path(root, "../outside.npz")

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "state.json"
            write_json_atomic(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})


if __name__ == "__main__":
    unittest.main()

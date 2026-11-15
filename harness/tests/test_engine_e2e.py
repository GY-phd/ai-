import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from harness.state import load_state

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "cpu-c-guide-20261110-v1"
DEFAULT_CONFIG = ROOT / "harness" / "configs" / "default.json"


def first_lines(path, count):
    with path.open(encoding="utf-8") as stream:
        return [next(stream).rstrip("\n") for _ in range(count)]


def shift_row(raw, factor=1.15):
    row = json.loads(raw)
    sensors = row["inputs"]["sensors"]
    for index in (2, 3, 7, 8):
        sensors[index] = round(float(sensors[index]) * factor, 6)
    return json.dumps(row, ensure_ascii=False)


def make_stream(root):
    root.mkdir(parents=True)
    train = first_lines(BASELINE / "public" / "train.jsonl", 480)
    validation = first_lines(BASELINE / "public" / "validation.jsonl", 240)
    batches = [
        ("batch-01", train[:240], validation[:120], False),
        ("batch-02", train[240:], validation[120:], True),
    ]
    for name, train_lines, validation_lines, shifted in batches:
        directory = root / name
        directory.mkdir()
        if shifted:
            train_lines = [shift_row(line) for line in train_lines]
            validation_lines = [shift_row(line) for line in validation_lines]
        (directory / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        (directory / "validation.jsonl").write_text("\n".join(validation_lines) + "\n",
                                                     encoding="utf-8")
    hidden = root / ".hidden-batch"
    hidden.mkdir()
    (hidden / "validation.jsonl").write_text(validation[0] + "\n", encoding="utf-8")
    return root


def make_config(root):
    raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["guard"] = {"rows": 100, "seed": 42}
    raw["experiments"].update({
        "min_attempts": 2, "max_candidates": 1, "epochs": 1, "exploration_every": 0,
        "time_budget_seconds": 120.0, "experiment_timeout_seconds": 120.0, "replay_rows": 0,
    })
    raw["gate"].update({
        "min_improvement": 0.0, "guard_max_degradation": 0.5,
        "static_tolerance": 0.5, "rollback_margin": 0.5,
    })
    raw["logging"]["swanlab"] = "off"
    path = root / "config.json"
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


def run_cli(arguments):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "harness.cli", *arguments],
        cwd=str(ROOT), env=environment, capture_output=True, text=True,
        encoding="utf-8", timeout=300, check=False,
    )


class EngineEndToEndTests(unittest.TestCase):
    def test_run_is_idempotent_and_writes_decisions_before_next_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stream = make_stream(root / "stream")
            config = make_config(root)
            workdir = root / "work"
            args = ["run", "--stream", str(stream), "--workdir", str(workdir),
                    "--config", str(config), "--quiet", "--json"]
            completed = run_cli(args)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(len(summary["decisions"]), 2)
            decisions = [item["decision"] for item in summary["decisions"]]
            self.assertTrue(all(item in ("promote", "hold", "rollback") for item in decisions))
            state = load_state(workdir / "state.json")
            self.assertEqual(len(state["batches"]), 2)
            self.assertGreaterEqual(state["counters"]["attempts"], 2)
            self.assertGreaterEqual(state["counters"]["successful_experiments"], 1)
            champion = workdir / state["champion"]["checkpoint"]
            self.assertTrue(champion.is_file())
            decision_files = sorted((workdir / "decisions").glob("*.json"))
            self.assertEqual(len(decision_files), 2)
            for path in decision_files:
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["format"], "harness.decision.v1")
                self.assertIn("drift", record)
                self.assertIn("experiments_planned", record)
                if record["experiments"]:
                    self.assertIn("gate", record["experiments"][0])
                    json.dumps(record, allow_nan=False)
            journal = (workdir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
            events = [json.loads(line)["event"] for line in journal]
            self.assertIn("decision", events)
            self.assertIn("batch_finished", events)

            before = load_state(workdir / "state.json")["counters"].copy()
            second = run_cli(args)
            self.assertEqual(second.returncode, 0, second.stderr)
            second_summary = json.loads(second.stdout)
            self.assertEqual(second_summary["decisions"], [])
            after = load_state(workdir / "state.json")["counters"]
            self.assertEqual(before, after)

    def test_dry_run_does_not_train_or_advance_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stream = make_stream(root / "stream")
            config = make_config(root)
            workdir = root / "dry"
            completed = run_cli(["run", "--stream", str(stream), "--workdir", str(workdir),
                                 "--config", str(config), "--dry-run", "--max-batches", "2",
                                 "--quiet", "--json"])
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(len(summary["decisions"]), 2)
            self.assertTrue(all(item["dry_run"] for item in summary["decisions"]))
            self.assertEqual([item["batch"]["index"] for item in summary["decisions"]], [1, 2])
            state = load_state(workdir / "state.json")
            self.assertEqual(state["batches"], [])
            self.assertEqual(state["counters"]["attempts"], 0)
            self.assertTrue(list((workdir / "decisions").glob("dry-run-*.json")))

    def test_force_reuses_stored_batch_index_after_ledger_reordering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stream = make_stream(root / "stream")
            config = make_config(root)
            workdir = root / "forced"
            first = run_cli(["run", "--stream", str(stream), "--workdir", str(workdir),
                             "--config", str(config), "--quiet", "--json"])
            self.assertEqual(first.returncode, 0, first.stderr)
            for batch_id, expected in (("batch-01", 1), ("batch-02", 2)):
                completed = run_cli(["run", "--batch", str(stream / batch_id),
                                     "--workdir", str(workdir), "--config", str(config),
                                     "--quiet", "--json", "--force"])
                self.assertEqual(completed.returncode, 0, completed.stderr)
                decision = json.loads(completed.stdout)["decisions"][0]
                self.assertEqual(decision["batch"]["index"], expected)
                state = load_state(workdir / "state.json")
                self.assertEqual(len(state["batches"]), 2)
                stored = {item["batch_id"]: item["batch_index"] for item in state["batches"]}
                self.assertEqual(stored, {"batch-01": 1, "batch-02": 2})


if __name__ == "__main__":
    unittest.main()

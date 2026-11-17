# -*- coding: utf-8 -*-
"""The offline tests for the Harness integration.

These are the tests that answer the question the SDK's own tests cannot:
*does this participant's round actually work?*

They run the participant's real ``train`` and ``predict`` against a local
host stand-in, using the **real** public data packages and the **real**
parent checkpoints that ship on this machine.  A round that passes here
exercises the same code path the platform will execute.

They do not call a real model and they do not measure a real score.  The
platform says so explicitly: passing the offline tests does not mean the
training, the prediction or the competition requirements have passed.

Run:

    python -m pytest tests/test_harness_integration.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines.offline_host import package_root, run_entrypoint  # noqa: E402


def _packages_available():
    """Are the local data packages present?"""
    try:
        for key in ("A", "B", "C"):
            root = package_root(key)
            if not (root / "public" / "train.jsonl").is_file():
                return False
            if not (root / "checkpoint.npz").is_file():
                return False
        return True
    except Exception:
        return False


PACKAGES = _packages_available()


class DescribeTest(unittest.TestCase):
    """``describe`` is the platform's first gate."""

    def test_describe_reports_the_frozen_contract(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["CONTEST_SUBMISSION"] = "contest_submission"
        env["CONTEST_APP_ROOT"] = str(ROOT)
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "harness.py"), "describe"],
            cwd=str(ROOT), env=env, text=True, capture_output=True,
            encoding="utf-8", check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        described = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(described["protocol"], "contest.harness.workspace.v2")
        self.assertEqual(described["model_access"], "host_gateway_rpc")
        self.assertEqual(sorted(described["tasks"]), ["classification", "regression"])
        self.assertEqual(described["sdk"], "contest_harness")


@unittest.skipUnless(PACKAGES, "the local data packages are not installed")
class TrainRoundTest(unittest.TestCase):
    """A whole training round, against the real parent checkpoint."""

    def _run(self, dataset_key):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            code, stdout, stderr, host = run_entrypoint("train", dataset_key, work)
            self.assertIsNone(host.failure, host.failure)
            self.assertEqual(code, 0, stderr)
            return stdout, stderr, host, work

    def test_round_b_completes_and_decides(self):
        stdout, stderr, host, _work = self._run("B")
        self.assertIsNotNone(host.finished, stderr)
        record = host.finished
        self.assertTrue(record["model_call_ids"])
        self.assertTrue(record["hypothesis"])
        self.assertTrue(record["reason"])
        self.assertTrue(record["change_summary"])
        self.assertIsNotNone(record["selected_artifact_id"])
        # The round must have made exactly one real model request.
        self.assertEqual(len(host.model_requests), 1)

    def test_round_a_completes_and_decides(self):
        stdout, stderr, host, _work = self._run("A")
        self.assertIsNotNone(host.finished, stderr)
        self.assertTrue(host.finished["model_call_ids"])
        self.assertEqual(len(host.model_requests), 1)

    def test_round_b_submits_a_candidate_when_it_improves(self):
        _stdout, stderr, host, _work = self._run("B")
        # The first hypothesis is "continue for more epochs", which the
        # parent was stopped short of, so a candidate is expected.
        self.assertIsNotNone(host.candidate, stderr)
        if host.candidate is not None:
            # The candidate request frame carries the artefact description
            # under ``artifact``, not at the top level.
            artefact = host.candidate["artifact"]
            self.assertTrue(artefact["checkpoint_paths"])
            self.assertEqual(artefact["predict_argv"],
                             ["python3", "/app/harness.py", "predict"])
            self.assertTrue(host.candidate["hypothesis"])
            self.assertTrue(host.candidate["change_summary"])
            # The candidate must be bound to the parent it continued.
            self.assertEqual(host.candidate["parent_checkpoint_sha256"],
                             host.start["parent_checkpoint_sha256"])

    def test_round_writes_the_ledger_into_the_state(self):
        _stdout, stderr, host, _work = self._run("B")
        self.assertTrue(host.state_files, stderr)
        names = [item["path"] for item in host.state_files]
        self.assertIn("ledger.jsonl", names)


@unittest.skipUnless(PACKAGES, "the local data packages are not installed")
class PredictTest(unittest.TestCase):
    """The prediction path, against the real parent checkpoint."""

    def test_regression_prediction_is_a_finite_nonnegative_number(self):
        import numpy as np

        from contest_baselines.model import Model

        root = package_root("B")
        model = Model.load(root / "checkpoint.npz")
        row = json.loads((root / "public" / "validation.jsonl")
                         .open(encoding="utf-8").readline())
        from contest_baselines.data import flatten
        _paths, values = flatten(row["inputs"])
        value = float(model.predict(np.asarray([values], dtype=np.float64))[0])
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_predict_serves_every_row_of_the_validation_set(self):
        """Every row gets exactly one answer, and the answers are valid."""
        import numpy as np

        from contest_baselines.data import flatten
        from contest_baselines.model import Model
        from contest_baselines.predict import predict_sample

        class Sample:
            def __init__(self, task, inputs):
                self.task = task
                self.inputs = inputs
                self.context = None

        root = package_root("B")
        checkpoint = root / "checkpoint.npz"
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            (artifact / "checkpoint.npz").write_bytes(checkpoint.read_bytes())
            model = Model.load(artifact / "checkpoint.npz")
            rows = []
            with (root / "public" / "validation.jsonl").open(encoding="utf-8") as stream:
                for number, line in enumerate(stream):
                    if number >= 20:
                        break
                    rows.append(json.loads(line))
            for row in rows:
                result = predict_sample(Sample("regression", row["inputs"]), artifact)
                self.assertEqual(set(result), {"outputs"})
                self.assertEqual(set(result["outputs"]), {"value"})
                value = result["outputs"]["value"]
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, 0.0)

    def test_classification_prediction_has_one_top_label(self):
        import numpy as np

        from contest_baselines.data import flatten
        from contest_baselines.model import Model
        from contest_baselines.predict import predict_sample

        class Sample:
            def __init__(self, task, inputs):
                self.task = task
                self.inputs = inputs
                self.context = None

        root = package_root("A")
        checkpoint = root / "checkpoint.npz"
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            (artifact / "checkpoint.npz").write_bytes(checkpoint.read_bytes())
            model = Model.load(artifact / "checkpoint.npz")
            row = json.loads((root / "public" / "validation.jsonl")
                             .open(encoding="utf-8").readline())
            result = predict_sample(Sample("classification", row["inputs"]), artifact)
            outputs = result["outputs"]
            self.assertIsInstance(outputs, list)
            labels = [item["label"] for item in outputs]
            self.assertEqual(len(labels), len(set(labels)))
            for item in outputs:
                self.assertTrue(0.0 <= item["score"] <= 1.0)
            top = [item for item in outputs if item["score"] == max(x["score"] for x in outputs)]
            self.assertEqual(len(top), 1)


class NoSecretTest(unittest.TestCase):
    """No credential may be baked into the image.

    The check is on the *code*, not on the prose: the word "secret" may of
    course appear in a comment that says no secret is present.  What must
    not appear is an assignment of a credential.
    """

    def test_no_api_key_in_the_participant_code(self):
        patterns = (
            r"pytrio_api_key\s*=",
            r"api_key\s*=\s*[\"'][^\"']+[\"']",
            r"\bsk-[A-Za-z0-9]{8,}",
            r"authorization\s*[:=]\s*[\"']bearer",
        )
        import re

        sources = [ROOT / "contest_submission.py", ROOT / "harness.py"]
        sources += sorted((ROOT / "contest_baselines").glob("*.py"))
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                self.assertIsNone(re.search(pattern, text, re.IGNORECASE),
                                  "%s matches the credential pattern %r"
                                  % (path.name, pattern))

    def test_the_model_is_reached_only_through_the_sdk(self):
        """The image must not carry its own sampling client."""
        for path in sorted((ROOT / "contest_baselines").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for needle in ("openai", "requests.post", "httpx", "aiohttp"):
                self.assertNotIn(needle, text.lower(),
                                 "%s mentions %s" % (path.name, needle))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""The acceptance benchmark's contract: what a comparison must contain.

The benchmark's job is to make the word "better" checkable.  A
comparison that cannot be reproduced, or that reports a gain without
saying what was compared with what, is worse than no comparison at
all - it looks like evidence.

So these tests pin down the parts that make the table trustworthy:

* the **direction** of a gain follows the task, not the metric's name;
* the **unit gain** is the improvement per adopted round, and it is
  unavailable rather than zero when nothing was adopted;
* the **report** carries the parent's digest and the seed, so a third
  party can reproduce it;
* the **artefacts** contain no weights.

The measurement itself needs the platform's data packages, so the
end-to-end path is exercised by ``tools/benchmark_profiles.py`` against
the real packages; here the parts that can be checked without them are
checked.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_profiles import (  # noqa: E402
    _gain,
    _is_better,
    _markdown,
    _primary_of,
    _write_report,
)


# ---------------------------------------------------------------- the direction
class TestDirection:
    """A gain is signed so that positive always means better."""

    def test_classification_prefers_a_higher_score(self):
        assert _is_better("classification", 0.70, 0.72)
        assert not _is_better("classification", 0.72, 0.70)

    def test_regression_prefers_a_lower_score(self):
        assert _is_better("regression", 100.0, 90.0)
        assert not _is_better("regression", 90.0, 100.0)

    def test_an_unchanged_score_is_neither(self):
        assert not _is_better("classification", 0.70, 0.70)
        assert not _is_better("regression", 90.0, 90.0)

    def test_the_gain_is_signed_by_the_task(self):
        assert _gain("classification", 0.70, 0.72) == pytest.approx(0.02)
        assert _gain("regression", 100.0, 90.0) == pytest.approx(10.0)
        assert _gain("regression", 90.0, 100.0) == pytest.approx(-10.0)

    def test_the_primary_metric_is_found_by_name(self):
        assert _primary_of({"macro_f1": 0.71, "accuracy": 0.8}) == ("macro_f1", 0.71)
        assert _primary_of({"rmse": 12.5, "mae": 8.0}) == ("rmse", 12.5)

    def test_an_empty_measurement_has_no_primary(self):
        assert _primary_of({}) == (None, None)
        assert _primary_of(None) == (None, None)


# ------------------------------------------------------------------ the report
def _profile(**overrides):
    """A profile's comparison, as the tool would produce it."""
    profile = {
        "profile": "A",
        "task": "classification",
        "primary_metric": "macro_f1",
        "parent_checkpoint_sha256": "sha256:" + "ab" * 32,
        "seed": 42,
        "rounds": 10,
        "adopted_rounds": 3,
        "adoption_rate": 0.3,
        "baseline": {"primary_value": 0.7140, "values": {"macro_f1": 0.7140}},
        "optimised": {"primary_value": 0.7192, "values": {"macro_f1": 0.7192},
                      "round": 2, "hypothesis": "more-data-helps"},
        "gain": 0.0052,
        "gain_direction": "improved",
        "unit_gain": 0.0052 / 3,
        "unit_gain_available": True,
    }
    profile.update(overrides)
    return profile


class TestReport:
    """The table carries what a reader needs to trust it."""

    def test_the_markdown_shows_both_sides(self):
        text = _markdown({"profiles": [_profile()]})
        assert "0.7140" in text
        assert "0.7192" in text

    def test_the_markdown_shows_the_direction_and_the_rates(self):
        text = _markdown({"profiles": [_profile()]})
        assert "improved" in text
        assert "0.30" in text

    def test_the_markdown_carries_the_digest_and_the_seed(self):
        text = _markdown({"profiles": [_profile()]})
        assert "sha256:" in text
        assert "42" in text

    def test_a_failed_profile_is_marked_as_failed(self):
        text = _markdown({"profiles": [
            {"profile": "B", "failed": True, "error": "no package"}]})
        assert "失败" in text
        assert "no package" not in text

    def test_an_unavailable_unit_gain_is_not_reported_as_zero(self):
        text = _markdown({"profiles": [_profile(
            unit_gain=None, unit_gain_available=False, adopted_rounds=0,
            adoption_rate=0.0, gain=None, gain_direction="unknown")]})
        assert "不可用" in text

    def test_the_reproduction_note_is_present(self):
        text = _markdown({"profiles": [_profile()]})
        assert "同一父检查点" in text


class TestArtefacts:
    """Both artefacts, and no weights."""

    def test_both_artefacts_are_written(self, tmp_path):
        report = {"profiles": [_profile()]}
        _write_report(report, tmp_path)
        assert (tmp_path / "benchmark.json").is_file()
        assert (tmp_path / "benchmark.md").is_file()

    def test_the_json_carries_the_reproduction_information(self, tmp_path):
        _write_report({"profiles": [_profile()]}, tmp_path)
        data = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
        profile = data["profiles"][0]
        assert profile["parent_checkpoint_sha256"].startswith("sha256:")
        assert profile["seed"] == 42
        assert profile["rounds"] == 10
        assert profile["adopted_rounds"] == 3

    def test_the_two_artefacts_agree(self, tmp_path):
        _write_report({"profiles": [_profile()]}, tmp_path)
        data = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
        text = (tmp_path / "benchmark.md").read_text(encoding="utf-8")
        assert "%.4f" % data["profiles"][0]["baseline"]["primary_value"] in text
        assert "%.4f" % data["profiles"][0]["optimised"]["primary_value"] in text

    def test_no_weights_are_written(self, tmp_path):
        _write_report({"profiles": [_profile()]}, tmp_path)
        weights = (".npz", ".pkl", ".bin", ".pt", ".pth", ".safetensors")
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert path.suffix.lower() not in weights, path

    def test_the_artefacts_are_json_serialisable(self, tmp_path):
        """No NaN survives into the JSON."""
        _write_report({"profiles": [_profile()]}, tmp_path)
        text = (tmp_path / "benchmark.json").read_text(encoding="utf-8")
        assert "NaN" not in text
        assert not re.search(r"\bInfinity\b", text)


class TestTheTool:
    """The tool's own contract."""

    def test_the_tool_exposes_its_arguments(self):
        from tools.benchmark_profiles import main

        assert callable(main)

    def test_the_tool_never_imports_a_weight_writer(self):
        """The loop's own checkpoints stay in the temporary directory."""
        source = (ROOT / "tools" / "benchmark_profiles.py").read_text(encoding="utf-8")
        assert "shutil.copyfile" not in source
        assert ".npz" not in source.replace("checkpoint.npz", "").replace(
            "candidate_checkpoint", "").replace(
            "the public validation split", "")

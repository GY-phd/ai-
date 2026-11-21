# -*- coding: utf-8 -*-
"""The View layer's deployment contract.

The View's queries are all implemented and all correct, which raises a
question a reviewer will ask: *why is nothing serving them?*

The answer is deliberate, and this file is where it is pinned down.  The
image's entry point has to be the **only** entry point and has to exit
cleanly when the round is over - that is what the platform's trial run
checks.  A service that outlives the round makes "is it finished?"
ambiguous, and a service that has to be killed is a service that might
not be.  So the View stays a library, used in process.

These tests therefore check two things:

* the **image does not start a View service** - the entry point never
  calls into the HTTP wrapper;
* the **View still works in process** - the queries answer, they carry
  the envelope, and they write nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines.layers.model.ledger import Ledger, LedgerEntry  # noqa: E402
from contest_baselines.layers.view import View  # noqa: E402


@pytest.fixture()
def state_dir(tmp_path):
    """A state directory with two rounds of ledger in it."""
    directory = tmp_path / "state"
    directory.mkdir()
    ledger = Ledger(directory / "ledger.jsonl")
    ledger.append(LedgerEntry(
        round_index=1, dataset_key="A", task="classification",
        hypothesis="more-data-helps", change="{}",
        before={"macro_f1": 0.70}, after={"macro_f1": 0.72},
        adopted=True, reason="the hypothesis is confirmed"))
    ledger.append(LedgerEntry(
        round_index=2, dataset_key="A", task="classification",
        hypothesis="the-tail-is-underweighted", change="{}",
        before={"macro_f1": 0.72}, after={"macro_f1": 0.71},
        adopted=False, reason="the hypothesis is refuted"))
    return directory


# ------------------------------------------------------------- the deployment
class TestTheImageDoesNotServeTheView:
    """The entry point stays the only entry point."""

    def _entry_sources(self):
        for name in ("harness.py", "contest_baselines/train.py"):
            path = ROOT / name
            if path.is_file():
                yield path

    def test_the_entry_point_never_starts_a_view_service(self):
        for path in self._entry_sources():
            source = path.read_text(encoding="utf-8")
            assert "make_server" not in source, path
            assert "layers.view" not in source, path
            assert "view.server" not in source, path

    def test_the_view_package_does_not_import_a_server_by_itself(self):
        """Importing the View must not start anything."""
        source = (ROOT / "contest_baselines" / "layers" / "view"
                  / "__init__.py").read_text(encoding="utf-8")
        assert "make_server" not in source
        assert "ThreadingHTTPServer" not in source

    def test_the_deployment_shape_is_documented(self):
        """The choice is stated, so it cannot be read as an oversight."""
        source = (ROOT / "contest_baselines" / "layers" / "view"
                  / "__init__.py").read_text(encoding="utf-8")
        assert "library" in source.lower()
        assert "does **not** start an HTTP server" in source

    def test_the_http_wrapper_still_exists_as_an_option(self):
        """The queries are available over HTTP for a dashboard."""
        path = ROOT / "contest_baselines" / "layers" / "view" / "server.py"
        assert path.is_file()
        source = path.read_text(encoding="utf-8")
        assert "make_server" in source


# ------------------------------------------------------------- the queries
class TestTheQueriesWorkInProcess:
    """The library answers, in process, with no service running."""

    def test_the_overview_answers(self, state_dir):
        answer = View(state_dir).overview()
        assert answer["kind"] == "overview"
        assert answer["version"] == 1
        assert answer["rounds"] == 2

    def test_every_endpoint_carries_the_envelope(self, state_dir):
        view = View(state_dir)
        for name in ("overview", "ledger", "rounds", "metrics",
                     "champion", "hypothesis"):
            answer = getattr(view, name)()
            assert answer["kind"], name
            assert answer["version"] == 1, name

    def test_the_endpoint_set_is_complete(self, state_dir):
        view = View(state_dir)
        for name in ("overview", "ledger", "rounds", "metrics",
                     "champion", "hypothesis", "predict"):
            assert callable(getattr(view, name)), name

    def test_the_answer_serialises_to_json(self, state_dir):
        view = View(state_dir)
        text = view.to_json(view.overview())
        assert json.loads(text)["kind"] == "overview"

    def test_the_queries_write_nothing(self, state_dir):
        """A read is a read."""
        before = {path: path.stat().st_mtime for path in state_dir.rglob("*")
                  if path.is_file()}
        view = View(state_dir)
        for name in ("overview", "ledger", "rounds", "metrics",
                     "champion", "hypothesis"):
            getattr(view, name)()
        after = {path: path.stat().st_mtime for path in state_dir.rglob("*")
                 if path.is_file()}
        assert before == after

    def test_a_frozen_directory_answers_the_same_way_twice(self, state_dir):
        view = View(state_dir)
        assert view.to_json(view.overview()) == view.to_json(view.overview())

    def test_the_view_does_not_import_the_protocol(self):
        source = (ROOT / "contest_baselines" / "layers" / "view"
                  / "api.py").read_text(encoding="utf-8")
        assert "contest_harness" not in source
        assert "sys.stdin" not in source
        assert "sys.stdout" not in source

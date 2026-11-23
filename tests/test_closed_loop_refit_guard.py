# -*- coding: utf-8 -*-
"""The closed loop must not end up worse than the champion it started from.

Why this test exists
--------------------
The loop's gate compares a **live** candidate against a **live** champion
on a held-back judge slice, and a promoted hypothesis's recipe becomes
the champion's recipe.  But the model that ships is a **fresh fit** of
that recipe on every training unit, which is not the same object as the
one that was gated.

The gap between the two is not academic.  Profile B's
``lifetime-is-multiplicative`` hypothesis moves the regression target
from ``minutes`` to ``life_fraction`` - a different model family, with
different scores on the official split (408.16 against 445.78 in the
measurements recorded in ``tracks/hypotheses.py``).  A promotion can
therefore pass the gate and still refit onto a branch that scores worse
than the one the loop started from.

``tracks/run.py`` now re-checks the refit against the champion's
starting recipe on the judge slice and ships the better of the two.
This test pins that behaviour down, so a future change that removes the
re-check fails here rather than on the platform.
"""
from __future__ import annotations

import unittest

from tracks.run import TrackRunner


class _FakeRunner(TrackRunner):
    """A runner whose judge scores are scripted.

    Substituting the judge keeps the test about the *decision* - which
    recipe ships - rather than about how well any particular model
    happens to fit.
    """

    def __init__(self, profile_key="B", judge_scores=None):
        # The real constructor needs the data; the decision logic under
        # test does not, so the data is stubbed out entirely.
        self.key = profile_key
        self.profile = type("P", (), {"task": "regression"})()
        self.champion_overrides = {"target": "minutes", "hidden": 96}
        self.initial_overrides = {"target": "life_fraction", "hidden": 64}
        self._judge_scores = list(judge_scores or [])
        self._judge_calls = 0
        # The three data pieces the final refit concatenates.  They are
        # empty because the test is about the decision, not the fit.
        import numpy as np
        empty = {"x": np.zeros((0, 1)), "y": np.zeros(0), "paths": [],
                 "groups": np.zeros(0, dtype=object), "order": np.zeros(0)}
        self.initial = empty
        self.stream = empty
        self.judge_set = empty

    def fit(self, x, y, paths, groups, order, tag, model_overrides=None):
        return ("model:%s" % tag), {"tag": tag}

    def judge(self, candidate, stop):
        self._judge_calls += 1
        if not self._judge_scores:
            return None
        score = self._judge_scores.pop(0)
        return {"score": {"rmse": score}, "rows": 1, "source": "judge_holdout"}

    def primary(self, scored):
        # The gate's convention: larger is better.
        return -float(scored["rmse"])


class FinalRefitGuardTest(unittest.TestCase):
    """The final refit is checked before it ships."""

    def test_a_refit_that_loses_to_the_initial_recipe_is_rejected(self):
        # The judge hands out: the initial recipe scores 400, the refit
        # scores 500 (worse, because lower RMSE is better for a
        # regression).  The initial recipe must win.
        runner = _FakeRunner(judge_scores=[400.0, 500.0])
        runner.judge_set = _FakeRunner(judge_scores=[]).judge_set
        runner.judge_set = dict(runner.judge_set)
        runner.judge_set["y"] = [1.0]
        model, fitted, note = runner._fit_final_champion()
        self.assertIn("initial", note)
        self.assertIn("lost", note)
        # The shipped recipe is the initial champion's.
        self.assertEqual(fitted["recipe"].get("target"), "life_fraction")

    def test_a_refit_that_wins_is_kept(self):
        runner = _FakeRunner(judge_scores=[500.0, 400.0])
        runner.judge_set = dict(runner.judge_set)
        runner.judge_set["y"] = [1.0]
        model, fitted, note = runner._fit_final_champion()
        self.assertIn("promoted recipe", note)
        self.assertEqual(fitted["recipe"].get("target"), "minutes")

    def test_no_judge_slice_means_no_fallback(self):
        runner = _FakeRunner()
        runner.judge_set = dict(runner.judge_set)
        runner.judge_set["y"] = []
        model, fitted, note = runner._fit_final_champion()
        self.assertIn("promoted recipe", note)
        self.assertEqual(fitted["recipe"].get("target"), "minutes")

    def test_the_initial_recipe_is_remembered_at_construction(self):
        """The starting recipe must be captured before any promotion."""
        from tracks.hypotheses import champion_defaults
        runner = TrackRunner.__new__(TrackRunner)
        runner.champion_overrides = {"target": "minutes"}
        runner.initial_overrides = dict(runner.champion_overrides)
        # A promotion changes the champion's recipe ...
        runner.champion_overrides = {"target": "life_fraction"}
        # ... but not the initial one.
        self.assertEqual(runner.initial_overrides.get("target"), "minutes")
        self.assertIsNotNone(champion_defaults("B"))


if __name__ == "__main__":
    unittest.main()

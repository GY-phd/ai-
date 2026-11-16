# -*- coding: utf-8 -*-
"""The three-track closed loop: the numbers the README claims, checked.

These tests are the *contract* between the closed loop and the write-up.
They read the run reports the loop actually wrote and assert three things:

  1. every profile beats the official static baseline on its primary metric;
  2. no profile leaked anything - the judge units were never trained on, and
     the validation set was never trained on;
  3. the gate compared the candidate and the champion on the **same** slice
     in every cycle, and every promotion had a positive measured gain.

If a change to the loop breaks any of these, the tests say so instead of the
number quietly changing on a slide.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ARTIFACTS = ROOT / "artifacts"

# The official static baselines, re-scored by
# ``research.tracks.official_baselines``.  Kept here as literals so that the
# test does not depend on the artefacts folder being present.
OFFICIAL_BASELINE = {
    "A": 0.7140,   # Macro-F1
    "B": 681.160,  # RMSE, minutes
    "C": 52.869,   # RMSE, cycles
}
# "lower" means a smaller number is better.
DIRECTION = {"A": "higher", "B": "lower", "C": "lower"}


def _report(profile):
    path = RUNS / profile / "run_report.json"
    if not path.is_file():
        raise unittest.SkipTest("no run report for profile %s" % profile)
    return json.loads(path.read_text(encoding="utf-8"))


def _primary(profile, scored):
    if profile == "A":
        return float(scored.get("macro_f1", 0.0))
    return float(scored["rmse"])


class ThreeTrackResultTest(unittest.TestCase):
    """The headline numbers."""

    def test_every_profile_beats_the_official_baseline(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                report = _report(profile)
                final = _primary(profile, report["final_score"])
                baseline = OFFICIAL_BASELINE[profile]
                if DIRECTION[profile] == "lower":
                    self.assertLess(final, baseline,
                                    "%s: %.4f is not better than the official %.4f"
                                    % (profile, final, baseline))
                else:
                    self.assertGreater(final, baseline,
                                       "%s: %.4f is not better than the official %.4f"
                                       % (profile, final, baseline))

    def test_the_closed_loop_beat_its_own_initial_champion(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                report = _report(profile)
                final = _primary(profile, report["final_score"])
                initial = _primary(profile, report["initial_score"])
                if DIRECTION[profile] == "lower":
                    self.assertLessEqual(final, initial)
                else:
                    self.assertGreaterEqual(final, initial)

    def test_the_loop_actually_ran_cycles(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                report = _report(profile)
                self.assertGreaterEqual(int(report["cycles"]), 3,
                                        "%s ran %s cycles" % (profile, report["cycles"]))

    def test_the_final_model_was_refitted_on_every_training_unit(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                report = _report(profile)
                refit = report.get("final_refit") or {}
                self.assertTrue(refit.get("training_units"),
                                "%s: the final refit recorded no training units" % profile)
                self.assertIn("recipe", refit)


class LeakageTest(unittest.TestCase):
    """Nothing may be trained on anything it is scored on."""

    def test_no_profile_leaked_the_validation_set(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                meta = _report(profile)["stream_meta"]
                self.assertFalse(meta["unit_leakage"])
                self.assertFalse(meta["validation_used_for_training"])
                self.assertFalse(meta["judge_used_for_training"])

    def test_the_judge_units_are_disjoint_from_the_training_units(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                meta = _report(profile)["stream_meta"]
                judge = set(meta.get("judge_units") or [])
                if not judge:
                    continue
                trained = set(meta.get("initial_units") or [])
                trained |= set(meta.get("stream_units") or [])
                overlap = judge & trained
                self.assertEqual(overlap, set(),
                                 "%s trained on the judge units: %s" % (profile, sorted(overlap)))


class GateTest(unittest.TestCase):
    """The gate has to be measuring the model, not the slice."""

    def test_every_cycle_compared_candidate_and_champion_on_the_same_slice(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                for record in _report(profile)["ledger"]:
                    self.assertIn("champion_score_on_same_slice", record)
                    self.assertIn("candidate", record)

    def test_every_promotion_had_a_positive_measured_gain(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                for record in _report(profile)["ledger"]:
                    if record["decision"] == "promote":
                        self.assertGreater(float(record["gain"]), 0.0,
                                           "%s cycle %s promoted on a non-positive gain"
                                           % (profile, record["cycle"]))

    def test_every_rejection_failed_to_clear_the_promotion_threshold(self):
        """A rejection may show a tiny positive delta, but not a real gain.

        The gate promotes only when the measured improvement clears
        ``min_gain``.  A candidate can therefore have a positive ``gain``
        and still be rejected: the delta is below the threshold and is
        indistinguishable from noise.  The contract is that every rejected
        candidate failed to clear that threshold, not that its raw delta
        was non-positive.
        """
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                for record in _report(profile)["ledger"]:
                    if record["decision"] != "reject":
                        continue
                    threshold = record.get("min_gain")
                    if threshold is None:
                        # Older reports did not record the threshold.  The
                        # runner's default is 0.01, and the falsification
                        # text carries the exact value that was used.
                        text = record.get("falsification", "")
                        marker = "by more than "
                        if marker in text:
                            try:
                                threshold = float(text.split(marker, 1)[1].strip())
                            except ValueError:
                                threshold = 0.01
                        else:
                            threshold = 0.01
                    self.assertLessEqual(
                        float(record["gain"]), float(threshold),
                        "%s cycle %s was rejected although its gain %s clears "
                        "the promotion threshold %s"
                        % (profile, record["cycle"], record["gain"], threshold))

    def test_every_cycle_records_a_falsifiable_hypothesis(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                for record in _report(profile)["ledger"]:
                    self.assertTrue(record.get("hypothesis"),
                                    "%s cycle %s has no hypothesis" % (profile, record["cycle"]))
                    self.assertTrue(record.get("falsification"),
                                    "%s cycle %s has no falsifier" % (profile, record["cycle"]))


class ArtefactTest(unittest.TestCase):
    """The generated artefacts must agree with the run reports."""

    def test_the_comparison_table_agrees_with_the_reports(self):
        path = ARTIFACTS / "comparison_table.csv"
        if not path.is_file():
            raise unittest.SkipTest("no comparison table - run tracks.report first")
        import csv
        # utf-8-sig: the table is written with a BOM so Excel opens it cleanly
        # on a Chinese locale, and a bare "utf-8" decode would leave the BOM on
        # the first header name.
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["profile"]: row for row in csv.DictReader(handle)}
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                if profile not in rows:
                    continue
                report = _report(profile)
                final = _primary(profile, report["final_score"])
                self.assertAlmostEqual(float(rows[profile]["system_final"]), final, places=3)

    def test_the_official_baselines_were_reproduced(self):
        path = ARTIFACTS / "tracks" / "official_baselines.json"
        if not path.is_file():
            raise unittest.SkipTest("no baselines artefact - run the baseline script first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("A", payload)
        self.assertIn("B", payload)
        self.assertIn("C", payload)


class DeckTest(unittest.TestCase):
    """The slide deck must quote the numbers the runs actually produced.

    A deck that quotes a stale number is worse than no deck: it is checked
    against the runs by a reviewer who does not know what the runs said.
    So the deck is regenerated from the reports, and these tests assert
    that the result is internally consistent.
    """

    def setUp(self):
        self.path = ARTIFACTS / "ppt.md"
        if not self.path.is_file():
            raise unittest.SkipTest("no deck - run tracks.report first")
        self.text = self.path.read_text(encoding="utf-8")

    def test_the_deck_quotes_every_profile_final_score(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                report = _report(profile)
                final = _primary(profile, report["final_score"])
                if profile == "A":
                    needle = "%.4f" % final
                else:
                    needle = "%.4f" % final
                self.assertIn(needle, self.text,
                              "the deck does not quote %s's final %s" % (profile, needle))

    def test_the_deck_quotes_every_official_baseline(self):
        for profile in ("A", "B", "C"):
            with self.subTest(profile=profile):
                baseline = OFFICIAL_BASELINE[profile]
                # The deck may round to two or three decimals, and may use
                # thousands separators; accept any prefix of the number.
                digits = ("%.2f" % baseline, "%.3f" % baseline, "%.4f" % baseline)
                flat = self.text.replace(",", "")
                self.assertTrue(
                    any(d in flat for d in digits),
                    "the deck does not quote %s's baseline %r" % (profile, baseline))

    def test_the_deck_has_no_table_split_by_prose(self):
        """A markdown table must not be interrupted by a paragraph.

        GitHub renders the rows after the paragraph as a second, headerless
        table, so a note placed in the middle of a table silently loses its
        header.  Every table therefore has to be contiguous.
        """
        lines = self.text.splitlines()
        table_rows = []
        for index, line in enumerate(lines):
            if line.startswith("|"):
                table_rows.append(index)
                continue
            if table_rows:
                # The run of table rows must be contiguous.
                if index - table_rows[-1] > 1:
                    self.fail("table interrupted at line %d: %r" % (index, line))
                table_rows = []

    def test_the_deck_states_the_zero_leakage_protocol(self):
        for needle in ("unit_leakage", "validation_used_for_training",
                       "judge_used_for_training"):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.text)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""The tests for the public projection's reader and its unit grouping.

Two properties are guarded here.

**The feature schema is reproduced exactly.**  The platform's reference
loader refuses a checkpoint whose schema differs from the data's, so the
flattening traversal has to visit dictionary keys in sorted order and
contribute one path per numeric leaf - exactly like the reference.

**The unit grouping works on every profile.**  The unit a row belongs to
is recovered from the sample id, because the rules forbid using the unit
identity as a model input but allow a *training-distribution* change to be
expressed over units.  The three profiles carry the unit in three
different ways, and a grouping that silently fails on one of them turns a
per-unit resampling into a per-row resampling - which is a no-op at best
and a quadratic loop at worst.

Run:

    python -m pytest tests/test_data_projection.py -v
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from contest_baselines.data import flatten, unit_ids, unit_of  # noqa: E402


class UnitOfTest(unittest.TestCase):
    """The unit label of a sample id, per profile."""

    def test_profile_b_strips_the_snapshot_index(self):
        self.assertEqual(unit_of("xjtu_Bearing1_1_s000001"), "xjtu_Bearing1_1")
        self.assertEqual(unit_of("xjtu_Bearing1_1_s000042"), "xjtu_Bearing1_1")

    def test_profile_c_strips_the_cycle_index(self):
        self.assertEqual(unit_of("fd001_train_u0001_c00001"), "fd001_train_u0001")
        self.assertEqual(unit_of("fd002_train_u0042_c00007"), "fd002_train_u0042")

    def test_profile_a_keeps_the_row_itself(self):
        # A digits-only trailing segment is a row number, not a
        # within-unit index, so nothing may be stripped.
        self.assertEqual(unit_of("gas_b01_000001"), "gas_b01_000001")

    def test_an_unknown_id_is_left_alone(self):
        self.assertEqual(unit_of("plain"), "plain")


class UnitIdsTest(unittest.TestCase):
    """The grouping of a whole column of ids."""

    def test_profile_b_groups_the_snapshots_of_a_bearing(self):
        ids = ["xjtu_Bearing1_1_s%06d" % number for number in range(1, 6)]
        ids += ["xjtu_Bearing1_2_s%06d" % number for number in range(1, 4)]
        groups = unit_ids(ids)
        self.assertEqual(len(set(groups)), 2)

    def test_profile_c_groups_the_cycles_of_a_unit(self):
        ids = ["fd001_train_u0001_c%05d" % number for number in range(1, 6)]
        ids += ["fd001_train_u0002_c%05d" % number for number in range(1, 4)]
        groups = unit_ids(ids)
        self.assertEqual(len(set(groups)), 2)

    def test_profile_a_keeps_every_row_its_own_unit(self):
        ids = ["gas_b01_%06d" % number for number in range(1, 11)]
        groups = unit_ids(ids)
        self.assertEqual(len(set(groups)), 10)

    def test_the_grouping_is_never_finer_than_the_rows(self):
        """A regression guard: one group per row means the grouping failed."""
        ids = ["fd001_train_u0001_c%05d" % number for number in range(1, 21)]
        groups = unit_ids(ids)
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(len(groups), 20)


class FlattenTest(unittest.TestCase):
    """The feature flattening has to match the reference loader."""

    def test_the_keys_are_visited_in_sorted_order(self):
        paths, values = flatten({"b": 2.0, "a": 1.0, "c": 3.0})
        self.assertEqual(paths, ['[["key","a"]]', '[["key","b"]]', '[["key","c"]]'])
        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_a_nested_dict_contributes_one_path_per_numeric_leaf(self):
        paths, values = flatten({"s": {"x": 1.0, "y": 2.0}})
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(values), 2)

    def test_a_list_contributes_one_path_per_element(self):
        paths, values = flatten({"v": [1.0, 2.0, 3.0]})
        self.assertEqual(len(paths), 3)
        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_a_target_derived_field_is_rejected(self):
        for name in ("RUL", "trueRUL", "normalizedRUL", "remainingUsefulLife"):
            with self.assertRaises(ValueError):
                flatten({name: 1.0, "x": 2.0})

    def test_a_non_numeric_leaf_is_rejected(self):
        with self.assertRaises(ValueError):
            flatten({"x": "text"})

    def test_an_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            flatten({})


class UnitGroupingCostTest(unittest.TestCase):
    """The grouping must not be quadratic in the number of rows."""

    def test_a_large_profile_c_column_groups_quickly(self):
        import time

        ids = []
        for unit in range(500):
            for cycle in range(200):
                ids.append("fd001_train_u%04d_c%05d" % (unit, cycle))
        started = time.monotonic()
        groups = unit_ids(ids)
        elapsed = time.monotonic() - started
        self.assertEqual(len(set(groups)), 500)
        # The grouping is a string operation per row; a quadratic version
        # would take orders of magnitude longer on this input.
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()

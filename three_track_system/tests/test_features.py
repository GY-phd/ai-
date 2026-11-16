# -*- coding: utf-8 -*-
"""Feature contract: schema order, transforms, and the guards around inputs."""
import numpy as np
import unittest

from rulkit.features import CHANNEL_FIELDS, TRANSFORMS, flatten, transform


def make_inputs(offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    channels = {}
    for channel in ("horizontal", "vertical"):
        channels[channel] = {name: float(rng.uniform(0.1, 5.0) + offset) for name in CHANNEL_FIELDS}
        channels[channel]["skewness"] = float(rng.normal())
        channels[channel]["spectral_kurtosis"] = 1.0
    return {"features": channels}


class FlattenTest(unittest.TestCase):
    def test_order_is_sorted_and_horizontal_first(self):
        inputs = {
            "features": {
                "vertical": {"b": 2.0, "a": 1.0},
                "horizontal": {"b": 4.0, "a": 3.0},
            }
        }
        paths, values = flatten(inputs)
        self.assertEqual(values, [3.0, 4.0, 1.0, 2.0])
        self.assertIn('"horizontal"', paths[0])
        self.assertIn('"vertical"', paths[2])

    def test_rejects_label_derived_field_names(self):
        for bad in ("rul", "TrueRUL", "total_snapshots", "normalized_RUL"):
            with self.assertRaises(ValueError):
                flatten({"features": {"horizontal": {bad: 1.0}}})

    def test_rejects_non_finite_and_empty(self):
        with self.assertRaises(ValueError):
            flatten({"features": {"horizontal": {"rms": float("nan")}}})
        with self.assertRaises(ValueError):
            flatten({"features": {}})

    def test_schema_has_34_numeric_leaves(self):
        paths, values = flatten(make_inputs())
        self.assertEqual(len(paths), 34)
        self.assertEqual(len(values), 34)


class TransformTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.uniform(0.01, 50.0, size=(5, 34))
        self.x[:, 4] = rng.normal(size=5)          # skewness may be negative
        self.x[:, 12] = rng.uniform(1e3, 1e5, 5)   # spectral kurtosis is heavy tailed

    def test_all_transforms_preserve_shape_and_finiteness(self):
        for mode in TRANSFORMS:
            z = transform(self.x, mode)
            self.assertEqual(z.shape, self.x.shape, mode)
            self.assertTrue(np.all(np.isfinite(z)), mode)

    def test_raw_is_identity(self):
        np.testing.assert_array_equal(transform(self.x, "raw"), self.x)

    def test_eng34_band_ratios_sum_to_one(self):
        z = transform(self.x, "eng34")
        for channel in range(2):
            block = z[:, channel * 17:(channel + 1) * 17]
            idx = [CHANNEL_FIELDS.index(name) for name in
                   ("band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")]
            shares = block[:, idx].sum(axis=1)
            np.testing.assert_allclose(shares, 1.0, rtol=1e-9)

    def test_unknown_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            transform(self.x, "nope")


if __name__ == "__main__":
    unittest.main()

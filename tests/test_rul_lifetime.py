# -*- coding: utf-8 -*-
"""Profile B's lifetime estimate: the defect that was found, and the fix.

The defect
----------
``rulkit.rul.safe_lifetime`` returns the **remaining** life implied by
the predicted fraction::

    instantaneous = f * t / (1 - f)

and ``LifeFractionModel.predict`` treated the returned value as the
bearing's **total lifetime**, multiplying by the fraction a second
time::

    product = fraction * scale        # = f * (f * t / (1 - f))

so the fraction was applied twice.  Under leave-one-bearing-out the
estimated lifetime came out at 32% to 59% of the truth on **every**
bearing, and a global multiplier fitted independently in each fold
came out at 2.02 to 2.13 - the same defect seen from the other side.

The ratio is also the wrong estimator on its own terms: it divides a
noisy numerator by a noisy denominator at every point, so Jensen's
inequality biases it **low**, more so the noisier the fraction model is.

The fix
-------
``rulkit.rul.fitted_lifetime`` fits the relation ``t = N * (1 - f)`` by
least squares over each unit's own history.  The relation is exact when
the fraction is right, so the fit is unbiased, and the fraction model's
noise averages out over the history instead of being amplified at every
point.

These tests pin the behaviour down with **synthetic** units, so they
run in a second and do not need the competition's data package:

* the fitted lifetime recovers a known lifetime from noisy fractions,
  while the ratio form is biased low;
* the fraction is applied exactly once;
* the champion's recipe does not use the ratio form or the affine map.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulkit.rul import fitted_lifetime, safe_lifetime  # noqa: E402


def _unit(lifetime, noise, seed):
    """A unit with a known lifetime and a noisy predicted fraction."""
    rng = np.random.default_rng(seed)
    elapsed = np.arange(lifetime, dtype=np.float64) + 1.0
    truth = (lifetime - elapsed + 1.0) / lifetime
    fraction = np.clip(truth + rng.normal(0.0, noise, lifetime), 0.0, 1.0 - 1e-3)
    return fraction, elapsed, truth


class FittedLifetimeTest(unittest.TestCase):
    """The fitted lifetime estimator."""

    def test_it_recovers_a_known_lifetime(self):
        for lifetime in (60, 200, 800):
            fraction, elapsed, _ = _unit(lifetime, noise=0.03, seed=lifetime)
            groups = np.zeros(lifetime, dtype=object)
            fitted = fitted_lifetime(fraction, elapsed, groups, floor=0.02,
                                     cap=3000.0)
            self.assertAlmostEqual(float(fitted[0]), float(lifetime),
                                   delta=lifetime * 0.15,
                                   msg="lifetime %d was not recovered" % lifetime)

    def test_it_is_unbiased_where_the_ratio_is_not(self):
        """The ratio is biased low; the fit is not.

        This is the whole reason the fix exists, so it is measured
        directly rather than asserted through a downstream score.
        """
        lifetime = 200
        fraction, elapsed, _ = _unit(lifetime, noise=0.05, seed=7)
        groups = np.zeros(lifetime, dtype=object)

        fitted = float(fitted_lifetime(fraction, elapsed, groups,
                                       floor=0.02, cap=3000.0)[0])
        # The ratio form, as the old code used it: the *scale* it
        # returns is the remaining life, and the old code then
        # multiplied by the fraction again.
        ratio_scale = float(np.median(safe_lifetime(fraction, elapsed,
                                                    floor=0.02, cap=3000.0,
                                                    rate_limit=0.25)))
        old_estimate = float(np.median(fraction * ratio_scale))

        self.assertGreater(fitted / lifetime, 0.75,
                           "the fitted lifetime is biased low: %.1f of %d"
                           % (fitted, lifetime))
        self.assertLess(old_estimate / lifetime, 0.75,
                        "the ratio form was expected to be biased low, but "
                        "it came out at %.1f of %d" % (old_estimate, lifetime))

    def test_the_fraction_is_applied_exactly_once(self):
        """``product = fraction * lifetime``, not ``fraction * scale * fraction``."""
        lifetime = 150
        fraction, elapsed, _ = _unit(lifetime, noise=0.02, seed=11)
        groups = np.zeros(lifetime, dtype=object)
        fitted = fitted_lifetime(fraction, elapsed, groups, floor=0.02,
                                 cap=3000.0)
        product = fraction * fitted
        # The remaining life of the unit, in the unit's own units.
        truth = (lifetime - np.arange(lifetime) - 1.0) / lifetime
        expected = truth * lifetime
        error = float(np.sqrt(np.mean((product - expected) ** 2)))
        self.assertLess(error, lifetime * 0.25,
                        "the product is not the remaining life: rmse %.1f"
                        % error)

    def test_it_falls_back_when_the_fit_is_degenerate(self):
        """A unit with a single observation cannot be fitted."""
        fraction = np.array([0.5])
        elapsed = np.array([10.0])
        groups = np.array(["only"])
        fitted = fitted_lifetime(fraction, elapsed, groups, floor=0.02,
                                 cap=3000.0)
        self.assertTrue(np.isfinite(fitted).all())
        self.assertGreater(float(fitted[0]), 0.0)

    def test_each_unit_gets_its_own_lifetime(self):
        """The fit is per unit, on that unit's own history."""
        first_fraction, first_elapsed, _ = _unit(80, noise=0.02, seed=1)
        second_fraction, second_elapsed, _ = _unit(500, noise=0.02, seed=2)
        fraction = np.concatenate([first_fraction, second_fraction])
        elapsed = np.concatenate([first_elapsed, second_elapsed])
        groups = np.array(["a"] * 80 + ["b"] * 500, dtype=object)
        fitted = fitted_lifetime(fraction, elapsed, groups, floor=0.02,
                                 cap=3000.0)
        self.assertAlmostEqual(float(fitted[0]), 80.0, delta=20.0)
        self.assertAlmostEqual(float(fitted[-1]), 500.0, delta=75.0)


class ChampionRecipeTest(unittest.TestCase):
    """The champion's recipe must not fall back on the biased form."""

    def test_the_champion_uses_the_fitted_estimator_and_no_affine_map(self):
        from tracks.hypotheses import CHAMPION_DEFAULTS

        recipe = CHAMPION_DEFAULTS["B"]
        self.assertEqual(recipe.get("lifetime_estimator"), "fitted")
        self.assertEqual(recipe.get("affine"), "off")


class ModelIntegrationTest(unittest.TestCase):
    """The model honours the two switches."""

    def _model(self, **overrides):
        from rulkit.rul import LifeFractionModel

        kwargs = dict(floor=0.02, cap=3000.0, rate_limit=0.25,
                      normalisation="slope", affine=False)
        kwargs.update(overrides)
        return LifeFractionModel(**kwargs)

    def test_the_estimator_is_selectable(self):
        self.assertEqual(self._model(lifetime_estimator="fitted")
                         .lifetime_estimator, "fitted")
        self.assertEqual(self._model(lifetime_estimator="ratio")
                         .lifetime_estimator, "ratio")
        with self.assertRaises(ValueError):
            self._model(lifetime_estimator="nonsense")

    def test_the_two_estimators_disagree(self):
        """The regression form must actually change the answer.

        If the two agreed, the fix would be a no-op and the test suite
        would say nothing about which one is in use.
        """
        model = self._model(lifetime_estimator="fitted")
        fraction = np.linspace(0.9, 0.1, 50)
        elapsed = np.arange(1.0, 51.0)
        groups = np.zeros(50, dtype=object)
        fitted = model._lifetime_scale(fraction, groups, elapsed)
        ratio = self._model(lifetime_estimator="ratio")._lifetime_scale(
            fraction, groups, elapsed)
        self.assertFalse(np.allclose(fitted, ratio),
                         "the two lifetime estimators produced the same answer")


if __name__ == "__main__":
    unittest.main()

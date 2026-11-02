"""Tests for the standalone McCrary density-discontinuity diagnostic."""

import unittest

import numpy as np
import pandas as pd

from cais.methods.regression_discontinuity.density_test import (
    McCraryDensityTest,
    mccrary_density_test,
)


def _continuous_increasing_data(n: int = 3000, seed: int = 3) -> pd.DataFrame:
    """Sample from a density proportional to (x + 11) on [-10, 10].

    The density is continuous and increasing through the cutoff at 0, so a
    density-discontinuity test should not reject. This is the case the old
    binomial 50/50 check gets wrong.
    """
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        x = rng.uniform(-10, 10)
        if rng.uniform() < (x + 11) / 21:
            samples.append(x)
    return pd.DataFrame({"score": samples})


def _uniform_data(n: int = 2000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"score": rng.uniform(-5, 5, n)})


def _bunched_data(n: int = 2000, n_bunch: int = 800, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(-5, 5, n)
    bunched = rng.uniform(0.0, 0.5, n_bunch)
    return pd.DataFrame({"score": np.concatenate([base, bunched])})


class TestMcCraryDensityTest(unittest.TestCase):

    def test_continuous_increasing_density_passes(self):
        result = mccrary_density_test(
            _continuous_increasing_data(), "score", cutoff_value=0.0
        )
        self.assertIsNotNone(result.passed)
        self.assertTrue(result.passed, msg=result.reasoning)
        self.assertIn("No evidence of a density discontinuity", result.reasoning)
        self.assertIn("p_value", result.details)
        self.assertEqual(result.details["test_type"], "rddensity_cjm")

    def test_uniform_density_passes(self):
        result = mccrary_density_test(_uniform_data(), "score", cutoff_value=0.0)
        self.assertTrue(result.passed, msg=result.reasoning)

    def test_bunching_at_cutoff_fails(self):
        result = mccrary_density_test(_bunched_data(), "score", cutoff_value=0.0)
        self.assertFalse(result.passed)
        self.assertLess(result.details["p_value"], 0.05)
        self.assertGreater(result.details["jump"], 0.0)
        self.assertIn("Density discontinuity", result.reasoning)

    def test_missing_running_variable_returns_unknown(self):
        df = pd.DataFrame({"score": np.linspace(-1, 1, 100)})
        result = mccrary_density_test(df, "missing", cutoff_value=0.0)
        self.assertIsNone(result.passed)
        self.assertIn("not a column", result.reasoning)

    def test_missing_cutoff_returns_unknown(self):
        result = mccrary_density_test(_uniform_data(), "score", cutoff_value=None)
        self.assertIsNone(result.passed)
        self.assertIn("cutoff value is missing", result.reasoning)

    def test_non_numeric_cutoff_returns_unknown(self):
        result = mccrary_density_test(_uniform_data(), "score", cutoff_value="zero")
        self.assertIsNone(result.passed)
        self.assertIn("not numeric", result.reasoning)

    def test_non_numeric_running_variable_returns_unknown(self):
        df = pd.DataFrame({"score": ["a"] * 50 + ["b"] * 50})
        result = mccrary_density_test(df, "score", cutoff_value=0.0)
        self.assertIsNone(result.passed)
        self.assertIn("no numeric observations", result.reasoning)

    def test_too_few_observations_returns_unknown(self):
        df = pd.DataFrame({"score": [-1.0] * 5 + [1.0] * 5})
        result = mccrary_density_test(df, "score", cutoff_value=0.0)
        self.assertIsNone(result.passed)
        self.assertIn("too few observations", result.reasoning)

    def test_cutoff_outside_support_returns_unknown(self):
        df = pd.DataFrame({"score": np.linspace(1, 5, 100)})
        result = mccrary_density_test(df, "score", cutoff_value=10.0)
        self.assertIsNone(result.passed)
        self.assertIn("outside the observed", result.reasoning)

    def test_discrete_running_variable_does_not_crash(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"score": rng.integers(-10, 11, 1500).astype(float)})
        result = mccrary_density_test(df, "score", cutoff_value=0.0)
        self.assertIsInstance(result.passed, (bool, type(None)))
        self.assertIn("p_value", result.details)

    def test_sensitivity_reports_variants(self):
        result = mccrary_density_test(
            _uniform_data(), "score", cutoff_value=0.0, sensitivity=True
        )
        variants = result.details.get("sensitivity", {})
        self.assertEqual(set(variants), {"comb", "each", "diff"})
        for variant in variants.values():
            self.assertTrue("p_value" in variant or variant.get("status") == "could_not_test")

    def test_deterministic(self):
        df = _uniform_data()
        first = mccrary_density_test(df, "score", cutoff_value=0.0)
        second = mccrary_density_test(df, "score", cutoff_value=0.0)
        self.assertEqual(first.passed, second.passed)
        self.assertAlmostEqual(first.details["p_value"], second.details["p_value"])

    def test_class_wrapper_delegates(self):
        wrapper = McCraryDensityTest()
        result = wrapper.test(_uniform_data(), "score", cutoff_value=0.0)
        self.assertIsNotNone(result.passed)
        self.assertIn("p_value", result.details)


if __name__ == "__main__":
    unittest.main()

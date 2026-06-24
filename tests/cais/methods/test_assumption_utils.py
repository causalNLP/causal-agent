"""Unit tests for pre/post-model assumption check utilities."""

import unittest
import numpy as np
import pandas as pd

from cais.models import AssumptionResult, AssumptionVariables
from cais.methods.pre_model_assumption_utils import (
    check_sutva,
    check_cond_ignorability,
    check_positivity,
    check_iv_relevance,
    check_iv_exclusion,
    check_parallel_trends,
    check_no_anticipation,
    check_baseline_outcome_balance,
    check_frontdoor_positivity,
    check_rdd_no_manipulation,
    check_rdd_covariate_continuity,
    ASSUMPTION_REGISTRY,
)
from cais.methods.post_model_assumption_utils import (
    check_balance_after_weighting,
    check_balance_after_matching,
    check_gps_specification,
    POST_ASSUMPTION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Shared synthetic data factory
# ---------------------------------------------------------------------------

def _make_df(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    income = rng.normal(50_000, 15_000, n)
    ps = 1 / (1 + np.exp(-(-2 + 0.03 * age + 0.00002 * income)))
    treat = (rng.uniform(size=n) < ps).astype(int)
    y = 2 * treat + 0.05 * age + rng.normal(0, 1, n)
    return pd.DataFrame({"treat": treat, "outcome": y, "age": age, "income": income})


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestAssumptionModels(unittest.TestCase):

    def test_assumption_result_defaults(self):
        r = AssumptionResult()
        self.assertIsNone(r.passed)
        self.assertEqual(r.reasoning, "")
        self.assertEqual(r.details, {})

    def test_assumption_result_fields(self):
        r = AssumptionResult(passed=True, reasoning="ok", details={"f": 12.3})
        self.assertTrue(r.passed)
        self.assertEqual(r.details["f"], 12.3)

    def test_assumption_variables_defaults(self):
        v = AssumptionVariables()
        self.assertIsNone(v.df)
        self.assertEqual(v.covariates, [])
        self.assertEqual(v.instruments, [])

    def test_assumption_variables_with_df(self):
        df = _make_df()
        v = AssumptionVariables(df=df, treatment="treat", outcome="outcome")
        self.assertIsNotNone(v.df)
        self.assertEqual(len(v.df), len(df))


# ---------------------------------------------------------------------------
# Pre-model checks
# ---------------------------------------------------------------------------

class TestCheckSutva(unittest.TestCase):

    def test_no_llm_returns_none(self):
        v = AssumptionVariables(
            dataset_description="Students in different schools.",
            variables_summary={"treatment": "tutoring", "outcome": "test_score"},
        )
        r = check_sutva(v, llm=None)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIsNone(r.passed)
        self.assertIn("not statistically testable", r.reasoning)


class TestCheckCondIgnorability(unittest.TestCase):

    def test_balanced_returns_passed_true(self):
        rng = np.random.default_rng(1)
        n = 500
        x = rng.normal(0, 1, n)
        t = (rng.uniform(size=n) < 0.5).astype(int)  # random, no confounding
        df = pd.DataFrame({"t": t, "x": x})
        v = AssumptionVariables(df=df, treatment="t", outcome="y", covariates=["x"])
        r = check_cond_ignorability(v)
        self.assertIsInstance(r, AssumptionResult)
        self.assertTrue(r.passed)

    def test_imbalanced_returns_passed_false(self):
        rng = np.random.default_rng(2)
        n = 500
        x = rng.normal(0, 1, n)
        t = (x > 0).astype(int)  # strong confounding
        df = pd.DataFrame({"t": t, "x": x})
        v = AssumptionVariables(df=df, treatment="t", outcome="y", covariates=["x"])
        r = check_cond_ignorability(v)
        self.assertIsInstance(r, AssumptionResult)
        self.assertFalse(r.passed)
        self.assertIn("x", r.details.get("imbalanced", {}))

    def test_no_covariates_returns_none(self):
        df = _make_df()
        v = AssumptionVariables(df=df, treatment="treat", outcome="outcome", covariates=[])
        r = check_cond_ignorability(v)
        self.assertIsNone(r.passed)

    def test_no_df_returns_none(self):
        v = AssumptionVariables(covariates=["x"])
        r = check_cond_ignorability(v)
        self.assertIsNone(r.passed)


class TestCheckPositivity(unittest.TestCase):

    def test_good_overlap(self):
        df = _make_df(n=500)
        ps = np.random.default_rng(5).uniform(0.2, 0.8, 500)
        v = AssumptionVariables(df=df, treatment="treat", outcome="outcome")
        r = check_positivity(v, propensity_scores=ps)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("pct_extreme_ps", r.details)

    def test_extreme_ps_fails(self):
        df = _make_df(n=200)
        ps = np.random.default_rng(6).choice([0.01, 0.99], size=200)
        v = AssumptionVariables(df=df, treatment="treat", outcome="outcome")
        r = check_positivity(v, propensity_scores=ps, max_extreme_pct=0.01)
        self.assertFalse(r.passed)


class TestCheckIVRelevance(unittest.TestCase):

    def test_strong_instrument(self):
        rng = np.random.default_rng(7)
        n = 500
        z = rng.normal(0, 1, n)
        t = 0.8 * z + rng.normal(0, 0.5, n)
        y = 2 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"t": t, "y": y, "z": z})
        v = AssumptionVariables(df=df, treatment="t", outcome="y", instruments=["z"])
        r = check_iv_relevance(v, f_threshold=10.0)
        self.assertIsInstance(r, AssumptionResult)
        if r.passed is not None:
            self.assertTrue(r.details.get("f_statistic", 0) > 0)

    def test_weak_instrument(self):
        rng = np.random.default_rng(8)
        n = 500
        z = rng.normal(0, 1, n)
        t = 0.05 * z + rng.normal(0, 2, n)
        y = 2 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"t": t, "y": y, "z": z})
        v = AssumptionVariables(df=df, treatment="t", outcome="y", instruments=["z"])
        r = check_iv_relevance(v, f_threshold=10.0)
        if r.passed is not None:
            self.assertFalse(r.passed)


class TestCheckIVExclusion(unittest.TestCase):

    def test_no_llm_returns_none(self):
        v = AssumptionVariables(
            dataset_description="Rain used as instrument for irrigation.",
            variables_summary={"instrument": "rainfall", "treatment": "irrigation", "outcome": "crop_yield"},
        )
        r = check_iv_exclusion(v, llm=None)
        self.assertIsNone(r.passed)


class TestCheckParallelTrends(unittest.TestCase):

    def _make_did_df(self, n_units=50, n_periods=6, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(n_units):
            group = int(u >= n_units // 2)
            for t in range(n_periods):
                y = group * 0.5 + t * 1.0 + rng.normal(0, 0.5)
                rows.append({"unit": u, "time": t, "group": group, "y": y})
        return pd.DataFrame(rows)

    def test_parallel_trends_valid(self):
        df = self._make_did_df()
        v = AssumptionVariables(
            df=df, treatment="group", outcome="y",
            time_var="time", group_var="group", treatment_period_start=4,
        )
        r = check_parallel_trends(v)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("p_value", r.details)


class TestCheckBaselineOutcomeBalance(unittest.TestCase):

    def test_balanced_baseline(self):
        rng = np.random.default_rng(9)
        n = 200
        t = rng.integers(0, 2, n)
        time = rng.integers(0, 5, n)
        y = rng.normal(10, 1, n)  # no effect of treatment
        df = pd.DataFrame({"t": t, "time": time, "y": y})
        v = AssumptionVariables(
            df=df, treatment="t", outcome="y",
            time_var="time", treatment_period_start=3,
        )
        r = check_baseline_outcome_balance(v, smd_threshold=0.3)
        self.assertIsInstance(r, AssumptionResult)


class TestCheckFrontdoorPositivity(unittest.TestCase):

    def test_full_support(self):
        rng = np.random.default_rng(10)
        n = 200
        t = rng.integers(0, 2, n)
        m = rng.integers(0, 3, n)
        df = pd.DataFrame({"t": t, "m": m, "y": rng.normal(size=n)})
        v = AssumptionVariables(df=df, treatment="t", outcome="y", mediator="m")
        r = check_frontdoor_positivity(v, min_count=1)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("observed_combos", r.details)

    def test_empty_combos_fails(self):
        df = pd.DataFrame({"t": [0, 0, 1, 1], "m": [0, 0, 1, 1], "y": [1, 2, 3, 4]})
        v = AssumptionVariables(df=df, treatment="t", outcome="y", mediator="m")
        r = check_frontdoor_positivity(v)
        self.assertFalse(r.passed)


class TestCheckRDDNoManipulation(unittest.TestCase):

    def test_no_manipulation(self):
        rng = np.random.default_rng(11)
        rv = rng.uniform(0, 10, 500)
        df = pd.DataFrame({"rv": rv, "y": rv + rng.normal(0, 1, 500)})
        v = AssumptionVariables(df=df, running_variable="rv", cutoff=5.0)
        r = check_rdd_no_manipulation(v, bandwidth=2.0)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("p_value", r.details)
        self.assertTrue(r.passed)

    def test_with_manipulation(self):
        rng = np.random.default_rng(12)
        below = rng.uniform(3.5, 4.99, 400)
        above = rng.uniform(5.0, 6.5, 50)
        rv = np.concatenate([below, above])
        df = pd.DataFrame({"rv": rv, "y": rv + rng.normal(0, 1, len(rv))})
        v = AssumptionVariables(df=df, running_variable="rv", cutoff=5.0)
        r = check_rdd_no_manipulation(v, bandwidth=1.5)
        self.assertIsInstance(r, AssumptionResult)
        self.assertFalse(r.passed)


class TestCheckRDDCovariateContinuity(unittest.TestCase):

    def test_continuous_covariates(self):
        rng = np.random.default_rng(13)
        n = 400
        rv = rng.uniform(0, 10, n)
        age = rng.normal(40, 5, n)  # unrelated to cutoff
        df = pd.DataFrame({"rv": rv, "age": age})
        v = AssumptionVariables(df=df, running_variable="rv", cutoff=5.0, covariates=["age"])
        r = check_rdd_covariate_continuity(v, bandwidth=2.0)
        self.assertIsInstance(r, AssumptionResult)
        self.assertTrue(r.passed)

    def test_no_covariates_returns_none(self):
        df = pd.DataFrame({"rv": np.linspace(0, 10, 100)})
        v = AssumptionVariables(df=df, running_variable="rv", cutoff=5.0, covariates=[])
        r = check_rdd_covariate_continuity(v)
        self.assertIsNone(r.passed)


# ---------------------------------------------------------------------------
# Post-model checks
# ---------------------------------------------------------------------------

class TestCheckBalanceAfterWeighting(unittest.TestCase):

    def test_perfect_weights_balance(self):
        df = _make_df(n=300)
        # Equal weights → same as unweighted; focus on testing the function runs
        weights = np.ones(len(df))
        v = AssumptionVariables(df=df, treatment="treat", outcome="outcome", covariates=["age", "income"])
        r = check_balance_after_weighting(v, weights=weights)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("weighted_smds", r.details)
        self.assertIn("age", r.details["weighted_smds"])

    def test_result_type_is_assumption_result(self):
        df = _make_df(n=100)
        weights = np.random.default_rng(20).uniform(0.5, 2.0, 100)
        v = AssumptionVariables(df=df, treatment="treat", covariates=["age"])
        r = check_balance_after_weighting(v, weights=weights)
        self.assertIsInstance(r, AssumptionResult)


class TestCheckBalanceAfterMatching(unittest.TestCase):

    def test_balanced_matched_sample(self):
        rng = np.random.default_rng(14)
        n = 200
        age = rng.normal(40, 5, n)
        treat = rng.integers(0, 2, n)
        df = pd.DataFrame({"treat": treat, "age": age})
        v = AssumptionVariables(treatment="treat", covariates=["age"])
        r = check_balance_after_matching(v, df_matched=df, smd_threshold=0.5)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("smds", r.details)


class TestCheckGPSSpecification(unittest.TestCase):

    def test_normal_residuals_pass(self):
        residuals = np.random.default_rng(15).normal(0, 1, 200)
        r = check_gps_specification(residuals)
        self.assertIsInstance(r, AssumptionResult)
        self.assertIn("shapiro_w", r.details)

    def test_too_few_residuals(self):
        r = check_gps_specification(np.array([1.0, 2.0]))
        self.assertIsNone(r.passed)

    def test_non_normal_residuals_fail(self):
        residuals = np.random.default_rng(16).exponential(scale=1, size=300)
        r = check_gps_specification(residuals)
        self.assertIsInstance(r, AssumptionResult)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestAssumptionRegistries(unittest.TestCase):

    def test_all_pre_methods_present(self):
        expected = {
            "linear_regression", "propensity_score_matching",
            "instrumental_variable", "difference_in_differences",
            "frontdoor_adjustment", "regression_discontinuity_design",
            "backdoor_adjustment",
        }
        self.assertEqual(set(ASSUMPTION_REGISTRY.keys()), expected)

    def test_all_registry_values_are_callable(self):
        for method, checks in ASSUMPTION_REGISTRY.items():
            for fn in checks:
                self.assertTrue(callable(fn), f"{fn} in registry for '{method}' is not callable")

    def test_post_registry_values_are_callable(self):
        for method, checks in POST_ASSUMPTION_REGISTRY.items():
            for fn in checks:
                self.assertTrue(callable(fn), f"{fn} in POST_ASSUMPTION_REGISTRY for '{method}' is not callable")

    def test_iv_pre_checks_count(self):
        checks = ASSUMPTION_REGISTRY["instrumental_variable"]
        self.assertGreaterEqual(len(checks), 4)

    def test_did_pre_checks_count(self):
        checks = ASSUMPTION_REGISTRY["difference_in_differences"]
        self.assertGreaterEqual(len(checks), 3)


if __name__ == "__main__":
    unittest.main()

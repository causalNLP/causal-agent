"""Tests for the standalone donut RDD estimator."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from cais.estimator_lib import Estimators
from cais.methods import METHOD_MAPPING
from cais.methods.regression_discontinuity.donut_estimator import (
    DonutRDD,
    estimate_effect,
)

BASE_RDD = "cais.methods.regression_discontinuity.donut_estimator.rdd_estimate_effect"


def _rdd_df(n: int = 200, cutoff: float = 50.0, effect: float = 10.0, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    running = rng.uniform(cutoff - 20, cutoff + 20, n)
    treated = (running >= cutoff).astype(int)
    outcome = 10.0 + 0.8 * running + effect * treated + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "treatment_indicator": treated,
            "running_var": running,
            "covariate1": rng.normal(0, 1, n),
        }
    )


def _base_result() -> dict:
    return {
        "effect_estimate": 10.2,
        "standard_error": 1.1,
        "p_value": 0.01,
        "confidence_interval": [8.0, 12.4],
        "method_details": "base RDD (Bandwidth: 5.0)",
        "method_used": "evan-magnusson/rdd",
    }


def _variables(**overrides):
    values = dict(
        treatment_variable="treatment_indicator",
        outcome_variable="outcome",
        running_variable="running_var",
        cutoff_value=50.0,
        treat_above_cutoff=True,
        confounders=[],
        covariates=[],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDonutRDD(unittest.TestCase):

    @patch(BASE_RDD)
    def test_drops_band_and_records_estimand(self, mock_rdd):
        mock_rdd.return_value = _base_result()
        df = _rdd_df()

        result = estimate_effect(
            df,
            treatment="treatment_indicator",
            outcome="outcome",
            running_variable="running_var",
            cutoff_value=50.0,
            donut_width=5.0,
        )

        filtered = mock_rdd.call_args.kwargs["df"]
        self.assertLess(len(filtered), len(df))
        self.assertTrue((filtered["running_var"] - 50.0).abs().ge(5.0).all())
        self.assertEqual(result["method_used"], "donut_rdd")
        self.assertEqual(result["donut_width"], 5.0)
        self.assertEqual(result["n_dropped"], len(df) - len(filtered))
        self.assertAlmostEqual(result["share_dropped"], (len(df) - len(filtered)) / len(df))
        self.assertIn("outside", result["method_details"].lower())
        self.assertIn("donut_width", result["diagnostics"])
        self.assertIn("estimand", result["diagnostics"])

    def test_donut_width_is_required(self):
        with self.assertRaisesRegex(ValueError, "explicit donut_width"):
            estimate_effect(
                _rdd_df(), "treatment_indicator", "outcome", "running_var", 50.0
            )

    def test_invalid_donut_width(self):
        for bad in (0, -1, "wide", float("nan")):
            with self.assertRaisesRegex(ValueError, "donut_width must be|not numeric"):
                estimate_effect(
                    _rdd_df(),
                    "treatment_indicator",
                    "outcome",
                    "running_var",
                    50.0,
                    donut_width=bad,
                )

    def test_missing_running_variable_column(self):
        with self.assertRaisesRegex(ValueError, "not a column"):
            estimate_effect(
                _rdd_df(), "treatment_indicator", "outcome", "missing", 50.0,
                donut_width=1.0,
            )

    def test_missing_cutoff(self):
        with self.assertRaisesRegex(ValueError, "requires a cutoff"):
            estimate_effect(
                _rdd_df(), "treatment_indicator", "outcome", "running_var", None,
                donut_width=1.0,
            )

    def test_cutoff_outside_support(self):
        with self.assertRaisesRegex(ValueError, "outside the observed"):
            estimate_effect(
                _rdd_df(), "treatment_indicator", "outcome", "running_var", 500.0,
                donut_width=1.0,
            )

    def test_too_few_retained_after_donut(self):
        with self.assertRaisesRegex(ValueError, "Too few observations remain"):
            estimate_effect(
                _rdd_df(), "treatment_indicator", "outcome", "running_var", 50.0,
                donut_width=19.0,
            )

    @patch(BASE_RDD)
    def test_no_rows_dropped_is_allowed_with_warning(self, mock_rdd):
        mock_rdd.return_value = _base_result()
        df = _rdd_df(n=50)
        # Cutoff is outside the donut band for all observations.
        result = estimate_effect(
            df, "treatment_indicator", "outcome", "running_var", 50.0,
            donut_width=0.000001,
        )
        self.assertEqual(result["n_dropped"], 0)
        self.assertEqual(result["method_used"], "donut_rdd")

    @patch(BASE_RDD)
    def test_class_wrapper_delegates(self, mock_rdd):
        mock_rdd.return_value = _base_result()
        method = DonutRDD()
        result = method.estimate_effect(
            _rdd_df(), _variables(), donut_width=5.0
        )
        self.assertEqual(result["method_used"], "donut_rdd")
        self.assertEqual(result["effect_estimate"], 10.2)
        self.assertEqual(result["donut_width"], 5.0)

    def test_class_wrapper_returns_error_without_width(self):
        method = DonutRDD()
        result = method.estimate_effect(_rdd_df(), _variables())
        self.assertIsNone(result["effect_estimate"])
        self.assertEqual(result["method_used"], "donut_rdd")
        self.assertIn("donut_width", result["error"])

    def test_registered_in_method_mapping_and_estimator_lib(self):
        self.assertIn("donut_rdd", METHOD_MAPPING)
        self.assertIn(DonutRDD.name, Estimators().estimators)

    def test_validate_assumptions_warns_about_estimand(self):
        response = DonutRDD().validate_assumptions(_rdd_df(), _variables())
        self.assertTrue(response["assumptions_valid"])
        self.assertTrue(any("outside the donut" in w for w in response["warnings"]))


if __name__ == "__main__":
    unittest.main()

"""Output sentences for advisory post-modeling checks.

Post-modeling checks run after estimation and are advisory: they never change
the estimate or stop the run. Their verdicts are rendered as one sentence each
in CAIS's output - affirming the check when it was valid, warning the user when
it failed or could not be evaluated.
"""

import numpy as np
import pandas as pd
import pytest

from cais.components.explanation_generator import explain_post_checks, generate_explanation
from cais.methods.propensity_score import weighting as psw
from cais.models import Variables


def synthetic_weighting_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.5 * x2)))
    treatment = rng.binomial(1, propensity, n)
    outcome = 1.5 * treatment + 2.0 * x1 + 0.5 * x2 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome})


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Keep the estimators deterministic and offline."""
    monkeypatch.setattr(psw, "get_llm_parameters", lambda *a, **k: {})
    monkeypatch.setattr(psw, "determine_optimal_weight_type", lambda *a, **k: "ATE")
    monkeypatch.setattr(psw, "determine_optimal_trim_threshold", lambda *a, **k: 0.0)


class TestExplainPostChecks:
    def test_passing_check_is_affirmed(self):
        sentences = explain_post_checks([{
            "name": "balance_after_weighting",
            "passed": True,
            "reasoning": "All balanced after IPW.",
        }])

        assert len(sentences) == 1
        assert "was valid" in sentences[0]
        assert "covariate balance after weighting" in sentences[0]
        assert "All balanced after IPW" in sentences[0]
        assert "WARNING" not in sentences[0]

    def test_failing_check_warns(self):
        sentences = explain_post_checks([{
            "name": "balance_after_matching",
            "passed": False,
            "reasoning": "Still imbalanced: ['x1'].",
        }])

        assert "WARNING" in sentences[0]
        assert "did not pass" in sentences[0]
        assert "Interpret the estimate with caution" in sentences[0]
        assert "Still imbalanced" in sentences[0]

    def test_unevaluated_check_warns(self):
        sentences = explain_post_checks([{
            "name": "gps_specification",
            "passed": None,
            "reasoning": "",
        }])

        assert "WARNING" in sentences[0]
        assert "could not be evaluated" in sentences[0]
        assert "Interpret the estimate with caution" in sentences[0]
        assert "GPS model specification" in sentences[0]

    def test_no_checks_means_no_sentences(self):
        assert explain_post_checks(None) == []
        assert explain_post_checks([]) == []


class TestOutputRendering:
    def test_generate_explanation_renders_post_check_section(self):
        explanation = generate_explanation(
            method_info={
                "selected_method": "propensity_score_weighting",
                "method_name": "Inverse Probability Weighting",
                "method_assumptions": [],
            },
            validation_result={
                "method": "propensity_score_weighting",
                "post_checks": [
                    {
                        "name": "balance_after_weighting",
                        "passed": False,
                        "reasoning": "Still imbalanced: ['x1'].",
                    },
                    {
                        "name": "gps_specification",
                        "passed": None,
                        "reasoning": "not implemented yet",
                    },
                ],
            },
            variables={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "covariates": ["x1"],
                "original_query": "Does treatment affect outcome?",
            },
            results={
                "effect_estimate": 1.0,
                "standard_error": 0.2,
                "confidence_interval": [0.6, 1.4],
                "p_value": 0.01,
                "method_used": "propensity_score_weighting",
            },
            dataset_analysis=None,
            dataset_description="Synthetic dataset.",
            llm=None,
        )

        text = explanation["final_explanation_text"]
        assert "**Post-estimation checks (advisory):**" in text
        assert "covariate balance after weighting" in text
        assert "did not pass" in text
        assert "could not be evaluated" in text
        assert "Interpret the estimate with caution" in text


class TestEstimatorPostChecks:
    def test_ipw_reports_balance_after_weighting(self):
        df = synthetic_weighting_data()
        result = psw.estimate_effect(df, "treatment", "outcome", ["x1", "x2"])

        post_checks = result["diagnostics"]["post_checks"]
        verdict = post_checks["balance_after_weighting"]
        assert verdict["passed"] in (True, False, None)
        assert isinstance(verdict["reasoning"], str) and verdict["reasoning"]
        assert "weighted_smds" in verdict["details"]

    def test_trimmed_ipw_reports_balance_after_weighting(self):
        df = synthetic_weighting_data()
        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["x1", "x2"],
            confounders=["x1", "x2"],
        )

        result = psw.TrimmedIPW(ps_bounds=(0.1, 0.9)).estimate_effect(df, variables)

        post_checks = result["diagnostics"]["post_checks"]
        assert "balance_after_weighting" in post_checks
        assert post_checks["balance_after_weighting"]["passed"] in (True, False, None)

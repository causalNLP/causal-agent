"""Tests for trimmed IPW: the decision tree's positivity-failure remedy.

The leaf is reached when the positivity/overlap gate fails in the binary
backdoor path. It runs IPW after dropping units whose propensity score falls
outside the configured overlap band (``positivity.extreme_ps_bounds``).
"""

import numpy as np
import pandas as pd
import pytest

from cais.models import Variables
from cais.methods import METHOD_MAPPING
from cais.methods.propensity_score import weighting as psw


def synthetic_overlap_violation(n=800, seed=7):
    """Binary treatment with strong covariate-driven selection.

    Enough units get extreme propensity scores that trimming to [0.1, 0.9]
    drops a non-trivial share of the sample while keeping both groups.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    logit = 2.5 * x1 - 1.5 * x2
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, propensity, n)
    outcome = 2.0 * treatment + 1.5 * x1 + 0.5 * x2 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome})


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Keep the estimator deterministic and offline."""
    monkeypatch.setattr(psw, "get_llm_parameters", lambda *a, **k: {})
    monkeypatch.setattr(psw, "determine_optimal_weight_type", lambda *a, **k: "ATE")
    monkeypatch.setattr(psw, "determine_optimal_trim_threshold", lambda *a, **k: 0.0)


def run_trimmed(df, bounds=(0.1, 0.9), **kwargs):
    return psw.estimate_effect(
        df,
        "treatment",
        "outcome",
        ["x1", "x2"],
        ps_bounds=bounds,
        **kwargs,
    )


class TestPsBoundsTrimming:
    def test_ps_bounds_drops_extreme_scores_and_records_trimming(self):
        df = synthetic_overlap_violation()

        result = run_trimmed(df)

        trimming = result["diagnostics"]["trimming"]
        assert result["parameters"]["trim_rule"] == "ps_bounds"
        assert result["parameters"]["ps_bounds"] == [0.1, 0.9]
        assert 0 < trimming["n_trimmed"] < len(df)
        assert trimming["n_after"] == len(df) - trimming["n_trimmed"]
        assert 0.0 < trimming["dropped_fraction"] < 1.0
        assert np.isfinite(result["effect_estimate"])
        assert result["effect_se"] > 0

    def test_no_bounds_keeps_the_full_sample(self):
        df = synthetic_overlap_violation()

        result = psw.estimate_effect(
            df, "treatment", "outcome", ["x1", "x2"], trim_threshold=0.0
        )

        trimming = result["diagnostics"]["trimming"]
        assert result["parameters"]["trim_rule"] == "none"
        assert trimming["n_trimmed"] == 0
        assert trimming["n_after"] == len(df)
        assert np.isfinite(result["effect_estimate"])

    def test_invalid_bounds_raise(self):
        df = synthetic_overlap_violation(n=80)

        with pytest.raises(ValueError):
            run_trimmed(df, bounds=(0.9, 0.1))
        with pytest.raises(ValueError):
            run_trimmed(df, bounds=(-0.1, 0.9))
        with pytest.raises(ValueError):
            run_trimmed(df, bounds=(0.5,))

    def test_trimming_that_empties_the_sample_raises(self, monkeypatch):
        df = synthetic_overlap_violation(n=60)
        monkeypatch.setattr(
            psw, "estimate_propensity_scores", lambda *a, **k: np.full(len(df), 0.02)
        )

        with pytest.raises(ValueError, match="All units removed"):
            run_trimmed(df)


class TestTrimmedIPWWrapper:
    def test_contract_and_estimand_annotation(self):
        df = synthetic_overlap_violation()
        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["x1", "x2"],
            confounders=["x1", "x2"],
        )

        result = psw.TrimmedIPW(ps_bounds=(0.1, 0.9)).estimate_effect(df, variables)

        assert result["method_used"] == "trimmed_ipw"
        assert result["standard_error"] == result["effect_se"]
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["parameters"]["ps_bounds"] == [0.1, 0.9]
        assert result["diagnostics"]["trimming"]["n_trimmed"] > 0
        assert "overlap-satisfying subpopulation" in result["diagnostics"]["estimand"]

    def test_default_bounds_come_from_config(self, monkeypatch):
        df = synthetic_overlap_violation()

        def fake_thresholds():
            return {"positivity": {"extreme_ps_bounds": [0.2, 0.8]}}

        monkeypatch.setattr("cais.config.load_thresholds", fake_thresholds)
        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["x1", "x2"],
            confounders=["x1", "x2"],
        )

        result = psw.TrimmedIPW().estimate_effect(df, variables)

        assert result["parameters"]["ps_bounds"] == [0.2, 0.8]
        assert "0.2, 0.8" in result["diagnostics"]["estimand"]

    def test_returns_error_dict_instead_of_raising_when_trimming_leaves_nobody(
        self, monkeypatch
    ):
        df = synthetic_overlap_violation(n=60)
        monkeypatch.setattr(
            psw, "estimate_propensity_scores", lambda *a, **k: np.full(len(df), 0.02)
        )
        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["x1", "x2"],
            confounders=["x1", "x2"],
        )

        result = psw.TrimmedIPW(ps_bounds=(0.1, 0.9)).estimate_effect(df, variables)

        assert result["effect_estimate"] is None
        assert result["method_used"] == "trimmed_ipw"
        assert "All units removed" in result["diagnostics"]["error"]

    def test_requires_covariates(self):
        df = synthetic_overlap_violation(n=60)
        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=[],
            confounders=[],
        )

        result = psw.TrimmedIPW(ps_bounds=(0.1, 0.9)).estimate_effect(df, variables)

        assert result["effect_estimate"] is None
        assert "covariate" in result["error"].lower()


class TestRegistration:
    def test_method_mapping_entry_exists_and_is_distinct(self):
        assert "trimmed_ipw" in METHOD_MAPPING
        assert METHOD_MAPPING["trimmed_ipw"] is not psw.estimate_effect

    def test_mapping_entry_normalizes_contract(self):
        df = synthetic_overlap_violation()

        result = METHOD_MAPPING["trimmed_ipw"](
            df=df,
            treatment="treatment",
            outcome="outcome",
            covariates=["x1", "x2"],
            dataset_description=None,
        )

        assert result["method_used"] == "trimmed_ipw"
        assert result["standard_error"] == result["effect_se"]
        assert "p_value" in result
        assert result["parameters"]["trim_rule"] == "ps_bounds"


class TestTreeIntegration:
    """Real tree checks -> trimmed_ipw leaf -> real estimator, no LLM/network."""

    def test_positivity_failure_routes_to_trimmed_ipw_and_estimates(self):
        from cais.components.decision_tree_v2 import select_method_v2

        df = synthetic_overlap_violation()
        result = select_method_v2(
            {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "treatment_variable_type": "binary",
                "covariates": ["x1", "x2"],
                "instrument_variable": None,
                "mediator_variable": None,
                "running_variable": None,
                "cutoff_value": None,
                "time_variable": None,
                "group_variable": "unit",
                "is_rct": False,
                "has_temporal_structure": False,
            },
            df=df,
            description="Synthetic selection-on-observables dataset.",
        )

        assert result.method == "trimmed_ipw"
        assert result.implemented
        assert not result.ended
        assert result.assumptions["positivity"].passed is False
        assert result.execution["method"] == "trimmed_ipw"
        assert result.execution["options"]["ps_bounds"] == [0.1, 0.9]
        assert "balance_after_weighting" in result.planned_post_checks

        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["x1", "x2"],
            confounders=["x1", "x2"],
        )
        output = psw.TrimmedIPW(
            ps_bounds=result.execution["options"]["ps_bounds"]
        ).estimate_effect(df, variables)

        assert np.isfinite(output["effect_estimate"])
        assert output["diagnostics"]["trimming"]["n_trimmed"] > 0
        assert "overlap-satisfying subpopulation" in output["diagnostics"]["estimand"]

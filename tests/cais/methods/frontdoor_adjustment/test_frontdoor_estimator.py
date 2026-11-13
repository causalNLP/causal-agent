"""Tests for the frontdoor adjustment estimator.

Synthetic data-generating processes put an unmeasured confounder U on the
treatment and outcome while the treatment's entire effect runs through a
discrete mediator, so the standard difference in means is biased but the
frontdoor formula is identifiable.
"""

import numpy as np
import pandas as pd
import pytest

from cais.models import Variables
from cais.methods import METHOD_MAPPING
from cais.methods.frontdoor_adjustment.estimator import (
    FrontdoorAdjustment,
    FrontdoorNotSupported,
    estimate_frontdoor_effect,
)


def synthetic_frontdoor(n=2000, seed=3):
    """Binary treatment, binary mediator, unmeasured T-Y confounding.

    U confounds T and Y; T affects Y only through M. True ATE = 2.0 * 0.6 = 1.2.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, n)
    treatment = (u + rng.normal(0.0, 1.0, n) > 0).astype(int)
    mediator = rng.binomial(1, 0.2 + 0.6 * treatment)
    outcome = 2.0 * mediator + u + rng.normal(0.0, 0.5, n)
    return pd.DataFrame({"u": u, "treatment": treatment, "mediator": mediator,
                         "outcome": outcome})


def synthetic_frontdoor_multilevel(n=3000, seed=11):
    """Three-level treatment; mediator probabilities 0.2 / 0.5 / 0.8.

    True ADRF contrasts vs level 0 are 2*(0.5-0.2)=0.6 and 2*(0.8-0.2)=1.2.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, n)
    latent = u + rng.normal(0.0, 1.0, n)
    treatment = np.digitize(latent, [-0.4, 0.4])
    p_mediator = {0: 0.2, 1: 0.5, 2: 0.8}
    mediator = rng.binomial(1, np.array([p_mediator[t] for t in treatment]))
    outcome = 2.0 * mediator + u + rng.normal(0.0, 0.5, n)
    return pd.DataFrame({"u": u, "treatment": treatment, "mediator": mediator,
                         "outcome": outcome})


def naive_difference(df):
    return df.loc[df.treatment == 1, "outcome"].mean() - df.loc[df.treatment == 0, "outcome"].mean()


class TestEstimator:
    def test_recovers_effect_despite_unmeasured_confounding(self):
        df = synthetic_frontdoor()

        result = estimate_frontdoor_effect(
            df, "treatment", "outcome", "mediator", n_bootstraps=100, seed=0
        )

        assert abs(result["effect_estimate"] - 1.2) < 0.15
        assert abs(naive_difference(df) - 1.2) > 0.4  # the naive estimate is biased
        assert result["standard_error"] > 0
        assert result["confidence_interval"][0] < result["effect_estimate"]
        assert result["confidence_interval"][1] > result["effect_estimate"]
        assert result["p_value"] < 0.05
        assert result["method_used"] == "frontdoor_adjustment"
        adrf = result["diagnostics"]["adrf"]
        assert set(adrf) == {"0", "1"}
        assert abs(adrf["1"] - adrf["0"] - 1.2) < 0.2

    def test_multi_level_treatment_reports_adrf_and_weighted_contrast(self):
        df = synthetic_frontdoor_multilevel()

        result = estimate_frontdoor_effect(
            df, "treatment", "outcome", "mediator", n_bootstraps=60, seed=0
        )

        diagnostics = result["diagnostics"]
        assert set(diagnostics["adrf"]) == {"0", "1", "2"}
        assert abs(diagnostics["contrasts_vs_reference"]["1"] - 0.6) < 0.2
        assert abs(diagnostics["contrasts_vs_reference"]["2"] - 1.2) < 0.2
        # reported scalar is the treatment-distribution-weighted ADRF summary
        assert abs(result["effect_estimate"] - 0.9) < 0.25
        assert "multi-level" in diagnostics["estimand"]

    def test_bootstrap_is_deterministic_with_seed(self):
        df = synthetic_frontdoor(n=800)

        first = estimate_frontdoor_effect(
            df, "treatment", "outcome", "mediator", n_bootstraps=40, seed=7
        )
        second = estimate_frontdoor_effect(
            df, "treatment", "outcome", "mediator", n_bootstraps=40, seed=7
        )

        assert first["effect_estimate"] == second["effect_estimate"]
        assert first["standard_error"] == second["standard_error"]
        assert first["confidence_interval"] == second["confidence_interval"]

    def test_refuses_continuous_mediator(self):
        df = synthetic_frontdoor(n=500)
        df["mediator"] = np.random.default_rng(0).normal(0.0, 1.0, len(df))

        with pytest.raises(FrontdoorNotSupported, match="discrete mediator"):
            estimate_frontdoor_effect(
                df, "treatment", "outcome", "mediator", n_bootstraps=10
            )

    def test_refuses_continuous_treatment(self):
        df = synthetic_frontdoor(n=500)
        df["treatment"] = np.random.default_rng(0).normal(0.0, 1.0, len(df))

        with pytest.raises(FrontdoorNotSupported, match="discrete treatment"):
            estimate_frontdoor_effect(
                df, "treatment", "outcome", "mediator", n_bootstraps=10
            )

    def test_refuses_sparse_cells(self):
        rng = np.random.default_rng(5)
        n = 100
        treatment = rng.integers(0, 2, n)
        mediator = np.where(treatment == 1, 1, 0)  # (T=0, M=1) cell is empty
        outcome = rng.normal(size=n)
        df = pd.DataFrame({"treatment": treatment, "mediator": mediator, "outcome": outcome})

        with pytest.raises(FrontdoorNotSupported, match="sparse"):
            estimate_frontdoor_effect(df, "treatment", "outcome", "mediator", n_bootstraps=10)


class TestWrapper:
    def _variables(self, **overrides):
        values = dict(
            treatment_variable="treatment",
            outcome_variable="outcome",
            mediator_variable="mediator",
            treatment_reference_level=0,
            covariates=[],
            confounders=[],
        )
        values.update(overrides)
        return Variables(**values)

    def test_contract(self):
        df = synthetic_frontdoor(n=1200)

        result = FrontdoorAdjustment().estimate_effect(
            df, self._variables(), n_bootstraps=40, seed=0
        )

        for key in ("effect_estimate", "standard_error", "confidence_interval",
                    "p_value", "method_used", "diagnostics"):
            assert key in result
        assert result["method_used"] == "frontdoor_adjustment"
        assert abs(result["effect_estimate"] - 1.2) < 0.2
        assert "adrf" in result["diagnostics"]

    def test_mediator_can_come_from_execution_option(self):
        df = synthetic_frontdoor(n=1200)
        variables = self._variables()
        variables.mediator_variable = None

        result = FrontdoorAdjustment().estimate_effect(
            df, variables, mediator_variable="mediator", n_bootstraps=30, seed=0
        )

        assert result["effect_estimate"] is not None
        assert result["diagnostics"]["adrf"]

    def test_missing_mediator_returns_error_dict(self):
        df = synthetic_frontdoor(n=200)
        variables = self._variables()
        variables.mediator_variable = None

        result = FrontdoorAdjustment().estimate_effect(df, variables)

        assert result["effect_estimate"] is None
        assert "mediator" in result["error"].lower()

    def test_unsupported_data_returns_error_dict(self):
        df = synthetic_frontdoor(n=500)
        df["mediator"] = np.random.default_rng(0).normal(0.0, 1.0, len(df))

        result = FrontdoorAdjustment().estimate_effect(df, self._variables(), n_bootstraps=10)

        assert result["effect_estimate"] is None
        assert "discrete mediator" in result["diagnostics"]["error"]


class TestRegistration:
    def test_method_mapping_entry(self):
        df = synthetic_frontdoor(n=800)

        result = METHOD_MAPPING["frontdoor_adjustment"](
            df=df,
            treatment="treatment",
            outcome="outcome",
            covariates=[],
            dataset_description=None,
            mediator_variable="mediator",
            n_bootstraps=20,
        )

        assert result["method_used"] == "frontdoor_adjustment"
        assert abs(result["effect_estimate"] - 1.2) < 0.25

    def test_method_mapping_requires_mediator(self):
        df = synthetic_frontdoor(n=200)

        with pytest.raises(FrontdoorNotSupported):
            METHOD_MAPPING["frontdoor_adjustment"](
                df=df, treatment="treatment", outcome="outcome", covariates=[]
            )


class TestTreeIntegration:
    """Real deterministic checks -> frontdoor leaf -> real estimator, no LLM."""

    def test_real_checks_route_to_frontdoor_and_estimate(self):
        from cais.components.decision_tree_v2 import select_method_v2

        df = synthetic_frontdoor(n=1500, seed=21)
        tree = select_method_v2(
            {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "treatment_variable_type": "binary",
                "covariates": [],
                "instrument_variable": None,
                "mediator_variable": "mediator",
                "running_variable": None,
                "cutoff_value": None,
                "time_variable": None,
                "group_variable": "unit",
                "is_rct": False,
                "has_temporal_structure": False,
                "treatment_reference_level": 0,
            },
            df=df,
            description="Synthetic mediated effect with an unmeasured confounder.",
        )

        assert tree.method == "frontdoor_adjustment"
        assert tree.implemented
        assert not tree.ended
        assert tree.assumptions["frontdoor_positivity"].passed is True
        assert tree.execution["method"] == "frontdoor_adjustment"
        assert tree.execution["options"]["mediator_variable"] == "mediator"

        variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            mediator_variable=None,
            treatment_reference_level=None,
            covariates=[],
            confounders=[],
        )
        output = FrontdoorAdjustment().estimate_effect(
            df, variables, n_bootstraps=40, seed=0, **tree.execution["options"]
        )

        assert abs(output["effect_estimate"] - 1.2) < 0.2

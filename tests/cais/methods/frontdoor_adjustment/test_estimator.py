import numpy as np
import pandas as pd
import pytest

from cais.estimator_lib import Estimators
from cais.methods import METHOD_MAPPING
from cais.methods.frontdoor_adjustment.estimator import (
    FrontdoorAdjustment,
    estimate_effect,
)
from cais.models import Variables


def make_frontdoor_data(n=30_000, seed=731460):
    """DGP from the front-door primer: the true risk-difference effect is .25."""
    rng = np.random.default_rng(seed)
    u = rng.binomial(1, 0.2, n)
    x = rng.binomial(1, 0.1 + 0.6 * u)
    m = rng.binomial(1, 0.3 + 0.5 * x)
    y = rng.binomial(1, 0.1 + 0.3 * u + 0.5 * m)
    return pd.DataFrame({"X": x, "M": m, "Y": y})


def test_frontdoor_formula_recovers_simulated_effect():
    result = estimate_effect(
        make_frontdoor_data(),
        treatment="X",
        mediator_variable="M",
        outcome="Y",
        n_bootstrap=120,
        random_state=7,
    )

    assert result["effect_estimate"] == pytest.approx(0.25, abs=0.035)
    assert result["standard_error"] > 0
    low, high = result["confidence_interval"]
    assert low < result["effect_estimate"] < high
    assert result["diagnostics"]["bootstrap_successful"] == 120


def test_frontdoor_pipeline_adapter_and_registry():
    variables = Variables(
        treatment_variable="X",
        mediator_variable="M",
        outcome_variable="Y",
        treatment_reference_level=0,
        frontdoor_criterion=True,
    )
    estimator = FrontdoorAdjustment()
    result = estimator.estimate_effect(make_frontdoor_data(n=15_000), variables)

    assert result["method_used"] == "Front-door Adjustment"
    assert result["effect_estimate"] == pytest.approx(0.25, abs=0.05)
    assert FrontdoorAdjustment.name in Estimators().names()
    assert METHOD_MAPPING["frontdoor_adjustment"] is estimate_effect


def test_frontdoor_rejects_empty_treatment_mediator_cells():
    df = pd.DataFrame(
        {
            "X": [0, 0, 1, 1],
            "M": [0, 0, 1, 1],
            "Y": [0.0, 1.0, 0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="positivity"):
        estimate_effect(
            df,
            treatment="X",
            mediator_variable="M",
            outcome="Y",
            n_bootstrap=0,
        )


def test_frontdoor_requires_a_mediator():
    with pytest.raises(ValueError, match="mediator"):
        estimate_effect(
            make_frontdoor_data(n=100),
            treatment="X",
            outcome="Y",
        )

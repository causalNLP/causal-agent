import numpy as np
import pandas as pd
import pytest

from cais.methods import METHOD_MAPPING
from cais.methods.fuzzy_regression_discontinuity.estimator import estimate_effect


def _fuzzy_rdd_data(seed=42, n=5000, effect=3.0, first_stage=0.65):
    rng = np.random.default_rng(seed)
    running = rng.uniform(-1.0, 1.0, n)
    eligibility = (running >= 0).astype(float)
    treatment_probability = 0.15 + first_stage * eligibility
    treatment = rng.binomial(1, treatment_probability)
    outcome = effect * treatment + 0.5 * running + rng.normal(0, 0.6, n)
    return pd.DataFrame({"running": running, "treatment": treatment, "outcome": outcome})


def test_fuzzy_rdd_recovers_local_wald_effect_and_is_registered():
    result = estimate_effect(
        _fuzzy_rdd_data(),
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff_value=0.0,
        bandwidth=0.5,
    )

    assert result["effect_estimate"] == pytest.approx(3.0, abs=0.25)
    assert result["standard_error"] > 0
    assert result["confidence_interval"][0] < 3.0 < result["confidence_interval"][1]
    assert result["diagnostics"]["first_stage_jump"] > 0.5
    assert result["diagnostics"]["first_stage_f_statistic"] > 10
    assert METHOD_MAPPING["fuzzy_regression_discontinuity_design"] is estimate_effect


def test_fuzzy_rdd_supports_treatment_below_cutoff():
    df = _fuzzy_rdd_data(seed=7)
    df["running"] *= -1

    result = estimate_effect(
        df,
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff_value=0.0,
        treat_above_cutoff=False,
        bandwidth=0.5,
    )

    assert result["effect_estimate"] == pytest.approx(3.0, abs=0.25)
    assert result["diagnostics"]["treat_above_cutoff"] is False


def test_fuzzy_rdd_rejects_weak_first_stage():
    df = _fuzzy_rdd_data(seed=9, first_stage=0.0)

    with pytest.raises(ValueError, match="first stage|coding"):
        estimate_effect(
            df,
            treatment="treatment",
            outcome="outcome",
            running_variable="running",
            cutoff_value=0.0,
            bandwidth=0.5,
        )


def test_fuzzy_rdd_rejects_nonbinary_treatment():
    df = _fuzzy_rdd_data(seed=11)
    df["treatment"] = df["treatment"].astype(float)
    df.loc[df.index[0], "treatment"] = 0.5

    with pytest.raises(ValueError, match="coded 0/1"):
        estimate_effect(
            df,
            treatment="treatment",
            outcome="outcome",
            running_variable="running",
            cutoff_value=0.0,
            bandwidth=0.5,
        )


def test_fuzzy_rdd_rejects_invalid_bandwidth():
    with pytest.raises(ValueError, match="bandwidth"):
        estimate_effect(
            _fuzzy_rdd_data(seed=13),
            treatment="treatment",
            outcome="outcome",
            running_variable="running",
            cutoff_value=0.0,
            bandwidth=0.0,
        )

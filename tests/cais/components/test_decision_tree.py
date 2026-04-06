import pytest

from cais.components.decision_tree import (
    select_method,
    METHOD_ASSUMPTIONS,
    LINEAR_REGRESSION,
    DIFF_IN_MEANS,
    DIFF_IN_DIFF,
    REGRESSION_DISCONTINUITY,
    PROPENSITY_SCORE_MATCHING,
    INSTRUMENTAL_VARIABLE,
)


def _make_props(**overrides):
    """Build a minimal dataset_properties dict for select_method."""
    base = {
        "treatment_variable": "T",
        "outcome_variable": "Y",
        "covariates": [],
        "is_rct": False,
        "instrument_variable": None,
        "running_variable": None,
        "cutoff_value": None,
        "time_variable": None,
        "has_temporal_structure": False,
        "treatment_variable_type": "binary",
        "frontdoor_criterion": False,
        "covariate_overlap_score": None,
    }
    base.update(overrides)
    return base


def test_no_covariates():
    """Observational, no covariates, no special design -> LINEAR_REGRESSION fallback."""
    props = _make_props(covariates=[])
    result = select_method(props)
    assert result["selected_method"] == LINEAR_REGRESSION
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[LINEAR_REGRESSION]


def test_rct_no_covariates():
    """RCT without covariates -> DIFF_IN_MEANS."""
    props = _make_props(is_rct=True, covariates=[])
    result = select_method(props)
    assert result["selected_method"] == DIFF_IN_MEANS
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[DIFF_IN_MEANS]


def test_rct_with_covariates():
    """RCT with covariates -> LINEAR_REGRESSION."""
    props = _make_props(is_rct=True, covariates=["X1", "X2"])
    result = select_method(props)
    assert result["selected_method"] == LINEAR_REGRESSION
    assert "rct" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[LINEAR_REGRESSION]


def test_observational_temporal():
    """Observational data with temporal structure -> DIFF_IN_DIFF."""
    props = _make_props(
        covariates=["X1"],
        has_temporal_structure=True,
        time_variable="year",
    )
    result = select_method(props)
    assert result["selected_method"] == DIFF_IN_DIFF
    assert "temporal" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[DIFF_IN_DIFF]


def test_observational_rdd():
    """Observational data with running variable and cutoff -> REGRESSION_DISCONTINUITY."""
    props = _make_props(running_variable="score", cutoff_value=50)
    result = select_method(props)
    assert result["selected_method"] == REGRESSION_DISCONTINUITY
    assert "running variable" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[REGRESSION_DISCONTINUITY]


def test_observational_iv():
    """Observational data with instrument -> INSTRUMENTAL_VARIABLE."""
    props = _make_props(instrument_variable="Z")
    result = select_method(props)
    assert result["selected_method"] == INSTRUMENTAL_VARIABLE
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE]


def test_observational_confounders_default_psm():
    """Observational data with covariates but no special design -> PROPENSITY_SCORE_MATCHING."""
    props = _make_props(covariates=["X1", "X2"])
    result = select_method(props)
    assert result["selected_method"] == PROPENSITY_SCORE_MATCHING
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[PROPENSITY_SCORE_MATCHING]


if __name__ == "__main__":
    pytest.main()

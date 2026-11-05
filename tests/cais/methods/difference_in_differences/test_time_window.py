"""Tests for the time-window DiD estimator."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cais.estimator_lib import Estimators
from cais.methods import METHOD_MAPPING
from cais.methods.difference_in_differences.time_window import (
    DiDTimeWindow,
    estimate_effect_time_window,
)


def _panel(n_units: int = 40, n_periods: int = 8, treatment_start: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        group = unit % 2
        unit_effect = rng.normal(0, 1)
        for year in range(n_periods):
            y = (
                2.0
                + 0.5 * year
                + group
                + 5.0 * group * (year >= treatment_start)
                + unit_effect
                + rng.normal(0, 0.5)
            )
            rows.append(
                {"unit": unit, "year": year, "group": group, "y": y}
            )
    return pd.DataFrame(rows)


def _kwargs(**overrides):
    values = dict(
        time_variable="year",
        group_variable="unit",
        treatment_period_start=5,
        anticipation_window_start=3,
    )
    values.update(overrides)
    return values


def test_time_window_did_removes_contaminated_rows():
    df = _panel()
    result = estimate_effect_time_window(
        df,
        treatment="group",
        outcome="y",
        covariates=[],
        **_kwargs(),
    )
    removed_years = 2  # years 3 and 4
    assert result["method_used"] == "difference_in_differences_time_window"
    assert result["n_rows_removed"] == 40 * removed_years
    assert result["anticipation_window_start"] == 3
    assert result["treatment_period_start"] == 5
    assert result["time_window_adjustment"] is True
    assert abs(result["effect_estimate"] - 5.0) < 1.0
    assert "anticipation_window_start" in result["diagnostics"]
    assert "dropped" in result["method_details"].lower()


def test_missing_window_raises():
    with pytest.raises(ValueError, match="anticipation_window_start"):
        estimate_effect_time_window(
            _panel(),
            treatment="group",
            outcome="y",
            covariates=[],
            **_kwargs(anticipation_window_start=None),
        )


def test_window_after_treatment_start_raises():
    with pytest.raises(ValueError, match="strictly before"):
        estimate_effect_time_window(
            _panel(),
            treatment="group",
            outcome="y",
            covariates=[],
            **_kwargs(anticipation_window_start=5),
        )


def test_binary_time_indicator_raises():
    df = _panel(n_periods=2)
    with pytest.raises(ValueError, match="three distinct time periods"):
        estimate_effect_time_window(
            df,
            treatment="group",
            outcome="y",
            covariates=[],
            **_kwargs(treatment_period_start=1, anticipation_window_start=0),
        )


def test_non_binary_treatment_raises():
    df = _panel()
    df["group"] = df["group"] * 2
    with pytest.raises(ValueError, match="binary treatment"):
        estimate_effect_time_window(
            df,
            treatment="group",
            outcome="y",
            covariates=[],
            **_kwargs(),
        )


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="not a column"):
        estimate_effect_time_window(
            _panel(),
            treatment="group",
            outcome="y",
            covariates=[],
            **_kwargs(time_variable="missing"),
        )


def test_class_wrapper_uses_variables_and_options():
    df = _panel()
    variables = SimpleNamespace(
        treatment_variable="group",
        outcome_variable="y",
        time_variable="year",
        group_variable="unit",
        confounders=[],
        covariates=[],
        treatment_time=5,
    )
    result = DiDTimeWindow().estimate_effect(
        df,
        variables,
        anticipation_window_start=3,
    )
    assert result["method_used"] == "difference_in_differences_time_window"
    assert result["n_rows_removed"] == 80


def test_class_wrapper_returns_error_without_window():
    variables = SimpleNamespace(
        treatment_variable="group",
        outcome_variable="y",
        time_variable="year",
        group_variable="unit",
        confounders=[],
        covariates=[],
        treatment_time=5,
    )
    result = DiDTimeWindow().estimate_effect(_panel(), variables)
    assert result["effect_estimate"] is None
    assert "anticipation_window_start" in result["error"]


def test_registered_in_mapping_and_estimator_lib():
    assert "difference_in_differences_time_window" in METHOD_MAPPING
    assert DiDTimeWindow.name in Estimators().estimators

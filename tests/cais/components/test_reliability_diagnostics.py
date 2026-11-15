import pandas as pd

from cais.components.reliability_diagnostics import (
    assess_binary_treatment_reliability,
    effective_sample_size,
)


def test_effective_sample_size_is_smaller_for_concentrated_weights():
    assert effective_sample_size([1, 1, 1, 1]) == 4.0
    assert effective_sample_size([1, 1, 1, 10]) < 4.0


def test_binary_reliability_reports_support_and_missingness():
    df = pd.DataFrame({"t": [0, 1, 0, 1], "x": [1.0, 2.0, None, 4.0]})
    result = assess_binary_treatment_reliability(df, "t", required_columns=["t", "x"])
    assert result["status"] == "supported"
    assert result["treated_n"] == 2
    assert result["control_n"] == 2
    assert result["missing_required_share"] == 0.25
    assert 0.0 <= result["reliability_score"] <= 1.0

import pandas as pd

from cais.components.sensitivity_analysis import run_sensitivity_analysis


def test_bootstrap_sensitivity_is_advisory_and_returns_interval():
    df = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1], "y": [0.0, 2.0, 1.0, 3.0, 0.5, 2.5]})

    def estimator(*, df, variables, query=None, **kwargs):
        return {"effect_estimate": float(df.loc[df.t == 1, "y"].mean() - df.loc[df.t == 0, "y"].mean())}

    result = run_sensitivity_analysis(
        df=df,
        estimator=estimator,
        variables={"treatment_variable": "t", "outcome_variable": "y"},
        query="Does t cause y?",
        base_results={"effect_estimate": 2.0},
        n_bootstrap=10,
    )
    assert result["advisory"] is True
    assert result["status"] == "supported"
    assert result["successful_resamples"] >= 5
    assert len(result["bootstrap_interval_95"]) == 2


def test_unavailable_fallback_sensitivity_does_not_block():
    result = run_sensitivity_analysis(
        df=pd.DataFrame({"y": [1, 2]}),
        estimator=None,
        variables={},
        query=None,
        base_results={"effect_estimate": 1.0},
    )
    assert result["status"] == "not_run"
    assert result["advisory"] is True

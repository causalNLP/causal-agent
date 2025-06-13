import pytest
import pandas as pd
import numpy as np
from auto_causal.methods.instrumental_variable.diagnostics import (
    calculate_first_stage_f_statistic,
    run_overidentification_test
)
from statsmodels.sandbox.regression.gmm import IV2SLS

# Fixture for basic IV data
@pytest.fixture
def iv_data():
    np.random.seed(42)
    n = 1000
    Z1 = np.random.normal(0, 1, n) # Instrument 1
    Z2 = np.random.normal(0, 1, n) # Instrument 2 (for over-ID test)
    W = np.random.normal(0, 1, n)  # Exogenous Covariate
    U = np.random.normal(0, 1, n)  # Unobserved Confounder

    # Strong Instrument Case
    T_strong = 0.5 * Z1 + 0.5 * W + 0.5 * U + np.random.normal(0, 1, n)
    Y_strong = 2.0 * T_strong + 1.0 * W + 1.0 * U + np.random.normal(0, 1, n)

    # Weak Instrument Case
    T_weak = 0.05 * Z1 + 0.5 * W + 0.5 * U + np.random.normal(0, 1, n)
    Y_weak = 2.0 * T_weak + 1.0 * W + 1.0 * U + np.random.normal(0, 1, n)

    df_strong = pd.DataFrame({'Y': Y_strong, 'T': T_strong, 'Z1': Z1, 'Z2': Z2, 'W': W, 'U': U})
    df_weak = pd.DataFrame({'Y': Y_weak, 'T': T_weak, 'Z1': Z1, 'Z2': Z2, 'W': W, 'U': U})

    return df_strong, df_weak


def test_calculate_first_stage_f_statistic_strong(iv_data):
    df_strong, _ = iv_data
    f_stat, p_val = calculate_first_stage_f_statistic(
        df=df_strong, treatment='T', instruments=['Z1'], covariates=['W']
    )
    assert f_stat is not None
    assert p_val is not None
    assert f_stat > 10 # Expect strong instrument
    assert p_val < 0.01 # Expect significance

def test_calculate_first_stage_f_statistic_weak(iv_data):
    _, df_weak = iv_data
    f_stat, p_val = calculate_first_stage_f_statistic(
        df=df_weak, treatment='T', instruments=['Z1'], covariates=['W']
    )
    assert f_stat is not None
    assert p_val is not None
    # Note: With random noise, weak instrument test might occasionally pass 10, but should be low
    assert f_stat < 15 # Check it's not extremely high
    # P-value might still be significant if sample size is large

def test_calculate_first_stage_f_statistic_no_instruments(caplog):
    """Test graceful handling when no instruments are provided."""
    df = pd.DataFrame({'T': [1, 2], 'W': [3, 4]})
    # Should now return (None, None) and log a warning, not raise Exception
    # with pytest.raises(Exception): # OLD assertion
    #      calculate_first_stage_f_statistic(
    #         df=df, treatment='T', instruments=[], covariates=['W']
    #      )
    f_stat, p_val = calculate_first_stage_f_statistic(
        df=df, treatment='T', instruments=[], covariates=['W']
    )
    assert f_stat is None
    assert p_val is None
    assert "No instruments provided" in caplog.text # Check log message


def test_run_overidentification_test_applicable(iv_data):
    df_strong, _ = iv_data
    # Need to run statsmodels IV first to get results object
    df_copy = df_strong.copy()
    df_copy['intercept'] = 1
    endog = df_copy['Y']
    exog_vars = ['intercept', 'W', 'T']
    instrument_vars = ['intercept', 'W', 'Z1', 'Z2'] # Z1, Z2 are instruments

    iv_model = IV2SLS(endog=endog, exog=df_copy[exog_vars], instrument=df_copy[instrument_vars])
    sm_results = iv_model.fit()

    stat, p_val, status = run_overidentification_test(
        sm_results=sm_results,
        df=df_strong,
        treatment='T',
        outcome='Y',
        instruments=['Z1', 'Z2'],
        covariates=['W']
    )

    assert "Test successful" in status
    assert stat is not None
    assert p_val is not None
    assert stat >= 0
    # In this correctly specified model, we expect the test to NOT reject H0 (p > 0.05)
    assert p_val > 0.05

def test_run_overidentification_test_not_applicable(iv_data):
    df_strong, _ = iv_data
    # Only one instrument
    stat, p_val, status = run_overidentification_test(
        sm_results=None, # Not needed if not applicable
        df=df_strong,
        treatment='T',
        outcome='Y',
        instruments=['Z1'],
        covariates=['W']
    )
    assert stat is None
    assert p_val is None
    assert "not applicable" in status.lower()

def test_run_overidentification_test_no_sm_results(iv_data):
    df_strong, _ = iv_data
    # More than one instrument, but no sm_results provided
    stat, p_val, status = run_overidentification_test(
        sm_results=None,
        df=df_strong,
        treatment='T',
        outcome='Y',
        instruments=['Z1', 'Z2'],
        covariates=['W']
    )
    assert stat is None
    assert p_val is None
    assert "object not available" in status.lower() 
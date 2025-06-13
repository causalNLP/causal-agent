import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from auto_causal.methods.linear_regression.diagnostics import run_lr_diagnostics

# Reuse the sample data fixture from estimator tests
@pytest.fixture
def sample_data():
    """Generates simple synthetic data for testing LR."""
    np.random.seed(42)
    n_samples = 100
    treatment_effect = 2.0
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(5, 2, n_samples)
    treatment = np.random.binomial(1, 0.5, n_samples)
    error = np.random.normal(0, 1, n_samples)
    outcome = 1.0 + treatment_effect * treatment + 0.5 * X1 - 1.5 * X2 + error
    
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'covariate1': X1,
        'covariate2': X2
    })
    return df

def test_run_lr_diagnostics_implementation(sample_data):
    """Tests the implemented diagnostics function with real results."""
    # Run a regression to get a real results object
    df_analysis = sample_data.dropna()
    covariates = ['covariate1', 'covariate2']
    X = df_analysis[['treatment'] + covariates]
    X = sm.add_constant(X)
    y = df_analysis['outcome']
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Run diagnostics
    diagnostics = run_lr_diagnostics(results, X) 
    
    assert isinstance(diagnostics, dict)
    assert diagnostics["status"] == "Success"
    assert "details" in diagnostics
    details = diagnostics["details"]
    
    # Check for key diagnostic metrics
    assert "r_squared" in details
    assert "adj_r_squared" in details
    assert "f_statistic" in details
    assert "f_p_value" in details
    assert "n_observations" in details
    assert "degrees_of_freedom_resid" in details
    
    # Check normality test results
    assert "residuals_normality_jb_stat" in details
    assert "residuals_normality_jb_p_value" in details
    assert "residuals_skewness" in details
    assert "residuals_kurtosis" in details
    assert "residuals_normality_status" in details
    assert isinstance(details["residuals_normality_status"], str)

    # Check homoscedasticity test results
    assert "homoscedasticity_bp_lm_stat" in details
    assert "homoscedasticity_bp_lm_p_value" in details
    assert "homoscedasticity_bp_f_stat" in details
    assert "homoscedasticity_bp_f_p_value" in details
    assert "homoscedasticity_status" in details
    assert isinstance(details["homoscedasticity_status"], str)
    
    # Check placeholder statuses
    assert "linearity_check" in details
    assert "multicollinearity_check" in details
    assert details["linearity_check"] == "Requires visual inspection (e.g., residual vs fitted plot)"
    assert details["multicollinearity_check"] == "Not Implemented (Requires VIF)"
    
    # Check types (basic)
    assert isinstance(details["r_squared"], float)
    assert isinstance(details["f_p_value"], float)
    assert isinstance(details["n_observations"], int)

def test_run_lr_diagnostics_failure():
    """Test diagnostic failure mode (e.g., passing wrong object)."""
    # Pass a non-results object
    diagnostics = run_lr_diagnostics("not a results object", pd.DataFrame({'const': [1]}))
    assert diagnostics["status"] == "Failed"
    assert "error" in diagnostics


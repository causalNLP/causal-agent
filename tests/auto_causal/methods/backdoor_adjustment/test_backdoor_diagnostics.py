import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from auto_causal.methods.backdoor_adjustment.diagnostics import run_backdoor_diagnostics

# --- Fixture for confounded data ---
@pytest.fixture
def sample_confounded_data():
    """Generates synthetic data with confounding for testing diagnostics."""
    np.random.seed(789)
    n_samples = 200
    W1 = np.random.normal(0, 1, n_samples)
    W2 = np.random.normal(2, 1, n_samples)
    treatment_prob = 1 / (1 + np.exp(-(0.5 * W1 - 0.5)))
    treatment = np.random.binomial(1, treatment_prob, n_samples)
    true_effect = 3.0
    error = np.random.normal(0, 1, n_samples)
    outcome = 10 + true_effect * treatment + 2.0 * W1 - 1.0 * W2 + error
    
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'confounder1': W1,
        'confounder2': W2
    })
    return df

def test_run_backdoor_diagnostics_success(sample_confounded_data):
    """Tests the diagnostics function with real results."""
    # Run a regression to get a real results object
    df_analysis = sample_confounded_data.dropna()
    treatment = 'treatment'
    covariates = ['confounder1', 'confounder2']
    X = df_analysis[[treatment] + covariates]
    X = sm.add_constant(X)
    y = df_analysis['outcome']
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Run diagnostics
    diagnostics = run_backdoor_diagnostics(results, X) 
    
    assert isinstance(diagnostics, dict)
    assert diagnostics["status"] == "Success"
    assert "details" in diagnostics
    details = diagnostics["details"]
    
    # Check for key OLS diagnostic metrics
    assert "r_squared" in details
    assert "adj_r_squared" in details
    assert "f_statistic" in details
    assert "f_p_value" in details
    assert "n_observations" in details
    assert "degrees_of_freedom_resid" in details
    assert "durbin_watson" in details
    
    # Check normality test results
    assert "residuals_normality_jb_stat" in details
    assert "residuals_normality_jb_p_value" in details
    assert "residuals_skewness" in details
    assert "residuals_kurtosis" in details
    assert "residuals_normality_status" in details

    # Check homoscedasticity test results
    assert "homoscedasticity_bp_lm_stat" in details
    assert "homoscedasticity_bp_lm_p_value" in details
    assert "homoscedasticity_status" in details
    
    # Check multicollinearity proxy
    assert "model_condition_number" in details
    assert "multicollinearity_status" in details
    
    # Check placeholder status
    assert "linearity_check" in details
    assert details["linearity_check"] == "Requires visual inspection (e.g., residual vs fitted plot)"
    
    # Check types (basic)
    assert isinstance(details["r_squared"], float)
    assert isinstance(details["f_p_value"], float)
    assert isinstance(details["n_observations"], int)

def test_run_backdoor_diagnostics_failure():
    """Test diagnostic failure mode (e.g., passing wrong object)."""
    # Pass a non-results object
    # Need a dummy X with matching columns expected by the function if it gets that far
    dummy_X = pd.DataFrame({'const': [1], 'treatment': [0], 'cov1': [1]}) 
    diagnostics = run_backdoor_diagnostics("not a results object", dummy_X)
    assert diagnostics["status"] == "Failed"
    assert "error" in diagnostics

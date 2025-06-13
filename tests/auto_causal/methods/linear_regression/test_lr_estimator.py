import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels import iolib
from causalscientist.auto_causal.methods.linear_regression.estimator import estimate_effect

# --- Fixtures ---

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
        'covariate2': X2,
        'other_col': np.random.rand(n_samples) # Unused column
    })
    return df

# --- Test Cases ---

def test_estimate_effect_no_covariates(sample_data):
    """Test estimating effect without covariates."""
    results = estimate_effect(sample_data, 'treatment', 'outcome')
    
    assert 'effect_estimate' in results
    assert 'p_value' in results
    assert 'confidence_interval' in results
    assert 'standard_error' in results
    assert 'formula' in results
    assert 'model_summary' in results
    assert 'diagnostics' in results # Placeholder check
    assert 'interpretation' in results # Placeholder check
    assert 'method_used' in results
    
    # Check if effect estimate is reasonably close (simple check)
    assert abs(results['effect_estimate'] - 2.0) < 0.5 
    assert 'treatment' in results['formula']
    assert 'covariate1' not in results['formula']
    assert results['method_used'] == 'Linear Regression (OLS)'

def test_estimate_effect_with_covariates(sample_data):
    """Test estimating effect with covariates."""
    covariates = ['covariate1', 'covariate2']
    results = estimate_effect(sample_data, 'treatment', 'outcome', covariates)
    
    assert 'effect_estimate' in results
    assert 'p_value' in results
    assert 'confidence_interval' in results
    assert 'standard_error' in results
    
    # Check if effect estimate is reasonably close to the true effect (2.0)
    assert abs(results['effect_estimate'] - 2.0) < 0.5 
    assert 'treatment' in results['formula']
    assert 'covariate1' in results['formula']
    assert 'covariate2' in results['formula']
    assert results['method_used'] == 'Linear Regression (OLS)'
    # Check summary type (basic check)
    assert isinstance(results['model_summary'], sm.iolib.summary.Summary)

def test_estimate_effect_missing_treatment(sample_data):
    """Test error handling for missing treatment column."""
    with pytest.raises(ValueError, match="Missing required columns:.*missing_treat.*"):
        estimate_effect(sample_data, 'missing_treat', 'outcome')

def test_estimate_effect_missing_outcome(sample_data):
    """Test error handling for missing outcome column."""
    with pytest.raises(ValueError, match="Missing required columns:.*missing_outcome.*"):
        estimate_effect(sample_data, 'treatment', 'missing_outcome')

def test_estimate_effect_missing_covariate(sample_data):
    """Test error handling for missing covariate column."""
    with pytest.raises(ValueError, match="Missing required columns:.*missing_cov.*"):
        estimate_effect(sample_data, 'treatment', 'outcome', ['covariate1', 'missing_cov'])

def test_estimate_effect_nan_data():
    """Test handling of data with NaNs resulting in empty analysis set."""
    df_nan = pd.DataFrame({
        'outcome': [1, np.nan, 3],
        'treatment': [0, np.nan, 1],
        'covariate1': [np.nan, 6, 7] # Add NaN here to ensure row 0 is dropped
    })
    # With this setup, row 0 has NaN in covariate1
    # Row 1 has NaN in outcome and treatment
    # Only row 2 is complete, but dropna() needs *all* specified cols
    # to be non-NA. Let's ensure dropna removes all rows.
    df_nan_all_removed = pd.DataFrame({
        'outcome': [1, np.nan, 3],
        'treatment': [0, 1, np.nan],
        'covariate1': [np.nan, 6, 7]
    })
    with pytest.raises(ValueError, match="No data remaining after dropping NaNs"):
        estimate_effect(df_nan_all_removed, 'treatment', 'outcome', ['covariate1'])

def test_formula_generation(sample_data):
    """Test the formula string generation."""
    # No covariates
    results_no_cov = estimate_effect(sample_data, 'treatment', 'outcome')
    assert results_no_cov['formula'] == "outcome ~ treatment + const"
    
    # With covariates
    results_with_cov = estimate_effect(sample_data, 'treatment', 'outcome', ['covariate1', 'covariate2'])
    assert results_with_cov['formula'] == "outcome ~ treatment + covariate1 + covariate2 + const"


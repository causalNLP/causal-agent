import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from unittest.mock import patch, MagicMock

from auto_causal.methods.diff_in_means.estimator import estimate_effect

# --- Fixtures ---

@pytest.fixture
def sample_rct_data():
    """Generates simple synthetic RCT data."""
    np.random.seed(456)
    n_samples = 150
    treatment_effect = 5.0
    treatment = np.random.binomial(1, 0.5, n_samples)
    error = np.random.normal(0, 2, n_samples)
    # Simple outcome model: baseline + treatment effect + noise
    outcome = 20.0 + treatment_effect * treatment + error
    
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'ignored_covariate': np.random.rand(n_samples)
    })
    return df

# --- Test Cases ---

@patch('auto_causal.methods.diff_in_means.estimator.run_dim_diagnostics')
@patch('auto_causal.methods.diff_in_means.estimator.interpret_dim_results')
def test_estimate_effect_basic(mock_interpret, mock_diagnostics, sample_rct_data):
    """Test basic execution and output structure."""
    mock_diagnostics.return_value = {"status": "Success", "details": {'control_group_stats': {}, 'treated_group_stats': {}}}
    mock_interpret.return_value = "LLM Interpretation"
    
    results = estimate_effect(sample_rct_data, 'treatment', 'outcome')
    
    assert 'effect_estimate' in results
    assert 'p_value' in results
    assert 'confidence_interval' in results
    assert 'standard_error' in results
    assert 'formula' in results
    assert 'model_summary' in results
    assert 'diagnostics' in results
    assert 'interpretation' in results
    assert 'method_used' in results
    
    # Check if effect estimate is reasonably close to the true effect (5.0)
    assert abs(results['effect_estimate'] - 5.0) < 1.0 # Allow some margin
    assert results['formula'] == "outcome ~ treatment + const"
    assert results['method_used'] == 'Difference in Means (OLS)'
    assert isinstance(results['model_summary'], sm.iolib.summary.Summary) 
    
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()

def test_estimate_effect_ignores_kwargs(sample_rct_data):
    """Test that extra kwargs (like covariates) are ignored."""
    # Should run without error and produce same results as basic test
    with patch('auto_causal.methods.diff_in_means.estimator.run_dim_diagnostics') as mock_diag, \
         patch('auto_causal.methods.diff_in_means.estimator.interpret_dim_results') as mock_interp:
        results = estimate_effect(sample_rct_data, 'treatment', 'outcome', covariates=['ignored_covariate'])

    assert results['formula'] == "outcome ~ treatment + const"
    assert abs(results['effect_estimate'] - 5.0) < 1.0 
    mock_diag.assert_called_once()
    mock_interp.assert_called_once()
    
def test_estimate_effect_missing_treatment(sample_rct_data):
    """Test error handling for missing treatment column."""
    with pytest.raises(ValueError, match="Missing required columns:.*missing_treat.*"):
        estimate_effect(sample_rct_data, 'missing_treat', 'outcome')

def test_estimate_effect_missing_outcome(sample_rct_data):
    """Test error handling for missing outcome column."""
    with pytest.raises(ValueError, match="Missing required columns:.*missing_outcome.*"):
        estimate_effect(sample_rct_data, 'treatment', 'missing_outcome')

def test_estimate_effect_non_binary_treatment(sample_rct_data):
    """Test warning for non-binary treatment column."""
    df_non_binary = sample_rct_data.copy()
    df_non_binary['treatment'] = np.random.randint(0, 3, size=len(df_non_binary)) # 0, 1, 2
    
    with pytest.warns(UserWarning, match="Treatment column 'treatment' contains values other than 0 and 1"):
        # We still expect it to run the OLS under the hood
         with patch('auto_causal.methods.diff_in_means.estimator.run_dim_diagnostics'), \
              patch('auto_causal.methods.diff_in_means.estimator.interpret_dim_results'):
            results = estimate_effect(df_non_binary, 'treatment', 'outcome')
            assert 'effect_estimate' in results # Check it still produced output

def test_estimate_effect_nan_data():
    """Test handling of data with NaNs resulting in empty analysis set."""
    df_nan = pd.DataFrame({
        'outcome': [np.nan, 2, np.nan], # Ensure all rows have NaN in required cols
        'treatment': [0, np.nan, 1],
    })
    with pytest.raises(ValueError, match="No data remaining after dropping NaNs"):
        estimate_effect(df_nan, 'treatment', 'outcome')

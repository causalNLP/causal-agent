import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from unittest.mock import patch, MagicMock

from causalscientist.auto_causal.methods.backdoor_adjustment.estimator import estimate_effect

# --- Fixtures ---

@pytest.fixture
def sample_confounded_data():
    """Generates synthetic data with confounding."""
    np.random.seed(789)
    n_samples = 200
    # Confounder affects both treatment and outcome
    W1 = np.random.normal(0, 1, n_samples)
    W2 = np.random.normal(2, 1, n_samples)
    # Treatment depends on confounder W1
    treatment_prob = 1 / (1 + np.exp(-(0.5 * W1 - 0.5)))
    treatment = np.random.binomial(1, treatment_prob, n_samples)
    # Outcome depends on treatment and confounders W1, W2
    true_effect = 3.0
    error = np.random.normal(0, 1, n_samples)
    outcome = 10 + true_effect * treatment + 2.0 * W1 - 1.0 * W2 + error
    
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'confounder1': W1,
        'confounder2': W2,
        'irrelevant_var': np.random.rand(n_samples) # Not a confounder
    })
    return df

# --- Test Cases ---

@patch('causalscientist.auto_causal.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics')
@patch('causalscientist.auto_causal.methods.backdoor_adjustment.estimator.interpret_backdoor_results')
def test_estimate_effect_basic(mock_interpret, mock_diagnostics, sample_confounded_data):
    """Test basic execution with a valid adjustment set."""
    mock_diagnostics.return_value = {"status": "Success", "details": {}}
    mock_interpret.return_value = "LLM Interpretation"
    adjustment_set = ['confounder1', 'confounder2']
    
    results = estimate_effect(sample_confounded_data, 'treatment', 'outcome', adjustment_set)
    
    assert 'effect_estimate' in results
    assert 'p_value' in results
    assert 'confidence_interval' in results
    assert 'standard_error' in results
    assert 'formula' in results
    assert 'model_summary' in results
    assert 'diagnostics' in results
    assert 'interpretation' in results
    assert 'method_used' in results
    
    # Check if effect estimate is reasonably close to the true effect (3.0)
    assert abs(results['effect_estimate'] - 3.0) < 1.0 
    assert "outcome ~ treatment + confounder1 + confounder2 + const" in results['formula']
    assert results['method_used'] == 'Backdoor Adjustment (OLS)'
    assert isinstance(results['model_summary'], sm.iolib.summary.Summary) 
    
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()

def test_estimate_effect_missing_treatment(sample_confounded_data):
    """Test error handling for missing treatment column."""
    with pytest.raises(ValueError, match="Missing required columns.*:.*missing_treat"):
        estimate_effect(sample_confounded_data, 'missing_treat', 'outcome', ['confounder1'])

def test_estimate_effect_missing_outcome(sample_confounded_data):
    """Test error handling for missing outcome column."""
    with pytest.raises(ValueError, match="Missing required columns.*:.*missing_outcome"):
        estimate_effect(sample_confounded_data, 'treatment', 'missing_outcome', ['confounder1'])

def test_estimate_effect_missing_covariate(sample_confounded_data):
    """Test error handling for missing covariate column in adjustment set."""
    with pytest.raises(ValueError, match="Missing required columns.*:.*missing_cov"):
        estimate_effect(sample_confounded_data, 'treatment', 'outcome', ['confounder1', 'missing_cov'])

def test_estimate_effect_empty_covariates(sample_confounded_data):
    """Test error handling when covariate list is empty."""
    with pytest.raises(ValueError, match="Backdoor Adjustment requires a non-empty list of covariates"):
        estimate_effect(sample_confounded_data, 'treatment', 'outcome', [])
    with pytest.raises(ValueError, match="Backdoor Adjustment requires a non-empty list of covariates"):
        estimate_effect(sample_confounded_data, 'treatment', 'outcome', None) # type: ignore

def test_estimate_effect_nan_data():
    """Test handling of data with NaNs resulting in empty analysis set."""
    df_nan = pd.DataFrame({
        'outcome': [np.nan, 2, 3, 4], # Ensure all rows have NaN in required cols
        'treatment': [0, np.nan, 1, 1],
        'covariate1': [5, 6, np.nan, np.nan]
    })
    with pytest.raises(ValueError, match="No data remaining after dropping NaNs"):
        estimate_effect(df_nan, 'treatment', 'outcome', ['covariate1'])

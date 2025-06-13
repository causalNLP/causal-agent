import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from auto_causal.methods.regression_discontinuity.estimator import estimate_effect

# --- Fixtures ---

@pytest.fixture
def sample_rdd_data():
    """Generates synthetic data suitable for RDD testing."""
    np.random.seed(123)
    n_samples = 200
    cutoff = 50.0
    treatment_effect = 10.0
    
    # Running variable centered around cutoff
    running_var = np.random.uniform(cutoff - 20, cutoff + 20, n_samples)
    # Treatment assigned based on cutoff
    treatment = (running_var >= cutoff).astype(int)
    # Covariate correlated with running variable
    covariate1 = 0.5 * running_var + np.random.normal(0, 5, n_samples)
    # Outcome depends on running var (parallel slopes), treatment, and covariate
    error = np.random.normal(0, 5, n_samples)
    outcome = (10 + 0.8 * running_var + 
               treatment_effect * treatment + 
               2.0 * covariate1 + error)
               
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment_indicator': treatment, # Actual treatment status
        'running_var': running_var,
        'covariate1': covariate1
    })
    return df

# --- Mocks for DoWhy --- 
@pytest.fixture
def mock_causal_model():
    """Fixture for a mocked DoWhy CausalModel."""
    mock_model_instance = MagicMock()
    # Mock the estimate_effect method
    mock_estimate = MagicMock()
    mock_estimate.value = 10.5 # Simulate DoWhy estimate
    mock_estimate.test_significance_pvalue = 0.01
    mock_estimate.confidence_interval = [8.0, 13.0]
    mock_estimate.standard_error = 1.25
    mock_model_instance.estimate_effect.return_value = mock_estimate
    
    # Patch the CausalModel class in the estimator module
    with patch('auto_causal.methods.regression_discontinuity.estimator.CausalModel') as MockCM:
        MockCM.return_value = mock_model_instance
        yield MockCM, mock_model_instance

# --- Test Cases ---

def test_estimate_effect_missing_args(sample_rdd_data):
    """Test that RDD estimation fails if required args are missing."""
    with pytest.raises(ValueError, match="Missing required RDD arguments"):
        estimate_effect(sample_rdd_data, 'treatment_indicator', 'outcome', running_variable=None, cutoff=50.0)
    with pytest.raises(ValueError, match="Missing required RDD arguments"):
        estimate_effect(sample_rdd_data, 'treatment_indicator', 'outcome', running_variable='running_var', cutoff=None)

@patch('auto_causal.methods.regression_discontinuity.estimator.run_rdd_diagnostics')
@patch('auto_causal.methods.regression_discontinuity.estimator.interpret_rdd_results')
def test_estimate_effect_dowhy_success(mock_interpret, mock_diagnostics, mock_causal_model, sample_rdd_data):
    """Test successful estimation using the mocked DoWhy path."""
    MockCM, mock_model_instance = mock_causal_model
    mock_diagnostics.return_value = {"status": "Success", "details": {"covariate_balance": "Checked"}}
    mock_interpret.return_value = "LLM Interpretation"
    
    results = estimate_effect(
        sample_rdd_data, 
        'treatment_indicator', 
        'outcome', 
        running_variable='running_var', 
        cutoff=50.0,
        bandwidth=5.0, # Specify bandwidth
        use_dowhy=True
    )
    
    MockCM.assert_called_once()
    mock_model_instance.estimate_effect.assert_called_once()
    call_args, call_kwargs = mock_model_instance.estimate_effect.call_args
    assert call_kwargs['method_name'] == "iv.regression_discontinuity"
    assert call_kwargs['method_params']['rd_variable_name'] == 'running_var'
    assert call_kwargs['method_params']['rd_threshold_value'] == 50.0
    assert call_kwargs['method_params']['rd_bandwidth'] == 5.0
    
    assert results['method_used'] == 'DoWhy RDD'
    assert results['effect_estimate'] == 10.5
    assert results['p_value'] == 0.01
    assert results['confidence_interval'] == [8.0, 13.0]
    assert results['standard_error'] == 1.25
    assert 'DoWhy RDD (Bandwidth: 5.000)' in results['method_details']
    assert 'diagnostics' in results
    assert 'interpretation' in results
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()

@patch('auto_causal.methods.regression_discontinuity.estimator.run_rdd_diagnostics')
@patch('auto_causal.methods.regression_discontinuity.estimator.interpret_rdd_results')
def test_estimate_effect_fallback_success(mock_interpret, mock_diagnostics, sample_rdd_data):
    """Test successful estimation using the fallback linear interaction method."""
    mock_diagnostics.return_value = {"status": "Success", "details": {"covariate_balance": "Checked"}}
    mock_interpret.return_value = "LLM Interpretation"

    results = estimate_effect(
        sample_rdd_data, 
        'treatment_indicator', 
        'outcome', 
        running_variable='running_var', 
        cutoff=50.0,
        covariates=['covariate1'],
        bandwidth=10.0,
        use_dowhy=False # Force fallback
    )
    
    assert results['method_used'] == 'Fallback RDD (Linear Interaction)'
    assert 'effect_estimate' in results
    assert 'p_value' in results
    assert 'confidence_interval' in results
    assert 'standard_error' in results
    assert 'model_summary' in results # Fallback provides summary
    assert 'Fallback Linear Interaction (Bandwidth: 10.000)' in results['method_details']
    # Check if estimate is reasonable (should be around 10)
    assert abs(results['effect_estimate'] - 10.0) < 20.0 
    assert 'diagnostics' in results
    assert 'interpretation' in results
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()

@patch('auto_causal.methods.regression_discontinuity.estimator.estimate_effect_dowhy')
@patch('auto_causal.methods.regression_discontinuity.estimator.estimate_effect_fallback')
def test_estimate_effect_dowhy_fails_fallback_succeeds(mock_fallback, mock_dowhy, sample_rdd_data):
    """Test that fallback is used when DoWhy fails."""
    mock_dowhy.side_effect = Exception("DoWhy broke")
    # Simulate successful fallback results
    mock_fallback.return_value = {
        'effect_estimate': 9.8,
        'p_value': 0.02,
        'confidence_interval': [1.0, 18.6],
        'standard_error': 4.0,
        'method_used': 'Fallback RDD (Linear Interaction)',
        'method_details': "Fallback Linear Interaction (Bandwidth: 10.000)",
        'formula': 'formula_str',
        'model_summary': 'summary_str'
    }
    
    # Need to also patch diagnostics and interpretation as they run after estimation
    with patch('auto_causal.methods.regression_discontinuity.estimator.run_rdd_diagnostics'), \
         patch('auto_causal.methods.regression_discontinuity.estimator.interpret_rdd_results'):
        
        results = estimate_effect(
            sample_rdd_data, 
            'treatment_indicator', 
            'outcome', 
            running_variable='running_var', 
            cutoff=50.0,
            bandwidth=10.0,
            use_dowhy=True # Try DoWhy first
        )

    mock_dowhy.assert_called_once()
    mock_fallback.assert_called_once()
    assert results['method_used'] == 'Fallback RDD (Linear Interaction)'
    assert results['effect_estimate'] == 9.8
    assert 'dowhy_error_info' in results # Check that DoWhy error was recorded
    assert "DoWhy broke" in results['dowhy_error_info']

@patch('auto_causal.methods.regression_discontinuity.estimator.estimate_effect_dowhy')
@patch('auto_causal.methods.regression_discontinuity.estimator.estimate_effect_fallback')
def test_estimate_effect_both_fail(mock_fallback, mock_dowhy, sample_rdd_data):
    """Test that an error is raised if both DoWhy and fallback fail."""
    mock_dowhy.side_effect = Exception("DoWhy broke")
    mock_fallback.side_effect = ValueError("Fallback broke")

    with pytest.raises(ValueError, match="RDD estimation failed using both DoWhy and fallback methods"):
        estimate_effect(
            sample_rdd_data, 
            'treatment_indicator', 
            'outcome', 
            running_variable='running_var', 
            cutoff=50.0,
            use_dowhy=True
        )
    mock_dowhy.assert_called_once()
    mock_fallback.assert_called_once()

def test_estimate_effect_no_data_in_bandwidth(sample_rdd_data):
    """Test error when bandwidth is too small, leading to no data."""
    # Use a very small bandwidth that excludes all data
    with pytest.raises(ValueError, match="No data within the specified bandwidth"):
         estimate_effect(
            sample_rdd_data, 
            'treatment_indicator', 
            'outcome', 
            running_variable='running_var', 
            cutoff=50.0,
            bandwidth=0.01, # Extremely small bandwidth
            use_dowhy=False # Force fallback for this specific error check
        )

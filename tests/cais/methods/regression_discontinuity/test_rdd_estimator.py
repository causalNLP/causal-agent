import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from cais.methods.regression_discontinuity.estimator import estimate_effect

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
        'treatment_indicator': treatment,
        'running_var': running_var,
        'covariate1': covariate1
    })
    return df


@pytest.fixture
def sample_rdd_data_treatment_below_cutoff():
    """Generates synthetic data suitable for RDD testing with treatment assigned below cutoff."""
    np.random.seed(123)
    n_samples = 200
    cutoff = 50.0
    treatment_effect = 10.0

    # Running variable centered around cutoff
    running_var = np.random.uniform(cutoff - 20, cutoff + 20, n_samples)
    # Treatment assigned below cutoff
    treatment = (running_var < cutoff).astype(int)
    # Covariate correlated with running variable
    covariate1 = 0.5 * running_var + np.random.normal(0, 5, n_samples)
    # Outcome depends on running var (parallel slopes), treatment, and covariate
    error = np.random.normal(0, 5, n_samples)
    outcome = (10 + 0.8 * running_var +
               treatment_effect * treatment +
               2.0 * covariate1 + error)

    df = pd.DataFrame({
        'outcome': outcome,
        'treatment_indicator': treatment,
        'running_var': running_var,
        'covariate1': covariate1
    })
    return df

# --- Test Cases ---

def test_estimate_effect_missing_args(sample_rdd_data):
    """Test that RDD estimation fails if required args are missing."""
    with pytest.raises(ValueError, match="Missing required RDD arguments"):
        estimate_effect(sample_rdd_data, 'treatment_indicator', 'outcome',
                        running_variable=None, cutoff_value=50.0)
    with pytest.raises(ValueError, match="Missing required RDD arguments"):
        estimate_effect(sample_rdd_data, 'treatment_indicator', 'outcome',
                        running_variable='running_var', cutoff_value=None)


@patch('cais.methods.regression_discontinuity.estimator.run_rdd_diagnostics')
@patch('cais.methods.regression_discontinuity.estimator.interpret_rdd_results')
@patch('cais.methods.regression_discontinuity.estimator.effect_estimate_rdd')
def test_estimate_effect_primary_success(mock_em_rdd, mock_interpret, mock_diagnostics, sample_rdd_data):
    """Test successful estimation using the mocked evan-magnusson/rdd path."""
    mock_em_rdd.return_value = {
        'effect_estimate': 10.5,
        'standard_error': 1.25,
        'p_value': 0.01,
        'confidence_interval': [8.0, 13.0],
        'method_details': 'RDD (evan-magnusson/rdd package, Bandwidth: 5.0000)',
        'bandwidth_used': 5.0,
        'formula': 'local linear',
        'model_summary': 'summary'
    }
    mock_diagnostics.return_value = {"status": "Success"}
    mock_interpret.return_value = "LLM Interpretation"

    results = estimate_effect(
        sample_rdd_data,
        'treatment_indicator',
        'outcome',
        running_variable='running_var',
        cutoff_value=50.0,
        bandwidth=5.0,
    )

    mock_em_rdd.assert_called_once()
    assert results['method_used'] == 'evan-magnusson/rdd'
    assert results['effect_estimate'] == 10.5
    assert results['p_value'] == 0.01
    assert results['confidence_interval'] == [8.0, 13.0]
    assert results['standard_error'] == 1.25
    assert 'diagnostics' in results
    assert 'interpretation' in results
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()


@patch('cais.methods.regression_discontinuity.estimator.run_rdd_diagnostics')
@patch('cais.methods.regression_discontinuity.estimator.interpret_rdd_results')
def test_estimate_effect_fallback_success(mock_interpret, mock_diagnostics, sample_rdd_data):
    """Test successful estimation using the fallback linear interaction method when primary fails."""
    mock_diagnostics.return_value = {"status": "Success"}
    mock_interpret.return_value = "LLM Interpretation"

    with patch('cais.methods.regression_discontinuity.estimator.effect_estimate_rdd',
               side_effect=Exception("Primary method unavailable")):
        results = estimate_effect(
            sample_rdd_data,
            'treatment_indicator',
            'outcome',
            running_variable='running_var',
            cutoff_value=50.0,
            covariates=['covariate1'],
            bandwidth=10.0,
        )

    assert 'Fallback' in results['method_used']
    assert 'effect_estimate' in results
    assert 'p_value' in results
    assert 'confidence_interval' in results
    assert 'standard_error' in results
    assert 'diagnostics' in results
    assert 'interpretation' in results
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()


@patch('cais.methods.regression_discontinuity.estimator.estimate_effect_fallback')
@patch('cais.methods.regression_discontinuity.estimator.effect_estimate_rdd')
def test_estimate_effect_primary_fails_fallback_succeeds(mock_em_rdd, mock_fallback, sample_rdd_data):
    """Test that fallback is used when primary method fails."""
    mock_em_rdd.side_effect = Exception("Primary method broke")
    mock_fallback.return_value = {
        'effect_estimate': 9.8,
        'p_value': 0.02,
        'confidence_interval': [1.0, 18.6],
        'standard_error': 4.0,
        'method_details': "Fallback Linear Interaction (Bandwidth: 10.000)",
        'formula': 'formula_str',
        'model_summary': 'summary_str'
    }

    with patch('cais.methods.regression_discontinuity.estimator.run_rdd_diagnostics'), \
         patch('cais.methods.regression_discontinuity.estimator.interpret_rdd_results'):

        results = estimate_effect(
            sample_rdd_data,
            'treatment_indicator',
            'outcome',
            running_variable='running_var',
            cutoff_value=50.0,
            bandwidth=10.0,
        )

    mock_em_rdd.assert_called_once()
    mock_fallback.assert_called_once()
    assert 'Fallback' in results['method_used']
    assert results['effect_estimate'] == 9.8


@patch('cais.methods.regression_discontinuity.estimator.estimate_effect_fallback')
@patch('cais.methods.regression_discontinuity.estimator.effect_estimate_rdd')
def test_estimate_effect_both_fail(mock_em_rdd, mock_fallback, sample_rdd_data):
    """Test that an error is raised if both primary and fallback methods fail."""
    mock_em_rdd.side_effect = Exception("Primary broke")
    mock_fallback.side_effect = ValueError("Fallback broke")

    with pytest.raises(ValueError, match="RDD estimation failed"):
        estimate_effect(
            sample_rdd_data,
            'treatment_indicator',
            'outcome',
            running_variable='running_var',
            cutoff_value=50.0,
        )
    mock_em_rdd.assert_called_once()
    mock_fallback.assert_called_once()


def test_estimate_effect_no_data_in_bandwidth(sample_rdd_data):
    """Test error when bandwidth is too small, leading to no data."""
    with patch('cais.methods.regression_discontinuity.estimator.effect_estimate_rdd',
               side_effect=Exception("Primary unavailable")):
        with pytest.raises(ValueError, match="No data within the specified bandwidth"):
            estimate_effect(
                sample_rdd_data,
                'treatment_indicator',
                'outcome',
                running_variable='running_var',
                cutoff_value=50.0,
                bandwidth=0.01,  # Extremely small bandwidth
            )


@patch('cais.methods.regression_discontinuity.estimator.run_rdd_diagnostics')
@patch('cais.methods.regression_discontinuity.estimator.interpret_rdd_results')
@patch('cais.methods.regression_discontinuity.estimator.effect_estimate_rdd')
def test_estimate_effect_primary_success_treatment_below_cutoff(mock_em_rdd, mock_interpret, mock_diagnostics, sample_rdd_data_treatment_below_cutoff):
    """Test successful estimation using the mocked evan-magnusson/rdd path with treatment assigned below cutoff."""
    mock_em_rdd.return_value = {
        'effect_estimate': 10.5,
        'standard_error': 1.25,
        'p_value': 0.01,
        'confidence_interval': [8.0, 13.0],
        'method_details': 'RDD (evan-magnusson/rdd package, Bandwidth: 5.0000)',
        'bandwidth_used': 5.0,
        'formula': 'local linear',
        'model_summary': 'summary'
    }
    mock_diagnostics.return_value = {"status": "Success"}
    mock_interpret.return_value = "LLM Interpretation"

    results = estimate_effect(
        sample_rdd_data_treatment_below_cutoff,
        'treatment_indicator',
        'outcome',
        running_variable='running_var',
        cutoff_value=50.0,
        bandwidth=5.0,
        treat_above_cutoff=False,
    )

    mock_em_rdd.assert_called_once()
    assert results['method_used'] == 'evan-magnusson/rdd'
    assert results['effect_estimate'] == 10.5
    assert results['p_value'] == 0.01
    assert results['confidence_interval'] == [8.0, 13.0]
    assert results['standard_error'] == 1.25
    assert 'diagnostics' in results
    assert 'interpretation' in results
    mock_diagnostics.assert_called_once()
    mock_interpret.assert_called_once()

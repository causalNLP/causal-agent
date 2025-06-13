import pytest
import pandas as pd
import numpy as np
from auto_causal.methods.regression_discontinuity.diagnostics import run_rdd_diagnostics

# --- Fixture for RDD data ---
@pytest.fixture
def sample_rdd_data():
    """Generates synthetic data suitable for RDD testing."""
    np.random.seed(123)
    n_samples = 200
    cutoff = 50.0
    treatment_effect = 10.0
    
    running_var = np.random.uniform(cutoff - 20, cutoff + 20, n_samples)
    treatment = (running_var >= cutoff).astype(int)
    # Covariate correlated with running variable (potential imbalance)
    covariate1 = 0.5 * running_var + np.random.normal(0, 5, n_samples)
    # Covariate uncorrelated (should be balanced)
    covariate2 = np.random.normal(10, 2, n_samples)
    error = np.random.normal(0, 5, n_samples)
    outcome = (10 + 0.8 * running_var + 
               treatment_effect * treatment + 
               1.2 * treatment * (running_var - cutoff) + 
               2.0 * covariate1 + 1.0 * covariate2 + error)
               
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment_indicator': treatment,
        'running_var': running_var,
        'covariate1': covariate1,
        'covariate2': covariate2
    })
    return df

# --- Test Cases ---

def test_run_rdd_diagnostics_success(sample_rdd_data):
    """Test the diagnostics function with covariates."""
    covariates = ['covariate1', 'covariate2']
    results = run_rdd_diagnostics(
        sample_rdd_data, 
        'outcome', 
        'running_var', 
        cutoff=50.0, 
        covariates=covariates,
        bandwidth=10.0 # Use a reasonable bandwidth
    )
    
    assert results["status"] == "Success (Partial Implementation)"
    assert "details" in results
    details = results["details"]
    
    assert "covariate_balance" in details
    balance = details['covariate_balance']
    assert isinstance(balance, dict)
    assert 'covariate1' in balance
    assert 'covariate2' in balance
    
    # Check structure of balance results
    assert 't_statistic' in balance['covariate1']
    assert 'p_value' in balance['covariate1']
    assert 'balanced' in balance['covariate1']
    assert 't_statistic' in balance['covariate2']
    assert 'p_value' in balance['covariate2']
    assert 'balanced' in balance['covariate2']
    
    # Check expected balance (covariate1 likely unbalanced, covariate2 likely balanced)
    # Due to random noise, these might occasionally fail, but should usually hold
    assert balance['covariate1']['balanced'].startswith("No") 
    assert balance['covariate2']['balanced'] == "Yes"
    
    # Check placeholders
    assert details['continuity_density_test'] == "Not Implemented (Requires specialized libraries like rdd)"
    assert details['visual_inspection'] == "Recommended (Plot outcome vs running variable with fits)"

def test_run_rdd_diagnostics_no_covariates(sample_rdd_data):
    """Test diagnostics when no covariates are provided."""
    results = run_rdd_diagnostics(
        sample_rdd_data, 'outcome', 'running_var', cutoff=50.0, covariates=None, bandwidth=10.0
    )
    assert results["status"] == "Success (Partial Implementation)"
    assert results["details"]['covariate_balance'] == "No covariates provided to check."

def test_run_rdd_diagnostics_small_bandwidth(sample_rdd_data):
    """Test diagnostics handles cases with insufficient data in bandwidth."""
    # Bandwidth so small it likely excludes one side
    results = run_rdd_diagnostics(
        sample_rdd_data, 'outcome', 'running_var', cutoff=50.0, covariates=['covariate1'], bandwidth=0.1 
    )
    assert results["status"] == "Skipped"
    assert "Insufficient data near cutoff" in results["reason"]

def test_run_rdd_diagnostics_missing_covariate(sample_rdd_data):
    """Test diagnostics handles missing covariate columns gracefully."""
    results = run_rdd_diagnostics(
        sample_rdd_data, 'outcome', 'running_var', cutoff=50.0, covariates=['covariate1', 'missing_cov'], bandwidth=10.0
    )
    assert results["status"] == "Success (Partial Implementation)"
    balance = results["details"]['covariate_balance']
    assert balance['missing_cov']['status'] == "Column Not Found"
    assert 't_statistic' in balance['covariate1'] # Check other covariate was still processed


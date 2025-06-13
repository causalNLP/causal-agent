import pytest
import pandas as pd
import numpy as np
from auto_causal.methods.diff_in_means.diagnostics import run_dim_diagnostics

# --- Fixture ---
@pytest.fixture
def sample_stats_data():
    """Generates data for testing diagnostic stats."""
    df = pd.DataFrame({
        'outcome': [10, 12, 11, 20, 25, 22, 100], # Group 0: 10, 11, 12; Group 1: 20, 22, 25; Group 2: 100
        'treatment': [ 0,  1,  0,  1,  1,  1,   2],
        'another': [1,1,1,1,1,1,1]
    })
    return df

# --- Test Cases ---

def test_run_dim_diagnostics_success(sample_stats_data):
    """Test successful calculation of group stats."""
    df = sample_stats_data[sample_stats_data['treatment'].isin([0, 1])] # Filter to binary
    results = run_dim_diagnostics(df, 'treatment', 'outcome')
    
    assert results['status'] == "Success"
    assert "details" in results
    details = results['details']
    
    assert "control_group_stats" in details
    assert "treated_group_stats" in details
    
    control = details['control_group_stats']
    treated = details['treated_group_stats']
    
    assert control['count'] == 2
    assert treated['count'] == 4
    assert np.isclose(control['mean'], 10.5)
    assert np.isclose(treated['mean'], 19.75)
    assert np.isclose(control['std'], np.std([10, 11], ddof=1)) # Pandas uses ddof=1
    assert np.isclose(treated['std'], np.std([12, 20, 25, 22], ddof=1))
    assert "variance_homogeneity_status" in details
    assert details['variance_homogeneity_status'] == "Potentially Unequal (ratio > 4 or < 0.25)" 

def test_run_dim_diagnostics_empty_group(sample_stats_data):
    """Test warning when one group is empty."""
    df_one_group = sample_stats_data[sample_stats_data['treatment'] == 1]
    results = run_dim_diagnostics(df_one_group, 'treatment', 'outcome')
    
    assert results['status'] == "Warning - Empty Group(s)"
    assert results['details']['control_group_stats']['count'] == 0
    assert results['details']['treated_group_stats']['count'] == 4

def test_run_dim_diagnostics_key_error(sample_stats_data):
    """Test that function runs successfully even with non-0/1 levels, but only calculates for 0/1."""
    # Use data with treatment = 2 present
    results = run_dim_diagnostics(sample_stats_data, 'treatment', 'outcome')
    
    # Expect success because groups 0 and 1 exist and stats can be calculated for them
    assert results['status'] == "Success" 
    details = results['details']
    assert 'control_group_stats' in details
    assert 'treated_group_stats' in details
    assert details['control_group_stats']['count'] == 2 # Should still find group 0
    assert details['treated_group_stats']['count'] == 4 # Should still find group 1

def test_run_dim_diagnostics_zero_variance(sample_stats_data):
    """Test handling when one group has zero variance."""
    df_zero_var = pd.DataFrame({
        'outcome': [10, 10, 20, 25, 22], 
        'treatment': [ 0,  0,  1,  1,  1],
    })
    results = run_dim_diagnostics(df_zero_var, 'treatment', 'outcome')
    assert results['status'] == "Success"
    assert results['details']['variance_homogeneity_status'] == "Could not calculate (zero variance in a group)"

import pytest
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from unittest.mock import patch, MagicMock

# Module containing the function to test
ESTIMATOR_MODULE = "auto_causal.methods.difference_in_differences.estimator"

# Import the function to test AFTER defining the module path
from auto_causal.methods.difference_in_differences.estimator import estimate_effect

# --- Fixtures ---

@pytest.fixture
def sample_did_data():
    """Generates synthetic panel data suitable for DiD testing."""
    np.random.seed(2024)
    n_units = 50
    n_periods = 10
    treatment_start_time = 5 # Treatment starts in period 5
    true_effect = 7.0

    units = np.arange(n_units)
    periods = np.arange(n_periods)

    # Create panel structure
    panel_index = pd.MultiIndex.from_product([units, periods], names=['unit_id', 'time_period'])
    df = pd.DataFrame(index=panel_index).reset_index()

    # Assign treatment group (first half of units)
    df['group'] = (df['unit_id'] < n_units // 2).astype(int)
    
    # Create post-treatment indicator - REMOVE this, let estimator call the helper
    # df['post'] = (df['time_period'] >= treatment_start_time).astype(int)

    # Create interaction term (true treatment effect applied here)
    # Need 'post' for this, so create it temporarily or adjust outcome formula
    # Let's adjust outcome formula to not rely on pre-calculated interaction
    # df['did_interaction'] = df['group'] * df['post'] 
    df['is_post_treatment'] = (df['time_period'] >= treatment_start_time).astype(int)
    
    # Create covariates
    df['covariate1'] = np.random.normal(5, 1, size=len(df))
    # Time-varying covariate
    df['covariate2'] = df['time_period'] * 0.2 + np.random.normal(0, 0.5, size=len(df))

    # Unit and time fixed effects
    unit_fe = np.random.normal(0, 3, n_units)
    time_fe = np.random.normal(0, 2, n_periods)
    df['unit_fe_val'] = df['unit_id'].map(dict(enumerate(unit_fe)))
    df['time_fe_val'] = df['time_period'].map(dict(enumerate(time_fe)))

    # Generate outcome
    error = np.random.normal(0, 1, len(df))
    df['outcome'] = (10 + 
                       true_effect * df['group'] * df['is_post_treatment'] + # Use group * post directly
                       df['unit_fe_val'] + 
                       df['time_fe_val'] + 
                       0.5 * df['covariate1'] + 
                       -0.3 * df['covariate2'] + 
                       error)
                       
    # Use 'group' as the main treatment indicator for some tests
    df['treatment'] = df['group'] 

    return df

# --- Test Cases ---

# Mock all imported helper functions from estimator module
@patch(f'{ESTIMATOR_MODULE}.identify_time_variable')
@patch(f'{ESTIMATOR_MODULE}.determine_treatment_period')
@patch(f'{ESTIMATOR_MODULE}.identify_treatment_group')
@patch(f'{ESTIMATOR_MODULE}.create_post_indicator')
@patch(f'{ESTIMATOR_MODULE}.validate_parallel_trends')
@patch(f'{ESTIMATOR_MODULE}.format_did_results') # Also mock the formatter
@patch(f'{ESTIMATOR_MODULE}.smf.ols') # Mock statsmodels itself
def test_estimate_effect_twfe_no_covariates(
    mock_ols, mock_formatter, mock_validate, mock_create_post, 
    mock_id_group, mock_det_period, mock_id_time, 
    sample_did_data
):
    """Test basic TWFE DiD estimation without covariates."""
    # Setup mocks for helpers
    mock_id_time.return_value = 'time_period'
    mock_det_period.return_value = 5 # Treatment start time
    mock_id_group.return_value = 'unit_id' # The ID variable for FE/clustering
    mock_create_post.return_value = (sample_did_data['time_period'] >= 5).astype(int) 
    mock_validate.return_value = {'valid': True, 'details': 'Mocked validation'}
    
    # Setup mock for statsmodels results
    mock_fit = MagicMock()
    interaction_term_formula = "Q('did_interaction')" # Key based on created col name
    mock_fit.params = pd.Series({interaction_term_formula: 7.1, 'other_coef': 1.0})
    mock_fit.bse = pd.Series({interaction_term_formula: 0.5, 'other_coef': 0.1})
    mock_fit.pvalues = pd.Series({interaction_term_formula: 0.001, 'other_coef': 0.1})
    conf_int_df = pd.DataFrame([[6.1, 8.1]], index=[interaction_term_formula], columns=[0, 1])
    mock_fit.conf_int.return_value = conf_int_df
    mock_fit.summary.return_value = "Mock Summary"
    mock_ols.return_value.fit.return_value = mock_fit
    
    # Setup mock for formatter to check its input
    mock_formatter.return_value = {"effect_estimate": 7.1, "method_used": "DiD.TWFE"} # Dummy return
    
    # Call the function (treatment='group' which is the binary indicator)
    results = estimate_effect(
        sample_did_data, 
        treatment='group', 
        outcome='outcome', 
        covariates=[],
        dataset_description={}
        )
    
    # Assertions
    mock_id_time.assert_called_once()
    mock_det_period.assert_called_once()
    mock_id_group.assert_called_once()
    mock_create_post.assert_called_once()
    mock_validate.assert_called_once()
    mock_ols.assert_called_once()
    mock_formatter.assert_called_once()
    
    # Check formula passed to OLS
    call_args, call_kwargs = mock_ols.call_args
    formula_used = call_kwargs['formula']
    assert "Q('outcome') ~ Q('did_interaction') + C(unit_id) + C(time_period)" == formula_used
    
    # Check clustering variable
    fit_call_args, fit_call_kwargs = mock_ols.return_value.fit.call_args
    assert fit_call_kwargs['cov_type'] == 'cluster'
    assert 'groups' in fit_call_kwargs['cov_kwds']
    # Check if the correct grouping column (unit_id) was used for clustering
    assert fit_call_kwargs['cov_kwds']['groups'].name == 'unit_id'

    # Check arguments passed to formatter
    format_call_args, format_call_kwargs = mock_formatter.call_args
    assert format_call_args[0] == mock_fit # Check results object
    assert format_call_args[1] == interaction_term_formula # Check interaction term key
    assert format_call_args[2]["parallel_trends"]['valid'] is True # Check diagnostics
    assert format_call_kwargs['parameters']['time_var'] == 'time_period'
    assert format_call_kwargs['parameters']['group_var'] == 'unit_id'
    assert format_call_kwargs['parameters']['treatment_indicator'] == 'group' # Identified correctly
    assert format_call_kwargs['parameters']['covariates'] == []
    
    # Check final output (dummy from formatter mock)
    assert results['effect_estimate'] == 7.1
    assert results['method_used'] == 'DiD.TWFE'

# Add more tests: with covariates, missing columns, variable identification scenarios etc.
# Example for missing column:
def test_estimate_effect_missing_outcome(sample_did_data):
    with pytest.raises(ValueError, match="Outcome variable 'missing_outcome' not found"):
         # Need to mock helpers even for early exit tests if they are called before check
         with patch(f'{ESTIMATOR_MODULE}.identify_time_variable', return_value='time_period'), \
              patch(f'{ESTIMATOR_MODULE}.identify_treatment_group', return_value='unit_id'), \
              patch(f'{ESTIMATOR_MODULE}.determine_treatment_period', return_value=5):
            estimate_effect(sample_did_data, treatment='group', outcome='missing_outcome', covariates=[], dataset_description={})

# Example for variable identification test
@patch(f'{ESTIMATOR_MODULE}.identify_time_variable')
@patch(f'{ESTIMATOR_MODULE}.determine_treatment_period')
@patch(f'{ESTIMATOR_MODULE}.identify_treatment_group')
@patch(f'{ESTIMATOR_MODULE}.create_post_indicator')
@patch(f'{ESTIMATOR_MODULE}.validate_parallel_trends')
@patch(f'{ESTIMATOR_MODULE}.format_did_results') 
@patch(f'{ESTIMATOR_MODULE}.smf.ols') 
def test_treatment_col_identification(
    mock_ols, mock_formatter, mock_validate, mock_create_post, 
    mock_id_group, mock_det_period, mock_id_time, 
    sample_did_data
):
    """Test identification of the binary treatment indicator."""
    mock_id_time.return_value = 'time_period'
    mock_det_period.return_value = 5 
    mock_id_group.return_value = 'unit_id' 
    mock_create_post.return_value = (sample_did_data['time_period'] >= 5).astype(int) 
    mock_validate.return_value = {'valid': True}
    mock_ols.return_value.fit.return_value = MagicMock() # Don't need detailed results mock
    mock_formatter.return_value = {}

    # Scenario 1: 'treatment' arg IS the binary indicator ('group' column)
    estimate_effect(sample_did_data, treatment='group', outcome='outcome', covariates=[], dataset_description={})
    format_call_args, format_call_kwargs = mock_formatter.call_args
    assert format_call_kwargs['parameters']['treatment_indicator'] == 'group'
    mock_formatter.reset_mock()

    # Scenario 2: 'treatment' arg is NOT binary, but 'group' col exists and IS binary
    # Keep the original 'group' column, just add the non-binary unit_id_str
    df_modified = sample_did_data.copy()
    # df_modified = sample_did_data.rename(columns={'group': 'treatment'}) # DON'T rename
    df_modified['unit_id_str'] = df_modified['unit_id'].astype(str) # Make unit_id non-binary
    mock_id_group.return_value = 'unit_id_str' # LLM identifies non-binary unit_id
    # Call estimate_effect with the non-binary unit_id_str as the treatment argument
    estimate_effect(df_modified, treatment='unit_id_str', outcome='outcome', covariates=[], dataset_description={})
    format_call_args, format_call_kwargs = mock_formatter.call_args
    # Should correctly identify the *original* 'group' column via Priority 2 logic
    assert format_call_kwargs['parameters']['treatment_indicator'] == 'group' 
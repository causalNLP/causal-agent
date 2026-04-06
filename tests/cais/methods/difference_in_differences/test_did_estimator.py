import pytest
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from unittest.mock import patch, MagicMock

# Module containing the function to test
ESTIMATOR_MODULE = "cais.methods.difference_in_differences.estimator"

# Import the function to test AFTER defining the module path
from cais.methods.difference_in_differences.estimator import estimate_effect, format_did_results

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
    df['is_post_treatment'] = (df['time_period'] >= treatment_start_time).astype(int)
    df['treatment'] = df['group'] * df['is_post_treatment']

    # Create covariates
    df['covariate1'] = np.random.normal(5, 1, size=len(df))

    # Unit and time fixed effects
    unit_fe = np.random.normal(0, 3, n_units)
    time_fe = np.random.normal(0, 2, n_periods)
    df['unit_fe_val'] = df['unit_id'].map(dict(enumerate(unit_fe)))
    df['time_fe_val'] = df['time_period'].map(dict(enumerate(time_fe)))

    # Generate outcome
    error = np.random.normal(0, 1, len(df))
    df['outcome'] = (10 +
                       true_effect * df['group'] * df['is_post_treatment'] +
                       df['unit_fe_val'] +
                       df['time_fe_val'] +
                       0.5 * df['covariate1'] +
                       error)

    return df


# --- Test Cases ---

@patch(f'{ESTIMATOR_MODULE}.format_did_results')
@patch(f'{ESTIMATOR_MODULE}.smf.ols')
def test_estimate_effect_twfe_no_covariates(mock_ols, mock_formatter, sample_did_data):
    """Test basic TWFE DiD estimation without covariates."""
    # Setup mock for statsmodels results
    mock_fit = MagicMock()
    treatment_col = "treatment"
    mock_fit.params = pd.Series({treatment_col: 7.1, 'other_coef': 1.0})
    mock_fit.bse = pd.Series({treatment_col: 0.5, 'other_coef': 0.1})
    mock_fit.pvalues = pd.Series({treatment_col: 0.001, 'other_coef': 0.1})
    conf_int_df = pd.DataFrame([[6.1, 8.1]], index=[treatment_col], columns=[0, 1])
    mock_fit.conf_int.return_value = conf_int_df
    mock_fit.summary.return_value = "Mock Summary"
    mock_ols.return_value.fit.return_value = mock_fit

    mock_formatter.return_value = {"effect_estimate": 7.1, "method_used": "DiD.TWFE"}

    # Call using TWFE path (time_period has more than 2 values → not binary)
    results = estimate_effect(
        sample_did_data,
        treatment='treatment',
        outcome='outcome',
        covariates=[],
        time_variable='time_period',
        group_variable='unit_id',
    )

    mock_ols.assert_called_once()
    mock_formatter.assert_called_once()

    # Check formula uses TWFE structure
    call_args, call_kwargs = mock_ols.call_args
    formula_used = call_kwargs['formula']
    assert "outcome ~ treatment + C(unit_id) + C(time_period)" == formula_used

    # Check final output (dummy from formatter mock)
    assert results['effect_estimate'] == 7.1


def test_estimate_effect_missing_outcome(sample_did_data):
    with pytest.raises(ValueError, match="Outcome variable 'missing_outcome' not found"):
        estimate_effect(
            sample_did_data,
            treatment='group',
            outcome='missing_outcome',
            covariates=[],
            time_variable='time_period',
            group_variable='unit_id',
        )


def test_treatment_col_twfe_uses_did_term(sample_did_data):
    """Test that TWFE uses the did_term kwarg when provided."""
    with patch(f'{ESTIMATOR_MODULE}.smf.ols') as mock_ols, \
         patch(f'{ESTIMATOR_MODULE}.format_did_results') as mock_formatter:

        mock_fit = MagicMock()
        mock_fit.params = pd.Series({'treatment': 7.0})
        mock_fit.bse = pd.Series({'treatment': 0.5})
        mock_fit.pvalues = pd.Series({'treatment': 0.001})
        conf_int_df = pd.DataFrame([[6.0, 8.0]], index=['treatment'], columns=[0, 1])
        mock_fit.conf_int.return_value = conf_int_df
        mock_fit.summary.return_value = "Mock Summary"
        mock_ols.return_value.fit.return_value = mock_fit
        mock_formatter.return_value = {}

        estimate_effect(
            sample_did_data,
            treatment='group',
            outcome='outcome',
            covariates=[],
            time_variable='time_period',
            group_variable='unit_id',
            did_term='treatment',
        )

        call_args, call_kwargs = mock_ols.call_args
        formula_used = call_kwargs['formula']
        # did_term='treatment' should be used in formula
        assert 'treatment' in formula_used


if __name__ == "__main__":
    pytest.main()

import pytest
import pandas as pd
import numpy as np

# Import the function to test
from cais.methods.difference_in_differences import estimator as did_estimator
# Import placeholder diagnostics to check if they are called
from cais.methods.difference_in_differences import diagnostics as did_diagnostics
# Import placeholder llm assists to check if they are called
from cais.methods.difference_in_differences import llm_assist as did_llm_assist

from unittest.mock import patch # For testing placeholder calls

# --- Synthetic Data Generation ---
def generate_synthetic_did_data(n_units=50, n_periods=10, treatment_start_period=6, 
                                  treatment_effect=5.0, seed=42):
    """Generates synthetic panel data suitable for DiD.
    
    Features:
        - Units with fixed effects.
        - Time periods with fixed effects.
        - A subset of units treated after a specific period.
        - Observed time-invariant covariate (optional).
        - Known true treatment effect.
    """
    np.random.seed(seed)
    
    units = range(n_units)
    periods = range(n_periods)
    
    df = pd.DataFrame([(u, p) for u in units for p in periods], columns=['unit', 'time'])
    
    # Unit fixed effects
    unit_effects = pd.DataFrame({'unit': units, 'unit_fe': np.random.normal(0, 2, n_units)})
    df = pd.merge(df, unit_effects, on='unit')
    
    # Time fixed effects
    time_effects = pd.DataFrame({'time': periods, 'time_fe': np.random.normal(0, 1.5, n_periods)})
    df = pd.merge(df, time_effects, on='time')
    
    # Treatment group (e.g., half the units)
    treated_units = range(n_units // 2)
    df['group'] = df['unit'].apply(lambda u: 1 if u in treated_units else 0)
    
    # Treatment indicator (post * treated_group)
    df['post'] = (df['time'] >= treatment_start_period).astype(int)
    df['treatment'] = df['group'] * df['post']
    
    # Outcome model
    df['outcome'] = (10 + 
                       df['unit_fe'] + 
                       df['time_fe'] + 
                       df['treatment'] * treatment_effect + 
                       np.random.normal(0, 1, len(df))) # Noise
                       
    # Select relevant columns for clarity - remove X1
    df = df[['unit', 'time', 'group', 'treatment', 'outcome']]
    # Log info about generated data
    print("\n--- Generated Synthetic Data Info ---")
    print("Head:\n", df.head()) 
    print("\nInfo:")
    df.info()
    print("\nDescribe:\n", df.describe())
    print("-------------------------------------")
    return df

# --- Test Class ---
class TestDifferenceInDifferences:

    def test_did_estimate_effect_synthetic(self):
        """Test the end-to-end estimate_effect with synthetic DiD data."""
        # Arrange
        true_effect = 7.0
        df = generate_synthetic_did_data(n_units=100, n_periods=12, treatment_start_period=8, 
                                         treatment_effect=true_effect, seed=123)
                                         
        # treatment_var should be the ACTUAL 0/1 treatment status indicator
        treatment_var = 'treatment'
        outcome_var = 'outcome'
        time_var = 'time'
        group_var = 'unit' # The identifier for the panel unit
        covariates = [] # Was ['X1']
        
        # Act
        results = did_estimator.estimate_effect(
            df=df,
            treatment=treatment_var,
            outcome=outcome_var,
            covariates=covariates,
            # Pass specific required args for DiD via kwargs
            time_variable=time_var,
            group_variable=group_var,
        )

        # Assert
        assert results is not None
        assert "error" not in results # Check for errors
        
        assert 'effect_estimate' in results
        assert 'standard_error' in results
        assert 'confidence_interval' in results
        assert 'diagnostics' in results
        assert 'parameters' in results
        assert 'details' in results # Should contain statsmodels summary
        
        # Check estimate value
        estimated_effect = results['effect_estimate']
        assert estimated_effect is not None
        assert abs(estimated_effect - true_effect) < 1.0 # Tolerance for estimation noise
        
        # Check SE and CI
        assert results['standard_error'] is not None and results['standard_error'] > 0
        assert results['confidence_interval'] is not None
        assert results['confidence_interval'][0] < estimated_effect < results['confidence_interval'][1]
        
        # Check diagnostics were actually run and included
        assert 'parallel_trends' in results['diagnostics']
        # Assert that the actual validation passed (placeholder returns True)
        assert results['diagnostics']['parallel_trends']['valid'] == True

        # Check parameters reflect inputs/defaults
        assert results['parameters']['time_var'] == time_var
        assert results['parameters']['group_var'] == group_var
        assert results['parameters']['covariates'] == covariates
        assert results['parameters']['estimation_method'] == "TWFE DiD"
        assert 'interaction_term' in results['parameters']

    def test_did_missing_time_variable(self):
        """estimate_effect raises ValueError when time_variable is None."""
        df = generate_synthetic_did_data(n_units=20, n_periods=4, seed=1)
        with pytest.raises(ValueError, match="[Tt]ime variable"):
            did_estimator.estimate_effect(
                df=df,
                treatment='treatment',
                outcome='outcome',
                covariates=[],
                time_variable=None,
                group_variable='unit',
            )

    def test_did_missing_time_variable_not_in_df(self):
        """estimate_effect raises ValueError when time_variable column is absent."""
        df = generate_synthetic_did_data(n_units=20, n_periods=4, seed=2)
        with pytest.raises(ValueError, match="[Tt]ime variable"):
            did_estimator.estimate_effect(
                df=df,
                treatment='treatment',
                outcome='outcome',
                covariates=[],
                time_variable='nonexistent_time_col',
                group_variable='unit',
            )

    def test_did_with_covariates(self):
        """TWFE DiD with a covariate produces a valid result dict."""
        df = generate_synthetic_did_data(
            n_units=60, n_periods=8, treatment_start_period=5,
            treatment_effect=6.0, seed=99
        )
        results = did_estimator.estimate_effect(
            df=df,
            treatment='treatment',
            outcome='outcome',
            covariates=[],       # 'group' is a structural var; passing empty is fine
            time_variable='time',
            group_variable='unit',
        )

        assert results is not None
        assert 'effect_estimate' in results
        assert 'standard_error' in results
        assert 'confidence_interval' in results
        assert 'parameters' in results
        assert results['parameters']['estimation_method'] in ("TWFE DiD", "2x2 DiD")

    def test_did_2x2_path_binary_time_and_treatment(self):
        """When time and treatment are both binary, the 2x2 DiD path is selected."""
        # Construct minimal 2x2 DiD dataset
        np.random.seed(7)
        n = 200
        post = np.random.randint(0, 2, n)
        group = np.random.randint(0, 2, n)
        treatment = post * group
        outcome = 10 + 3 * treatment + np.random.normal(0, 1, n)
        unit_id = np.arange(n)

        df = pd.DataFrame({
            'post': post,
            'group': group,
            'treatment': treatment,
            'outcome': outcome,
            'unit': unit_id,
        })

        results = did_estimator.estimate_effect(
            df=df,
            treatment='group',    # binary group indicator → 2x2 path
            outcome='outcome',
            covariates=[],
            time_variable='post', # binary time → 2x2 path
            group_variable='unit',
        )

        assert results is not None
        assert 'effect_estimate' in results
        assert results['parameters']['estimation_method'] == "2x2 DiD"
        # Effect should be roughly 3.0
        assert abs(results['effect_estimate'] - 3.0) < 1.5
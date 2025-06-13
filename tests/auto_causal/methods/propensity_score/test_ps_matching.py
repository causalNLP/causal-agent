import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the function to test
from auto_causal.methods.propensity_score import matching as ps_matching

# Helper function to generate synthetic data for PSM
def generate_synthetic_psm_data(n_samples=1000, treatment_effect=5.0, seed=42):
    """Generates synthetic data suitable for PSM testing.
    
    Features:
        - Binary treatment based on covariates (logistic function).
        - Confounding: Covariates affect both treatment and outcome.
        - Known true treatment effect.
    """
    np.random.seed(seed)
    
    # Covariates
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.binomial(1, 0.6, n_samples)
    
    # Propensity score (true probability of treatment)
    logit_p = -0.5 + 1.5 * X1 - 0.8 * X2 + np.random.normal(0, 0.5, n_samples)
    p_treatment = 1 / (1 + np.exp(-logit_p))
    
    # Treatment assignment
    treatment = np.random.binomial(1, p_treatment, n_samples)
    
    # Outcome model (linear)
    # Base outcome depends on covariates (confounding)
    base_outcome = 10 + 2.0 * X1 + 3.0 * X2 + np.random.normal(0, 2, n_samples)
    # Treatment adds a fixed effect
    outcome = base_outcome + treatment * treatment_effect
    
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'treatment': treatment,
        'outcome': outcome
    })
    return data

# Test Class
class TestPropensityScoreMatching:

    def test_estimate_effect_synthetic_data(self):
        """Test the end-to-end estimate_effect with synthetic data."""
        df = generate_synthetic_psm_data(n_samples=2000, treatment_effect=5.0, seed=123)
        covariates = ['X1', 'X2']
        treatment = 'treatment'
        outcome = 'outcome'
        
        # Run the matching estimator
        # Use default parameters for now (caliper=0.2, n_neighbors=1, logistic model)
        results = ps_matching.estimate_effect(df, treatment, outcome, covariates)
        
        assert 'effect_estimate' in results
        assert 'effect_se' in results
        assert 'confidence_interval' in results
        assert 'diagnostics' in results
        assert 'parameters' in results
        
        # Check if the estimated effect is reasonably close to the true effect
        # Allow for some tolerance due to estimation noise
        true_effect = 5.0
        estimated_effect = results['effect_estimate']
        assert abs(estimated_effect - true_effect) < 1.0 # Adjust tolerance as needed
        
        # Check if standard error and CI are plausible
        assert results['effect_se'] > 0
        assert results['confidence_interval'][0] < estimated_effect
        assert results['confidence_interval'][1] > estimated_effect
        
        # Check diagnostics structure (based on current placeholder implementation)
        assert 'balance_metrics' in results['diagnostics']
        assert 'balance_achieved' in results['diagnostics']
        assert 'plots' in results['diagnostics']
        assert 'percent_treated_matched' in results['diagnostics']

    @patch('auto_causal.methods.propensity_score.matching.get_llm_parameters')
    @patch('auto_causal.methods.propensity_score.matching.determine_optimal_caliper')
    @patch('auto_causal.methods.propensity_score.matching.select_propensity_model')
    def test_llm_parameter_usage(self, mock_select_model, mock_determine_caliper, mock_get_llm_params):
        """Test that LLM helper functions are called and their results are potentially used."""
        df = generate_synthetic_psm_data(n_samples=100, seed=456) # Smaller sample for this test
        covariates = ['X1', 'X2']
        treatment = 'treatment'
        outcome = 'outcome'
        query = "What is the effect?"
        
        # Configure mocks
        # Simulate LLM providing some parameters
        mock_get_llm_params.return_value = {
            "parameters": {"caliper": 0.1, "n_neighbors": 2}, 
            "validation": {}
        }
        # Ensure other helpers return defaults if LLM doesn't provide everything
        mock_determine_caliper.return_value = 0.2 # Fallback if LLM doesn't provide caliper
        mock_select_model.return_value = 'logistic' # Fallback model

        # Call the function with a query to trigger LLM pathway
        results = ps_matching.estimate_effect(df, treatment, outcome, covariates, query=query)
        
        # Assertions
        mock_get_llm_params.assert_called_once_with(df, query, "PS.Matching")
        # determine_optimal_caliper should NOT be called if LLM provides the caliper
        mock_determine_caliper.assert_not_called() 
        # select_propensity_model should NOT be called if get_llm_parameters provides it (it doesn't in this mock setup)
        # --> actually, it WILL be called because the mock get_llm_parameters doesn't provide 'propensity_model_type'
        mock_select_model.assert_called_once() 
        
        # Check if the parameters used in the results reflect the LLM suggestions
        assert results['parameters']['caliper'] == 0.1
        assert results['parameters']['n_neighbors'] == 2
        assert results['parameters']['propensity_model'] == 'logistic' # Came from fallback

    # TODO: Add tests for diagnostic outputs (checking balance improvement)
    # TODO: Add tests for edge cases (e.g., no matches found)
    # TODO: Add tests for different parameter inputs (e.g., specifying caliper directly) 
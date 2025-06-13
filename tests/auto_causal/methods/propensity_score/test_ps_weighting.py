import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the function to test
from auto_causal.methods.propensity_score import weighting as ps_weighting

# Reuse the synthetic data generation function from matching tests
# (or redefine if weighting requires different data characteristics)
from .test_ps_matching import generate_synthetic_psm_data 

# Test Class
class TestPropensityScoreWeighting:

    def test_estimate_effect_synthetic_data_ate(self):
        """Test the end-to-end estimate_effect with synthetic data for ATE."""
        df = generate_synthetic_psm_data(n_samples=2000, treatment_effect=5.0, seed=123)
        covariates = ['X1', 'X2']
        treatment = 'treatment'
        outcome = 'outcome'
        
        # Run the weighting estimator for ATE
        results = ps_weighting.estimate_effect(df, treatment, outcome, covariates, weight_type='ATE')
        
        assert 'effect_estimate' in results
        assert 'effect_se' in results
        assert 'confidence_interval' in results
        assert 'diagnostics' in results
        assert 'parameters' in results
        
        # Check if the estimated effect is reasonably close to the true effect
        # ATE might be slightly different from ATT depending on effect heterogeneity (none here)
        true_effect = 5.0
        estimated_effect = results['effect_estimate']
        assert abs(estimated_effect - true_effect) < 1.0 # Adjust tolerance
        
        # Check if standard error and CI are plausible
        assert results['effect_se'] > 0
        assert results['confidence_interval'][0] < estimated_effect
        assert results['confidence_interval'][1] > estimated_effect
        
        # Check diagnostics structure (based on current placeholder implementation)
        assert 'min_weight' in results['diagnostics']
        assert 'max_weight' in results['diagnostics']
        assert 'effective_sample_size' in results['diagnostics']
        assert 'propensity_score_model' in results['diagnostics']

    def test_estimate_effect_synthetic_data_att(self):
        """Test the end-to-end estimate_effect with synthetic data for ATT."""
        df = generate_synthetic_psm_data(n_samples=2000, treatment_effect=5.0, seed=456)
        covariates = ['X1', 'X2']
        treatment = 'treatment'
        outcome = 'outcome'
        
        # Run the weighting estimator for ATT
        results = ps_weighting.estimate_effect(df, treatment, outcome, covariates, weight_type='ATT')
        
        # Check if the estimated effect is reasonably close to the true effect
        true_effect = 5.0
        estimated_effect = results['effect_estimate']
        assert abs(estimated_effect - true_effect) < 1.0 # Adjust tolerance

    @patch('auto_causal.methods.propensity_score.weighting.get_llm_parameters')
    @patch('auto_causal.methods.propensity_score.weighting.determine_optimal_weight_type')
    @patch('auto_causal.methods.propensity_score.weighting.determine_optimal_trim_threshold')
    @patch('auto_causal.methods.propensity_score.weighting.select_propensity_model')
    def test_llm_parameter_usage(self, mock_select_model, mock_determine_trim, mock_determine_weight, mock_get_llm_params):
        """Test that LLM helper functions are called and their results are potentially used."""
        df = generate_synthetic_psm_data(n_samples=100, seed=789) # Smaller sample
        covariates = ['X1', 'X2']
        treatment = 'treatment'
        outcome = 'outcome'
        query = "What is the ATT?"

        # Configure mocks
        mock_get_llm_params.return_value = {
            "parameters": {"weight_type": "ATT", "trim_threshold": 0.01}, 
            "validation": {}
        }
        mock_determine_weight.return_value = 'ATE' # Fallback
        mock_determine_trim.return_value = None # Fallback (no trim)
        mock_select_model.return_value = 'logistic' # Fallback

        # Call the function
        results = ps_weighting.estimate_effect(df, treatment, outcome, covariates, query=query)

        # Assertions
        mock_get_llm_params.assert_called_once_with(df, query, "PS.Weighting")
        # Helpers should not be called if LLM provided the value
        mock_determine_weight.assert_not_called()
        mock_determine_trim.assert_not_called()
        # Model selection will still be called as it wasn't in mock LLM params
        mock_select_model.assert_called_once()

        # Check parameters reflect LLM suggestions
        assert results['parameters']['weight_type'] == 'ATT'
        assert results['parameters']['trim_threshold'] == 0.01
        assert results['parameters']['propensity_model'] == 'logistic'

    # TODO: Add tests for weight trimming effects
    # TODO: Add tests for diagnostic outputs (e.g., checking weight distribution stats)
    # TODO: Add tests for edge cases (e.g., extreme weights) 
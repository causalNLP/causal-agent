"""Unit tests for propensity score matching."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from causal_agent.methods.propensity_score.matching import estimate_effect, _perform_matching_and_get_att
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, DatasetType, SyntheticDataConfig
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestPropensityScoreMatching(MethodTestCase):
    """Test cases for propensity score matching."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=300, treatment_effect=2.0, random_seed=123)
        )
        self.observational_data = self.generator.generate_observational_data()
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    def test_estimate_effect_basic(self, mock_assess_balance, mock_llm_params):
        """Test basic PSM estimation."""
        # Setup mocks
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_balance.return_value = {"balance_score": 0.8}
        
        covariates = self.observational_data.attrs.get('confounders', ['confounder_0', 'confounder_1'])
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates
        )
        
        # Validate result structure
        expected_keys = [
            'effect_estimate', 'effect_se', 'confidence_interval',
            'diagnostics', 'method_details', 'parameters'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check types and ranges
        self.assertIsInstance(results['effect_estimate'], (int, float))
        self.assertIsInstance(results['effect_se'], (int, float))
        self.assertGreater(results['effect_se'], 0.0)
        
        # Check confidence interval structure
        ci = results['confidence_interval']
        self.assertIsInstance(ci, list)
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])
        
        # Check method details
        self.assertIn("PSM", results['method_details'])
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    def test_estimate_effect_with_parameters(self, mock_assess_balance, mock_llm_params):
        """Test PSM with specific parameters."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_balance.return_value = {"balance_score": 0.75}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            caliper=0.1,
            n_neighbors=2,
            n_bootstraps=50,
            propensity_model_type='logistic'
        )
        
        # Check that parameters are reflected in results
        params = results['parameters']
        self.assertEqual(params['caliper'], 0.1)
        self.assertEqual(params['n_neighbors'], 2)
        self.assertEqual(params['n_bootstraps_config'], 50)
        self.assertEqual(params['propensity_model'], 'logistic')
    
    def test_perform_matching_and_get_att_basic(self):
        """Test the helper function for ATT calculation."""
        covariates = ['confounder_0', 'confounder_1']
        
        att = _perform_matching_and_get_att(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            propensity_model_type='logistic',
            n_neighbors=1,
            caliper=None,
            perform_bias_adjustment=False
        )
        
        # Should return a numeric ATT estimate
        self.assertIsInstance(att, (int, float))
        self.assertFalse(np.isnan(att))
    
    def test_perform_matching_and_get_att_with_bias_adjustment(self):
        """Test ATT calculation with bias adjustment."""
        covariates = ['confounder_0', 'confounder_1']
        
        att = _perform_matching_and_get_att(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            propensity_model_type='logistic',
            n_neighbors=1,
            caliper=0.2,
            perform_bias_adjustment=True
        )
        
        # Should return a numeric ATT estimate
        self.assertIsInstance(att, (int, float))
        self.assertFalse(np.isnan(att))
    
    def test_perform_matching_and_get_att_empty_groups(self):
        """Test ATT calculation with empty treatment groups."""
        # Create data with only treated units
        treated_only_data = pd.DataFrame({
            'treatment': [1, 1, 1, 1],
            'outcome': [2, 3, 4, 5],
            'confounder_0': [0.1, 0.2, 0.3, 0.4],
            'confounder_1': [1.1, 1.2, 1.3, 1.4]
        })
        
        att = _perform_matching_and_get_att(
            treated_only_data,
            'treatment',
            'outcome',
            ['confounder_0', 'confounder_1'],
            propensity_model_type='logistic',
            n_neighbors=1,
            caliper=None,
            perform_bias_adjustment=False
        )
        
        # Should return NaN when no control units
        self.assertTrue(np.isnan(att))
    
    def test_perform_matching_and_get_att_with_caliper(self):
        """Test ATT calculation with caliper restriction."""
        covariates = ['confounder_0', 'confounder_1']
        
        # Test with very restrictive caliper
        att_restrictive = _perform_matching_and_get_att(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            propensity_model_type='logistic',
            n_neighbors=1,
            caliper=0.01,  # Very restrictive
            perform_bias_adjustment=False
        )
        
        # Test with permissive caliper
        att_permissive = _perform_matching_and_get_att(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            propensity_model_type='logistic',
            n_neighbors=1,
            caliper=1.0,  # Very permissive
            perform_bias_adjustment=False
        )
        
        # Both should be numeric (though restrictive might be NaN if no matches)
        self.assertTrue(isinstance(att_restrictive, (int, float)))
        self.assertTrue(isinstance(att_permissive, (int, float)))
        self.assertFalse(np.isnan(att_permissive))
    
    @pytest.mark.parametrize("n_neighbors", [1, 2, 3])
    def test_perform_matching_different_neighbors(self, n_neighbors):
        """Test ATT calculation with different numbers of neighbors."""
        covariates = ['confounder_0', 'confounder_1']
        
        att = _perform_matching_and_get_att(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            propensity_model_type='logistic',
            n_neighbors=n_neighbors,
            caliper=None,
            perform_bias_adjustment=False
        )
        
        # Should work with different neighbor counts
        self.assertIsInstance(att, (int, float))
        self.assertFalse(np.isnan(att))
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    def test_estimate_effect_bootstrap_se(self, mock_assess_balance, mock_llm_params):
        """Test that bootstrap standard errors are calculated."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_balance.return_value = {"balance_score": 0.8}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            n_bootstraps=20  # Small number for testing
        )
        
        # Should have positive standard error from bootstrap
        self.assertGreater(results['effect_se'], 0.0)
        
        # Check diagnostics include bootstrap info
        diagnostics = results['diagnostics']
        self.assertIn('bootstrap_iterations_for_se', diagnostics)
        self.assertGreater(diagnostics['bootstrap_iterations_for_se'], 0)
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    def test_estimate_effect_missing_columns(self, mock_assess_balance, mock_llm_params):
        """Test error handling for missing columns."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_balance.return_value = {"balance_score": 0.8}
        
        with self.assertRaises(KeyError):
            estimate_effect(
                self.observational_data,
                'missing_treatment',
                'outcome',
                ['confounder_0']
            )
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    def test_estimate_effect_with_query(self, mock_assess_balance, mock_llm_params):
        """Test PSM with query parameter."""
        mock_llm_params.return_value = {
            "parameters": {
                "caliper": 0.15,
                "n_neighbors": 2,
                "propensity_model_type": "logistic"
            }
        }
        mock_assess_balance.return_value = {"balance_score": 0.85}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            query="What is the effect of treatment on outcome?"
        )
        
        # Should use LLM-suggested parameters
        params = results['parameters']
        self.assertEqual(params['caliper'], 0.15)
        self.assertEqual(params['n_neighbors'], 2)
    
    def test_perform_matching_nan_propensity_scores(self):
        """Test handling of NaN propensity scores."""
        # Create data that might lead to NaN propensity scores
        problematic_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'outcome': [1, 2, 1, 2, 1, 2],
            'confounder_0': [np.nan, 0.5, np.nan, 1.0, np.nan, 1.5],  # NaN values
            'confounder_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        })
        
        att = _perform_matching_and_get_att(
            problematic_data,
            'treatment',
            'outcome',
            ['confounder_0', 'confounder_1'],
            propensity_model_type='logistic',
            n_neighbors=1,
            caliper=None,
            perform_bias_adjustment=False
        )
        
        # Should handle NaN gracefully (might return NaN or valid estimate)
        self.assertIsInstance(att, (int, float))
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    @patch('causal_agent.methods.propensity_score.matching.CausalModel')
    def test_estimate_effect_dowhy_fallback(self, mock_causal_model, mock_assess_balance, mock_llm_params):
        """Test fallback to custom PSM when DoWhy fails."""
        # Setup mocks
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_balance.return_value = {"balance_score": 0.8}
        
        # Mock DoWhy to raise an exception
        mock_causal_model.side_effect = Exception("DoWhy PSM failed")
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates
        )
        
        # Should still return valid results using fallback
        self.assertIn('effect_estimate', results)
        self.assertIsInstance(results['effect_estimate'], (int, float))
        
        # Check that fallback method is indicated
        diagnostics = results['diagnostics']
        self.assertEqual(diagnostics['att_estimation_method'], 'Fallback Custom PSM')
    
    @patch('causal_agent.methods.propensity_score.matching.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.matching.assess_balance')
    def test_estimate_effect_diagnostics_structure(self, mock_assess_balance, mock_llm_params):
        """Test that diagnostics have expected structure."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_balance.return_value = {
            "balance_score": 0.8,
            "standardized_mean_differences": {"confounder_0": 0.1, "confounder_1": 0.05}
        }
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates
        )
        
        diagnostics = results['diagnostics']
        
        # Check expected diagnostic keys
        expected_diagnostic_keys = [
            'att_estimation_method',
            'propensity_score_model',
            'bootstrap_iterations_for_se',
            'final_caliper_used',
            'unmatched_treated_count',
            'percent_treated_matched'
        ]
        
        for key in expected_diagnostic_keys:
            self.assertIn(key, diagnostics)
        
        # Check types
        self.assertIsInstance(diagnostics['unmatched_treated_count'], int)
        self.assertIsInstance(diagnostics['percent_treated_matched'], (int, float))
        self.assertGreaterEqual(diagnostics['percent_treated_matched'], 0.0)
        self.assertLessEqual(diagnostics['percent_treated_matched'], 100.0)


if __name__ == '__main__':
    pytest.main([__file__])
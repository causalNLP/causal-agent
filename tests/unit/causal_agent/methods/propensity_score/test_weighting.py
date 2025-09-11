"""Unit tests for propensity score weighting."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from causal_agent.methods.propensity_score.weighting import estimate_effect
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, DatasetType, SyntheticDataConfig
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestPropensityScoreWeighting(MethodTestCase):
    """Test cases for propensity score weighting (IPW)."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=400, treatment_effect=1.5, random_seed=456)
        )
        self.observational_data = self.generator.generate_observational_data()
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_ate_basic(self, mock_assess_weights, mock_llm_params):
        """Test basic ATE estimation using IPW."""
        # Setup mocks
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = self.observational_data.attrs.get('confounders', ['confounder_0', 'confounder_1'])
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            weight_type='ATE'
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
        self.assertEqual(results['method_details'], "PS.Weighting")
        
        # Check parameters
        params = results['parameters']
        self.assertEqual(params['weight_type'], 'ATE')
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_att_basic(self, mock_assess_weights, mock_llm_params):
        """Test basic ATT estimation using IPW."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            weight_type='ATT'
        )
        
        # Validate basic structure
        self.assertIn('effect_estimate', results)
        self.assertIn('effect_se', results)
        
        # Check parameters reflect ATT
        params = results['parameters']
        self.assertEqual(params['weight_type'], 'ATT')
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_with_trimming(self, mock_assess_weights, mock_llm_params):
        """Test IPW with propensity score trimming."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            weight_type='ATE',
            trim_threshold=0.05  # Trim 5% from each tail
        )
        
        # Should complete successfully with trimming
        self.assertIn('effect_estimate', results)
        
        # Check that trimming parameter is recorded
        params = results['parameters']
        self.assertEqual(params['trim_threshold'], 0.05)
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_different_propensity_models(self, mock_assess_weights, mock_llm_params):
        """Test IPW with different propensity score models."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        # Test with logistic regression
        results_logistic = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            propensity_model_type='logistic'
        )
        
        # Should work with logistic model
        self.assertIn('effect_estimate', results_logistic)
        params = results_logistic['parameters']
        self.assertEqual(params['propensity_model'], 'logistic')
    
    def test_estimate_effect_unsupported_weight_type(self):
        """Test error handling for unsupported weight types."""
        covariates = ['confounder_0', 'confounder_1']
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.observational_data,
                'treatment',
                'outcome',
                covariates,
                weight_type='UNSUPPORTED'
            )
        
        self.assertIn("Unsupported weight type", str(context.exception))
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_robust_se(self, mock_assess_weights, mock_llm_params):
        """Test IPW with robust standard errors."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        # Test with robust SE enabled
        results_robust = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            robust_se=True
        )
        
        # Test with robust SE disabled
        results_non_robust = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            robust_se=False
        )
        
        # Both should work
        self.assertIn('effect_estimate', results_robust)
        self.assertIn('effect_estimate', results_non_robust)
        
        # Check parameter recording
        self.assertTrue(results_robust['parameters']['robust_se'])
        self.assertFalse(results_non_robust['parameters']['robust_se'])
    
    @pytest.mark.parametrize("weight_type", ["ATE", "ATT"])
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_parametrized_weight_types(self, mock_assess_weights, mock_llm_params, weight_type):
        """Test IPW with different weight types parametrically."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            weight_type=weight_type
        )
        
        # Should work for both weight types
        self.assertIn('effect_estimate', results)
        self.assertIsInstance(results['effect_estimate'], (int, float))
        self.assertFalse(np.isnan(results['effect_estimate']))
        
        # Check weight type is recorded correctly
        self.assertEqual(results['parameters']['weight_type'], weight_type)
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_extreme_trimming(self, mock_assess_weights, mock_llm_params):
        """Test IPW with extreme trimming that removes all data."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "poor"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.observational_data,
                'treatment',
                'outcome',
                covariates,
                trim_threshold=0.5  # Remove 50% from each tail (100% total)
            )
        
        self.assertIn("All units removed after trimming", str(context.exception))
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_with_query_and_llm_params(self, mock_assess_weights, mock_llm_params):
        """Test IPW with query and LLM-suggested parameters."""
        # Mock LLM to suggest specific parameters
        mock_llm_params.return_value = {
            "parameters": {
                "weight_type": "ATT",
                "trim_threshold": 0.1,
                "propensity_model_type": "logistic"
            }
        }
        mock_assess_weights.return_value = {"weight_quality": "excellent"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates,
            query="What is the average treatment effect on the treated?"
        )
        
        # Should use LLM-suggested parameters
        params = results['parameters']
        self.assertEqual(params['weight_type'], 'ATT')
        self.assertEqual(params['trim_threshold'], 0.1)
        self.assertEqual(params['propensity_model'], 'logistic')
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_missing_columns(self, mock_assess_weights, mock_llm_params):
        """Test error handling for missing columns."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        with self.assertRaises(KeyError):
            estimate_effect(
                self.observational_data,
                'missing_treatment',
                'outcome',
                ['confounder_0']
            )
        
        with self.assertRaises(KeyError):
            estimate_effect(
                self.observational_data,
                'treatment',
                'missing_outcome',
                ['confounder_0']
            )
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_diagnostics_structure(self, mock_assess_weights, mock_llm_params):
        """Test that diagnostics have expected structure."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {
            "weight_quality": "good",
            "mean_weight": 1.0,
            "max_weight": 5.2,
            "weight_variance": 2.1
        }
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates
        )
        
        diagnostics = results['diagnostics']
        
        # Check that weight assessment results are included
        self.assertIn('weight_quality', diagnostics)
        self.assertIn('propensity_score_model', diagnostics)
        
        # Check that propensity model is recorded
        self.assertIsInstance(diagnostics['propensity_score_model'], str)
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_confidence_interval_calculation(self, mock_assess_weights, mock_llm_params):
        """Test confidence interval calculation."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "good"}
        
        covariates = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.observational_data,
            'treatment',
            'outcome',
            covariates
        )
        
        # Check confidence interval properties
        effect = results['effect_estimate']
        se = results['effect_se']
        ci = results['confidence_interval']
        
        # CI should be approximately effect ± 1.96 * SE
        expected_lower = effect - 1.96 * se
        expected_upper = effect + 1.96 * se
        
        self.assertAlmostEqual(ci[0], expected_lower, places=6)
        self.assertAlmostEqual(ci[1], expected_upper, places=6)
        
        # Effect should be within CI
        self.assertGreaterEqual(effect, ci[0])
        self.assertLessEqual(effect, ci[1])
    
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_small_dataset(self, mock_assess_weights, mock_llm_params):
        """Test IPW with very small dataset."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "poor"}
        
        # Create minimal dataset
        small_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'outcome': [1, 3, 2, 4, 1.5, 3.5],
            'confounder_0': [0.1, 0.8, 0.2, 0.9, 0.15, 0.85],
            'confounder_1': [1.1, 1.8, 1.2, 1.9, 1.15, 1.85]
        })
        
        results = estimate_effect(
            small_data,
            'treatment',
            'outcome',
            ['confounder_0', 'confounder_1']
        )
        
        # Should handle small dataset gracefully
        self.assertIn('effect_estimate', results)
        self.assertIsInstance(results['effect_estimate'], (int, float))
    
    @patch('causal_agent.methods.propensity_score.weighting.estimate_propensity_scores')
    @patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters')
    @patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_propensity_score_failure(self, mock_assess_weights, mock_llm_params, mock_ps_estimation):
        """Test handling of propensity score estimation failure."""
        mock_llm_params.return_value = {"parameters": {}}
        mock_assess_weights.return_value = {"weight_quality": "poor"}
        
        # Mock propensity score estimation to fail
        mock_ps_estimation.side_effect = Exception("Propensity score estimation failed")
        
        covariates = ['confounder_0', 'confounder_1']
        
        with self.assertRaises(Exception):
            estimate_effect(
                self.observational_data,
                'treatment',
                'outcome',
                covariates
            )


if __name__ == '__main__':
    pytest.main([__file__])
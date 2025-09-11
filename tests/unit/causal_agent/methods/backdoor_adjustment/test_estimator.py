"""Unit tests for backdoor adjustment estimator."""

import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary import Summary
from unittest.mock import patch, MagicMock

from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, DatasetType, SyntheticDataConfig
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestBackdoorAdjustmentEstimator(MethodTestCase):
    """Test cases for backdoor adjustment estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=200, treatment_effect=3.0, random_seed=789)
        )
        self.confounded_data = self.generator.generate_observational_data()
    
    @patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics')
    @patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results')
    def test_estimate_effect_basic(self, mock_interpret, mock_diagnostics):
        """Test basic execution with a valid adjustment set."""
        # Setup mocks
        mock_diagnostics.return_value = {"status": "Success", "details": {}}
        mock_interpret.return_value = "LLM Interpretation"
        
        # Get confounders from dataset metadata
        adjustment_set = self.confounded_data.attrs.get('confounders', ['confounder_0', 'confounder_1'])
        
        # Run estimation
        results = estimate_effect(
            self.confounded_data, 
            'treatment', 
            'outcome', 
            adjustment_set
        )
        
        # Validate result structure
        self.validate_method_output(results, 'Backdoor Adjustment (OLS)')
        
        # Check specific keys
        expected_keys = [
            'effect_estimate', 'p_value', 'confidence_interval', 
            'standard_error', 'formula', 'model_summary', 
            'diagnostics', 'interpretation', 'method_used'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check if effect estimate is reasonably close to the true effect (3.0)
        self.assertIsInstance(results['effect_estimate'], (int, float))
        self.assertLess(abs(results['effect_estimate'] - 3.0), 1.5)  # Allow some variance
        
        # Check formula construction
        expected_formula_parts = ['outcome', 'treatment'] + adjustment_set + ['const']
        for part in expected_formula_parts:
            self.assertIn(part, results['formula'])
        
        # Check model summary type
        self.assertIsInstance(results['model_summary'], Summary)
        
        # Verify mocks were called
        mock_diagnostics.assert_called_once()
        mock_interpret.assert_called_once()
    
    def test_estimate_effect_missing_treatment(self):
        """Test error handling for missing treatment column."""
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.confounded_data, 
                'missing_treatment', 
                'outcome', 
                ['confounder_0']
            )
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("missing_treatment", str(context.exception))
    
    def test_estimate_effect_missing_outcome(self):
        """Test error handling for missing outcome column."""
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.confounded_data, 
                'treatment', 
                'missing_outcome', 
                ['confounder_0']
            )
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("missing_outcome", str(context.exception))
    
    def test_estimate_effect_missing_covariate(self):
        """Test error handling for missing covariate column."""
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.confounded_data, 
                'treatment', 
                'outcome', 
                ['confounder_0', 'missing_covariate']
            )
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("missing_covariate", str(context.exception))
    
    def test_estimate_effect_empty_covariates(self):
        """Test error handling when covariate list is empty."""
        with self.assertRaises(ValueError) as context:
            estimate_effect(self.confounded_data, 'treatment', 'outcome', [])
        self.assertIn("non-empty list of covariates", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(self.confounded_data, 'treatment', 'outcome', None)
        self.assertIn("non-empty list of covariates", str(context.exception))
    
    def test_estimate_effect_nan_data(self):
        """Test handling of data with NaNs resulting in empty analysis set."""
        # Create data where all rows have NaN in required columns
        df_nan = pd.DataFrame({
            'outcome': [np.nan, 2, 3, 4],
            'treatment': [0, np.nan, 1, 1],
            'covariate1': [5, 6, np.nan, np.nan]
        })
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(df_nan, 'treatment', 'outcome', ['covariate1'])
        self.assertIn("No data remaining after dropping NaNs", str(context.exception))
    
    @patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics')
    @patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results')
    def test_estimate_effect_with_llm(self, mock_interpret, mock_diagnostics):
        """Test estimation with LLM integration."""
        # Setup mocks
        mock_diagnostics.return_value = {"status": "Success", "details": {}}
        mock_interpret.return_value = "Detailed LLM interpretation of results"
        mock_llm = MagicMock()
        
        adjustment_set = ['confounder_0', 'confounder_1']
        
        results = estimate_effect(
            self.confounded_data,
            'treatment',
            'outcome',
            adjustment_set,
            query="What is the effect of treatment on outcome?",
            llm=mock_llm
        )
        
        # Verify LLM was passed to interpretation function
        mock_interpret.assert_called_once()
        call_args = mock_interpret.call_args
        self.assertEqual(call_args[1]['llm'], mock_llm)
    
    @pytest.mark.parametrize("n_samples,treatment_effect", [
        (100, 0.5),
        (500, 1.0),
        (1000, 2.0),
    ])
    @patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics')
    @patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results')
    def test_estimate_effect_parametrized(self, mock_interpret, mock_diagnostics, n_samples, treatment_effect):
        """Test estimation with different sample sizes and effect sizes."""
        mock_diagnostics.return_value = {"status": "Success", "details": {}}
        mock_interpret.return_value = "Parametrized test interpretation"
        
        # Generate data with specific parameters
        config = SyntheticDataConfig(
            n_samples=n_samples, 
            treatment_effect=treatment_effect,
            random_seed=42
        )
        generator = SyntheticDataGenerator(config)
        data = generator.generate_observational_data()
        
        adjustment_set = data.attrs.get('confounders', ['confounder_0', 'confounder_1'])
        
        results = estimate_effect(data, 'treatment', 'outcome', adjustment_set)
        
        # Validate basic structure
        self.validate_method_output(results, 'Backdoor Adjustment (OLS)')
        
        # Check that effect estimate is in reasonable range
        self.assertIsInstance(results['effect_estimate'], (int, float))
        # Allow for estimation error, especially with smaller samples
        tolerance = max(0.5, treatment_effect * 0.5)
        self.assertLess(abs(results['effect_estimate'] - treatment_effect), tolerance)
    
    def test_estimate_effect_confidence_interval_structure(self):
        """Test that confidence intervals have correct structure."""
        with patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics') as mock_diag, \
             patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results') as mock_interp:
            
            mock_diag.return_value = {"status": "Success", "details": {}}
            mock_interp.return_value = "Test interpretation"
            
            adjustment_set = ['confounder_0', 'confounder_1']
            results = estimate_effect(
                self.confounded_data, 
                'treatment', 
                'outcome', 
                adjustment_set
            )
            
            # Check confidence interval structure
            ci = results['confidence_interval']
            self.assertIsInstance(ci, list)
            self.assertEqual(len(ci), 2)
            self.assertLess(ci[0], ci[1])  # Lower bound < upper bound
            
            # Effect estimate should be within confidence interval
            effect = results['effect_estimate']
            self.assertGreaterEqual(effect, ci[0])
            self.assertLessEqual(effect, ci[1])
    
    def test_estimate_effect_statistical_properties(self):
        """Test statistical properties of the estimation."""
        with patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics') as mock_diag, \
             patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results') as mock_interp:
            
            mock_diag.return_value = {"status": "Success", "details": {}}
            mock_interp.return_value = "Statistical properties test"
            
            adjustment_set = ['confounder_0', 'confounder_1']
            results = estimate_effect(
                self.confounded_data, 
                'treatment', 
                'outcome', 
                adjustment_set
            )
            
            # Check statistical properties
            self.assertIsInstance(results['p_value'], (int, float))
            self.assertGreaterEqual(results['p_value'], 0.0)
            self.assertLessEqual(results['p_value'], 1.0)
            
            self.assertIsInstance(results['standard_error'], (int, float))
            self.assertGreater(results['standard_error'], 0.0)
    
    def test_estimate_effect_regression_failure(self):
        """Test handling of regression failures."""
        # Create problematic data (perfect multicollinearity)
        problematic_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'outcome': [1, 2, 1, 2],
            'covariate1': [1, 2, 1, 2],
            'covariate2': [1, 2, 1, 2]  # Identical to covariate1
        })
        
        # This should raise an exception due to perfect multicollinearity
        with self.assertRaises(Exception):
            estimate_effect(
                problematic_data, 
                'treatment', 
                'outcome', 
                ['covariate1', 'covariate2']
            )


if __name__ == '__main__':
    pytest.main([__file__])
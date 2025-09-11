"""Unit tests for propensity score base functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from causal_agent.methods.propensity_score.base import (
    estimate_propensity_scores,
    format_ps_results,
    select_propensity_model
)
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


class TestPropensityScoreBase(MethodTestCase):
    """Test cases for propensity score base functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=200, treatment_effect=1.0, random_seed=789)
        )
        self.observational_data = self.generator.generate_observational_data()
    
    def test_estimate_propensity_scores_logistic(self):
        """Test propensity score estimation with logistic regression."""
        covariates = ['confounder_0', 'confounder_1']
        
        ps_scores = estimate_propensity_scores(
            self.observational_data,
            'treatment',
            covariates,
            model_type='logistic'
        )
        
        # Check output properties
        self.assertIsInstance(ps_scores, np.ndarray)
        self.assertEqual(len(ps_scores), len(self.observational_data))
        
        # Check that scores are probabilities (between 0 and 1)
        self.assertTrue(np.all(ps_scores >= 0.0))
        self.assertTrue(np.all(ps_scores <= 1.0))
        
        # Check that scores are clipped (not exactly 0 or 1)
        self.assertTrue(np.all(ps_scores >= 0.01))
        self.assertTrue(np.all(ps_scores <= 0.99))
        
        # Check that there's variation in scores
        self.assertGreater(np.std(ps_scores), 0.01)
    
    def test_estimate_propensity_scores_with_parameters(self):
        """Test propensity score estimation with custom parameters."""
        covariates = ['confounder_0', 'confounder_1']
        
        ps_scores = estimate_propensity_scores(
            self.observational_data,
            'treatment',
            covariates,
            model_type='logistic',
            max_iter=2000,
            C=0.5,
            penalty='l1',
            solver='liblinear'
        )
        
        # Should still produce valid propensity scores
        self.assertIsInstance(ps_scores, np.ndarray)
        self.assertEqual(len(ps_scores), len(self.observational_data))
        self.assertTrue(np.all(ps_scores >= 0.01))
        self.assertTrue(np.all(ps_scores <= 0.99))
    
    def test_estimate_propensity_scores_unsupported_model(self):
        """Test error handling for unsupported model types."""
        covariates = ['confounder_0', 'confounder_1']
        
        with self.assertRaises(ValueError) as context:
            estimate_propensity_scores(
                self.observational_data,
                'treatment',
                covariates,
                model_type='unsupported_model'
            )
        
        self.assertIn("Unsupported propensity score model type", str(context.exception))
    
    def test_estimate_propensity_scores_missing_columns(self):
        """Test error handling for missing columns."""
        with self.assertRaises(KeyError):
            estimate_propensity_scores(
                self.observational_data,
                'missing_treatment',
                ['confounder_0']
            )
        
        with self.assertRaises(KeyError):
            estimate_propensity_scores(
                self.observational_data,
                'treatment',
                ['missing_covariate']
            )
    
    def test_estimate_propensity_scores_single_covariate(self):
        """Test propensity score estimation with single covariate."""
        ps_scores = estimate_propensity_scores(
            self.observational_data,
            'treatment',
            ['confounder_0'],
            model_type='logistic'
        )
        
        # Should work with single covariate
        self.assertIsInstance(ps_scores, np.ndarray)
        self.assertEqual(len(ps_scores), len(self.observational_data))
        self.assertTrue(np.all(ps_scores >= 0.01))
        self.assertTrue(np.all(ps_scores <= 0.99))
    
    def test_estimate_propensity_scores_perfect_separation(self):
        """Test propensity score estimation with perfect separation."""
        # Create data with perfect separation
        perfect_sep_data = pd.DataFrame({
            'treatment': [0, 0, 0, 1, 1, 1],
            'covariate': [0, 0.1, 0.2, 0.8, 0.9, 1.0]  # Clear separation
        })
        
        # Should handle perfect separation gracefully
        ps_scores = estimate_propensity_scores(
            perfect_sep_data,
            'treatment',
            ['covariate'],
            model_type='logistic'
        )
        
        # Should still produce clipped scores
        self.assertTrue(np.all(ps_scores >= 0.01))
        self.assertTrue(np.all(ps_scores <= 0.99))
    
    def test_estimate_propensity_scores_with_nan_data(self):
        """Test propensity score estimation with NaN data."""
        # Create data with NaN values
        nan_data = self.observational_data.copy()
        nan_data.loc[0, 'confounder_0'] = np.nan
        nan_data.loc[1, 'confounder_1'] = np.nan
        
        # Should handle NaN values (sklearn will handle this)
        with self.assertRaises(ValueError):
            estimate_propensity_scores(
                nan_data,
                'treatment',
                ['confounder_0', 'confounder_1'],
                model_type='logistic'
            )
    
    def test_format_ps_results_basic(self):
        """Test basic result formatting."""
        effect_estimate = 0.5
        effect_se = 0.1
        diagnostics = {"balance_score": 0.8}
        method_details = "Test Method"
        parameters = {"param1": "value1"}
        
        results = format_ps_results(
            effect_estimate,
            effect_se,
            diagnostics,
            method_details,
            parameters
        )
        
        # Check structure
        expected_keys = [
            'effect_estimate', 'effect_se', 'confidence_interval',
            'diagnostics', 'method_details', 'parameters'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check values
        self.assertEqual(results['effect_estimate'], 0.5)
        self.assertEqual(results['effect_se'], 0.1)
        self.assertEqual(results['method_details'], "Test Method")
        self.assertEqual(results['parameters'], {"param1": "value1"})
        self.assertEqual(results['diagnostics'], {"balance_score": 0.8})
    
    def test_format_ps_results_confidence_interval(self):
        """Test confidence interval calculation in result formatting."""
        effect_estimate = 1.0
        effect_se = 0.2
        
        results = format_ps_results(
            effect_estimate,
            effect_se,
            {},
            "Test",
            {}
        )
        
        # Check confidence interval calculation (effect ± 1.96 * SE)
        expected_lower = 1.0 - 1.96 * 0.2
        expected_upper = 1.0 + 1.96 * 0.2
        
        ci = results['confidence_interval']
        self.assertAlmostEqual(ci[0], expected_lower, places=6)
        self.assertAlmostEqual(ci[1], expected_upper, places=6)
    
    def test_format_ps_results_type_conversion(self):
        """Test that result formatting converts types appropriately."""
        # Use numpy types that should be converted to Python types
        effect_estimate = np.float64(0.75)
        effect_se = np.float32(0.15)
        
        results = format_ps_results(
            effect_estimate,
            effect_se,
            {},
            "Test",
            {}
        )
        
        # Check that values are converted to Python float
        self.assertIsInstance(results['effect_estimate'], float)
        self.assertIsInstance(results['effect_se'], float)
        self.assertIsInstance(results['confidence_interval'][0], float)
        self.assertIsInstance(results['confidence_interval'][1], float)
    
    def test_select_propensity_model_default(self):
        """Test default propensity model selection."""
        covariates = ['confounder_0', 'confounder_1']
        
        model_type = select_propensity_model(
            self.observational_data,
            'treatment',
            covariates
        )
        
        # Should return default model type
        self.assertEqual(model_type, 'logistic')
    
    def test_select_propensity_model_with_query(self):
        """Test propensity model selection with query."""
        covariates = ['confounder_0', 'confounder_1']
        
        model_type = select_propensity_model(
            self.observational_data,
            'treatment',
            covariates,
            query="What is the best model for this data?"
        )
        
        # Should still return default (placeholder implementation)
        self.assertEqual(model_type, 'logistic')
    
    @pytest.mark.parametrize("n_samples,n_features", [
        (50, 2),
        (200, 3),
        (500, 5),
    ])
    def test_estimate_propensity_scores_different_sizes(self, n_samples, n_features):
        """Test propensity score estimation with different dataset sizes."""
        # Generate data with specific size
        config = SyntheticDataConfig(
            n_samples=n_samples,
            n_features=n_features,
            random_seed=42
        )
        generator = SyntheticDataGenerator(config)
        data = generator.generate_observational_data()
        
        covariates = [f'confounder_{i}' for i in range(n_features)]
        
        ps_scores = estimate_propensity_scores(
            data,
            'treatment',
            covariates,
            model_type='logistic'
        )
        
        # Should work regardless of size
        self.assertEqual(len(ps_scores), n_samples)
        self.assertTrue(np.all(ps_scores >= 0.01))
        self.assertTrue(np.all(ps_scores <= 0.99))
    
    def test_estimate_propensity_scores_binary_treatment_validation(self):
        """Test that propensity score estimation works with binary treatment."""
        covariates = ['confounder_0', 'confounder_1']
        
        # Verify treatment is binary
        treatment_values = self.observational_data['treatment'].unique()
        self.assertEqual(set(treatment_values), {0, 1})
        
        ps_scores = estimate_propensity_scores(
            self.observational_data,
            'treatment',
            covariates,
            model_type='logistic'
        )
        
        # Should produce valid scores for binary treatment
        self.assertIsInstance(ps_scores, np.ndarray)
        self.assertTrue(np.all(ps_scores >= 0.01))
        self.assertTrue(np.all(ps_scores <= 0.99))
    
    def test_estimate_propensity_scores_convergence_issues(self):
        """Test handling of convergence issues in propensity score estimation."""
        covariates = ['confounder_0', 'confounder_1']
        
        # Use very low max_iter to potentially cause convergence issues
        ps_scores = estimate_propensity_scores(
            self.observational_data,
            'treatment',
            covariates,
            model_type='logistic',
            max_iter=1  # Very low iteration limit
        )
        
        # Should still produce scores (sklearn handles convergence warnings)
        self.assertIsInstance(ps_scores, np.ndarray)
        self.assertEqual(len(ps_scores), len(self.observational_data))


if __name__ == '__main__':
    pytest.main([__file__])
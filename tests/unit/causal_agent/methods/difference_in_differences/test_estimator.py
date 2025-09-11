"""Unit tests for difference-in-differences estimator."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from causal_agent.methods.difference_in_differences.estimator import (
    estimate_effect,
    format_did_results
)
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, DatasetType, SyntheticDataConfig
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestDifferenceInDifferencesEstimator(MethodTestCase):
    """Test cases for difference-in-differences estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(
                n_samples=400, 
                treatment_effect=1.5, 
                n_periods=20,
                n_units=50,
                treatment_start_period=10,
                random_seed=123
            )
        )
        self.did_data = self.generator.generate_did_data()
    
    def test_format_did_results_basic(self):
        """Test basic DiD result formatting."""
        # Mock statsmodels results
        mock_results = MagicMock()
        mock_results.params = {'did_interaction': 1.5}
        mock_results.bse = {'did_interaction': 0.2}
        mock_results.pvalues = {'did_interaction': 0.01}
        mock_results.conf_int.return_value = pd.DataFrame({
            0: [1.1], 1: [1.9]
        }, index=['did_interaction'])
        mock_results.summary.return_value = "Mock summary"
        
        validation_results = {"parallel_trends": {"valid": True}}
        parameters = {"time_var": "period", "group_var": "unit"}
        
        results = format_did_results(
            mock_results,
            'did_interaction',
            validation_results,
            "Test DiD Method",
            parameters
        )
        
        # Check structure
        expected_keys = [
            'effect_estimate', 'standard_error', 'p_value',
            'confidence_interval', 'diagnostics', 'parameters', 'details'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check values
        self.assertEqual(results['effect_estimate'], 1.5)
        self.assertEqual(results['standard_error'], 0.2)
        self.assertEqual(results['p_value'], 0.01)
        self.assertEqual(results['confidence_interval'], [1.1, 1.9])
    
    def test_format_did_results_missing_interaction_term(self):
        """Test DiD result formatting when interaction term is missing."""
        # Mock statsmodels results without the expected interaction term
        mock_results = MagicMock()
        mock_results.params = {'other_term': 1.0}
        mock_results.params.index.tolist.return_value = ['other_term']
        
        results = format_did_results(
            mock_results,
            'missing_interaction',
            {},
            "Test Method",
            {}
        )
        
        # Should handle missing term gracefully with NaN values
        self.assertTrue(np.isnan(results['effect_estimate']))
        self.assertTrue(np.isnan(results['standard_error']))
        self.assertTrue(np.isnan(results['p_value']))
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    def test_estimate_effect_basic_twfe(self, mock_interpret, mock_validate, mock_determine_period, 
                                       mock_identify_group, mock_identify_time, mock_get_llm):
        """Test basic DiD estimation with TWFE (multiple periods)."""
        # Setup mocks
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = 10
        mock_validate.return_value = {"valid": True, "test_statistic": 0.5}
        mock_interpret.return_value = "DiD interpretation"
        
        covariates = ['covariate']
        
        results = estimate_effect(
            self.did_data,
            'treatment',
            'outcome',
            covariates
        )
        
        # Check basic structure
        expected_keys = [
            'effect_estimate', 'standard_error', 'p_value',
            'confidence_interval', 'diagnostics', 'parameters'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check that TWFE method was used (multiple periods)
        self.assertIn('TWFE', results['parameters']['estimation_method'])
        
        # Check that effect estimate is numeric
        self.assertIsInstance(results['effect_estimate'], (int, float))
        self.assertFalse(np.isnan(results['effect_estimate']))
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    def test_estimate_effect_2x2_did(self, mock_interpret, mock_validate, mock_determine_period,
                                    mock_identify_group, mock_identify_time, mock_get_llm):
        """Test DiD estimation with 2x2 design (two periods)."""
        # Setup mocks
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = 1
        mock_validate.return_value = {"valid": True}
        mock_interpret.return_value = "2x2 DiD interpretation"
        
        # Create 2x2 DiD data (2 periods, 2 groups)
        did_2x2_data = pd.DataFrame({
            'unit': [1, 1, 2, 2, 3, 3, 4, 4],
            'period': [0, 1, 0, 1, 0, 1, 0, 1],
            'treatment': [0, 1, 0, 1, 0, 0, 0, 0],  # Treatment group indicator
            'outcome': [1, 3, 2, 4, 1.5, 2.5, 2, 3],
            'covariate': [0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.25, 0.25]
        })
        
        results = estimate_effect(
            did_2x2_data,
            'treatment',
            'outcome',
            ['covariate']
        )
        
        # Check that 2x2 method was used
        self.assertIn('2x2', results['parameters']['estimation_method'])
        
        # Check interaction term format for 2x2
        self.assertIn(':', results['parameters']['interaction_term_coefficient_name'])
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    def test_estimate_effect_missing_time_variable(self, mock_determine_period, mock_identify_group, 
                                                  mock_identify_time, mock_get_llm):
        """Test error handling when time variable cannot be identified."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = None  # Cannot identify time variable
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.did_data,
                'treatment',
                'outcome',
                ['covariate']
            )
        
        self.assertIn("Time variable could not be identified", str(context.exception))
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    def test_estimate_effect_missing_group_variable(self, mock_identify_group, mock_identify_time, mock_get_llm):
        """Test error handling when group variable cannot be identified."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = None  # Cannot identify group variable
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.did_data,
                'treatment',
                'outcome',
                ['covariate']
            )
        
        self.assertIn("Group/Unit variable could not be identified", str(context.exception))
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    def test_estimate_effect_missing_outcome(self, mock_validate, mock_determine_period,
                                           mock_identify_group, mock_identify_time, mock_get_llm):
        """Test error handling when outcome variable is missing."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = 10
        mock_validate.return_value = {"valid": True}
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.did_data,
                'treatment',
                'missing_outcome',
                ['covariate']
            )
        
        self.assertIn("Outcome variable 'missing_outcome' not found", str(context.exception))
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    def test_estimate_effect_with_kwargs(self, mock_interpret, mock_validate, mock_determine_period,
                                        mock_identify_group, mock_identify_time, mock_get_llm):
        """Test DiD estimation with explicit kwargs."""
        mock_get_llm.return_value = MagicMock()
        mock_validate.return_value = {"valid": True}
        mock_interpret.return_value = "Kwargs test interpretation"
        
        results = estimate_effect(
            self.did_data,
            'treatment',
            'outcome',
            ['covariate'],
            time_variable='period',
            group_variable='unit',
            treatment_period_start=10,
            query_str="What is the treatment effect?"
        )
        
        # Should use provided kwargs instead of calling identification functions
        mock_identify_time.assert_not_called()
        mock_identify_group.assert_not_called()
        mock_determine_period.assert_not_called()
        
        # Check that parameters reflect the provided kwargs
        params = results['parameters']
        self.assertEqual(params['time_var'], 'period')
        self.assertEqual(params['group_var'], 'unit')
        self.assertEqual(params['treatment_period_start'], 10)
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    def test_estimate_effect_parallel_trends_violation(self, mock_interpret, mock_validate, 
                                                      mock_determine_period, mock_identify_group, 
                                                      mock_identify_time, mock_get_llm):
        """Test DiD estimation when parallel trends assumption is violated."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = 10
        mock_validate.return_value = {"valid": False, "warning": "Parallel trends violated"}
        mock_interpret.return_value = "Interpretation with warning"
        
        results = estimate_effect(
            self.did_data,
            'treatment',
            'outcome',
            ['covariate']
        )
        
        # Should still produce results but with warning in diagnostics
        self.assertIn('effect_estimate', results)
        self.assertFalse(results['diagnostics']['parallel_trends']['valid'])
        self.assertIn('warning', results['diagnostics']['parallel_trends'])
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    def test_estimate_effect_binary_treatment_detection(self, mock_interpret, mock_validate,
                                                       mock_determine_period, mock_identify_group,
                                                       mock_identify_time, mock_get_llm):
        """Test detection of binary treatment group indicator."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = 10
        mock_validate.return_value = {"valid": True}
        mock_interpret.return_value = "Binary detection test"
        
        # Create data with explicit binary group column
        did_data_with_group = self.did_data.copy()
        did_data_with_group['group'] = (did_data_with_group['treated_unit']).astype(int)
        
        results = estimate_effect(
            did_data_with_group,
            'treatment',
            'outcome',
            ['covariate']
        )
        
        # Should detect and use the 'group' column
        params = results['parameters']
        self.assertEqual(params['treatment_indicator'], 'group')
    
    @pytest.mark.parametrize("n_periods,n_units", [
        (2, 10),   # 2x2 DiD
        (5, 20),   # Short panel
        (20, 50),  # Long panel
    ])
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    def test_estimate_effect_different_panel_dimensions(self, mock_interpret, mock_validate,
                                                       mock_determine_period, mock_identify_group,
                                                       mock_identify_time, mock_get_llm,
                                                       n_periods, n_units):
        """Test DiD estimation with different panel dimensions."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = n_periods // 2
        mock_validate.return_value = {"valid": True}
        mock_interpret.return_value = f"Panel test {n_periods}x{n_units}"
        
        # Generate data with specific dimensions
        config = SyntheticDataConfig(
            n_periods=n_periods,
            n_units=n_units,
            treatment_start_period=n_periods // 2,
            random_seed=42
        )
        generator = SyntheticDataGenerator(config)
        panel_data = generator.generate_did_data()
        
        results = estimate_effect(
            panel_data,
            'treatment',
            'outcome',
            ['covariate']
        )
        
        # Should work with different panel dimensions
        self.assertIn('effect_estimate', results)
        self.assertIsInstance(results['effect_estimate'], (int, float))
        
        # Check method selection based on number of periods
        if n_periods == 2:
            self.assertIn('2x2', results['parameters']['estimation_method'])
        else:
            self.assertIn('TWFE', results['parameters']['estimation_method'])
    
    @patch('causal_agent.methods.difference_in_differences.estimator.get_llm_client')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_time_variable')
    @patch('causal_agent.methods.difference_in_differences.estimator.identify_treatment_group')
    @patch('causal_agent.methods.difference_in_differences.estimator.determine_treatment_period')
    @patch('causal_agent.methods.difference_in_differences.estimator.validate_parallel_trends')
    @patch('causal_agent.methods.difference_in_differences.estimator.interpret_did_results')
    @patch('causal_agent.methods.difference_in_differences.estimator.smf.ols')
    def test_estimate_effect_regression_failure(self, mock_ols, mock_interpret, mock_validate,
                                               mock_determine_period, mock_identify_group,
                                               mock_identify_time, mock_get_llm):
        """Test handling of regression estimation failure."""
        mock_get_llm.return_value = MagicMock()
        mock_identify_time.return_value = 'period'
        mock_identify_group.return_value = 'unit'
        mock_determine_period.return_value = 10
        mock_validate.return_value = {"valid": True}
        
        # Mock OLS to fail
        mock_ols.side_effect = Exception("Regression failed")
        
        with self.assertRaises(ValueError) as context:
            estimate_effect(
                self.did_data,
                'treatment',
                'outcome',
                ['covariate']
            )
        
        self.assertIn("DiD estimation failed", str(context.exception))
    
    def test_estimate_effect_confidence_interval_structure(self):
        """Test that confidence intervals have correct structure."""
        with patch.multiple(
            'causal_agent.methods.difference_in_differences.estimator',
            get_llm_client=MagicMock(return_value=MagicMock()),
            identify_time_variable=MagicMock(return_value='period'),
            identify_treatment_group=MagicMock(return_value='unit'),
            determine_treatment_period=MagicMock(return_value=10),
            validate_parallel_trends=MagicMock(return_value={"valid": True}),
            interpret_did_results=MagicMock(return_value="CI test")
        ):
            results = estimate_effect(
                self.did_data,
                'treatment',
                'outcome',
                ['covariate']
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


if __name__ == '__main__':
    pytest.main([__file__])
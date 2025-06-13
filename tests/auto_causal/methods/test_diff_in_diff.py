import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the function to test
from auto_causal.methods.diff_in_diff import estimate_effect

class TestDifferenceInDifferences(unittest.TestCase):

    def setUp(self):
        '''Set up dummy panel data for testing.'''
        # Simple 2 groups, 2 periods example
        self.df = pd.DataFrame({
            'unit': [1, 1, 2, 2, 3, 3, 4, 4], # 2 treated (1,2), 2 control (3,4)
            'time': [0, 1, 0, 1, 0, 1, 0, 1],
            'treatment_group': [1, 1, 1, 1, 0, 0, 0, 0], # Group indicator
            'outcome': [10, 12, 11, 14, 9, 9.5, 10, 10.5], # Treated increase more in period 1
            'covariate1': [1, 1, 2, 2, 1, 1, 2, 2] 
        })
        self.treatment = 'treatment_group' # This identifies the group
        self.outcome = 'outcome'
        self.covariates = ['covariate1']
        self.time_var = 'time'
        self.group_var = 'unit'

    # Mock all helper/validation functions within diff_in_diff.py
    @patch('auto_causal.methods.diff_in_diff.identify_time_variable')
    @patch('auto_causal.methods.diff_in_diff.identify_treatment_group')
    @patch('auto_causal.methods.diff_in_diff.determine_treatment_period')
    @patch('auto_causal.methods.diff_in_diff.validate_parallel_trends')
    # Mock estimate_did_model to avoid actual regression, return mock results
    @patch('auto_causal.methods.diff_in_diff.estimate_did_model')
    def test_estimate_effect_structure_and_types(self, mock_estimate_model, mock_validate_trends, 
                                                 mock_determine_period, mock_identify_group, mock_identify_time):
        '''Test the basic structure and types of the DiD estimate_effect output.'''
        # Configure mocks
        mock_identify_time.return_value = self.time_var
        mock_identify_group.return_value = self.group_var
        mock_determine_period.return_value = 1 # Assume treatment starts at time 1
        mock_validate_trends.return_value = {"valid": True, "p_value": 0.9}
        
        # Mock the statsmodels result object
        mock_model_results = MagicMock()
        # Define the interaction term based on how construct_did_formula names it
        # Assuming treatment='treatment_group', post='post'
        interaction_term = f"{self.treatment}_x_post"
        mock_model_results.params = {interaction_term: 2.5, 'Intercept': 10.0}
        mock_model_results.bse = {interaction_term: 0.5, 'Intercept': 0.2}
        mock_model_results.pvalues = {interaction_term: 0.01, 'Intercept': 0.001}
        # Mock the summary() method if format_did_results uses it
        mock_model_results.summary.return_value = "Mocked Model Summary"
        mock_estimate_model.return_value = mock_model_results

        # Call the function (passing explicit vars to bypass internal identification mocks if desired)
        result = estimate_effect(self.df, self.treatment, self.outcome, self.covariates, 
                                 time_var=self.time_var, group_var=self.group_var, query="Test query")

        # Assertions
        self.assertIsInstance(result, dict)
        expected_keys = ["effect_estimate", "effect_se", "confidence_interval", "p_value", 
                         "diagnostics", "method_details", "parameters", "model_summary"]
        for key in expected_keys:
            self.assertIn(key, result, f"Key '{key}' missing from result")

        self.assertEqual(result["method_details"], "DiD.TWFE")
        self.assertIsInstance(result["effect_estimate"], float)
        self.assertIsInstance(result["effect_se"], float)
        self.assertIsInstance(result["confidence_interval"], list)
        self.assertEqual(len(result["confidence_interval"]), 2)
        self.assertIsInstance(result["diagnostics"], dict)
        self.assertIsInstance(result["parameters"], dict)
        self.assertIn("time_var", result["parameters"])
        self.assertIn("group_var", result["parameters"])
        self.assertIn("interaction_term", result["parameters"])
        self.assertEqual(result["parameters"]["interaction_term"], interaction_term)
        self.assertIn("valid", result["diagnostics"])
        self.assertIn("model_summary", result)

if __name__ == '__main__':
    unittest.main() 
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the function to test
from auto_causal.methods.propensity_score.weighting import estimate_effect

class TestPropensityScoreWeighting(unittest.TestCase):

    def setUp(self):
        '''Set up a dummy DataFrame for testing.'''
        self.df = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'outcome': [10, 12, 11, 13, 9, 14, 10, 15, 11, 16],
            'covariate1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            'covariate2': [5.5, 6.5, 5.8, 6.2, 5.1, 6.8, 5.3, 6.1, 5.9, 6.3]
        })
        self.treatment = 'treatment'
        self.outcome = 'outcome'
        self.covariates = ['covariate1', 'covariate2']

    @patch('auto_causal.methods.propensity_score.weighting.get_llm_parameters')
    @patch('auto_causal.methods.propensity_score.weighting.determine_optimal_weight_type')
    @patch('auto_causal.methods.propensity_score.weighting.determine_optimal_trim_threshold')
    @patch('auto_causal.methods.propensity_score.weighting.select_propensity_model')
    @patch('auto_causal.methods.propensity_score.weighting.estimate_propensity_scores')
    @patch('auto_causal.methods.propensity_score.weighting.assess_weight_distribution')
    def test_estimate_effect_structure_and_types(self, mock_assess_weights, mock_estimate_ps, 
                                                 mock_select_model, mock_determine_trim, 
                                                 mock_determine_weight, mock_get_llm_params):
        '''Test the basic structure and types of the estimate_effect output.'''
        # Configure mocks
        mock_get_llm_params.return_value = {"parameters": {"weight_type": "ATE", "trim_threshold": 0.0}, "validation": {}}
        mock_determine_weight.return_value = 'ATE'
        mock_determine_trim.return_value = 0.0 # No trimming
        mock_select_model.return_value = 'logistic'
        # Simulate propensity scores
        mock_estimate_ps.return_value = np.random.uniform(0.1, 0.9, size=len(self.df))
        # Simulate diagnostics output
        mock_assess_weights.return_value = {
            "min_weight": 0.5, "max_weight": 5.0, "mean_weight": 1.0, "std_dev_weight": 0.5,
            "effective_sample_size": len(self.df) * 0.8, "potential_issues": False
        }

        # Call the function
        result = estimate_effect(self.df, self.treatment, self.outcome, self.covariates, query="Test query")

        # Assertions
        self.assertIsInstance(result, dict)
        expected_keys = ["effect_estimate", "effect_se", "confidence_interval", 
                         "diagnostics", "method_details", "parameters"]
        for key in expected_keys:
            self.assertIn(key, result, f"Key '{key}' missing from result")

        self.assertEqual(result["method_details"], "PS.Weighting")
        self.assertIsInstance(result["effect_estimate"], float)
        self.assertIsInstance(result["effect_se"], float)
        self.assertIsInstance(result["confidence_interval"], list)
        self.assertEqual(len(result["confidence_interval"]), 2)
        self.assertIsInstance(result["diagnostics"], dict)
        self.assertIsInstance(result["parameters"], dict)
        self.assertIn("weight_type", result["parameters"])
        self.assertIn("trim_threshold", result["parameters"])
        self.assertIn("propensity_model", result["parameters"])
        self.assertIn("effective_sample_size", result["diagnostics"])

if __name__ == '__main__':
    unittest.main() 
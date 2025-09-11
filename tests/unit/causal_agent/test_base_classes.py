"""Test the base test classes to ensure they work correctly."""

import unittest
import pandas as pd
import numpy as np
from tests.base import CausalAgentTestCase, MethodTestCase, IntegrationTestCase


class TestCausalAgentTestCase(CausalAgentTestCase):
    """Test the CausalAgentTestCase base class."""
    
    def test_create_mock_dataset(self):
        """Test mock dataset creation."""
        dataset = self.create_mock_dataset(n_samples=50, n_features=3)
        
        # Check structure
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertEqual(len(dataset), 50)
        self.assertEqual(len(dataset.columns), 5)  # 3 features + treatment + outcome
        
        # Check required columns exist
        self.assertIn("treatment", dataset.columns)
        self.assertIn("outcome", dataset.columns)
        self.assertIn("feature_0", dataset.columns)
        
        # Check data types
        self.assertTrue(dataset["treatment"].dtype in [int, bool])
        self.assertTrue(dataset["outcome"].dtype in [int, float])
    
    def test_create_mock_llm_response(self):
        """Test mock LLM response creation."""
        response = self.create_mock_llm_response("method_selection")
        
        # Check structure
        self.assertIn("recommended_method", response)
        self.assertIn("confidence", response)
        self.assertIn("reasoning", response)
        
        # Check types
        self.assertIsInstance(response["confidence"], (int, float))
        self.assertIsInstance(response["recommended_method"], str)
    
    def test_assert_dataframe_structure(self):
        """Test DataFrame structure assertion."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        # Should pass
        self.assert_dataframe_structure(df, ["col1", "col2"], min_rows=2)
        
        # Should fail with missing column
        with self.assertRaises(AssertionError):
            self.assert_dataframe_structure(df, ["col1", "missing_col"])
    
    def test_assert_causal_result_structure(self):
        """Test causal result structure assertion."""
        valid_result = {
            "effect_estimate": 0.5,
            "confidence_interval": [0.2, 0.8],
            "method_used": "test_method"
        }
        
        # Should pass
        self.assert_causal_result_structure(valid_result)
        
        # Should fail with missing key
        invalid_result = {"effect_estimate": 0.5}
        with self.assertRaises(AssertionError):
            self.assert_causal_result_structure(invalid_result)


class TestMethodTestCase(MethodTestCase):
    """Test the MethodTestCase base class."""
    
    def test_validate_method_output(self):
        """Test method output validation."""
        valid_output = {
            "effect_estimate": 0.3,
            "confidence_interval": [0.1, 0.5],
            "method_used": "test_method",
            "p_value": 0.05
        }
        
        # Should pass
        self.validate_method_output(valid_output, "test_method")
        
        # Should fail with wrong method name
        with self.assertRaises(AssertionError):
            self.validate_method_output(valid_output, "different_method")
    
    def test_create_method_test_scenarios(self):
        """Test method test scenario creation."""
        scenarios = self.create_method_test_scenarios()
        
        # Check structure
        self.assertIsInstance(scenarios, list)
        self.assertGreater(len(scenarios), 0)
        
        # Check scenario structure
        for scenario in scenarios:
            self.assertIn("name", scenario)
            self.assertIn("dataset", scenario)
            self.assertIn("treatment", scenario)
            self.assertIn("outcome", scenario)
            self.assertIsInstance(scenario["dataset"], pd.DataFrame)


class TestIntegrationTestCase(IntegrationTestCase):
    """Test the IntegrationTestCase base class."""
    
    def test_create_integration_test_query(self):
        """Test integration test query creation."""
        query = self.create_integration_test_query("causal_effect")
        
        # Check structure
        self.assertIn("query", query)
        self.assertIn("treatment_variable", query)
        self.assertIn("outcome_variable", query)
        self.assertIsInstance(query["query"], str)
    
    def test_validate_workflow_execution(self):
        """Test workflow execution validation."""
        workflow_result = {
            "stages_completed": ["data_analysis", "method_selection", "estimation"],
            "final_result": {
                "effect_estimate": 0.4,
                "confidence_interval": [0.2, 0.6],
                "method_used": "test_method"
            }
        }
        
        expected_stages = ["data_analysis", "method_selection", "estimation"]
        
        # Should pass
        self.validate_workflow_execution(workflow_result, expected_stages)
        
        # Should fail with missing stage
        with self.assertRaises(AssertionError):
            self.validate_workflow_execution(workflow_result, expected_stages + ["missing_stage"])


if __name__ == "__main__":
    unittest.main()
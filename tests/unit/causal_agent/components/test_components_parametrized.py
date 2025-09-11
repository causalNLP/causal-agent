"""Parametrized tests for causal_agent components."""

import unittest
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import tempfile
import os

from causal_agent.components.dataset_analyzer import analyze_dataset, _categorize_columns
from causal_agent.components.decision_tree import select_method
from causal_agent.components.input_parser import parse_input
from tests.base import CausalAgentTestCase


class TestComponentsParametrized(CausalAgentTestCase):
    """Parametrized tests for multiple components."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create various test datasets
        self.datasets = {
            "small": self.create_mock_dataset(n_samples=50, n_features=2),
            "medium": self.create_mock_dataset(n_samples=200, n_features=5),
            "large": self.create_mock_dataset(n_samples=1000, n_features=10)
        }
        
        # Create temporary files for each dataset
        self.temp_files = {}
        for name, data in self.datasets.items():
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            data.to_csv(temp_file.name, index=False)
            temp_file.close()
            self.temp_files[name] = temp_file.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        for temp_file in self.temp_files.values():
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_dataset_analyzer_different_sizes(self):
        """Test dataset analyzer with different dataset sizes."""
        test_cases = [("small", 50), ("medium", 200), ("large", 1000)]
        
        for dataset_size, expected_min_rows in test_cases:
            with self.subTest(dataset_size=dataset_size):
                dataset_path = self.temp_files[dataset_size]
                
                result = analyze_dataset(dataset_path)
                
                # Should handle all dataset sizes
                self.assertIsInstance(result, dict)
                self.assertNotIn("error", result)
                self.assertGreaterEqual(result["dataset_info"]["num_rows"], expected_min_rows)
    
    def test_categorize_columns_different_types(self):
        """Test column categorization with different data types."""
        test_cases = [
            ("binary_numeric", [0, 1, 0, 1, 0], "binary"),
            ("continuous_numeric", [1.5, 2.7, 3.2, 4.1, 5.8], ["continuous_numeric", "categorical_numeric"]),
            ("categorical_numeric", [1, 2, 1, 3, 2], "categorical_numeric"),
            ("binary_categorical", ['A', 'B', 'A', 'B', 'A'], "binary_categorical"),
            ("categorical", ['X', 'Y', 'Z', 'X', 'Y'], "categorical"),
            ("discrete_numeric", [10, 20, 30, 40, 50], ["discrete_numeric", "categorical_numeric"])
        ]
        
        for column_type, test_values, expected_category in test_cases:
            with self.subTest(column_type=column_type):
                test_df = pd.DataFrame({column_type: test_values})
                
                result = _categorize_columns(test_df)
                
                if isinstance(expected_category, list):
                    self.assertIn(result[column_type], expected_category)
                else:
                    self.assertEqual(result[column_type], expected_category)
    
    def test_decision_tree_method_selection_scenarios(self):
        """Test decision tree method selection for different scenarios."""
        method_scenarios = [
        {
            "name": "basic_observational",
            "properties": {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "covariates": ["feature_0", "feature_1"],
                "is_rct": False,
                "has_temporal_structure": False,
                "treatment_variable_type": "binary"
            },
            "expected_methods": ["propensity_score_matching", "propensity_score_weighting", "linear_regression"]
        },
        {
            "name": "rct_with_covariates",
            "properties": {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "covariates": ["feature_0"],
                "is_rct": True,
                "has_temporal_structure": False,
                "treatment_variable_type": "binary"
            },
            "expected_methods": ["linear_regression", "diff_in_means"]
        },
        {
            "name": "instrumental_variable",
            "properties": {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "instrument_variable": "instrument",
                "covariates": ["feature_0"],
                "is_rct": False,
                "has_temporal_structure": False,
                "treatment_variable_type": "binary"
            },
            "expected_methods": ["instrumental_variable"]
        },
        {
            "name": "temporal_data",
            "properties": {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "time_variable": "year",
                "covariates": ["feature_0"],
                "is_rct": False,
                "has_temporal_structure": True,
                "treatment_variable_type": "binary"
            },
            "expected_methods": ["difference_in_differences"]
        },
        {
            "name": "regression_discontinuity",
            "properties": {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "running_variable": "score",
                "cutoff_value": 0.5,
                "covariates": [],
                "is_rct": False,
                "has_temporal_structure": False,
                "treatment_variable_type": "binary"
            },
            "expected_methods": ["regression_discontinuity_design"]
        }
        ]
        
        for method_scenario in method_scenarios:
            with self.subTest(scenario=method_scenario["name"]):
                properties = method_scenario["properties"]
                expected_methods = method_scenario["expected_methods"]
                
                result = select_method(properties)
                
                # Check that a method was selected
                self.assertIn("selected_method", result)
                selected_method = result["selected_method"]
                
                # Check that selected method is one of the expected methods
                self.assertIn(selected_method, expected_methods)
                
                # Check result structure
                self.assertIn("method_justification", result)
                self.assertIn("method_assumptions", result)
                self.assertIsInstance(result["method_assumptions"], list)
    
    def test_input_parser_query_types(self):
        """Test input parser with different query types."""
        query_scenarios = [
        {
            "query": "What is the effect of treatment on outcome?",
            "expected_type": "EFFECT_ESTIMATION",
            "expected_variables": {"treatment": ["treatment"], "outcome": ["outcome"]}
        },
        {
            "query": "Is there a correlation between age and income?",
            "expected_type": "CORRELATION",
            "expected_variables": {"treatment": [], "outcome": []}
        },
        {
            "query": "Show me the average salary by department",
            "expected_type": "DESCRIPTIVE",
            "expected_variables": {"treatment": [], "outcome": []}
        },
        {
            "query": "What would happen if we increased the budget?",
            "expected_type": "COUNTERFACTUAL",
            "expected_variables": {"treatment": [], "outcome": []}
        }
        ]
        
        for query_scenario in query_scenarios:
            with self.subTest(query=query_scenario["query"]):
                query = query_scenario["query"]
                expected_type = query_scenario["expected_type"]
                
                # Mock LLM response
                mock_llm = Mock()
                
                with patch('causal_agent.components.input_parser._extract_query_information_with_llm') as mock_extract:
                    from causal_agent.components.input_parser import ParsedQueryInfo, ParsedVariables
                    
                    mock_extract.return_value = ParsedQueryInfo(
                        query_type=expected_type,
                        variables=ParsedVariables(**query_scenario["expected_variables"]),
                        constraints=[],
                        dataset_path_mentioned=None
                    )
                    
                    result = parse_input(query, llm=mock_llm)
                
                self.assertEqual(result["query_type"], expected_type)
                self.assertIsInstance(result["extracted_variables"], dict)
    
    def test_dataset_analyzer_data_quality(self):
        """Test dataset analyzer with different data quality levels."""
        dataset_characteristics_list = [
        {
            "name": "high_quality",
            "missing_rate": 0.0,
            "outlier_rate": 0.0,
            "n_samples": 1000
        },
        {
            "name": "medium_quality",
            "missing_rate": 0.05,
            "outlier_rate": 0.02,
            "n_samples": 500
        },
        {
            "name": "low_quality",
            "missing_rate": 0.15,
            "outlier_rate": 0.05,
            "n_samples": 100
        }
        ]
        
        for dataset_characteristics in dataset_characteristics_list:
            with self.subTest(quality=dataset_characteristics["name"]):
                # Create dataset with specified characteristics
                n_samples = dataset_characteristics["n_samples"]
                missing_rate = dataset_characteristics["missing_rate"]
                outlier_rate = dataset_characteristics["outlier_rate"]
                
                # Generate base dataset
                data = self.create_mock_dataset(n_samples=n_samples, n_features=3)
                
                # Introduce missing values
                if missing_rate > 0:
                    n_missing = int(n_samples * missing_rate)
                    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
                    data.loc[missing_indices, 'feature_0'] = np.nan
                
                # Introduce outliers
                if outlier_rate > 0:
                    n_outliers = int(n_samples * outlier_rate)
                    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
                    data.loc[outlier_indices, 'outcome'] = data['outcome'].mean() + 5 * data['outcome'].std()
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                data.to_csv(temp_file.name, index=False)
                temp_file.close()
                
                try:
                    result = analyze_dataset(temp_file.name)
                    
                    # Should handle all quality levels
                    self.assertIsInstance(result, dict)
                    self.assertEqual(result["dataset_info"]["num_rows"], n_samples)
                    
                    # Should not crash on data quality issues
                    self.assertNotIn("error", result)
                    
                finally:
                    os.unlink(temp_file.name)
    
    def test_components_with_different_effect_sizes(self):
        """Test components with datasets having different treatment effect sizes."""
        effect_sizes = [0.0, 0.1, 0.3, 0.5, 1.0]
        
        for treatment_effect_size in effect_sizes:
            with self.subTest(effect_size=treatment_effect_size):
                # Create dataset with specified treatment effect
                data = self.create_mock_dataset(
                    n_samples=200,
                    n_features=3,
                    treatment_effect=treatment_effect_size
                )
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                data.to_csv(temp_file.name, index=False)
                temp_file.close()
                
                try:
                    # Test dataset analyzer
                    analysis_result = analyze_dataset(temp_file.name)
                    self.assertIsInstance(analysis_result, dict)
                    self.assertNotIn("error", analysis_result)
                    
                    # Test decision tree with this data structure
                    properties = {
                        "treatment_variable": "treatment",
                        "outcome_variable": "outcome",
                        "covariates": ["feature_0", "feature_1", "feature_2"],
                        "is_rct": False,
                        "has_temporal_structure": False,
                        "treatment_variable_type": "binary"
                    }
                    
                    method_result = select_method(properties)
                    self.assertIn("selected_method", method_result)
                    
                finally:
                    os.unlink(temp_file.name)
    
    def test_decision_tree_method_exclusion(self):
        """Test decision tree with different method exclusion scenarios."""
        exclusion_scenarios = [
        {
            "excluded": ["propensity_score_matching"],
            "should_work": True
        },
        {
            "excluded": ["propensity_score_matching", "propensity_score_weighting"],
            "should_work": True
        },
        {
            "excluded": ["propensity_score_matching", "propensity_score_weighting", "linear_regression"],
            "should_work": True
        }
        ]
        
        for exclusion_scenario in exclusion_scenarios:
            with self.subTest(excluded=exclusion_scenario["excluded"]):
                properties = {
                    "treatment_variable": "treatment",
                    "outcome_variable": "outcome",
                    "covariates": ["feature_0"],
                    "is_rct": False,
                    "has_temporal_structure": False,
                    "treatment_variable_type": "binary"
                }
                
                excluded_methods = exclusion_scenario["excluded"]
                should_work = exclusion_scenario["should_work"]
                
                if should_work:
                    result = select_method(properties, excluded_methods=excluded_methods)
                    
                    # Should select a method not in excluded list
                    selected_method = result["selected_method"]
                    self.assertNotIn(selected_method, excluded_methods)
                    
                    # Should include excluded methods in result
                    self.assertEqual(set(result["excluded_methods"]), set(excluded_methods))
                else:
                    # Should raise error when all methods excluded
                    with self.assertRaises(RuntimeError):
                        select_method(properties, excluded_methods=excluded_methods)


if __name__ == '__main__':
    unittest.main()
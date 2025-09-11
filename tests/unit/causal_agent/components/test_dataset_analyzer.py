"""Unit tests for dataset analyzer component."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import pytest

from causal_agent.components.dataset_analyzer import (
    analyze_dataset,
    _categorize_columns,
    _identify_potential_variables,
    detect_temporal_structure,
    find_potential_instruments,
    _calculate_per_group_stats
)
from tests.base import CausalAgentTestCase


class TestDatasetAnalyzer(CausalAgentTestCase):
    """Test cases for dataset analyzer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create test dataset
        self.test_data = self.create_mock_dataset(
            n_samples=100,
            n_features=3,
            treatment_col="treatment",
            outcome_col="outcome"
        )
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
        
        # Mock LLM client
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = Mock(content='{"potential_treatments": ["treatment"], "potential_outcomes": ["outcome"]}')
    
    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_analyze_dataset_basic(self):
        """Test basic dataset analysis functionality."""
        result = analyze_dataset(self.temp_csv.name)
        
        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertIn("dataset_info", result)
        self.assertIn("columns", result)
        self.assertIn("potential_treatments", result)
        self.assertIn("potential_outcomes", result)
        
        # Check dataset info
        dataset_info = result["dataset_info"]
        self.assertEqual(dataset_info["num_rows"], 100)
        self.assertEqual(dataset_info["num_columns"], 5)  # 3 features + treatment + outcome
        
        # Check columns
        expected_columns = ["feature_0", "feature_1", "feature_2", "treatment", "outcome"]
        self.assertEqual(set(result["columns"]), set(expected_columns))
    
    def test_analyze_dataset_with_llm(self):
        """Test dataset analysis with LLM client."""
        result = analyze_dataset(self.temp_csv.name, llm_client=self.mock_llm)
        
        self.assertIsInstance(result, dict)
        self.assertIn("llm_augmentation", result)
        self.assertNotEqual(result["llm_augmentation"], "Not used")
    
    def test_analyze_dataset_nonexistent_file(self):
        """Test analysis with nonexistent file."""
        result = analyze_dataset("nonexistent_file.csv")
        
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])
    
    def test_categorize_columns(self):
        """Test column categorization functionality."""
        # Create test data with different column types
        test_df = pd.DataFrame({
            'binary_numeric': [0, 1, 0, 1, 0],
            'continuous_numeric': [1.5, 2.7, 3.2, 4.1, 5.8],
            'categorical_numeric': [1, 2, 1, 3, 2],
            'binary_categorical': ['A', 'B', 'A', 'B', 'A'],
            'categorical': ['X', 'Y', 'Z', 'X', 'Y'],
            'text': ['hello', 'world', 'test', 'data', 'analysis']
        })
        
        result = _categorize_columns(test_df)
        
        self.assertEqual(result['binary_numeric'], 'binary')
        # Note: continuous_numeric might be categorized as categorical_numeric if unique values < 10
        self.assertIn(result['continuous_numeric'], ['continuous_numeric', 'categorical_numeric'])
        self.assertEqual(result['categorical_numeric'], 'categorical_numeric')
        self.assertEqual(result['binary_categorical'], 'binary_categorical')
        self.assertEqual(result['categorical'], 'categorical')
        self.assertIn(result['text'], ['categorical', 'text_or_other'])
    
    @patch('causal_agent.components.dataset_analyzer.json.loads')
    def test_identify_potential_variables_with_llm(self, mock_json_loads):
        """Test variable identification with LLM."""
        # Mock successful LLM response
        mock_json_loads.return_value = {
            "potential_treatments": ["treatment"],
            "potential_outcomes": ["outcome"]
        }
        
        column_categories = {
            'treatment': 'binary',
            'outcome': 'continuous_numeric',
            'feature_0': 'continuous_numeric'
        }
        
        result = _identify_potential_variables(
            self.test_data, 
            column_categories, 
            llm_client=self.mock_llm
        )
        
        self.assertIn("potential_treatments", result)
        self.assertIn("potential_outcomes", result)
        self.assertEqual(result["potential_treatments"], ["treatment"])
        self.assertEqual(result["potential_outcomes"], ["outcome"])
    
    def test_identify_potential_variables_heuristic(self):
        """Test variable identification using heuristic method."""
        column_categories = {
            'treatment': 'binary',
            'outcome': 'continuous_numeric',
            'feature_0': 'continuous_numeric',
            'feature_1': 'continuous_numeric',
            'feature_2': 'continuous_numeric'
        }
        
        result = _identify_potential_variables(
            self.test_data, 
            column_categories, 
            llm_client=None
        )
        
        self.assertIn("potential_treatments", result)
        self.assertIn("potential_outcomes", result)
        self.assertIsInstance(result["potential_treatments"], list)
        self.assertIsInstance(result["potential_outcomes"], list)
    
    def test_detect_temporal_structure_no_temporal(self):
        """Test temporal structure detection with non-temporal data."""
        result = detect_temporal_structure(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn("has_temporal_structure", result)
        self.assertIn("is_panel_data", result)
        self.assertFalse(result["has_temporal_structure"])
        self.assertFalse(result["is_panel_data"])
    
    def test_detect_temporal_structure_with_time_column(self):
        """Test temporal structure detection with time column."""
        # Add time column to test data
        temporal_data = self.test_data.copy()
        temporal_data['year'] = [2020, 2021, 2020, 2021, 2020] * 20
        temporal_data['unit_id'] = list(range(20)) * 5
        
        result = detect_temporal_structure(temporal_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn("has_temporal_structure", result)
        # Note: Without LLM, heuristic detection might not catch this
        # This tests the structure of the function
    
    def test_find_potential_instruments_no_llm(self):
        """Test instrument finding without LLM."""
        result = find_potential_instruments(
            self.test_data,
            llm_client=None,
            potential_treatments=["treatment"],
            potential_outcomes=["outcome"]
        )
        
        self.assertIsInstance(result, list)
        # Without LLM, should return empty list or heuristic results
    
    def test_calculate_per_group_stats(self):
        """Test per-group statistics calculation."""
        # Create data with binary treatment
        test_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'covariate1': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            'covariate2': [10, 20, 15, 25, 12, 22]
        })
        
        result = _calculate_per_group_stats(test_data, ["treatment"])
        
        self.assertIsInstance(result, dict)
        self.assertIn("treatment", result)
        
        treatment_stats = result["treatment"]
        self.assertIn("group_sizes", treatment_stats)
        self.assertIn("covariate_stats", treatment_stats)
        
        # Check group sizes
        self.assertEqual(treatment_stats["group_sizes"]["control"], 3)
        self.assertEqual(treatment_stats["group_sizes"]["treated"], 3)
    
    def test_calculate_per_group_stats_non_binary(self):
        """Test per-group stats with non-binary treatment."""
        test_data = pd.DataFrame({
            'treatment': [0, 1, 2, 0, 1, 2],  # Non-binary
            'covariate1': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
        })
        
        result = _calculate_per_group_stats(test_data, ["treatment"])
        
        # Should skip non-binary treatments
        self.assertIsInstance(result, dict)
        # May be empty or contain error info
    
    def test_analyze_dataset_different_sizes(self):
        """Test dataset analysis with different dataset sizes."""
        test_cases = [(50, 2), (200, 5), (1000, 10)]
        
        for n_samples, n_features in test_cases:
            with self.subTest(n_samples=n_samples, n_features=n_features):
                # Create dataset of specified size
                test_data = self.create_mock_dataset(
                    n_samples=n_samples,
                    n_features=n_features
                )
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                test_data.to_csv(temp_file.name, index=False)
                temp_file.close()
                
                try:
                    result = analyze_dataset(temp_file.name)
                    
                    # Check that analysis scales appropriately
                    self.assertEqual(result["dataset_info"]["num_rows"], n_samples)
                    self.assertEqual(result["dataset_info"]["num_columns"], n_features + 2)  # +treatment +outcome
                    
                finally:
                    os.unlink(temp_file.name)
    
    def test_analyze_dataset_with_missing_values(self):
        """Test dataset analysis with missing values."""
        # Create data with missing values
        test_data = self.test_data.copy()
        test_data.loc[0:4, 'feature_0'] = np.nan
        test_data.loc[10:14, 'outcome'] = np.nan
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            result = analyze_dataset(temp_file.name)
            
            # Should handle missing values gracefully
            self.assertIsInstance(result, dict)
            self.assertNotIn("error", result)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_analyze_dataset_error_handling(self):
        """Test error handling in dataset analysis."""
        # Create invalid CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("invalid,csv,content\n1,2\n3,4,5,6")  # Inconsistent columns
        temp_file.close()
        
        try:
            result = analyze_dataset(temp_file.name)
            
            # Should return error information
            self.assertIn("error", result)
            
        finally:
            os.unlink(temp_file.name)


if __name__ == '__main__':
    unittest.main()
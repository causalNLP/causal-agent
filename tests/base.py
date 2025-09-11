"""Base test classes for causal_agent testing infrastructure."""

import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


class CausalAgentTestCase(unittest.TestCase):
    """Base test class with common utilities for causal_agent tests.
    
    Provides shared functionality for all test classes including:
    - Common setup and teardown
    - Mock data generation utilities
    - Assertion helpers
    - Test configuration management
    """
    
    def setUp(self):
        """Set up common test fixtures and configuration."""
        self.test_data_dir = Path(__file__).parent / "fixtures" / "data"
        self.temp_dir = None
        self.mock_llm_responses = {}
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Default test configuration
        self.test_config = {
            "mock_llm": True,
            "use_synthetic_data": True,
            "coverage_threshold": 0.8,
            "timeout_seconds": 30
        }
    
    def tearDown(self):
        """Clean up test fixtures and temporary files."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_mock_dataset(self, 
                           n_samples: int = 100, 
                           n_features: int = 5,
                           treatment_col: str = "treatment",
                           outcome_col: str = "outcome",
                           **kwargs) -> pd.DataFrame:
        """Create a mock dataset for testing.
        
        Args:
            n_samples: Number of samples in the dataset
            n_features: Number of feature columns
            treatment_col: Name of the treatment column
            outcome_col: Name of the outcome column
            **kwargs: Additional parameters for data generation
            
        Returns:
            pd.DataFrame: Mock dataset with specified structure
        """
        np.random.seed(42)  # For reproducible tests
        
        data = {}
        
        # Generate feature columns
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.normal(0, 1, n_samples)
        
        # Generate treatment column (binary)
        data[treatment_col] = np.random.binomial(1, 0.5, n_samples)
        
        # Generate outcome column with some treatment effect
        treatment_effect = kwargs.get("treatment_effect", 0.5)
        data[outcome_col] = (
            np.random.normal(0, 1, n_samples) + 
            treatment_effect * data[treatment_col]
        )
        
        return pd.DataFrame(data)
    
    def create_mock_llm_response(self, 
                                response_type: str = "method_selection",
                                **kwargs) -> Dict[str, Any]:
        """Create mock LLM response for testing.
        
        Args:
            response_type: Type of LLM response to mock
            **kwargs: Additional parameters for response generation
            
        Returns:
            Dict[str, Any]: Mock LLM response
        """
        if response_type == "method_selection":
            return {
                "recommended_method": kwargs.get("method", "backdoor_adjustment"),
                "confidence": kwargs.get("confidence", 0.8),
                "reasoning": kwargs.get("reasoning", "Test reasoning"),
                "assumptions": kwargs.get("assumptions", ["Test assumption"])
            }
        elif response_type == "variable_identification":
            return {
                "treatment_variable": kwargs.get("treatment", "treatment"),
                "outcome_variable": kwargs.get("outcome", "outcome"),
                "confounders": kwargs.get("confounders", ["feature_0", "feature_1"]),
                "instruments": kwargs.get("instruments", [])
            }
        else:
            return {"response": "Mock response", "type": response_type}
    
    def assert_dataframe_structure(self, 
                                  df: pd.DataFrame, 
                                  expected_columns: List[str],
                                  min_rows: int = 1):
        """Assert that a DataFrame has the expected structure.
        
        Args:
            df: DataFrame to check
            expected_columns: List of expected column names
            min_rows: Minimum number of rows expected
        """
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), min_rows)
        
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' not found in DataFrame")
    
    def assert_causal_result_structure(self, result: Dict[str, Any]):
        """Assert that a causal analysis result has the expected structure.
        
        Args:
            result: Causal analysis result dictionary
        """
        required_keys = ["effect_estimate", "confidence_interval", "method_used"]
        for key in required_keys:
            self.assertIn(key, result, f"Required key '{key}' not found in result")
        
        # Check that effect estimate is numeric
        self.assertIsInstance(result["effect_estimate"], (int, float, np.number))
        
        # Check confidence interval structure
        if "confidence_interval" in result and result["confidence_interval"]:
            ci = result["confidence_interval"]
            self.assertIsInstance(ci, (list, tuple))
            self.assertEqual(len(ci), 2)
            self.assertLessEqual(ci[0], ci[1])


class MethodTestCase(CausalAgentTestCase):
    """Base class for causal method tests.
    
    Extends CausalAgentTestCase with method-specific utilities:
    - Method validation helpers
    - Performance benchmarking
    - Statistical test utilities
    """
    
    def setUp(self):
        """Set up method-specific test fixtures."""
        super().setUp()
        
        # Method-specific configuration
        self.method_config = {
            "max_execution_time": 10.0,  # seconds
            "min_accuracy_threshold": 0.7,
            "statistical_significance": 0.05
        }
        
        # Create standard test datasets
        self.rct_data = self.create_mock_dataset(
            n_samples=200, 
            treatment_effect=0.5,
            dataset_type="rct"
        )
        
        self.observational_data = self.create_mock_dataset(
            n_samples=500,
            treatment_effect=0.3,
            dataset_type="observational"
        )
    
    def validate_method_output(self, 
                              result: Dict[str, Any], 
                              method_name: str):
        """Validate that method output meets requirements.
        
        Args:
            result: Method execution result
            method_name: Name of the causal method
        """
        # Check basic result structure
        self.assert_causal_result_structure(result)
        
        # Check method-specific requirements
        self.assertEqual(result["method_used"], method_name)
        
        # Validate statistical properties
        if "p_value" in result:
            self.assertIsInstance(result["p_value"], (int, float, np.number))
            self.assertGreaterEqual(result["p_value"], 0.0)
            self.assertLessEqual(result["p_value"], 1.0)
        
        # Validate confidence interval
        if "confidence_interval" in result and result["confidence_interval"]:
            ci = result["confidence_interval"]
            effect = result["effect_estimate"]
            self.assertLessEqual(ci[0], effect)
            self.assertGreaterEqual(ci[1], effect)
    
    def benchmark_method_performance(self, 
                                   method_func, 
                                   dataset: pd.DataFrame,
                                   **kwargs) -> Dict[str, float]:
        """Benchmark method performance metrics.
        
        Args:
            method_func: Method function to benchmark
            dataset: Dataset to use for benchmarking
            **kwargs: Additional arguments for method function
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        import time
        import psutil
        import os
        
        # Measure execution time
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute method
        result = method_func(dataset, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        performance_metrics = {
            "execution_time": end_time - start_time,
            "memory_usage_mb": end_memory - start_memory,
            "dataset_size": len(dataset)
        }
        
        # Validate performance thresholds
        self.assertLess(
            performance_metrics["execution_time"], 
            self.method_config["max_execution_time"],
            f"Method execution time exceeded threshold"
        )
        
        return performance_metrics
    
    def create_method_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create standard test scenarios for method validation.
        
        Returns:
            List[Dict[str, Any]]: List of test scenarios
        """
        scenarios = [
            {
                "name": "rct_scenario",
                "dataset": self.rct_data,
                "treatment": "treatment",
                "outcome": "outcome",
                "expected_effect_range": (0.3, 0.7)
            },
            {
                "name": "observational_scenario", 
                "dataset": self.observational_data,
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["feature_0", "feature_1"],
                "expected_effect_range": (0.1, 0.5)
            }
        ]
        
        return scenarios


class IntegrationTestCase(CausalAgentTestCase):
    """Base class for integration tests.
    
    Extends CausalAgentTestCase with integration-specific utilities:
    - End-to-end workflow testing
    - Component interaction validation
    - System-level assertions
    """
    
    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()
        
        # Integration test configuration
        self.integration_config = {
            "workflow_timeout": 60.0,  # seconds
            "max_memory_usage_mb": 500,
            "test_data_variants": 3
        }
        
        # Mock external dependencies
        self.setup_integration_mocks()
    
    def setup_integration_mocks(self):
        """Set up mocks for external dependencies in integration tests."""
        # Mock LLM API calls
        self.llm_patcher = patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm.return_value = self.create_mock_llm_response()
        
        # Mock file I/O operations
        self.file_patcher = patch('builtins.open')
        self.mock_file = self.file_patcher.start()
    
    def tearDown(self):
        """Clean up integration test mocks."""
        super().tearDown()
        
        # Stop all patchers
        if hasattr(self, 'llm_patcher'):
            self.llm_patcher.stop()
        if hasattr(self, 'file_patcher'):
            self.file_patcher.stop()
    
    def validate_workflow_execution(self, 
                                   workflow_result: Dict[str, Any],
                                   expected_stages: List[str]):
        """Validate that a workflow executed all expected stages.
        
        Args:
            workflow_result: Result from workflow execution
            expected_stages: List of expected workflow stages
        """
        self.assertIn("stages_completed", workflow_result)
        completed_stages = workflow_result["stages_completed"]
        
        for stage in expected_stages:
            self.assertIn(stage, completed_stages, 
                         f"Expected stage '{stage}' not completed")
        
        # Validate final result structure
        if "final_result" in workflow_result:
            self.assert_causal_result_structure(workflow_result["final_result"])
    
    def create_integration_test_query(self, 
                                     query_type: str = "causal_effect") -> Dict[str, Any]:
        """Create a test query for integration testing.
        
        Args:
            query_type: Type of query to create
            
        Returns:
            Dict[str, Any]: Test query structure
        """
        if query_type == "causal_effect":
            return {
                "query": "What is the effect of treatment on outcome?",
                "dataset_path": str(self.test_data_dir / "test_data.csv"),
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "method_preference": None  # Let system choose
            }
        elif query_type == "method_comparison":
            return {
                "query": "Compare different methods for causal analysis",
                "dataset_path": str(self.test_data_dir / "test_data.csv"),
                "methods_to_compare": ["backdoor_adjustment", "propensity_score"],
                "treatment_variable": "treatment",
                "outcome_variable": "outcome"
            }
        else:
            return {"query": f"Test query for {query_type}"}
    
    def assert_component_integration(self, 
                                   component_a_output: Any,
                                   component_b_input: Any):
        """Assert that output from component A is compatible with component B input.
        
        Args:
            component_a_output: Output from first component
            component_b_input: Expected input format for second component
        """
        # This is a placeholder for component compatibility checks
        # Specific implementations would depend on actual component interfaces
        self.assertIsNotNone(component_a_output)
        self.assertIsNotNone(component_b_input)
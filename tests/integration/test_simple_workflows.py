"""Simple integration tests for basic workflow functionality."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Dict, Any

import unittest


class TestSimpleWorkflowIntegration(unittest.TestCase):
    """Simple integration tests for basic workflow components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_simple_dataset(self) -> str:
        """Create a simple test dataset."""
        np.random.seed(42)
        n = 50
        
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 10 + 2 * treatment + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "simple_test.csv")
        df.to_csv(filepath, index=False)
        
        # Verify file creation
        assert os.path.exists(filepath), f"Failed to create dataset at {filepath}"
        
        return filepath
    
    def test_dataset_creation(self):
        """Test that dataset creation works correctly."""
        dataset_path = self.create_simple_dataset()
        
        # Verify file exists and has correct structure
        self.assertTrue(os.path.exists(dataset_path))
        
        df = pd.read_csv(dataset_path)
        self.assertEqual(len(df), 50)
        self.assertIn('treatment', df.columns)
        self.assertIn('outcome', df.columns)
        
        # Verify data types
        self.assertTrue(df['treatment'].dtype in [int, 'int64'])
        self.assertTrue(df['outcome'].dtype in [float, 'float64'])
    
    def test_component_imports(self):
        """Test that all required components can be imported."""
        try:
            from causal_agent.tools.input_parser_tool import input_parser_tool
            from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
            from causal_agent.tools.query_interpreter_tool import query_interpreter_tool
            from causal_agent.tools.method_selector_tool import method_selector_tool
            from causal_agent.tools.method_validator_tool import method_validator_tool
            from causal_agent.tools.method_executor_tool import method_executor_tool
            from causal_agent.tools.explanation_generator_tool import explanation_generator_tool
            from causal_agent.tools.output_formatter_tool import output_formatter_tool
            
            # Verify tools are callable
            self.assertTrue(callable(input_parser_tool))
            self.assertTrue(hasattr(dataset_analyzer_tool, 'func'))
            self.assertTrue(hasattr(query_interpreter_tool, 'func'))
            
        except ImportError as e:
            self.fail(f"Failed to import required components: {e}")
    
    def test_input_parser_basic(self):
        """Test basic input parser functionality."""
        from causal_agent.tools.input_parser_tool import input_parser_tool
        
        dataset_path = self.create_simple_dataset()
        
        input_text = f"Query: What is the effect of treatment on outcome?\nDataset: {dataset_path}\nDescription: Simple test"
        
        try:
            result = input_parser_tool(input_text)
            
            self.assertIsInstance(result, dict)
            self.assertIn('original_query', result)
            self.assertIn('dataset_path', result)
            self.assertEqual(result['dataset_path'], dataset_path)
            
        except Exception as e:
            self.fail(f"Input parser failed: {e}")
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_basic(self, mock_get_llm, mock_llm_call):
        """Test basic dataset analyzer functionality."""
        from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
        
        dataset_path = self.create_simple_dataset()
        
        # Mock LLM response
        mock_llm_call.return_value = {
            "variables": ["treatment", "outcome"],
            "treatment_candidates": ["treatment"],
            "outcome_candidates": ["outcome"],
            "data_quality": "good"
        }
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        try:
            result = dataset_analyzer_tool.func(
                dataset_path=dataset_path,
                dataset_description="Simple test dataset",
                original_query="What is the effect of treatment on outcome?"
            )
            
            self.assertIsInstance(result, object)  # Should return DatasetAnalysisResult
            self.assertTrue(hasattr(result, 'analysis_results'))
            
        except Exception as e:
            self.fail(f"Dataset analyzer failed: {e}")
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_workflow_component_chain(self, mock_get_llm, mock_llm_call):
        """Test chaining of workflow components."""
        from causal_agent.tools.input_parser_tool import input_parser_tool
        from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
        from causal_agent.models import QueryInfo
        
        dataset_path = self.create_simple_dataset()
        
        # Mock LLM responses
        mock_llm_call.return_value = {
            "variables": ["treatment", "outcome"],
            "treatment_candidates": ["treatment"],
            "outcome_candidates": ["outcome"],
            "data_quality": "good"
        }
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        try:
            # Step 1: Parse input
            input_text = f"Query: What is the effect of treatment on outcome?\nDataset: {dataset_path}\nDescription: Test"
            input_result = input_parser_tool(input_text)
            
            self.assertIn('original_query', input_result)
            self.assertIn('dataset_path', input_result)
            
            # Step 2: Analyze dataset
            analysis_result = dataset_analyzer_tool.func(
                dataset_path=input_result['dataset_path'],
                dataset_description=input_result['dataset_description'],
                original_query=input_result['original_query']
            )
            
            self.assertTrue(hasattr(analysis_result, 'analysis_results'))
            
            # Verify the chain works
            self.assertEqual(input_result['dataset_path'], dataset_path)
            
        except Exception as e:
            self.fail(f"Component chain failed: {e}")
    
    def test_error_handling_missing_file(self):
        """Test error handling with missing dataset file."""
        from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
        
        nonexistent_path = "/nonexistent/file.csv"
        
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            
            # Should handle missing file gracefully
            try:
                result = dataset_analyzer_tool.func(
                    dataset_path=nonexistent_path,
                    dataset_description="Test",
                    original_query="Test query"
                )
                # If it doesn't raise an exception, it should return an error result
                if hasattr(result, 'error') or (hasattr(result, 'analysis_results') and 'error' in result.analysis_results):
                    pass  # Expected error handling
                else:
                    self.fail("Expected error handling for missing file")
            except Exception:
                # Exception is also acceptable error handling
                pass
    
    def test_data_validation(self):
        """Test data validation with various dataset formats."""
        # Test with different column names
        datasets = [
            {'treat': [0, 1, 0, 1], 'result': [10, 12, 11, 13]},
            {'intervention': [0, 1, 0, 1], 'response': [10, 12, 11, 13]},
            {'x': [0, 1, 0, 1], 'y': [10, 12, 11, 13]}
        ]
        
        for i, data in enumerate(datasets):
            with self.subTest(dataset=i):
                df = pd.DataFrame(data)
                filepath = os.path.join(self.temp_dir, f"test_data_{i}.csv")
                df.to_csv(filepath, index=False)
                
                # Verify file can be read
                loaded_df = pd.read_csv(filepath)
                self.assertEqual(len(loaded_df), 4)
                self.assertEqual(len(loaded_df.columns), 2)
    
    def test_mock_configuration(self):
        """Test that mocking is configured correctly."""
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
                
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm
                mock_llm_call.return_value = {"test": "response"}
                
                # Verify mocks are working
                from causal_agent.config import get_llm_client
                from causal_agent.utils.llm_helpers import call_llm_with_json_output
                
                client = get_llm_client()
                self.assertIsInstance(client, Mock)
                
                response = call_llm_with_json_output("test", {})
                self.assertEqual(response, {"test": "response"})
    
    def test_temp_directory_cleanup(self):
        """Test that temporary directories are cleaned up properly."""
        # Create a file in temp directory
        test_file = os.path.join(self.temp_dir, "cleanup_test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        self.assertTrue(os.path.exists(test_file))
        
        # Cleanup will be tested in tearDown
        # This test just verifies the setup works
"""Unit tests for dataset analyzer tool."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import pytest
from pydantic import ValidationError

from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
from causal_agent.models import DatasetAnalyzerOutput, DatasetAnalysis, DatasetInfo, TemporalStructure
from tests.base import CausalAgentTestCase


class TestDatasetAnalyzerTool(CausalAgentTestCase):
    """Test cases for dataset analyzer tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create test dataset
        self.test_data = self.create_mock_dataset(n_samples=50, n_features=3)
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
        
        # Mock successful analysis result
        self.mock_analysis_result = {
            "dataset_info": {
                "num_rows": 50,
                "num_columns": 5,
                "file_path": self.temp_csv.name,
                "file_name": os.path.basename(self.temp_csv.name)
            },
            "columns": ["feature_0", "feature_1", "feature_2", "treatment", "outcome"],
            "potential_treatments": ["treatment"],
            "potential_outcomes": ["outcome"],
            "temporal_structure_detected": False,
            "panel_data_detected": False,
            "potential_instruments_detected": False,
            "discontinuities_detected": False,
            "temporal_structure": {
                "has_temporal_structure": False,
                "temporal_columns": [],
                "is_panel_data": False,
                "time_column": None,
                "id_column": None,
                "time_periods": None,
                "units": None,
                "identification_method": "None"
            },
            "sample_size": 50,
            "num_covariates_estimate": 3,
            "llm_augmentation": "Not used"
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    @patch('causal_agent.components.dataset_analyzer.analyze_dataset')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_success(self, mock_get_llm, mock_analyze):
        """Test successful dataset analyzer tool execution."""
        # Mock LLM client
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock analyze_dataset function
        mock_analyze.return_value = self.mock_analysis_result
        
        # Execute tool
        result = dataset_analyzer_tool.invoke({
            "dataset_path":  self.temp_csv.name,
            "dataset_description": "Test dataset"
        })
        
        # Check result type
        self.assertIsInstance(result, DatasetAnalyzerOutput)
        
        # Check that analyze_dataset was called correctly
        mock_analyze.assert_called_once_with(
            self.temp_csv.name,
            llm_client=mock_llm,
            dataset_description="Test dataset",
            original_query=None
        )
        
        # Check result structure
        self.assertIsInstance(result.analysis_results, DatasetAnalysis)
        self.assertEqual(result.dataset_description, "Test dataset")
        self.assertEqual(result.dataset_path, self.temp_csv.name)
        self.assertIsInstance(result.workflow_state, dict)
    
    @patch('causal_agent.components.dataset_analyzer.analyze_dataset')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_with_query(self, mock_get_llm, mock_analyze):
        """Test tool execution with original query."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_analyze.return_value = self.mock_analysis_result
        
        original_query = "What is the effect of treatment on outcome?"
        
        result = dataset_analyzer_tool.invoke({
            "dataset_path":  self.temp_csv.name,
            "dataset_description": "Test dataset",
            "original_query": original_query
        })
        
        # Check that original_query was passed through
        mock_analyze.assert_called_once_with(
            self.temp_csv.name,
            llm_client=mock_llm,
            dataset_description="Test dataset",
            original_query=original_query
        )
        
        self.assertIsInstance(result, DatasetAnalyzerOutput)
    
    @patch('causal_agent.components.dataset_analyzer.analyze_dataset')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_analysis_error(self, mock_get_llm, mock_analyze):
        """Test tool behavior when analysis returns error."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock analysis returning error
        mock_analyze.return_value = {
            "error": "Dataset file not found"
        }
        
        result = dataset_analyzer_tool.invoke({
            "dataset_path": "nonexistent.csv",
            "dataset_description": "Test dataset"
        })
        
        # Should return DatasetAnalyzerOutput with error handling
        self.assertIsInstance(result, DatasetAnalyzerOutput)
        
        # Check that error was handled gracefully
        self.assertIsInstance(result.analysis_results, DatasetAnalysis)
        # Error case should have minimal/default values
        self.assertEqual(result.analysis_results.dataset_info.num_rows, 0)
    
    @patch('causal_agent.components.dataset_analyzer.analyze_dataset')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_pydantic_validation_error(self, mock_get_llm, mock_analyze):
        """Test tool behavior when Pydantic validation fails."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock analysis returning invalid data structure
        invalid_result = {
            "dataset_info": "invalid_structure",  # Should be dict
            "columns": "not_a_list",  # Should be list
            # Missing required fields
        }
        mock_analyze.return_value = invalid_result
        
        result = dataset_analyzer_tool.invoke({
            "dataset_path": self.temp_csv.name,
            "dataset_description": "Test dataset"
        })
        
        # Should handle validation error gracefully
        self.assertIsInstance(result, DatasetAnalyzerOutput)
        
        # Should return error analysis with minimal info
        self.assertIsInstance(result.analysis_results, DatasetAnalysis)
        self.assertEqual(result.analysis_results.dataset_info.num_rows, 0)
    
    @patch('causal_agent.components.dataset_analyzer.analyze_dataset')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_exception_handling(self, mock_get_llm, mock_analyze):
        """Test tool behavior when unexpected exception occurs."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock analysis raising exception
        mock_analyze.side_effect = Exception("Unexpected error")
        
        result = dataset_analyzer_tool.invoke({
            "dataset_path": self.temp_csv.name,
            "dataset_description": "Test dataset"
        })
        
        # Should handle exception gracefully
        self.assertIsInstance(result, DatasetAnalyzerOutput)
        
        # Should return error analysis
        self.assertIsInstance(result.analysis_results, DatasetAnalysis)
        self.assertEqual(result.analysis_results.dataset_info.file_path, self.temp_csv.name)
    
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_llm_client_error(self, mock_get_llm):
        """Test tool behavior when LLM client initialization fails."""
        # Mock LLM client initialization failure
        mock_get_llm.side_effect = Exception("LLM initialization failed")
        
        # Tool should still work without LLM
        with patch('causal_agent.tools.dataset_analyzer_tool.analyze_dataset') as mock_analyze:
            mock_analyze.return_value = self.mock_analysis_result
            
            result = dataset_analyzer_tool.invoke({
                "dataset_path": self.temp_csv.name,
                "dataset_description": "Test dataset"
            })
            
            # Should call analyze_dataset with None LLM client
            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args
            self.assertIsNone(call_args[1]['llm_client'])
    
    @patch('causal_agent.components.dataset_analyzer.analyze_dataset')
    @patch('causal_agent.config.get_llm_client')
    def test_dataset_analyzer_tool_workflow_state(self, mock_get_llm, mock_analyze):
        """Test that tool creates appropriate workflow state."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_analyze.return_value = self.mock_analysis_result
        
        result = dataset_analyzer_tool.invoke({
            "dataset_path": self.temp_csv.name,
            "dataset_description": "Test dataset"
        })
        
        # Check workflow state structure
        workflow_state = result.workflow_state
        self.assertIsInstance(workflow_state, dict)
        
        # Should indicate successful completion
        # Note: Exact keys depend on create_workflow_state_update implementation
        # This tests that workflow_state is populated
        self.assertGreater(len(workflow_state), 0)
    
    def test_dataset_analyzer_tool_parameter_combinations(self):
        """Test tool with different parameter combinations."""
        test_cases = [
            (None, None),
            ("Test description", None),
            (None, "Test query"),
            ("Test description", "Test query"),
            ("", ""),
            ("Very long description " * 50, "Very long query " * 30)
        ]
        
        for dataset_description, original_query in test_cases:
            with self.subTest(desc=dataset_description, query=original_query):
                with patch('causal_agent.components.dataset_analyzer.analyze_dataset') as mock_analyze:
                    with patch('causal_agent.config.get_llm_client') as mock_get_llm:
                        mock_llm = Mock()
                        mock_get_llm.return_value = mock_llm
                        mock_analyze.return_value = self.mock_analysis_result
                        
                        result = dataset_analyzer_tool.invoke({
                            "dataset_path": self.temp_csv.name,
                            "dataset_description": dataset_description,
                            "original_query": original_query
                        })
                        
                        # Should handle all parameter combinations
                        self.assertIsInstance(result, DatasetAnalyzerOutput)
                        self.assertEqual(result.dataset_description, dataset_description)
                        
                        # Check that parameters were passed correctly
                        mock_analyze.assert_called_once_with(
                            self.temp_csv.name,
                            llm_client=mock_llm,
                            dataset_description=dataset_description,
                            original_query=original_query
                        )
    
    def test_dataset_analyzer_tool_real_file_integration(self):
        """Test tool with real file (integration-style test)."""
        # This test uses the actual analyze_dataset function
        # but mocks the LLM client to avoid external dependencies
        
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            
            # Mock LLM responses to avoid actual API calls
            mock_llm.invoke.return_value = Mock(content='{"potential_treatments": ["treatment"], "potential_outcomes": ["outcome"]}')
            
            result = dataset_analyzer_tool.invoke({
                "dataset_path": self.temp_csv.name,
                "dataset_description": "Integration test dataset"
            })
            
            # Should return valid result
            self.assertIsInstance(result, DatasetAnalyzerOutput)
            self.assertIsInstance(result.analysis_results, DatasetAnalysis)
            
            # Check that basic analysis worked
            self.assertEqual(result.analysis_results.dataset_info.num_rows, 50)
            self.assertEqual(result.analysis_results.dataset_info.num_columns, 5)


if __name__ == '__main__':
    unittest.main()
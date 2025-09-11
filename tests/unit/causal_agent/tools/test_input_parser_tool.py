"""Unit tests for input parser tool."""

import unittest
from unittest.mock import Mock, patch
import pytest

from causal_agent.tools.input_parser_tool import input_parser_tool
from tests.base import CausalAgentTestCase


class TestInputParserTool(CausalAgentTestCase):
    """Test cases for input parser tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock successful parse_input result
        self.mock_parse_result = {
            "original_query": "What is the effect of treatment on outcome?",
            "dataset_path": "data/test.csv",
            "query_type": "EFFECT_ESTIMATION",
            "extracted_variables": {
                "treatment": ["treatment"],
                "outcome": ["outcome"],
                "covariates_mentioned": ["age"],
                "grouping_vars": [],
                "instruments_mentioned": []
            },
            "constraints": []
        }
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_structured_input(self, mock_get_llm, mock_parse_input):
        """Test tool with structured input format."""
        # Mock LLM client
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock parse_input function
        mock_parse_input.return_value = self.mock_parse_result
        
        # Structured input text
        input_text = """My question is: What is the effect of treatment on outcome?
The dataset is located at: data/test.csv
Dataset Description: This is a test dataset for causal analysis with treatment and outcome variables."""
        
        result = input_parser_tool(input_text)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("query", result)
        self.assertIn("dataset_path", result)
        self.assertIn("dataset_description", result)
        
        # Check extracted values
        self.assertEqual(result["query"], "What is the effect of treatment on outcome?")
        self.assertEqual(result["dataset_path"], "data/test.csv")
        self.assertIn("test dataset", result["dataset_description"])
        
        # Check that parse_input was called correctly
        mock_parse_input.assert_called_once()
        call_args = mock_parse_input.call_args
        self.assertEqual(call_args[0][0], "What is the effect of treatment on outcome?")
        self.assertEqual(call_args[1]["dataset_path_arg"], "data/test.csv")
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_unstructured_input(self, mock_get_llm, mock_parse_input):
        """Test tool with unstructured input."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        # Unstructured input text
        input_text = "I want to analyze the causal effect of education on income using my dataset."
        
        result = input_parser_tool(input_text)
        
        # Should handle unstructured input
        self.assertIsInstance(result, dict)
        
        # Should use the entire input as query when no structured format found
        self.assertEqual(result["query"], input_text)
        self.assertIsNone(result["dataset_path"])
        self.assertIsNone(result["dataset_description"])
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_partial_structure(self, mock_get_llm, mock_parse_input):
        """Test tool with partially structured input."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        # Partially structured input (only query and path)
        input_text = """My question is: Does training improve performance?
The dataset is located at: training_data.csv"""
        
        result = input_parser_tool(input_text)
        
        # Should extract available structured information
        self.assertEqual(result["query"], "Does training improve performance?")
        self.assertEqual(result["dataset_path"], "training_data.csv")
        self.assertIsNone(result["dataset_description"])
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_multiline_description(self, mock_get_llm, mock_parse_input):
        """Test tool with multiline dataset description."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        input_text = """My question is: What is the effect of treatment on outcome?
The dataset is located at: data/test.csv
Dataset Description: This is a comprehensive dataset
containing information about treatments and outcomes.
It includes multiple covariates and was collected
over a period of 5 years."""
        
        result = input_parser_tool(input_text)
        
        # Should capture multiline description
        description = result["dataset_description"]
        self.assertIn("comprehensive dataset", description)
        self.assertIn("5 years", description)
        self.assertIn("covariates", description)
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_case_insensitive(self, mock_get_llm, mock_parse_input):
        """Test tool with different case variations."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        # Mixed case input
        input_text = """MY QUESTION IS: What is the effect?
the dataset is located at: data.csv
dataset description: Test data"""
        
        result = input_parser_tool(input_text)
        
        # Should handle case variations
        self.assertEqual(result["query"], "What is the effect?")
        self.assertEqual(result["dataset_path"], "data.csv")
        self.assertEqual(result["dataset_description"], "Test data")
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_llm_error(self, mock_get_llm, mock_parse_input):
        """Test tool behavior when LLM initialization fails."""
        # Mock LLM initialization failure
        mock_get_llm.side_effect = Exception("LLM initialization failed")
        
        mock_parse_input.return_value = self.mock_parse_result
        
        input_text = "My question is: Test query?"
        
        result = input_parser_tool(input_text)
        
        # Should still work without LLM
        self.assertIsInstance(result, dict)
        self.assertEqual(result["query"], "Test query?")
        
        # Should call parse_input with None LLM
        call_args = mock_parse_input.call_args
        self.assertIsNone(call_args[1]["llm"])
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_parse_input_error(self, mock_get_llm, mock_parse_input):
        """Test tool behavior when parse_input fails."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock parse_input raising exception
        mock_parse_input.side_effect = Exception("Parse input failed")
        
        input_text = "My question is: Test query?"
        
        result = input_parser_tool(input_text)
        
        # Should handle error gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("Parse input failed", result["error"])
        
        # Should still include basic extracted information
        self.assertEqual(result["query"], "Test query?")
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_workflow_state(self, mock_get_llm, mock_parse_input):
        """Test workflow state creation."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        input_text = "My question is: Test query?"
        
        result = input_parser_tool(input_text)
        
        # Should include workflow state information
        # The exact keys depend on create_workflow_state_update implementation
        workflow_keys = [k for k in result.keys() if 'step' in k or 'tool' in k or 'reason' in k]
        self.assertGreater(len(workflow_keys), 0)
    
    def test_input_parser_tool_different_formats(self):
        """Test tool with different input formats."""
        input_formats = [
            {
                "text": "My question is: Effect of X on Y?\nThe dataset is located at: data.csv\nDataset Description: Test",
                "expected_query": "Effect of X on Y?",
                "expected_path": "data.csv",
                "expected_desc": "Test"
            },
            {
                "text": "Just a simple question about causality",
                "expected_query": "Just a simple question about causality",
                "expected_path": None,
                "expected_desc": None
            },
            {
                "text": "My question is: Complex query?\nDataset Description: Only description, no path",
                "expected_query": "Complex query?",
                "expected_path": None,
                "expected_desc": "Only description, no path"
            }
        ]
        
        for input_format in input_formats:
            with self.subTest(text=input_format["text"][:30]):
                with patch('causal_agent.components.input_parser.parse_input') as mock_parse_input:
                    with patch('causal_agent.config.get_llm_client') as mock_get_llm:
                        mock_llm = Mock()
                        mock_get_llm.return_value = mock_llm
                        mock_parse_input.return_value = self.mock_parse_result
                        
                        result = input_parser_tool(input_format["text"])
                        
                        # Check extracted values match expectations
                        self.assertEqual(result["query"], input_format["expected_query"])
                        self.assertEqual(result["dataset_path"], input_format["expected_path"])
                        self.assertEqual(result["dataset_description"], input_format["expected_desc"])
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_empty_input(self, mock_get_llm, mock_parse_input):
        """Test tool with empty input."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        result = input_parser_tool("")
        
        # Should handle empty input gracefully
        self.assertIsInstance(result, dict)
        self.assertEqual(result["query"], "")
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_special_characters(self, mock_get_llm, mock_parse_input):
        """Test tool with special characters in input."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_parse_input.return_value = self.mock_parse_result
        
        input_text = """My question is: What's the effect of X@#$% on Y!??
The dataset is located at: /path/with spaces/data-file_v2.csv
Dataset Description: Dataset with émojis 🚀 and symbols ∑∆"""
        
        result = input_parser_tool(input_text)
        
        # Should handle special characters
        self.assertIn("X@#$%", result["query"])
        self.assertIn("spaces", result["dataset_path"])
        self.assertIn("🚀", result["dataset_description"])
    
    @patch('causal_agent.components.input_parser.parse_input')
    @patch('causal_agent.config.get_llm_client')
    def test_input_parser_tool_integration_with_parse_input(self, mock_get_llm, mock_parse_input):
        """Test integration between tool and parse_input function."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Return different parse_input result to test integration
        custom_parse_result = {
            "original_query": "Custom query",
            "dataset_path": "custom_path.csv",
            "query_type": "CORRELATION",
            "extracted_variables": {"treatment": [], "outcome": []},
            "constraints": ["age > 25"]
        }
        mock_parse_input.return_value = custom_parse_result
        
        input_text = "My question is: Custom query?"
        
        result = input_parser_tool(input_text)
        
        # Should integrate parse_input results
        self.assertIn("parsed_query_info", result)
        parsed_info = result["parsed_query_info"]
        self.assertEqual(parsed_info["query_type"], "CORRELATION")
        self.assertEqual(parsed_info["constraints"], ["age > 25"])


if __name__ == '__main__':
    unittest.main()
"""Unit tests for input parser component."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import pytest
from pydantic import ValidationError

from causal_agent.components.input_parser import (
    parse_input,
    extract_dataset_path,
    extract_dataset_path_regex,
    _extract_query_information_with_llm,
    _validate_llm_output,
    _build_llm_prompt,
    ParsedQueryInfo,
    ParsedVariables
)
from tests.base import CausalAgentTestCase


class TestInputParser(CausalAgentTestCase):
    """Test cases for input parser component."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock LLM client
        self.mock_llm = Mock()
        
        # Create test dataset info
        self.dataset_info = {
            "columns": ["treatment", "outcome", "age", "income"],
            "column_types": {
                "treatment": "int64",
                "outcome": "float64", 
                "age": "int64",
                "income": "float64"
            },
            "sample_rows": [
                {"treatment": 1, "outcome": 5.2, "age": 25, "income": 50000},
                {"treatment": 0, "outcome": 3.8, "age": 30, "income": 45000}
            ]
        }
        
        # Create temporary CSV file for path testing
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write("treatment,outcome,age\n1,5.2,25\n0,3.8,30\n")
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_parse_input_basic(self):
        """Test basic input parsing functionality."""
        query = "What is the effect of treatment on outcome?"
        
        result = parse_input(query, llm=self.mock_llm)
        
        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertIn("original_query", result)
        self.assertIn("query_type", result)
        self.assertIn("extracted_variables", result)
        self.assertIn("constraints", result)
        
        self.assertEqual(result["original_query"], query)
    
    def test_parse_input_with_dataset_path(self):
        """Test parsing with explicit dataset path."""
        query = "Analyze the effect of treatment on outcome"
        dataset_path = self.temp_csv.name
        
        result = parse_input(query, dataset_path_arg=dataset_path, llm=self.mock_llm)
        
        self.assertEqual(result["dataset_path"], dataset_path)
    
    def test_parse_input_with_dataset_info(self):
        """Test parsing with dataset context information."""
        query = "What is the effect of treatment on outcome?"
        
        # Mock successful LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_type": "EFFECT_ESTIMATION",
            "variables": {
                "treatment": ["treatment"],
                "outcome": ["outcome"],
                "covariates_mentioned": ["age"],
                "grouping_vars": [],
                "instruments_mentioned": []
            },
            "constraints": [],
            "dataset_path_mentioned": None
        })
        
        with patch('causal_agent.components.input_parser._extract_query_information_with_llm') as mock_extract:
            mock_extract.return_value = ParsedQueryInfo(
                query_type="EFFECT_ESTIMATION",
                variables=ParsedVariables(
                    treatment=["treatment"],
                    outcome=["outcome"],
                    covariates_mentioned=["age"]
                ),
                constraints=[],
                dataset_path_mentioned=None
            )
            
            result = parse_input(
                query, 
                dataset_info=self.dataset_info,
                llm=self.mock_llm
            )
        
        self.assertEqual(result["query_type"], "EFFECT_ESTIMATION")
        self.assertIn("treatment", result["extracted_variables"]["treatment"])
        self.assertIn("outcome", result["extracted_variables"]["outcome"])
    
    def test_extract_dataset_path_regex_patterns(self):
        """Test dataset path extraction with various patterns."""
        test_cases = [
            ("Use dataset at 'data/test.csv'", "data/test.csv"),
            ("Analyze the file /path/to/dataset.csv", "/path/to/dataset.csv"),
            ("Load data from dataset.csv", "dataset.csv"),
            ('Use "data/analysis.csv" for this study', "data/analysis.csv"),
            ("No dataset mentioned here", None)
        ]
        
        for query, expected_path in test_cases:
            result = extract_dataset_path_regex(query)
            if expected_path:
                self.assertEqual(result, expected_path)
            else:
                self.assertIsNone(result)
    
    def test_extract_dataset_path_with_llm_fallback(self):
        """Test dataset path extraction with LLM fallback."""
        query = "Analyze the impact using my research data"
        
        # Mock LLM response for path extraction
        mock_llm_response = Mock()
        mock_llm_response.dataset_path = "research_data.csv"
        
        with patch('causal_agent.components.input_parser._call_llm_for_path') as mock_llm_path:
            mock_llm_path.return_value = "research_data.csv"
            
            result = extract_dataset_path(query, llm=self.mock_llm)
            
            self.assertEqual(result, "research_data.csv")
    
    def test_extract_dataset_path_existing_file(self):
        """Test path extraction when file actually exists."""
        query = f"Use dataset at {self.temp_csv.name}"
        
        result = extract_dataset_path_regex(query)
        
        self.assertEqual(result, self.temp_csv.name)
    
    def test_build_llm_prompt(self):
        """Test LLM prompt building."""
        query = "What is the effect of treatment on outcome?"
        
        # Test without dataset info
        prompt = _build_llm_prompt(query)
        self.assertIn(query, prompt)
        self.assertIn("No dataset context provided", prompt)
        
        # Test with dataset info
        prompt_with_info = _build_llm_prompt(query, self.dataset_info)
        self.assertIn(query, prompt_with_info)
        self.assertIn("treatment", prompt_with_info)
        self.assertIn("outcome", prompt_with_info)
        self.assertNotIn("No dataset context provided", prompt_with_info)
    
    def test_validate_llm_output_valid(self):
        """Test LLM output validation with valid input."""
        valid_output = ParsedQueryInfo(
            query_type="EFFECT_ESTIMATION",
            variables=ParsedVariables(
                treatment=["treatment"],
                outcome=["outcome"]
            ),
            constraints=[],
            dataset_path_mentioned=None
        )
        
        # Should not raise exception
        result = _validate_llm_output(valid_output, self.dataset_info)
        self.assertTrue(result)
    
    def test_validate_llm_output_invalid_query_type(self):
        """Test LLM output validation with invalid query type."""
        invalid_output = ParsedQueryInfo(
            query_type="INVALID_TYPE",
            variables=ParsedVariables(),
            constraints=[],
            dataset_path_mentioned=None
        )
        
        with self.assertRaises(AssertionError):
            _validate_llm_output(invalid_output)
    
    def test_validate_llm_output_missing_variables_for_effect(self):
        """Test validation when effect query missing variables."""
        invalid_output = ParsedQueryInfo(
            query_type="EFFECT_ESTIMATION",
            variables=ParsedVariables(
                treatment=[],  # Empty treatment
                outcome=["outcome"]
            ),
            constraints=[],
            dataset_path_mentioned=None
        )
        
        with self.assertRaises(AssertionError):
            _validate_llm_output(invalid_output)
    
    def test_validate_llm_output_unknown_variables(self):
        """Test validation with variables not in dataset."""
        invalid_output = ParsedQueryInfo(
            query_type="EFFECT_ESTIMATION",
            variables=ParsedVariables(
                treatment=["unknown_treatment"],  # Not in dataset
                outcome=["outcome"]
            ),
            constraints=[],
            dataset_path_mentioned=None
        )
        
        with self.assertRaises(AssertionError):
            _validate_llm_output(invalid_output, self.dataset_info)
    
    @patch('causal_agent.components.input_parser._validate_llm_output')
    def test_extract_query_information_with_llm_success(self, mock_validate):
        """Test successful LLM query information extraction."""
        mock_validate.return_value = True
        
        # Mock structured LLM response
        mock_structured_llm = Mock()
        mock_parsed_info = ParsedQueryInfo(
            query_type="EFFECT_ESTIMATION",
            variables=ParsedVariables(
                treatment=["treatment"],
                outcome=["outcome"]
            ),
            constraints=[],
            dataset_path_mentioned=None
        )
        mock_structured_llm.invoke.return_value = mock_parsed_info
        
        with patch.object(self.mock_llm, 'with_structured_output', return_value=mock_structured_llm):
            result = _extract_query_information_with_llm(
                "What is the effect of treatment on outcome?",
                self.dataset_info,
                self.mock_llm
            )
        
        self.assertEqual(result, mock_parsed_info)
    
    def test_extract_query_information_with_llm_no_client(self):
        """Test LLM extraction without client."""
        result = _extract_query_information_with_llm(
            "What is the effect of treatment on outcome?",
            self.dataset_info,
            llm=None
        )
        
        self.assertIsNone(result)
    
    @pytest.mark.parametrize("query_type,expected_variables", [
        ("EFFECT_ESTIMATION", {"treatment": ["treatment"], "outcome": ["outcome"]}),
        ("CORRELATION", {"treatment": [], "outcome": []}),
        ("DESCRIPTIVE", {"treatment": [], "outcome": []})
    ])
    def test_parse_input_different_query_types(self, query_type, expected_variables):
        """Test parsing different types of queries."""
        query = "Test query"
        
        # Mock LLM response for different query types
        mock_parsed_info = ParsedQueryInfo(
            query_type=query_type,
            variables=ParsedVariables(**expected_variables),
            constraints=[],
            dataset_path_mentioned=None
        )
        
        with patch('causal_agent.components.input_parser._extract_query_information_with_llm') as mock_extract:
            mock_extract.return_value = mock_parsed_info
            
            result = parse_input(query, llm=self.mock_llm)
        
        self.assertEqual(result["query_type"], query_type)
    
    def test_parse_input_with_constraints(self):
        """Test parsing queries with constraints."""
        query = "Effect of treatment on outcome for age > 30"
        
        mock_parsed_info = ParsedQueryInfo(
            query_type="EFFECT_ESTIMATION",
            variables=ParsedVariables(
                treatment=["treatment"],
                outcome=["outcome"]
            ),
            constraints=["age > 30"],
            dataset_path_mentioned=None
        )
        
        with patch('causal_agent.components.input_parser._extract_query_information_with_llm') as mock_extract:
            mock_extract.return_value = mock_parsed_info
            
            result = parse_input(query, llm=self.mock_llm)
        
        self.assertEqual(result["constraints"], ["age > 30"])
    
    def test_parse_input_llm_failure_fallback(self):
        """Test parsing when LLM extraction fails."""
        query = "What is the effect of treatment on outcome?"
        
        with patch('causal_agent.components.input_parser._extract_query_information_with_llm') as mock_extract:
            mock_extract.return_value = None  # LLM extraction failed
            
            result = parse_input(query, llm=self.mock_llm)
        
        # Should still return basic structure with defaults
        self.assertEqual(result["query_type"], "OTHER")
        self.assertIsInstance(result["extracted_variables"], dict)
        self.assertIsInstance(result["constraints"], list)
    
    def test_pydantic_models_validation(self):
        """Test Pydantic model validation."""
        # Test valid ParsedVariables
        valid_vars = ParsedVariables(
            treatment=["treatment"],
            outcome=["outcome"],
            covariates_mentioned=["age"]
        )
        self.assertEqual(valid_vars.treatment, ["treatment"])
        
        # Test valid ParsedQueryInfo
        valid_query_info = ParsedQueryInfo(
            query_type="EFFECT_ESTIMATION",
            variables=valid_vars,
            constraints=["age > 25"]
        )
        self.assertEqual(valid_query_info.query_type, "EFFECT_ESTIMATION")
        
        # Test invalid query type should work (no enum constraint in current model)
        # If enum constraint is added later, this test should be updated
    
    def test_parse_input_edge_cases(self):
        """Test parsing with edge cases."""
        # Empty query
        result = parse_input("", llm=self.mock_llm)
        self.assertEqual(result["original_query"], "")
        
        # Very long query
        long_query = "What is the effect of treatment on outcome? " * 100
        result = parse_input(long_query, llm=self.mock_llm)
        self.assertEqual(result["original_query"], long_query)
        
        # Query with special characters
        special_query = "Effect of treatment@#$% on outcome!??"
        result = parse_input(special_query, llm=self.mock_llm)
        self.assertEqual(result["original_query"], special_query)


if __name__ == '__main__':
    unittest.main()
"""
Tests for the LLM helpers module.
"""

import unittest
import json
from unittest.mock import MagicMock, patch
import logging

from causal_agent.utils.llm_helpers import call_llm_with_json_output


class TestLLMHelpers(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.test_prompt = "Test prompt"
    
    def test_call_llm_with_json_output_success(self):
        """Test successful LLM call with valid JSON response."""
        test_json = {"key": "value", "number": 42}
        mock_response = MagicMock()
        mock_response.content = json.dumps(test_json)
        self.mock_llm.invoke.return_value = mock_response
        
        result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
        
        self.assertEqual(result, test_json)
        self.mock_llm.invoke.assert_called_once_with(self.test_prompt)
    
    def test_call_llm_with_json_output_with_markdown_fences(self):
        """Test LLM call with JSON wrapped in markdown code fences."""
        test_json = {"method": "linear_regression", "confidence": 0.95}
        json_with_fences = f"```json\n{json.dumps(test_json)}\n```"
        
        mock_response = MagicMock()
        mock_response.content = json_with_fences
        self.mock_llm.invoke.return_value = mock_response
        
        result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
        
        self.assertEqual(result, test_json)
    
    def test_call_llm_with_json_output_with_plain_backticks(self):
        """Test LLM call with JSON wrapped in plain backticks."""
        test_json = {"treatment": "education", "outcome": "income"}
        json_with_backticks = f"```\n{json.dumps(test_json)}\n```"
        
        mock_response = MagicMock()
        mock_response.content = json_with_backticks
        self.mock_llm.invoke.return_value = mock_response
        
        result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
        
        self.assertEqual(result, test_json)
    
    def test_call_llm_with_json_output_with_extra_text(self):
        """Test LLM call with JSON surrounded by extra text."""
        test_json = {"selected_method": "difference_in_differences"}
        response_text = f"Here is the analysis:\n```json\n{json.dumps(test_json)}\n```\nThis concludes the analysis."
        
        mock_response = MagicMock()
        mock_response.content = response_text
        self.mock_llm.invoke.return_value = mock_response
        
        result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
        
        self.assertEqual(result, test_json)
    
    def test_call_llm_with_json_output_none_llm(self):
        """Test function behavior when LLM is None."""
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(None, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.warning.assert_called_once()
    
    def test_call_llm_with_json_output_invalid_json(self):
        """Test function behavior with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON"
        self.mock_llm.invoke.return_value = mock_response
        
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.error.assert_called()
    
    def test_call_llm_with_json_output_empty_response(self):
        """Test function behavior with empty response."""
        mock_response = MagicMock()
        mock_response.content = ""
        self.mock_llm.invoke.return_value = mock_response
        
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.error.assert_called()
    
    def test_call_llm_with_json_output_non_dict_json(self):
        """Test function behavior when JSON is valid but not a dictionary."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(["list", "instead", "of", "dict"])
        self.mock_llm.invoke.return_value = mock_response
        
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.error.assert_called()
    
    def test_call_llm_with_json_output_llm_exception(self):
        """Test function behavior when LLM call raises an exception."""
        self.mock_llm.invoke.side_effect = Exception("LLM API error")
        
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.error.assert_called()
    
    def test_call_llm_with_json_output_no_content_attribute(self):
        """Test function behavior when response has no content attribute."""
        mock_response = MagicMock()
        del mock_response.content  # Remove content attribute
        self.mock_llm.invoke.return_value = mock_response
        
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.error.assert_called()
    
    def test_call_llm_with_json_output_non_string_content(self):
        """Test function behavior when content is not a string."""
        mock_response = MagicMock()
        mock_response.content = 12345  # Non-string content
        self.mock_llm.invoke.return_value = mock_response
        
        with patch('causal_agent.utils.llm_helpers.logger') as mock_logger:
            result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
            
            self.assertIsNone(result)
            mock_logger.error.assert_called()
    
    def test_call_llm_with_json_output_complex_json(self):
        """Test function with complex nested JSON structure."""
        test_json = {
            "selected_method": "instrumental_variable",
            "method_justification": "Strong instrument available",
            "alternative_methods": ["propensity_score_matching", "linear_regression"],
            "assumptions": {
                "relevance": True,
                "exclusion": True,
                "monotonicity": "assumed"
            },
            "confidence": 0.85
        }
        
        mock_response = MagicMock()
        mock_response.content = json.dumps(test_json, indent=2)
        self.mock_llm.invoke.return_value = mock_response
        
        result = call_llm_with_json_output(self.mock_llm, self.test_prompt)
        
        self.assertEqual(result, test_json)


if __name__ == '__main__':
    unittest.main()
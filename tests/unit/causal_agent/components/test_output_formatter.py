"""
Tests for causal_agent.components.output_formatter module
"""
import pytest
from unittest.mock import Mock, patch
from causal_agent.components.output_formatter import format_output


class TestOutputFormatter:
    """Test output formatting functions"""
    
    def test_format_output_basic(self):
        """Test basic output formatting"""
        query = "What is the effect of treatment on outcome?"
        method = "linear_regression"
        results = {
            'effect_estimate': 2.5,
            'standard_error': 0.5,
            'p_value': 0.01,
            'confidence_interval': [1.5, 3.5]
        }
        explanation = {
            'final_explanation_text': 'The treatment has a positive effect.'
        }
        
        result = format_output(query, method, results, explanation)
        
        # Should return a FormattedOutput object
        assert result is not None
        assert hasattr(result, 'query')
        assert hasattr(result, 'method')
        assert hasattr(result, 'results')
        assert hasattr(result, 'explanation')
    
    def test_format_output_with_dataset_info(self):
        """Test output formatting with dataset analysis"""
        query = "What is the effect of treatment on outcome?"
        method = "propensity_score_matching"
        results = {'effect_estimate': 1.8}
        explanation = {'final_explanation_text': 'Analysis complete.'}
        dataset_analysis = {
            'dataset_info': {'num_rows': 100, 'num_columns': 5}
        }
        dataset_description = "Test dataset"
        
        result = format_output(
            query, method, results, explanation, 
            dataset_analysis, dataset_description
        )
        
        assert result is not None
    
    def test_format_output_minimal_inputs(self):
        """Test output formatting with minimal inputs"""
        query = "Simple query"
        method = "difference_in_means"
        results = {}
        explanation = {}
        
        result = format_output(query, method, results, explanation)
        
        assert result is not None
    
    def test_format_output_with_error_results(self):
        """Test output formatting when results contain errors"""
        query = "What is the effect?"
        method = "instrumental_variable"
        results = {
            'error': 'Estimation failed',
            'effect_estimate': None
        }
        explanation = {
            'final_explanation_text': 'Analysis could not be completed.'
        }
        
        result = format_output(query, method, results, explanation)
        
        assert result is not None
    
    def test_format_output_none_values(self):
        """Test output formatting with None values"""
        query = "Test query"
        method = "linear_regression"
        results = {
            'effect_estimate': None,
            'standard_error': None,
            'p_value': None
        }
        explanation = {'final_explanation_text': None}
        
        result = format_output(query, method, results, explanation)
        
        assert result is not None
    
    def test_format_output_empty_strings(self):
        """Test output formatting with empty strings"""
        query = ""
        method = ""
        results = {}
        explanation = {'final_explanation_text': ""}
        
        result = format_output(query, method, results, explanation)
        
        assert result is not None
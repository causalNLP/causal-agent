"""Unit tests for explanation generator component."""

import unittest
from unittest.mock import Mock, patch
import pytest

from causal_agent.components.explanation_generator import (
    generate_explanation,
    get_method_explanation,
    explain_assumptions,
    explain_application,
    explain_limitations,
    generate_interpretation_guide
)
from tests.base import CausalAgentTestCase


class TestExplanationGenerator(CausalAgentTestCase):
    """Test cases for explanation generator component."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock method info
        self.method_info = {
            "method": "backdoor_adjustment",
            "assumptions": [
                "No unmeasured confounders",
                "Positivity assumption",
                "Consistency assumption"
            ],
            "justification": "Sufficient confounders available"
        }
        
        # Mock validation result
        self.validation_result = {
            "valid": True,
            "concerns": [],
            "recommendations": []
        }
        
        # Mock variables
        self.variables = {
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "covariates": ["age", "income", "education"]
        }
        
        # Mock results
        self.results = {
            "effect_estimate": 0.45,
            "confidence_interval": [0.2, 0.7],
            "p_value": 0.02,
            "method_used": "backdoor_adjustment"
        }
        
        # Mock dataset analysis
        self.dataset_analysis = {
            "sample_size": 1000,
            "num_covariates_estimate": 3,
            "data_quality_score": 0.85
        }
    
    def test_generate_explanation_basic(self):
        """Test basic explanation generation."""
        result = generate_explanation(
            method_info=self.method_info,
            validation_result=self.validation_result,
            variables=self.variables,
            results=self.results
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)
        
        # Check that explanation contains key information
        explanation = result["explanation"]
        self.assertIn("backdoor adjustment", explanation.lower())
        self.assertIn("treatment", explanation)
        self.assertIn("outcome", explanation)
    
    def test_generate_explanation_with_dataset_info(self):
        """Test explanation generation with dataset information."""
        result = generate_explanation(
            method_info=self.method_info,
            validation_result=self.validation_result,
            variables=self.variables,
            results=self.results,
            dataset_analysis=self.dataset_analysis,
            dataset_description="Test dataset for causal analysis"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)
        
        # Should incorporate dataset information
        explanation = result["explanation"]
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
    
    def test_generate_explanation_invalid_method(self):
        """Test explanation generation when method validation fails."""
        invalid_validation = {
            "valid": False,
            "recommended_method": "propensity_score",
            "concerns": ["Insufficient overlap in propensity scores"],
            "recommendations": ["Consider alternative methods"]
        }
        
        result = generate_explanation(
            method_info=self.method_info,
            validation_result=invalid_validation,
            variables=self.variables,
            results=self.results
        )
        
        # Should handle invalid method gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)
        
        # Should mention the recommended method
        explanation = result["explanation"]
        self.assertIn("propensity", explanation.lower())
    
    def test_get_method_explanation(self):
        """Test method explanation retrieval."""
        # Test known methods
        known_methods = [
            "backdoor_adjustment",
            "propensity_score",
            "instrumental_variable",
            "regression_discontinuity",
            "difference_in_differences"
        ]
        
        for method in known_methods:
            explanation = get_method_explanation(method)
            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)
            # Should contain method name or related terms
            self.assertTrue(
                any(term in explanation.lower() for term in method.split('_'))
            )
    
    def test_get_method_explanation_unknown_method(self):
        """Test method explanation for unknown method."""
        explanation = get_method_explanation("unknown_method")
        self.assertIsInstance(explanation, str)
        # Should provide generic explanation
        self.assertIn("causal", explanation.lower())
    
    def test_explain_assumptions(self):
        """Test assumption explanation generation."""
        assumptions = [
            "No unmeasured confounders",
            "Positivity assumption",
            "Consistency assumption"
        ]
        
        explanation = explain_assumptions(assumptions)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        
        # Should mention key assumption concepts
        for assumption in assumptions:
            # Check if key terms from assumptions appear in explanation
            key_terms = assumption.lower().split()
            self.assertTrue(
                any(term in explanation.lower() for term in key_terms if len(term) > 3)
            )
    
    def test_explain_assumptions_empty_list(self):
        """Test assumption explanation with empty list."""
        explanation = explain_assumptions([])
        self.assertIsInstance(explanation, str)
        # Should provide generic message about assumptions
        self.assertIn("assumption", explanation.lower())
    
    def test_explain_application(self):
        """Test application explanation generation."""
        explanation = explain_application(
            method="backdoor_adjustment",
            treatment="treatment",
            outcome="outcome",
            covariates=["age", "income"],
            variables=self.variables
        )
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        
        # Should mention key variables
        self.assertIn("treatment", explanation)
        self.assertIn("outcome", explanation)
        self.assertIn("age", explanation)
        self.assertIn("income", explanation)
    
    def test_explain_application_no_covariates(self):
        """Test application explanation without covariates."""
        explanation = explain_application(
            method="diff_in_means",
            treatment="treatment",
            outcome="outcome",
            covariates=[],
            variables={"treatment_variable": "treatment", "outcome_variable": "outcome"}
        )
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        # Should handle case with no covariates
    
    def test_explain_limitations(self):
        """Test limitations explanation generation."""
        concerns = [
            "Potential unmeasured confounding",
            "Limited sample size",
            "Measurement error in outcome"
        ]
        
        explanation = explain_limitations("backdoor_adjustment", concerns)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        
        # Should mention concerns
        for concern in concerns:
            key_terms = concern.lower().split()
            self.assertTrue(
                any(term in explanation.lower() for term in key_terms if len(term) > 3)
            )
    
    def test_explain_limitations_no_concerns(self):
        """Test limitations explanation with no concerns."""
        explanation = explain_limitations("backdoor_adjustment", [])
        self.assertIsInstance(explanation, str)
        # Should provide general limitations information
        self.assertIn("limitation", explanation.lower())
    
    def test_generate_interpretation_guide(self):
        """Test interpretation guide generation."""
        guide = generate_interpretation_guide(
            method="backdoor_adjustment",
            treatment="treatment",
            outcome="outcome"
        )
        
        self.assertIsInstance(guide, str)
        self.assertGreater(len(guide), 0)
        
        # Should provide guidance on interpreting results
        self.assertIn("interpret", guide.lower())
        self.assertIn("effect", guide.lower())
    
    def test_generate_explanation_different_methods(self):
        """Test explanation generation for different methods."""
        methods = [
            "backdoor_adjustment",
            "propensity_score",
            "instrumental_variable",
            "regression_discontinuity",
            "difference_in_differences",
            "linear_regression"
        ]
        
        for method in methods:
            with self.subTest(method=method):
                method_info = self.method_info.copy()
                method_info["method"] = method
                
                result = generate_explanation(
                    method_info=method_info,
                    validation_result=self.validation_result,
                    variables=self.variables,
                    results=self.results
                )
                
                # Should work for all methods
                self.assertIsInstance(result, dict)
                self.assertIn("explanation", result)
                
                explanation = result["explanation"]
                self.assertIsInstance(explanation, str)
                self.assertGreater(len(explanation), 0)
    
    def test_generate_explanation_with_llm(self):
        """Test explanation generation with LLM client."""
        mock_llm = Mock()
        
        result = generate_explanation(
            method_info=self.method_info,
            validation_result=self.validation_result,
            variables=self.variables,
            results=self.results,
            llm=mock_llm
        )
        
        # Should handle LLM parameter (even if not used in current implementation)
        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)
    
    def test_generate_explanation_edge_cases(self):
        """Test explanation generation with edge cases."""
        # Test with minimal information
        minimal_method_info = {"method": "unknown_method"}
        minimal_variables = {"treatment_variable": "T", "outcome_variable": "Y"}
        minimal_results = {}
        
        result = generate_explanation(
            method_info=minimal_method_info,
            validation_result={},
            variables=minimal_variables,
            results=minimal_results
        )
        
        # Should handle minimal information gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)
        
        # Test with None validation_result
        result_none_validation = generate_explanation(
            method_info=self.method_info,
            validation_result=None,
            variables=self.variables,
            results=self.results
        )
        
        self.assertIsInstance(result_none_validation, dict)
        self.assertIn("explanation", result_none_validation)
    
    def test_explanation_content_quality(self):
        """Test that generated explanations meet quality standards."""
        result = generate_explanation(
            method_info=self.method_info,
            validation_result=self.validation_result,
            variables=self.variables,
            results=self.results,
            dataset_analysis=self.dataset_analysis
        )
        
        explanation = result["explanation"]
        
        # Quality checks
        self.assertGreater(len(explanation), 100)  # Should be reasonably detailed
        self.assertLess(len(explanation), 5000)    # But not excessively long
        
        # Should contain key causal inference concepts
        causal_terms = ["causal", "effect", "treatment", "outcome"]
        for term in causal_terms:
            self.assertIn(term, explanation.lower())
        
        # Should be properly formatted (no obvious formatting issues)
        self.assertNotIn("None", explanation)  # No None values leaked into text
        self.assertNotIn("[]", explanation)    # No empty list representations


if __name__ == '__main__':
    unittest.main()
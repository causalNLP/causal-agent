"""Unit tests for decision tree component."""

import unittest
from unittest.mock import Mock, patch
import pytest

from causal_agent.components.decision_tree import (
    select_method,
    rule_based_select_method,
    DecisionTreeEngine,
    BACKDOOR_ADJUSTMENT,
    LINEAR_REGRESSION,
    DIFF_IN_MEANS,
    DIFF_IN_DIFF,
    REGRESSION_DISCONTINUITY,
    PROPENSITY_SCORE_MATCHING,
    INSTRUMENTAL_VARIABLE,
    PROPENSITY_SCORE_WEIGHTING,
    METHOD_ASSUMPTIONS
)
from tests.base import CausalAgentTestCase


class TestDecisionTree(CausalAgentTestCase):
    """Test cases for decision tree component."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Basic dataset properties for testing
        self.basic_properties = {
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "covariates": ["feature_0", "feature_1"],
            "is_rct": False,
            "has_temporal_structure": False,
            "treatment_variable_type": "binary"
        }
        
        # Mock LLM client
        self.mock_llm = Mock()
    
    def test_select_method_basic_observational(self):
        """Test method selection for basic observational data."""
        result = select_method(self.basic_properties)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("selected_method", result)
        self.assertIn("method_justification", result)
        self.assertIn("method_assumptions", result)
        self.assertIn("alternatives", result)
        
        # Should select a reasonable method for observational data
        selected_method = result["selected_method"]
        self.assertIn(selected_method, [
            PROPENSITY_SCORE_MATCHING,
            PROPENSITY_SCORE_WEIGHTING,
            LINEAR_REGRESSION,
            BACKDOOR_ADJUSTMENT
        ])
        
        # Check assumptions are provided
        self.assertIsInstance(result["method_assumptions"], list)
        self.assertGreater(len(result["method_assumptions"]), 0)
    
    def test_select_method_rct(self):
        """Test method selection for RCT data."""
        rct_properties = self.basic_properties.copy()
        rct_properties["is_rct"] = True
        
        result = select_method(rct_properties)
        
        # Should prefer RCT-appropriate methods
        selected_method = result["selected_method"]
        self.assertIn(selected_method, [
            DIFF_IN_MEANS,
            LINEAR_REGRESSION,
            INSTRUMENTAL_VARIABLE
        ])
    
    def test_select_method_with_instrument(self):
        """Test method selection with instrumental variable."""
        iv_properties = self.basic_properties.copy()
        iv_properties["instrument_variable"] = "instrument"
        
        result = select_method(iv_properties)
        
        # Should prefer instrumental variable method
        self.assertEqual(result["selected_method"], INSTRUMENTAL_VARIABLE)
        
        # Check IV-specific assumptions
        assumptions = result["method_assumptions"]
        self.assertIn("instrument is correlated with treatment (relevance)", assumptions)
        self.assertIn("instrument affects outcome only through treatment (exclusion restriction)", assumptions)
    
    def test_select_method_with_temporal_structure(self):
        """Test method selection with temporal data."""
        temporal_properties = self.basic_properties.copy()
        temporal_properties["has_temporal_structure"] = True
        temporal_properties["time_variable"] = "year"
        
        result = select_method(temporal_properties)
        
        # Should consider difference-in-differences
        self.assertEqual(result["selected_method"], DIFF_IN_DIFF)
        
        # Check DiD-specific assumptions
        assumptions = result["method_assumptions"]
        self.assertIn("parallel trends between treatment and control groups before treatment", assumptions)
    
    def test_select_method_with_rdd(self):
        """Test method selection with regression discontinuity design."""
        rdd_properties = self.basic_properties.copy()
        rdd_properties["running_variable"] = "score"
        rdd_properties["cutoff_value"] = 0.5
        
        result = select_method(rdd_properties)
        
        # Should select regression discontinuity
        self.assertEqual(result["selected_method"], REGRESSION_DISCONTINUITY)
        
        # Check RDD-specific assumptions
        assumptions = result["method_assumptions"]
        self.assertIn("units cannot precisely manipulate the running variable around the cutoff", assumptions)
    
    def test_select_method_continuous_treatment(self):
        """Test method selection with continuous treatment."""
        continuous_properties = self.basic_properties.copy()
        continuous_properties["treatment_variable_type"] = "continuous"
        
        result = select_method(continuous_properties)
        
        # Should handle continuous treatment appropriately
        selected_method = result["selected_method"]
        self.assertIn(selected_method, [
            LINEAR_REGRESSION,
            INSTRUMENTAL_VARIABLE
        ])
    
    def test_select_method_excluded_methods(self):
        """Test method selection with excluded methods."""
        excluded = [PROPENSITY_SCORE_MATCHING, PROPENSITY_SCORE_WEIGHTING]
        
        result = select_method(self.basic_properties, excluded_methods=excluded)
        
        # Should not select excluded methods
        self.assertNotIn(result["selected_method"], excluded)
        
        # Should include excluded methods in result
        self.assertEqual(set(result["excluded_methods"]), set(excluded))
    
    def test_select_method_all_excluded_error(self):
        """Test error when all viable methods are excluded."""
        # Exclude all possible methods for this scenario
        all_methods = [
            PROPENSITY_SCORE_MATCHING,
            PROPENSITY_SCORE_WEIGHTING,
            LINEAR_REGRESSION,
            BACKDOOR_ADJUSTMENT,
            INSTRUMENTAL_VARIABLE
        ]
        
        with self.assertRaises(RuntimeError):
            select_method(self.basic_properties, excluded_methods=all_methods)
    
    def test_select_method_missing_variables(self):
        """Test error handling with missing required variables."""
        incomplete_properties = {
            "treatment_variable": "treatment"
            # Missing outcome_variable
        }
        
        with self.assertRaises(ValueError):
            select_method(incomplete_properties)
    
    def test_rule_based_select_method(self):
        """Test the rule-based wrapper function."""
        # Mock dataset analysis
        dataset_analysis = {
            "temporal_structure": {
                "has_temporal_structure": False
            }
        }
        
        # Mock variables
        variables = {
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "covariates": ["feature_0"],
            "treatment_variable_type": "binary"
        }
        
        result = rule_based_select_method(
            dataset_analysis=dataset_analysis,
            variables=variables,
            is_rct=False,
            llm=self.mock_llm,
            dataset_description="Test dataset",
            original_query="Test query"
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("selected_method", result)
        self.assertIn("method_justification", result)
    
    def test_method_assumptions_completeness(self):
        """Test that all methods have defined assumptions."""
        # Get all method constants
        method_names = [
            BACKDOOR_ADJUSTMENT,
            LINEAR_REGRESSION,
            DIFF_IN_MEANS,
            DIFF_IN_DIFF,
            REGRESSION_DISCONTINUITY,
            PROPENSITY_SCORE_MATCHING,
            INSTRUMENTAL_VARIABLE,
            PROPENSITY_SCORE_WEIGHTING
        ]
        
        for method in method_names:
            self.assertIn(method, METHOD_ASSUMPTIONS)
            assumptions = METHOD_ASSUMPTIONS[method]
            self.assertIsInstance(assumptions, list)
            self.assertGreater(len(assumptions), 0)
            
            # Check that assumptions are strings
            for assumption in assumptions:
                self.assertIsInstance(assumption, str)
                self.assertGreater(len(assumption), 0)


class TestDecisionTreeEngine(CausalAgentTestCase):
    """Test cases for DecisionTreeEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        self.engine = DecisionTreeEngine(verbose=False)
        self.test_data = self.create_mock_dataset()
        
        # Mock dataset analysis
        self.dataset_analysis = {
            "temporal_structure": {
                "has_temporal_structure": False
            },
            "potential_instruments": [],
            "discontinuities_detected": False
        }
        
        # Mock query details
        self.query_details = {
            "treatment_variable_type": "binary",
            "is_rct": False,
            "covariate_overlap_result": 0.5
        }
    
    def test_engine_initialization(self):
        """Test DecisionTreeEngine initialization."""
        engine = DecisionTreeEngine(verbose=True)
        self.assertTrue(engine.verbose)
        
        engine_default = DecisionTreeEngine()
        self.assertFalse(engine_default.verbose)
    
    def test_select_method_basic(self):
        """Test basic method selection through engine."""
        result = self.engine.select_method(
            df=self.test_data,
            treatment="treatment",
            outcome="outcome",
            covariates=["feature_0", "feature_1"],
            dataset_analysis=self.dataset_analysis,
            query_details=self.query_details
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("selected_method", result)
        self.assertIn("method_justification", result)
        self.assertIn("method_assumptions", result)
        self.assertIn("decision_path", result)
        
        # Check decision path
        decision_path = result["decision_path"]
        self.assertIsInstance(decision_path, list)
        self.assertGreater(len(decision_path), 0)
    
    def test_select_method_with_rct(self):
        """Test method selection with RCT through engine."""
        rct_query_details = self.query_details.copy()
        rct_query_details["is_rct"] = True
        
        result = self.engine.select_method(
            df=self.test_data,
            treatment="treatment",
            outcome="outcome",
            covariates=["feature_0"],
            dataset_analysis=self.dataset_analysis,
            query_details=rct_query_details
        )
        
        # Should select RCT-appropriate method
        selected_method = result["selected_method"]
        self.assertIn(selected_method, [DIFF_IN_MEANS, LINEAR_REGRESSION])
    
    def test_select_method_with_temporal(self):
        """Test method selection with temporal data through engine."""
        temporal_analysis = self.dataset_analysis.copy()
        temporal_analysis["temporal_structure"]["has_temporal_structure"] = True
        
        temporal_query_details = self.query_details.copy()
        temporal_query_details["time_variable"] = "year"
        
        result = self.engine.select_method(
            df=self.test_data,
            treatment="treatment",
            outcome="outcome",
            covariates=["feature_0"],
            dataset_analysis=temporal_analysis,
            query_details=temporal_query_details
        )
        
        # Should consider temporal methods
        self.assertEqual(result["selected_method"], DIFF_IN_DIFF)
    
    def test_get_decision_path(self):
        """Test decision path generation for different methods."""
        test_methods = [
            LINEAR_REGRESSION,
            PROPENSITY_SCORE_MATCHING,
            INSTRUMENTAL_VARIABLE,
            REGRESSION_DISCONTINUITY,
            DIFF_IN_DIFF
        ]
        
        for method in test_methods:
            path = self.engine._get_decision_path(method)
            self.assertIsInstance(path, list)
            self.assertGreater(len(path), 0)
            
            # Check that path contains reasonable steps
            path_str = " ".join(path).lower()
            if method == LINEAR_REGRESSION:
                self.assertIn("randomized", path_str)
            elif method == PROPENSITY_SCORE_MATCHING:
                self.assertIn("observational", path_str)
                self.assertIn("overlap", path_str)
    
    def test_decision_path_content(self):
        """Test that decision paths contain expected keywords."""
        test_cases = [
            (LINEAR_REGRESSION, ["randomized", "experiment"]),
            (PROPENSITY_SCORE_MATCHING, ["observational", "overlap"]),
            (INSTRUMENTAL_VARIABLE, ["instrument"]),
            (REGRESSION_DISCONTINUITY, ["discontinuity"]),
            (DIFF_IN_DIFF, ["temporal", "panel"])
        ]
        
        for method_name, expected_keywords in test_cases:
            with self.subTest(method=method_name):
                path = self.engine._get_decision_path(method_name)
                path_text = " ".join(path).lower()
                
                for keyword in expected_keywords:
                    self.assertIn(keyword, path_text)


if __name__ == '__main__':
    unittest.main()
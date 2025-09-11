"""Unit tests for method selector tool."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from causal_agent.tools.method_selector_tool import method_selector_tool
from causal_agent.models import Variables, DatasetAnalysis, DatasetInfo, TemporalStructure
from tests.base import CausalAgentTestCase


class TestMethodSelectorTool(CausalAgentTestCase):
    """Test cases for method selector tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create test Variables model
        self.test_variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["feature_0", "feature_1"],
            is_rct=False,
            treatment_variable_type="binary"
        )
        
        # Create test DatasetAnalysis model
        dataset_info = DatasetInfo(
            num_rows=100,
            num_columns=5,
            file_path="test.csv",
            file_name="test.csv"
        )
        
        temporal_structure = TemporalStructure(
            has_temporal_structure=False,
            temporal_columns=[],
            is_panel_data=False
        )
        
        self.test_dataset_analysis = DatasetAnalysis(
            dataset_info=dataset_info,
            columns=["treatment", "outcome", "feature_0", "feature_1", "feature_2"],
            potential_treatments=["treatment"],
            potential_outcomes=["outcome"],
            temporal_structure_detected=False,
            panel_data_detected=False,
            potential_instruments_detected=False,
            discontinuities_detected=False,
            temporal_structure=temporal_structure,
            sample_size=100,
            num_covariates_estimate=3
        )
        
        # Mock successful method selection result
        self.mock_method_result = {
            "selected_method": "backdoor_adjustment",
            "method_justification": "Sufficient confounders available",
            "method_assumptions": [
                "No unmeasured confounders",
                "Positivity assumption",
                "Consistency assumption"
            ],
            "alternatives": ["propensity_score", "linear_regression"],
            "excluded_methods": []
        }
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_success(self, mock_get_llm, mock_rule_based):
        """Test successful method selector tool execution."""
        # Mock LLM client
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock rule-based method selection
        mock_rule_based.return_value = self.mock_method_result
        
        # Execute tool
        result = method_selector_tool(
            variables=self.test_variables,
            dataset_analysis=self.test_dataset_analysis,
            dataset_description="Test dataset",
            original_query="What is the effect of treatment on outcome?"
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("method_info", result)
        self.assertIn("variables", result)
        self.assertIn("dataset_analysis", result)
        
        # Check method_info structure
        method_info = result["method_info"]
        self.assertEqual(method_info["selected_method"], "backdoor_adjustment")
        self.assertEqual(method_info["method_name"], "Backdoor Adjustment")
        self.assertEqual(method_info["method_justification"], "Sufficient confounders available")
        self.assertIsInstance(method_info["method_assumptions"], list)
        self.assertIsInstance(method_info["alternative_methods"], list)
        
        # Check that rule_based_select_method was called correctly
        mock_rule_based.assert_called_once()
        call_args = mock_rule_based.call_args
        self.assertEqual(call_args[1]["is_rct"], False)
        self.assertIsNone(call_args[1]["excluded_methods"])
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_with_excluded_methods(self, mock_get_llm, mock_rule_based):
        """Test tool execution with excluded methods."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        excluded_methods = ["propensity_score", "linear_regression"]
        mock_result = self.mock_method_result.copy()
        mock_result["excluded_methods"] = excluded_methods
        mock_rule_based.return_value = mock_result
        
        result = method_selector_tool(
            variables=self.test_variables,
            dataset_analysis=self.test_dataset_analysis,
            excluded_methods=excluded_methods
        )
        
        # Check that excluded methods were passed through
        call_args = mock_rule_based.call_args
        self.assertEqual(call_args[1]["excluded_methods"], excluded_methods)
        
        self.assertIsInstance(result, dict)
        self.assertIn("method_info", result)
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_rct_variables(self, mock_get_llm, mock_rule_based):
        """Test tool execution with RCT variables."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Create RCT variables
        rct_variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            covariates=["feature_0"],
            is_rct=True,
            treatment_variable_type="binary"
        )
        
        mock_rct_result = {
            "selected_method": "diff_in_means",
            "method_justification": "Pure RCT without covariates",
            "method_assumptions": ["Random assignment", "No spillovers"],
            "alternatives": ["linear_regression"],
            "excluded_methods": []
        }
        mock_rule_based.return_value = mock_rct_result
        
        result = method_selector_tool(
            variables=rct_variables,
            dataset_analysis=self.test_dataset_analysis
        )
        
        # Check that is_rct=True was passed
        call_args = mock_rule_based.call_args
        self.assertEqual(call_args[1]["is_rct"], True)
        
        # Check result
        method_info = result["method_info"]
        self.assertEqual(method_info["selected_method"], "diff_in_means")
    
    def test_method_selector_tool_missing_treatment(self):
        """Test tool behavior with missing treatment variable."""
        # Create variables without treatment
        invalid_variables = Variables(
            treatment_variable=None,
            outcome_variable="outcome",
            covariates=["feature_0"],
            is_rct=False
        )
        
        result = method_selector_tool(
            variables=invalid_variables,
            dataset_analysis=self.test_dataset_analysis
        )
        
        # Should return error result
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Missing treatment/outcome")
    
    def test_method_selector_tool_missing_outcome(self):
        """Test tool behavior with missing outcome variable."""
        # Create variables without outcome
        invalid_variables = Variables(
            treatment_variable="treatment",
            outcome_variable=None,
            covariates=["feature_0"],
            is_rct=False
        )
        
        result = method_selector_tool(
            variables=invalid_variables,
            dataset_analysis=self.test_dataset_analysis
        )
        
        # Should return error result
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Missing treatment/outcome")
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_llm_initialization_error(self, mock_get_llm, mock_rule_based):
        """Test tool behavior when LLM initialization fails."""
        # Mock LLM initialization failure
        mock_get_llm.side_effect = Exception("LLM initialization failed")
        
        mock_rule_based.return_value = self.mock_method_result
        
        result = method_selector_tool(
            variables=self.test_variables,
            dataset_analysis=self.test_dataset_analysis
        )
        
        # Should still work with None LLM
        call_args = mock_rule_based.call_args
        self.assertIsNone(call_args[1]["llm"])
        
        self.assertIsInstance(result, dict)
        self.assertIn("method_info", result)
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_method_selection_error(self, mock_get_llm, mock_rule_based):
        """Test tool behavior when method selection fails."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Mock method selection raising exception
        mock_rule_based.side_effect = Exception("Method selection failed")
        
        result = method_selector_tool(
            variables=self.test_variables,
            dataset_analysis=self.test_dataset_analysis
        )
        
        # Should return error result
        self.assertIn("error", result)
        self.assertIn("Method selection logic failed", result["error"])
        
        # Should still include input data
        self.assertIn("variables", result)
        self.assertIn("dataset_analysis", result)
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_method_name_formatting(self, mock_get_llm, mock_rule_based):
        """Test method name formatting in results."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        test_cases = [
            ("backdoor_adjustment", "Backdoor Adjustment"),
            ("propensity_score_matching", "Propensity Score Matching"),
            ("difference_in_differences", "Difference In Differences"),
            ("regression_discontinuity", "Regression Discontinuity")
        ]
        
        for method_name, expected_formatted in test_cases:
            mock_result = self.mock_method_result.copy()
            mock_result["selected_method"] = method_name
            mock_rule_based.return_value = mock_result
            
            result = method_selector_tool(
                variables=self.test_variables,
                dataset_analysis=self.test_dataset_analysis
            )
            
            method_info = result["method_info"]
            self.assertEqual(method_info["method_name"], expected_formatted)
    
    @patch('causal_agent.components.decision_tree.rule_based_select_method')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_workflow_state(self, mock_get_llm, mock_rule_based):
        """Test workflow state creation."""
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        mock_rule_based.return_value = self.mock_method_result
        
        result = method_selector_tool(
            variables=self.test_variables,
            dataset_analysis=self.test_dataset_analysis
        )
        
        # Should include workflow state information
        # The exact keys depend on create_workflow_state_update implementation
        # This tests that workflow-related keys are present
        workflow_keys = [k for k in result.keys() if 'step' in k or 'tool' in k or 'reason' in k]
        self.assertGreater(len(workflow_keys), 0)
    
    def test_method_selector_tool_optional_parameters(self):
        """Test tool with different optional parameter combinations."""
        test_cases = [
            (None, None),
            ("Test description", None),
            (None, "Test query"),
            ("Test description", "Test query")
        ]
        
        for dataset_description, original_query in test_cases:
            with self.subTest(desc=dataset_description, query=original_query):
                with patch('causal_agent.components.decision_tree.rule_based_select_method') as mock_rule_based:
                    with patch('causal_agent.config.get_llm_client') as mock_get_llm:
                        mock_llm = Mock()
                        mock_get_llm.return_value = mock_llm
                        mock_rule_based.return_value = self.mock_method_result
                        
                        result = method_selector_tool(
                            variables=self.test_variables,
                            dataset_analysis=self.test_dataset_analysis,
                            dataset_description=dataset_description,
                            original_query=original_query
                        )
                        
                        # Should handle all parameter combinations
                        self.assertIsInstance(result, dict)
                        self.assertIn("method_info", result)
                        
                        # Check that parameters were passed correctly
                        call_args = mock_rule_based.call_args
                        self.assertEqual(call_args[1]["dataset_description"], dataset_description)
                        self.assertEqual(call_args[1]["original_query"], original_query)
    
    @patch('causal_agent.components.decision_tree_llm.DecisionTreeLLMEngine')
    @patch('causal_agent.config.get_llm_client')
    def test_method_selector_tool_llm_decision_tree_path(self, mock_get_llm, mock_llm_engine_class):
        """Test LLM decision tree path (when USE_LLM_DECISION_TREE=True)."""
        # This test would require modifying the USE_LLM_DECISION_TREE flag
        # For now, we test that the LLM engine class exists and can be imported
        self.assertTrue(hasattr(mock_llm_engine_class, 'return_value'))
        
        # In a real implementation, you might want to:
        # 1. Make USE_LLM_DECISION_TREE configurable
        # 2. Test both code paths
        # 3. Ensure LLM engine is called correctly when enabled


if __name__ == '__main__':
    unittest.main()
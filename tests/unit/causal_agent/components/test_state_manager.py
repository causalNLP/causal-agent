"""Unit tests for state manager component."""

import unittest
from causal_agent.components.state_manager import create_workflow_state_update
from tests.base import CausalAgentTestCase


class TestStateManager(CausalAgentTestCase):
    """Test cases for state manager component."""
    
    def test_create_workflow_state_update_basic(self):
        """Test basic workflow state update creation."""
        result = create_workflow_state_update(
            current_step="input_processing",
            step_completed_flag=True,
            next_tool="dataset_analyzer_tool",
            next_step_reason="Need to analyze dataset structure"
        )
        
        # Check structure
        self.assertIsInstance(result, dict)
        self.assertIn("workflow_state", result)
        
        workflow_state = result["workflow_state"]
        self.assertEqual(workflow_state["current_step"], "input_processing")
        self.assertEqual(workflow_state["input_processing_completed"], True)
        self.assertEqual(workflow_state["next_tool"], "dataset_analyzer_tool")
        self.assertEqual(workflow_state["next_step_reason"], "Need to analyze dataset structure")
        self.assertNotIn("error_message", workflow_state)
    
    def test_create_workflow_state_update_with_error(self):
        """Test workflow state update with error message."""
        result = create_workflow_state_update(
            current_step="method_execution",
            step_completed_flag=False,
            next_tool="error_handler_tool",
            next_step_reason="Method execution failed",
            error="Convergence failure in optimization"
        )
        
        workflow_state = result["workflow_state"]
        self.assertEqual(workflow_state["current_step"], "method_execution")
        self.assertEqual(workflow_state["method_execution_completed"], False)
        self.assertEqual(workflow_state["next_tool"], "error_handler_tool")
        self.assertEqual(workflow_state["next_step_reason"], "Method execution failed")
        self.assertEqual(workflow_state["error_message"], "Convergence failure in optimization")
    
    def test_create_workflow_state_update_different_steps(self):
        """Test state updates for different workflow steps."""
        test_cases = [
            {
                "current_step": "data_analysis",
                "step_completed_flag": True,
                "next_tool": "query_interpreter_tool",
                "next_step_reason": "Dataset analyzed successfully"
            },
            {
                "current_step": "method_selection",
                "step_completed_flag": True,
                "next_tool": "method_validator_tool",
                "next_step_reason": "Method selected, need validation"
            },
            {
                "current_step": "result_interpretation",
                "step_completed_flag": True,
                "next_tool": "output_formatter_tool",
                "next_step_reason": "Results interpreted, ready for formatting"
            }
        ]
        
        for case in test_cases:
            result = create_workflow_state_update(**case)
            
            workflow_state = result["workflow_state"]
            self.assertEqual(workflow_state["current_step"], case["current_step"])
            self.assertEqual(workflow_state[f"{case['current_step']}_completed"], case["step_completed_flag"])
            self.assertEqual(workflow_state["next_tool"], case["next_tool"])
            self.assertEqual(workflow_state["next_step_reason"], case["next_step_reason"])
    
    def test_create_workflow_state_update_step_completed_flag_types(self):
        """Test different types for step_completed_flag."""
        # Test with boolean True
        result_true = create_workflow_state_update(
            current_step="test_step",
            step_completed_flag=True,
            next_tool="next_tool",
            next_step_reason="Test reason"
        )
        self.assertEqual(result_true["workflow_state"]["test_step_completed"], True)
        
        # Test with boolean False
        result_false = create_workflow_state_update(
            current_step="test_step",
            step_completed_flag=False,
            next_tool="next_tool",
            next_step_reason="Test reason"
        )
        self.assertEqual(result_false["workflow_state"]["test_step_completed"], False)
        
        # Test with string (should work as the function doesn't enforce type)
        result_string = create_workflow_state_update(
            current_step="test_step",
            step_completed_flag="dataset_analyzed",
            next_tool="next_tool",
            next_step_reason="Test reason"
        )
        self.assertEqual(result_string["workflow_state"]["test_step_completed"], "dataset_analyzed")
    
    def test_create_workflow_state_update_empty_strings(self):
        """Test state update with empty strings."""
        result = create_workflow_state_update(
            current_step="",
            step_completed_flag=True,
            next_tool="",
            next_step_reason=""
        )
        
        workflow_state = result["workflow_state"]
        self.assertEqual(workflow_state["current_step"], "")
        self.assertEqual(workflow_state["_completed"], True)  # Empty step name
        self.assertEqual(workflow_state["next_tool"], "")
        self.assertEqual(workflow_state["next_step_reason"], "")
    
    def test_create_workflow_state_update_none_error(self):
        """Test state update with None error (should not include error_message)."""
        result = create_workflow_state_update(
            current_step="test_step",
            step_completed_flag=True,
            next_tool="next_tool",
            next_step_reason="Test reason",
            error=None
        )
        
        workflow_state = result["workflow_state"]
        self.assertNotIn("error_message", workflow_state)
    
    def test_create_workflow_state_update_empty_error(self):
        """Test state update with empty string error."""
        result = create_workflow_state_update(
            current_step="test_step",
            step_completed_flag=False,
            next_tool="error_tool",
            next_step_reason="Error occurred",
            error=""
        )
        
        workflow_state = result["workflow_state"]
        # Empty string is falsy, so error_message should not be included
        self.assertNotIn("error_message", workflow_state)
    
    def test_create_workflow_state_update_long_strings(self):
        """Test state update with long strings."""
        long_reason = "This is a very long reason " * 20
        long_error = "This is a very long error message " * 15
        
        result = create_workflow_state_update(
            current_step="long_test_step",
            step_completed_flag=False,
            next_tool="long_tool_name",
            next_step_reason=long_reason,
            error=long_error
        )
        
        workflow_state = result["workflow_state"]
        self.assertEqual(workflow_state["next_step_reason"], long_reason)
        self.assertEqual(workflow_state["error_message"], long_error)
    
    def test_create_workflow_state_update_special_characters(self):
        """Test state update with special characters."""
        result = create_workflow_state_update(
            current_step="step_with_@#$%",
            step_completed_flag=True,
            next_tool="tool_with_!&*()",
            next_step_reason="Reason with émojis 🚀 and symbols ∑∆",
            error="Error with unicode: αβγ"
        )
        
        workflow_state = result["workflow_state"]
        self.assertEqual(workflow_state["current_step"], "step_with_@#$%")
        self.assertEqual(workflow_state["next_tool"], "tool_with_!&*()")
        self.assertIn("🚀", workflow_state["next_step_reason"])
        self.assertIn("αβγ", workflow_state["error_message"])


if __name__ == '__main__':
    unittest.main()
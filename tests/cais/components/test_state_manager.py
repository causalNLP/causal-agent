import unittest
from cais.components.state_manager import create_workflow_state_update


class TestStateManagerUtils(unittest.TestCase):

    def test_create_workflow_state_update(self):
        """Test the happy-path workflow state update."""
        current = "step_A"
        next_tool = "tool_B"
        reason = "Reason for B"

        expected_output = {
            "workflow_state": {
                "current_step": current,
                "step_A_completed": True,
                "next_tool": next_tool,
                "next_step_reason": reason,
            }
        }

        actual_output = create_workflow_state_update(current, True, next_tool, reason)
        self.assertDictEqual(actual_output, expected_output)

    def test_create_workflow_state_update_with_error(self):
        """When an error is passed, it should appear as error_message in the state."""
        current = "step_B"
        next_tool = "tool_C"
        reason = "Retry after error"
        error_msg = "Something went wrong during step_B"

        result = create_workflow_state_update(current, False, next_tool, reason, error=error_msg)

        self.assertIn("workflow_state", result)
        ws = result["workflow_state"]
        self.assertEqual(ws["current_step"], current)
        self.assertFalse(ws["step_B_completed"])
        self.assertEqual(ws["next_tool"], next_tool)
        self.assertEqual(ws["next_step_reason"], reason)
        self.assertEqual(ws["error_message"], error_msg)

    def test_create_workflow_state_update_step_not_completed(self):
        """When step_completed_flag is False, the flag in state should be False."""
        result = create_workflow_state_update("step_X", False, "tool_Y", "Not done yet")

        ws = result["workflow_state"]
        self.assertFalse(ws["step_X_completed"])
        # No error key when no error is passed
        self.assertNotIn("error_message", ws)

    def test_create_workflow_state_update_no_error_key_by_default(self):
        """error_message should NOT appear in state when no error is provided."""
        result = create_workflow_state_update("step_A", True, "tool_B", "All good")
        self.assertNotIn("error_message", result["workflow_state"])


if __name__ == '__main__':
    unittest.main()

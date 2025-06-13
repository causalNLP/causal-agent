import unittest
from auto_causal.components.state_manager import create_workflow_state_update

class TestStateManagerUtils(unittest.TestCase):

    def test_create_workflow_state_update(self):
        '''Test the workflow state update utility function.'''
        current = "step_A"
        flag = "step_A_done"
        next_tool = "tool_B"
        reason = "Reason for B"

        expected_output = {
            "workflow_state": {
                "current_step": current,
                flag: True,
                "next_tool": next_tool,
                "next_step_reason": reason
            }
        }

        actual_output = create_workflow_state_update(current, flag, next_tool, reason)
        self.assertDictEqual(actual_output, expected_output)

if __name__ == '__main__':
    unittest.main() 
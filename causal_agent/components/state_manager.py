"""
State management utilities for the causal_agent workflow.

This module provides utility functions to create standardized state updates
for passing between tools in the causal_agent workflow.
"""

from typing import Dict, Any, Optional

def create_workflow_state_update(
    current_step: str,
    step_completed_flag: bool,
    next_tool: str,
    next_step_reason: str,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized workflow state update dictionary.
    
    Args:
        current_step: Current step in the workflow (e.g., "input_processing")
        step_completed_flag: Flag indicating which step was completed (e.g., "query_parsed")
        next_tool: Name of the next tool to call
        next_step_reason: Reason message for the next step
        error: Optional error message if the step failed
        
    Returns:
        Dictionary containing the workflow_state sub-dictionary
    """
    state_update = {
        "workflow_state": {
            "current_step": current_step,
            current_step + "_completed": step_completed_flag,
            "next_tool": next_tool,
            "next_step_reason": next_step_reason
        }
    }
    if error:
        state_update["workflow_state"]["error_message"] = error
    return state_update 
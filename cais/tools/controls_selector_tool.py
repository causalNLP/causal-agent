"""
Controls Selector Tool for selecting control variables.

This module provides a LangChain tool for selecting appropriate
control variables based on the chosen causal inference method and dataset characteristics.
"""

import logging
from typing import Dict, Any, Optional
from langchain.tools import tool

# Import component function and central LLM factory
from cais.components.controls_selector import select_controls
from cais.config import get_llm_client
from cais.components.state_manager import create_workflow_state_update

# Import shared models from central location
from cais.models import (
    Variables,
    DatasetAnalysis,
    ControlsSelectorInput
)

logger = logging.getLogger(__name__)

@tool(args_schema=ControlsSelectorInput)
def controls_selector_tool(
    method_name: str,
    variables: Variables,
    dataset_analysis: DatasetAnalysis,
    dataset_description: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Select control variables for causal estimation based on the chosen method.

    Control variables are used in methods like DiD, RDD, IV, and Linear Regression
    to improve precision of causal effect estimates by reducing standard errors.
    They are not the same as confounders used in propensity score methods.

    Args:
        method_name: The selected causal inference method name.
        variables: Pydantic model containing identified variables (T, O, instrument, running_var, time_var, etc.).
        dataset_analysis: Pydantic model containing results of dataset analysis.
        dataset_description: Optional textual description of the dataset.
        original_query: Optional original user query string.

    Returns:
        Dictionary with selected controls, updated variables, context for next step, and workflow state.
    """
    logger.info(f"Running controls_selector_tool for method: {method_name}")

    # Access data directly from arguments (they are already Pydantic models)
    variables_model = variables
    dataset_analysis_model = dataset_analysis

    # Convert Pydantic models to dicts for component call
    variables_dict = variables_model.model_dump()
    dataset_analysis_dict = dataset_analysis_model.model_dump()

    # Extract required data from dataset_analysis
    columns = dataset_analysis_dict.get("columns", [])
    column_categories = dataset_analysis_dict.get("column_categories", {})

    # Get LLM instance (optional for component)
    try:
        llm_instance = get_llm_client()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM for controls_selector_tool: {e}. Using heuristic selection.")
        llm_instance = None

    # Call the component function to select controls
    try:
        selected_controls = select_controls(
            method_name=method_name,
            variables=variables_dict,
            columns=columns,
            column_categories=column_categories,
            query=original_query or "",
            description=dataset_description or "",
            llm=llm_instance
        )

        if not isinstance(selected_controls, list):
            raise TypeError(f"select_controls component did not return a list. Got: {type(selected_controls)}")

        logger.info(f"Selected {len(selected_controls)} control variables: {selected_controls}")

    except Exception as e:
        logger.error(f"Error during controls selection: {e}", exc_info=True)
        # Construct error output
        workflow_update = create_workflow_state_update(
            current_step="controls_selection",
            step_completed_flag=False,
            next_tool="error_handler_tool",
            next_step_reason=f"Component failed: {e}",
            error=f"Component failed: {e}"
        )
        return {
            "error": f"Controls selection failed: {e}",
            "selected_controls": [],
            "variables": variables_dict,
            "method_name": method_name,
            "dataset_analysis": dataset_analysis_dict,
            "dataset_description": dataset_description,
            "original_query": original_query,
            **workflow_update.get('workflow_state', {})
        }

    # Update variables dict with selected controls
    # Note: For methods not requiring controls, select_controls returns empty list
    variables_dict["covariates"] = selected_controls

    # Recreate Variables model with updated controls
    updated_variables = Variables(**variables_dict)

    # Prepare output dictionary for next tool (method_validator)
    result = {
        "selected_controls": selected_controls,
        "variables": updated_variables.model_dump(),  # Pass updated variables as dict
        "method_name": method_name,
        "dataset_analysis": dataset_analysis_dict,
        "dataset_description": dataset_description,
        "original_query": original_query
    }

    # Determine workflow state
    controls_selected_flag = True  # Always true if no error
    next_tool_name = "method_validator_tool"
    next_reason = f"Controls selected ({len(selected_controls)} variables). Proceeding to method validation."

    workflow_update = create_workflow_state_update(
        current_step="controls_selection",
        step_completed_flag=controls_selected_flag,
        next_tool=next_tool_name,
        next_step_reason=next_reason
    )
    result.update(workflow_update.get('workflow_state', {}))

    logger.info(f"controls_selector_tool finished. Selected {len(selected_controls)} controls.")
    return result

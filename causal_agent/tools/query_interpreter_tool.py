"""
Tool for interpreting causal queries in the context of a dataset.

This module provides a LangChain tool for matching query concepts to actual
dataset variables, identifying treatment, outcome, and covariate variables.
"""

from typing import Dict, List, Any, Optional, Union 
import logging

from causal_agent.models import (
    TemporalStructure,
    DatasetInfo,
    DatasetAnalysis,
    QueryInfo,
    QueryInterpreterInput,
    Variables,
    QueryInterpreterOutput
)



logger = logging.getLogger(__name__)

from langchain.tools import tool
from causal_agent.components.query_interpreter import interpret_query
from causal_agent.components.state_manager import create_workflow_state_update


@tool()
# Modify signature to accept individual Pydantic models/types as arguments
def query_interpreter_tool(
    query_info: QueryInfo, 
    dataset_analysis: DatasetAnalysis, 
    dataset_description: str, 
    original_query: Optional[str] = None # Keep optional original_query
) -> QueryInterpreterOutput:
    """
    Interpret a causal query in the context of a specific dataset.

    Args:
        query_info: Pydantic model with parsed query information.
        dataset_analysis: Pydantic model with dataset analysis results.
        dataset_description: String description of the dataset.
        original_query: The original user query string (optional).
        
    Returns:
        A Pydantic model containing identified variables (including is_rct), dataset analysis, description, and workflow state.
    """
    logger.info("Running query_interpreter_tool with direct arguments...")
    
    # Use arguments directly, dump models to dicts for the component call
    query_info_dict = query_info.model_dump()
    dataset_analysis_dict = dataset_analysis.model_dump()
    # dataset_description is already a string
    # Call the component function 
    try:
        # Assume interpret_query returns a dictionary compatible with Variables model
        # AND that interpret_query now attempts to determine is_rct
        interpretation_dict = interpret_query(query_info_dict, dataset_analysis_dict, dataset_description)
        if not isinstance(interpretation_dict, dict):
             raise TypeError(f"interpret_query component did not return a dictionary. Got: {type(interpretation_dict)}")
             
        # Validate and structure the interpretation using Pydantic
        # This will raise validation error if interpret_query didn't return expected fields
        variables_output = Variables(**interpretation_dict)
        
    except Exception as e:
        logger.error(f"Error during query interpretation component call: {e}", exc_info=True)
        workflow_update = create_workflow_state_update(
            current_step="variable_identification",
            step_completed_flag=False, 
            next_tool="query_interpreter_tool", # Or error handler
            next_step_reason=f"Component execution failed: {e}"
        )
        error_vars = Variables() 
        # Use the passed dataset_analysis object directly in case of error
        error_analysis = dataset_analysis 
        # Return Pydantic output even on error
        return QueryInterpreterOutput(
            variables=error_vars, 
            dataset_analysis=error_analysis, 
            dataset_description=dataset_description,
            original_query=original_query, # Pass original query if available
            workflow_state=workflow_update.get('workflow_state', {})
        )

    # Create workflow state update for success
    workflow_update = create_workflow_state_update(
        current_step="variable_identification",
        step_completed_flag="variables_identified",
        next_tool="method_selector_tool",
        next_step_reason="Now that we have identified the variables, we can select an appropriate causal inference method"
    )
    
    # Construct the Pydantic output object
    output = QueryInterpreterOutput(
        variables=variables_output,
        # Pass the original dataset_analysis Pydantic model 
        dataset_analysis=dataset_analysis, 
        dataset_description=dataset_description,
        original_query=original_query, # Pass along original query
        workflow_state=workflow_update.get('workflow_state', {}) # Extract state dict
    )
    
    logger.info("query_interpreter_tool finished successfully.")
    return output
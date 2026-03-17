"""
Method Selector Tool for selecting causal inference methods.

This module provides a LangChain tool for selecting appropriate
causal inference methods based on dataset characteristics and query details.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union
from langchain_core.tools import tool 

# Import component function and central LLM factory
from cais.components.decision_tree import rule_based_select_method 
from cais.components.decision_tree_llm import DecisionTreeLLMEngine 
from cais.config import get_llm_client 
from cais.components.state_manager import create_workflow_state_update

# Import shared models from central location
from cais.models import (
    Variables, 
    DatasetAnalysis, 
    MethodSelectorInput # Still needed for args_schema
)

logger = logging.getLogger(__name__)

@tool(args_schema=MethodSelectorInput)
# Option 1: Modify signature to match args_schema fields
def method_selector_tool(
    variables: Variables, 
    dataset_analysis: DatasetAnalysis, 
    dataset_description: Optional[str] = None, 
    original_query: Optional[str] = None,
    excluded_methods: Optional[List[str]] = None,
    use_decision_tree: Optional[bool] = True
) -> Dict[str, Any]:
    """
    Select the most appropriate causal inference method based on structured input.
    
    Applies decision logic based on dataset analysis and identified variables (including is_rct).
    
    Args:
        variables: Pydantic model containing identified variables (T, O, C, IV, RDD, is_rct, etc.).
        dataset_analysis: Pydantic model containing results of dataset analysis.
        dataset_description: Optional textual description of the dataset.
        original_query: Optional original user query string.
        excluded_methods: Optional list of method names to exclude from selection.
        
    Returns:
        Dictionary with method selection details, context for next step, and workflow state.
    """
    logger.info("Running method_selector_tool with individual args...")

    is_rct_flag = variables.is_rct # Get is_rct directly from variables argument

    # Convert Pydantic models to dicts for the component call (select_method expects dicts)
    variables_dict = variables.model_dump()
    dataset_analysis_dict = dataset_analysis.model_dump()
    
    # Basic validation
    treatment = variables_dict.get("treatment_variable")
    outcome = variables_dict.get("outcome_variable")
    if not all([treatment, outcome]):
        logger.critical(f"Missing treatment or outcome variable in input. Treament:{treatment} Outcome: {outcome}")
        # Construct error output, including passed-along context
        workflow_update = create_workflow_state_update(
            current_step="method_selection", 
            step_completed_flag=False, 
            next_tool="method_selector_tool", 
            next_step_reason="Missing treatment/outcome variable in input",
            error="Missing treatment/outcome variable in input"
        )
        # Use model_dump() for analysis dict
        return { "error": "Missing treatment/outcome", 
                 "variables": variables_dict,
                 "dataset_analysis": dataset_analysis_dict, 
                 "dataset_description": dataset_description,
                 **workflow_update.get('workflow_state', {})}
        
    # Get LLM instance (optional for component)
    try:
        llm_instance = get_llm_client()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM for method_selector_tool: {e}. Proceeding without LLM features.")
        llm_instance = None
        

    # Call the component function
    try:
        if use_decision_tree:
            #print('USING DECISION TREE')
            logger.info("Using LLM-based Decision Tree Engine for method selection.")
            if not llm_instance:
                logger.warning("LLM instance is required for DecisionTreeLLMEngine but not available. Falling back to rule-based or error.")
                # Potentially raise an error or explicitly call rule-based here if LLM is mandatory for this path
                # For now, it will proceed and DecisionTreeLLMEngine will handle the missing llm

            llm_engine = DecisionTreeLLMEngine(verbose=True) # You can set verbosity as needed
            method_selection_dict = llm_engine.select_method_llm(
                dataset_analysis=dataset_analysis_dict,
                variables=variables_dict,
                is_rct=is_rct_flag if isinstance(is_rct_flag, bool) else False,
                llm=llm_instance,
                excluded_methods=excluded_methods
            )
        else:
            #print('NOT USING DECISION TREE')
            logger.info("Using Rule-based Decision Tree Engine for method selection.")
            # Pass dicts and the is_rct flag
            method_selection_dict = rule_based_select_method(
                 dataset_analysis=dataset_analysis_dict, 
                 variables=variables_dict,
                 is_rct=is_rct_flag if isinstance(is_rct_flag, bool) else False, # Handle None case
                 llm=llm_instance, 
                 dataset_description = dataset_description,
                 original_query = original_query,
                 excluded_methods = excluded_methods
            )
    except Exception as e:
        logger.error(f"Error during method selection execution: {e}", exc_info=True)
        # Construct error output
        workflow_update = create_workflow_state_update(
            current_step="method_selection", 
            step_completed_flag=False, 
            next_tool="error_handler_tool", 
            next_step_reason=f"Component failed: {e}",
            error=f"Component failed: {e}"
        )
        return { "error": f"Method selection logic failed: {e}",
                 "variables": variables_dict, 
                 "dataset_analysis": dataset_analysis_dict, 
                 "dataset_description": dataset_description,
                 **workflow_update.get('workflow_state', {})}

    # --- Prepare Output Dictionary --- 
    method_selected_flag = bool(method_selection_dict.get("selected_method") and method_selection_dict["selected_method"] != "Error")
    
    # Create the 'method_info' sub-dictionary required by the validator
    # Include alternative_methods if present in the selection output
    method_info = {
         "selected_method": method_selection_dict.get("selected_method"),
         "method_name": method_selection_dict.get("selected_method", "").replace("_", " ").title() if method_selected_flag else None, 
         "method_justification": method_selection_dict.get("method_justification"),
         "method_assumptions": method_selection_dict.get("method_assumptions", []),
         "alternative_methods": method_selection_dict.get("alternatives", []), # Include alternatives
         "decision_tree": method_selection_dict.get("decision_tree", {})
    }
    
    # Create the final output dictionary for the agent
    result = {
        "method_info": method_info,
        "variables": variables_dict,
        "dataset_analysis": dataset_analysis_dict, 
        "dataset_description": dataset_description,
        "original_query": original_query # Pass original query argument
    }
    
    # Determine workflow state for the next step
    next_tool_name = "controls_selector_tool" if method_selected_flag else "error_handler_tool"
    next_reason = "Now we need to select control variables for the chosen method" if method_selected_flag else "Method selection failed or returned an error."
    workflow_update = create_workflow_state_update(
        current_step="method_selection",
        step_completed_flag=method_selected_flag, 
        next_tool=next_tool_name, 
        next_step_reason=next_reason
    )
    result.update(workflow_update.get('workflow_state', {})) # Add workflow state dict
    
    logger.info(f"method_selector_tool finished. Selected: {method_info.get('selected_method')}")
    return result 

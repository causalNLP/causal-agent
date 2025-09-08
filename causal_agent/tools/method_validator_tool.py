"""
Method validator tool for causal inference methods.

This tool validates the selected causal inference method against
dataset characteristics and available variables.
"""

from typing import Dict, Any, Optional, List, Union
from langchain.tools import tool
import logging

from causal_agent.components.method_validator import validate_method
from causal_agent.components.state_manager import create_workflow_state_update
from causal_agent.components.decision_tree import rule_based_select_method

from causal_agent.models import (
    Variables, 
    TemporalStructure, 
    DatasetInfo,       
    DatasetAnalysis,
    MethodInfo, 
    MethodValidatorInput
)

logger = logging.getLogger(__name__)

def extract_properties_from_inputs(inputs: MethodValidatorInput) -> Dict[str, Any]:
    """
    Helper function to extract dataset properties from MethodValidatorInput
    for use with the decision tree.
    """
    variables_dict = inputs.variables.model_dump()
    dataset_analysis_dict = inputs.dataset_analysis.model_dump()
    
    return {
        "treatment_variable": variables_dict.get("treatment_variable"),
        "outcome_variable": variables_dict.get("outcome_variable"),
        "instrument_variable": variables_dict.get("instrument_variable"),
        "covariates": variables_dict.get("covariates", []),
        "time_variable": variables_dict.get("time_variable"),
        "running_variable": variables_dict.get("running_variable"),
        "treatment_variable_type": variables_dict.get("treatment_variable_type", "binary"),
        "has_temporal_structure": dataset_analysis_dict.get("temporal_structure", {}).get("has_temporal_structure", False),
        "frontdoor_criterion": variables_dict.get("frontdoor_criterion", False),
        "cutoff_value": variables_dict.get("cutoff_value"),
        "covariate_overlap_score": variables_dict.get("covariate_overlap_result", 0),
        "is_rct": variables_dict.get("is_rct", False)
    }

 
@tool
def method_validator_tool(inputs: MethodValidatorInput) -> Dict[str, Any]: # Use Pydantic Input
    """
    Validate the assumptions of the selected causal method using structured input.
    
    Args:
        inputs: Pydantic model containing method_info, dataset_analysis, variables, and dataset_description.
        
    Returns:
        Dictionary with validation results, context for next step, and workflow state.
    """
    logger.info(f"Running method_validator_tool for method: {inputs.method_info.selected_method}")
    
    # Access data from input model (converting to dicts for component)
    method_info_dict = inputs.method_info.model_dump()
    dataset_analysis_dict = inputs.dataset_analysis.model_dump()
    variables_dict = inputs.variables.model_dump()
    dataset_description_str = inputs.dataset_description
    
    # Call the component function to validate the method
    try:
        validation_results = validate_method(method_info_dict, dataset_analysis_dict, variables_dict)
        if not isinstance(validation_results, dict):
            raise TypeError(f"validate_method component did not return a dict. Got: {type(validation_results)}")
            
    except Exception as e:
        logger.error(f"Error during validate_method execution: {e}", exc_info=True)
        # Construct error output
        workflow_update = create_workflow_state_update(
            current_step="method_validation", method_validated=False, error=f"Component failed: {e}"
        )
        # Pass context even on error
        return {"error": f"Method validation component failed: {e}",
                "variables": variables_dict,
                "dataset_analysis": dataset_analysis_dict,
                "dataset_description": dataset_description_str,
                **workflow_update.get('workflow_state', {})}

    # Determine if assumptions are valid based on component output
    assumptions_valid = validation_results.get("valid", False) 
    failed_assumptions = validation_results.get("concerns", [])
    original_method = method_info_dict.get("selected_method")
    recommended_method = validation_results.get("recommended_method", original_method)
    
    # If validation failed, attempt to backtrack through decision tree
    if not assumptions_valid and failed_assumptions:
        logger.info(f"Method {original_method} failed validation due to: {failed_assumptions}")
        logger.info("Attempting to backtrack and select alternative method...")
        
        try:
            # Extract properties for decision tree
            dataset_props = extract_properties_from_inputs(inputs)
            
            # Get LLM instance (may be None)
            from causal_agent.config import get_llm_client
            try:
                llm_instance = get_llm_client()
            except Exception as e:
                logger.warning(f"Failed to get LLM instance: {e}")
                llm_instance = None
            
            # Re-run decision tree with failed method excluded
            excluded_methods = [original_method]
            new_selection = rule_based_select_method(
                dataset_analysis=inputs.dataset_analysis.model_dump(),
                variables=inputs.variables.model_dump(),
                is_rct=inputs.variables.is_rct or False,
                llm=llm_instance,
                dataset_description=inputs.dataset_description,
                original_query=inputs.original_query,
                excluded_methods=excluded_methods
            )
            
            recommended_method = new_selection.get("selected_method", original_method)
            logger.info(f"Backtracking selected new method: {recommended_method}")
            
            # Update validation results to include backtracking info
            validation_results["backtrack_attempted"] = True
            validation_results["backtrack_method"] = recommended_method
            validation_results["excluded_methods"] = excluded_methods
            
        except Exception as e:
            logger.error(f"Backtracking failed: {e}")
            validation_results["backtrack_attempted"] = True
            validation_results["backtrack_error"] = str(e)
            # Keep original recommended method
    
    # Prepare output dictionary for the next tool (method_executor)
    result = {
        # --- Data for Method Executor --- 
        "method": recommended_method, # Use recommended method going forward
        "variables": variables_dict, # Pass along all identified variables
        "dataset_path": dataset_analysis_dict.get('dataset_info',{}).get('file_path'), # Extract path
        "dataset_analysis": dataset_analysis_dict, # Pass full analysis
        "dataset_description": dataset_description_str, # Pass description string
        "original_query": inputs.original_query, # Pass original query
        
        # --- Validation Results --- 
        "validation_info": {
            "original_method": method_info_dict.get("selected_method"),
            "recommended_method": recommended_method,
            "assumptions_valid": assumptions_valid,
            "failed_assumptions": failed_assumptions,
            "warnings": validation_results.get("warnings", []),
            "suggestions": validation_results.get("suggestions", [])
        }
    }
    
    # Determine workflow state
    method_validated_flag = assumptions_valid # Or perhaps always True if validation ran?
    next_tool_name = "method_executor_tool" if method_validated_flag else "error_handler_tool" # Go to executor even if assumptions failed?
    next_reason = "Method assumptions checked. Proceeding to execution." if method_validated_flag else "Method assumptions failed validation."
    workflow_update = create_workflow_state_update(
        current_step="method_validation",
        step_completed_flag=method_validated_flag,
        next_tool=next_tool_name, 
        next_step_reason=next_reason
    )
    result.update(workflow_update) # Add workflow state
    
    logger.info(f"method_validator_tool finished. Assumptions valid: {assumptions_valid}")
    return result 
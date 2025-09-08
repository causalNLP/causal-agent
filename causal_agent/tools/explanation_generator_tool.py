"""
Explanation generator tool for causal inference methods.

This tool generates explanations for the selected causal inference method,
including what the method does, its assumptions, and how it will be applied.
"""

from typing import Dict, Any, Optional, List, Union
from langchain.tools import tool
import logging

from causal_agent.components.explanation_generator import generate_explanation
from causal_agent.components.state_manager import create_workflow_state_update
from causal_agent.config import get_llm_client

# Import shared models from central location
from causal_agent.models import (
    ExplainerInput, MethodInfo, Variables, DatasetAnalysis
)

logger = logging.getLogger(__name__)



# --- Tool Definition --- 
@tool(args_schema=ExplainerInput)
# Change signature to accept individual arguments 
def explanation_generator_tool(
    method_info: MethodInfo,
    variables: Variables, 
    results: Dict[str, Any],
    dataset_analysis: DatasetAnalysis,
    validation_info: Optional[Dict[str, Any]] = None,
    dataset_description: Optional[str] = None,
    original_query: Optional[str] = None # Get original query if passed
) -> Dict[str, Any]:
    """
    Generate a single comprehensive explanation string using structured Pydantic input.
    
    Args:
        method_info: Pydantic model with method details.
        variables: Pydantic model with identified variables.
        results: Dictionary containing numerical results from execution.
        dataset_analysis: Pydantic model with dataset analysis results.
        validation_info: Optional dictionary with validation results.
        dataset_description: Optional string description of the dataset.
        original_query: Optional original user query string.
        
    Returns:
        Dictionary with the final explanation text, context, and workflow state.
    """
    logger.info("Running explainer_tool with direct arguments...")
    
    # Use arguments directly, dump models to dicts if needed by component
    method_info_dict = method_info.model_dump()
    validation_result_dict = validation_info # Already dict or None
    variables_dict = variables.model_dump()
    # results is already a dict
    dataset_analysis_dict = dataset_analysis.model_dump()
    # dataset_description is already str or None

    # Include original_query in variables_dict if the component expects it there
    if original_query:
        variables_dict['original_query'] = original_query

    # Get LLM instance if needed by generate_explanation
    llm_instance = None
    try:
        llm_instance = get_llm_client() 
    except Exception as e:
        logger.warning(f"Could not get LLM client for explainer: {e}")

    # Call component to generate the single explanation string
    try:
        explanation_dict = generate_explanation(
            method_info=method_info_dict,
            validation_result=validation_result_dict,
            variables=variables_dict,
            results=results, # Pass results dict directly
            dataset_analysis=dataset_analysis_dict, 
            dataset_description=dataset_description,
            llm=llm_instance # Pass LLM if component uses it
            )
        if not isinstance(explanation_dict, dict):
             raise TypeError(f"generate_explanation component did not return a dict. Got: {type(explanation_dict)}")

    except Exception as e:
        logger.error(f"Error during generate_explanation execution: {e}", exc_info=True)
        # Provide missing args for the error state update
        workflow_update = create_workflow_state_update(
            current_step="result_explanation", 
            step_completed_flag=False, 
            error=f"Component failed: {e}",
            next_tool="explanation_generator_tool", # Indicate failed tool
            next_step_reason=f"Explanation generation component failed: {e}" # Provide reason
        )
        # Return structure consistent with success case, but with error info
        return {
            "error": f"Explanation generation component failed: {e}",
            # Pass necessary context for potential retry or next step
            "query": original_query or "N/A",
            "method": method_info_dict.get('selected_method', "N/A"),
            "results": results, # Include results even if explanation failed
            "explanation": {"error": str(e)}, # Include error in explanation part
            "dataset_analysis": dataset_analysis_dict,
            "dataset_description": dataset_description,
             **workflow_update.get('workflow_state', {})
        }

    # Create workflow state update
    workflow_update = create_workflow_state_update(
        current_step="result_explanation",
        step_completed_flag="results_explained",
        next_tool="output_formatter_tool", # Step 8: Format output
        next_step_reason="Finally, we need to format the output for presentation"
    )
    
    # Prepare result dict for the next tool (formatter)
    result_for_formatter = {
        # Pass the necessary pieces for the formatter
        "query": original_query or "N/A", # Use original_query directly
        "method": method_info_dict.get('selected_method', 'N/A'),
        "results": results, # Pass the numerical results directly
        "explanation": explanation_dict, # Pass the structured explanation
        # Avoid passing full analysis if not needed by formatter? Check formatter needs.
        # For now, keep them.
        "dataset_analysis": dataset_analysis_dict, 
        "dataset_description": dataset_description 
    }
    
    # Add workflow state to the result
    result_for_formatter.update(workflow_update)
    
    logger.info("explanation_generator_tool finished successfully.")
    return result_for_formatter 
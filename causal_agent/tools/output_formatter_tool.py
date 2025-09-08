"""
Output formatter tool for causal inference results.

This tool provides the LangChain interface for the output formatter component.
"""


from typing import Dict, Any, Optional
import logging
import json 

# Add import for @tool decorator
from langchain.tools import tool

from causal_agent.components import output_formatter

from causal_agent.models import FormattedOutput



# --- Tool Definition --- 
logger = logging.getLogger(__name__)

@tool
def output_formatter_tool(
    query: str,
    method: str,
    results: Dict[str, Any], 
    explanation: Dict[str, Any], 
    dataset_analysis: Optional[Dict[str, Any]] = None, 
    dataset_description: Optional[str] = None 
) -> Dict[str, Any]:
    """
    Formats the final explanation and results using the output_formatter component,
    packages it into a dictionary, adds workflow state, and a JSON representation.

    Args:
        query: Original user query.
        method: The method used (string name).
        results: Numerical results dict from method_executor_tool.
        explanation: Structured explanation dict from explainer_tool.
        dataset_analysis: Optional results from dataset_analyzer_tool.
        dataset_description: Optional initial description string.
        
    Returns:
        Dict containing the formatted output fields, workflow state, and a JSON string.
    """
    logger.info("Running output_formatter_tool...")

    try:
        # Call component function - it now returns a FormattedOutput Pydantic model
        formatted_output_model: FormattedOutput = output_formatter.format_output(
            query=query,
            method=method,
            results=results, 
            explanation=explanation, # Pass explanation dict directly
            # Pass analysis dict directly, handle None case for component
            dataset_analysis=dataset_analysis if dataset_analysis else None,
            dataset_description=dataset_description
        )
        
        # Convert the Pydantic model back to a dictionary for tool output
        # Use model_dump() for Pydantic v2+, or .dict() for v1
        try:
            # Attempt model_dump first (Pydantic v2)
            formatted_output_dict = formatted_output_model.model_dump(mode='json') # mode='json' handles complex types
        except AttributeError:
            # Fallback to dict() (Pydantic v1)
            formatted_output_dict = formatted_output_model.dict()

        # Generate JSON representation of the dictionary
        try:
            # Exclude workflow_state if it accidentally got included in the model dump
            dict_for_json = {k: v for k, v in formatted_output_dict.items() if k != 'workflow_state'}
            json_output_str = json.dumps(dict_for_json, indent=4)
            formatted_output_dict["json_output"] = json_output_str
        except TypeError as json_err:
            logger.error(f"Failed to serialize output to JSON: {json_err}")
            formatted_output_dict["json_output"] = f'{{"error": "Failed to serialize output to JSON: {json_err}"}}'

        # Add workflow state information - analysis is complete
        formatted_output_dict["workflow_state"] = {
            "current_step": "output_formatting",
            "analysis_complete": True
        }
        
        logger.info("Output formatting successful.")
        return formatted_output_dict # Return the final dictionary
        
    except Exception as e:
        logger.error(f"Error during output formatting: {e}", exc_info=True)
        # Return error structure
        return {
            "error": f"Failed to format output: {e}",
            "workflow_state": {
                "current_step": "output_formatting",
                "analysis_complete": False, # Indicate failure
                "error": f"Formatting component failed: {e}"
            }
        } 
"""
Method Executor Tool for the causal inference agent.

Executes the selected causal inference method using its implementation function.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Union
from langchain.tools import tool
import traceback # For error logging
import logging # Add logging

# Import the mapping and potentially preprocessing utils
from causal_agent.methods import METHOD_MAPPING
from causal_agent.methods.utils import preprocess_data # Assuming preprocess exists
from causal_agent.components.state_manager import create_workflow_state_update
from causal_agent.config import get_llm_client # IMPORT LLM Client Factory

# Import shared models from central location
from causal_agent.models import (
    Variables, 
    TemporalStructure, # Needed indirectly by DatasetAnalysis
    DatasetInfo,       # Needed indirectly by DatasetAnalysis
    DatasetAnalysis,
    MethodExecutorInput
)

# Add this module-level variable, typically near imports or at the top
CURRENT_OUTPUT_LOG_FILE = None

logger = logging.getLogger(__name__)

@tool
def method_executor_tool(inputs: MethodExecutorInput, original_query: Optional[str] = None) -> Dict[str, Any]: # Use Pydantic Input
    '''Execute the selected causal inference method function using structured input.

    Args:
        inputs: Pydantic model containing method, variables, dataset_path, 
                dataset_analysis, and dataset_description.
        
    Returns:
        Dict with numerical results, context for next step, and workflow state.
    '''
    # Access data from input model
    method = inputs.method
    variables_dict = inputs.variables.model_dump()
    dataset_path = inputs.dataset_path
    dataset_analysis_dict = inputs.dataset_analysis.model_dump()
    dataset_description_str = inputs.dataset_description
    validation_info = inputs.validation_info # Can be passed if needed

    logger.info(f"Executing method: {method}")
        
    try:
        # --- Get LLM Instance --- 
        llm_instance = None
        try:
            llm_instance = get_llm_client()
        except Exception as llm_e:
             logger.warning(f"Could not get LLM client in method_executor_tool: {llm_e}. LLM-dependent features in method will be disabled.")

        # 1. Load Data
        if not dataset_path:
             raise ValueError("Dataset path is missing.")
        df = pd.read_csv(dataset_path)
        
        # 2. Extract Key Variables needed by estimate_func signature
        treatment = variables_dict.get("treatment_variable")
        outcome = variables_dict.get("outcome_variable")
        covariates = variables_dict.get("covariates", [])
        query_str = original_query if original_query is not None else inputs.original_query
        
        if not all([treatment, outcome]):
            raise ValueError("Treatment or Outcome variable not found in 'variables' dict.")
            
        # 3. Preprocess Data 
        required_cols_for_method = [treatment, outcome] + covariates 
        # Add method-specific required vars from the variables_dict
        if method == "instrumental_variable" and variables_dict.get("instrument_variable"):
            required_cols_for_method.append(variables_dict["instrument_variable"])
        elif method == "regression_discontinuity" and variables_dict.get("running_variable"):
             required_cols_for_method.append(variables_dict["running_variable"])
        
        missing_df_cols = [col for col in required_cols_for_method if col not in df.columns]
        if missing_df_cols:
             raise ValueError(f"Dataset at {dataset_path} is missing required columns for method '{method}': {missing_df_cols}")
             
        df_processed, updated_treatment, updated_outcome, updated_covariates, column_mappings = \
            preprocess_data(df, treatment, outcome, covariates, verbose=False)

        # 4. Get the correct method execution function
        if method not in METHOD_MAPPING:
            raise ValueError(f"Method '{method}' not found in METHOD_MAPPING.")
        estimate_func = METHOD_MAPPING[method]
        
        # 5. Execute the method
        # Pass only necessary args from variables_dict as kwargs
        # (e.g., instrument_variable, running_variable, cutoff_value, etc.)
        # Avoid passing the entire variables_dict as estimate_func expects specific args
        kwargs_for_method = {}
        for key in ["instrument_variable", "time_variable", "group_variable", 
                    "running_variable", "cutoff_value"]:
            if key in variables_dict and variables_dict[key] is not None:
                 kwargs_for_method[key] = variables_dict[key]
        
        # Add new fields from the Variables model (which is inputs.variables)
        if hasattr(inputs, 'variables'): # ensure variables object exists on inputs
            if inputs.variables.treatment_reference_level is not None:
                kwargs_for_method['treatment_reference_level'] = inputs.variables.treatment_reference_level
            if inputs.variables.interaction_term_suggested is not None: # boolean, so check for None to allow False
                kwargs_for_method['interaction_term_suggested'] = inputs.variables.interaction_term_suggested
            if inputs.variables.interaction_variable_candidate is not None:
                kwargs_for_method['interaction_variable_candidate'] = inputs.variables.interaction_variable_candidate

        # Add query if needed by llm_assist functions within the method
        kwargs_for_method['query'] = query_str
        kwargs_for_method['column_mappings'] = column_mappings
        
                 
        results_dict = estimate_func(
            df=df_processed, 
            treatment=updated_treatment,
            outcome=updated_outcome,
            covariates=updated_covariates,
            dataset_description=dataset_description_str, 
            query_str=query_str,
            llm=llm_instance,
            **kwargs_for_method # Pass specific args needed by the method
        )
        
        # 6. Prepare output
        logger.info(f"Method execution successful. Effect estimate: {results_dict.get('effect_estimate')}")
            
        # Add workflow state
        workflow_update = create_workflow_state_update(
            current_step="method_execution", 
            step_completed_flag="method_executed",
            next_tool="explainer_tool", 
            next_step_reason="Now we need to explain the results and their implications"
        )
        
        # --- Prepare Output Dictionary --- 
        # Structure required by explainer_tool: context + nested "results"
        final_output = {
            # Nested dictionary for numerical results and diagnostics
            "results": {
                # Core estimation results (extracted from results_dict)
                "effect_estimate": results_dict.get("effect_estimate"),
                "confidence_interval": results_dict.get("confidence_interval"),
                "standard_error": results_dict.get("standard_error"),
                "p_value": results_dict.get("p_value"),
                "method_used": results_dict.get("method_used"),
                "llm_assumption_check": results_dict.get("llm_assumption_check"),
                "raw_results": results_dict.get("raw_results"),
                # Diagnostics and Refutation results
                "diagnostics": results_dict.get("diagnostics"),
                "refutation_results": results_dict.get("refutation_results")
            },
            # Top-level context to be passed along
            "variables": variables_dict, 
            "dataset_analysis": dataset_analysis_dict,
            "dataset_description": dataset_description_str,
            "validation_info": validation_info, # Pass validation info
            "original_query": inputs.original_query,
            "column_mappings": column_mappings # Add column_mappings to the output
            # Workflow state will be added next
        }
        
        # Add workflow state to the final output
        final_output.update(workflow_update.get('workflow_state', {}))

        # --- Logging logic (moved from output_formatter.py) ---
        # Prepare a summary dict for logging
        summary_keys = {"query", "method_used", "causal_effect", "standard_error", "confidence_interval"}
        # Try to get these from the available context
        summary_dict = {
            "query": inputs.original_query if hasattr(inputs, 'original_query') else None,
            "method_used": results_dict.get("method_used"),
            "causal_effect": results_dict.get("effect_estimate"),
            "standard_error": results_dict.get("standard_error"),
            "confidence_interval": results_dict.get("confidence_interval")
        }
        print(f"summary_dict: {summary_dict}")
        print(f"CURRENT_OUTPUT_LOG_FILE: {CURRENT_OUTPUT_LOG_FILE}")
        if CURRENT_OUTPUT_LOG_FILE and summary_dict:
            try:
                import json
                log_entry = {"type": "analysis_result", "data": summary_dict}
                with open(CURRENT_OUTPUT_LOG_FILE, mode='a', encoding='utf-8') as log_file:
                    log_file.write('\n' + json.dumps(log_entry) + '\n')
            except Exception as e:
                print(f"[ERROR] method_executor_tool.py: Failed to write analysis results to log file '{CURRENT_OUTPUT_LOG_FILE}': {e}")

        return final_output

    except Exception as e:
        error_message = f"Error executing method {method}: {str(e)}"
        logger.error(error_message, exc_info=True)
            
        # Return error state, include context if available
        workflow_update = create_workflow_state_update(
            current_step="method_execution",
            step_completed_flag=False, 
            next_tool="explainer_tool", # Or error handler?
            next_step_reason=f"Failed during method execution: {error_message}"
        )
        # Ensure error output still contains necessary context keys if possible
        error_result = {"error": error_message, 
                        "variables": variables_dict if 'variables_dict' in locals() else {},
                        "dataset_analysis": dataset_analysis_dict if 'dataset_analysis_dict' in locals() else {},
                        "dataset_description": dataset_description_str if 'dataset_description_str' in locals() else None,
                        "original_query": inputs.original_query if hasattr(inputs, 'original_query') else None,
                        "column_mappings": column_mappings if 'column_mappings' in locals() else {} # Also add to error output
                        }
        error_result.update(workflow_update.get('workflow_state', {}))
        return error_result 
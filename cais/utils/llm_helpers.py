"""
Utility functions for LLM interactions within the cais module.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import logging
import json
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage 

logger = logging.getLogger(__name__)

def call_llm_with_json_output(llm: Optional[BaseChatModel], prompt: str) -> Optional[Dict[str, Any]]:
    """
    Calls the provided LLM with a prompt, expecting a JSON object in the response.
    It parses the JSON string (after attempting to remove markdown fences)
    and returns it as a Python dictionary.

    Args:
        llm: An instance of BaseChatModel (e.g., from Langchain). If None,
             the function will log a warning and return None.
        prompt: The prompt string to send to the LLM.

    Returns:
        A dictionary parsed from the LLM's JSON response, or None if:
        - llm is None.
        - The LLM call fails.
        - The LLM response content cannot be extracted as a string.
        - The response content is empty after stripping markdown.
        - The response is not valid JSON.
        - The parsed JSON is not a dictionary.
    """
    if not llm:
        logger.warning("LLM client (BaseChatModel) not provided to call_llm_with_json_output. Cannot make LLM call.")
        return None

    logger.info(f"Attempting LLM call with {type(llm).__name__} for JSON output.")
    # Full prompt logging can be verbose, using DEBUG level.
    logger.debug(f"LLM Prompt for JSON output:\\n{prompt}")

    raw_response_content = ""  # For logging in case of errors before parsing
    processed_content_for_json = "" # For logging in case of JSON parsing error

    try:
        llm_response_obj = llm.invoke(prompt)

        # Extract string content from LLM response object
        if hasattr(llm_response_obj, 'content') and isinstance(llm_response_obj.content, str):
            raw_response_content = llm_response_obj.content
        elif isinstance(llm_response_obj, str):
            raw_response_content = llm_response_obj
        else:
            # Fallback for other potential response structures
            logger.warning(
                f"LLM response is not a string and has no '.content' attribute of type string. "
                f"Type: {type(llm_response_obj)}. Trying '.text' attribute."
            )
            if hasattr(llm_response_obj, 'text') and isinstance(llm_response_obj.text, str):
                raw_response_content = llm_response_obj.text

        if not raw_response_content:
            logger.warning(f"LLM invocation returned no extractable string content. Response object type: {type(llm_response_obj)}")
            return None

        # Prepare content for JSON parsing: strip whitespace and markdown fences.
        # Using the same stripping logic as in llm_identify_temporal_and_unit_vars for consistency.
        processed_content_for_json = raw_response_content.strip()

        if processed_content_for_json.startswith("```json"):
            # Removes "```json" prefix and "```" suffix, then strips whitespace.
            # Assumes the format is "```json\\nCONTENT\\n```" or similar.
            processed_content_for_json = processed_content_for_json[7:-3].strip()
        elif processed_content_for_json.startswith("```"):
            # Removes generic "```" prefix and "```" suffix, then strips.
            processed_content_for_json = processed_content_for_json[3:-3].strip()
        
        if not processed_content_for_json: # Check if empty after stripping
            logger.warning(
                "LLM response content became empty after attempting to strip markdown. "
                f"Original raw content snippet: '{raw_response_content[:200]}...'"
            )
            return None

        parsed_json = json.loads(processed_content_for_json)

        if not isinstance(parsed_json, dict):
            logger.warning(
                "LLM response was successfully parsed as JSON, but it is not a dictionary. "
                f"Type: {type(parsed_json)}. Parsed content snippet: '{str(parsed_json)[:200]}...'"
            )
            return None

        logger.info(f"Successfully received and parsed JSON response from {type(llm).__name__}.")
        return parsed_json

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON from LLM response. Error: {e}. "
            f"Content processed for parsing (snippet): '{processed_content_for_json[:500]}...'"
        )
        return None
    except Exception as e:
        # This catches errors from llm.invoke() or other unexpected issues.
        logger.error(f"An unexpected error occurred during LLM call or JSON processing: {e}", exc_info=True)
        # Log raw content if available and different from processed, for better debugging
        if raw_response_content and raw_response_content[:500] != processed_content_for_json[:500]:
             logger.debug(f"Original raw LLM response content (snippet): '{raw_response_content[:500]}...'")
        return None

# Placeholder for processing LLM response
def process_llm_response(response: Dict[str, Any], method: str) -> Dict[str, Any]:
    # Validate and structure the LLM response based on the method
    # For now, just return the response
    return response

# Placeholder for getting column info
def get_columns_info(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def analyze_dataset_for_method(df: pd.DataFrame, query: str, method: str) -> Dict[str, Any]:
    """Use LLM to analyze dataset for appropriate method parameters.
    
    Args:
        df: Input DataFrame
        query: User's causal query
        method: The causal method being considered
        
    Returns:
        Dictionary with suggested parameters and validation checks from LLM.
    """
    # Prepare prompt with dataset information
    columns_info = get_columns_info(df)
    try:
        # Attempt to get sample data safely
        sample_data = df.head(5).to_dict(orient='records')
    except Exception:
        sample_data = "Error retrieving sample data."
    
    
    prompt = f"""
    Given the dataset with columns {columns_info} and the causal query "{query}",
    suggest SENSIBLE INITIAL DEFAULT parameters for applying the {method} method.
    Do NOT attempt complex optimization; provide common starting points.

    The first 5 rows of data look like:
    {sample_data}

    Specifically for {method}:
    - If PS.Matching:
        - For 'caliper': Suggest a common heuristic value like 0.01, 0.02, or 0.05 (this is relative to std dev of logit score, but just suggest the number). If unsure, suggest 0.02.
        - For 'n_neighbors': Suggest 1.
        - For 'propensity_model_type': Suggest 'logistic' unless the context strongly implies a more complex model is needed.
    - If PS.Weighting:
        - For 'weight_type': Suggest 'ATE' unless the query specifically asks for ATT or ATC.
        - For 'trim_threshold': Suggest a small value like 0.01 or 0.05 if the data seems noisy or has extreme propensity scores, otherwise suggest null (no trimming). Default to null if unsure.
    - Add other parameters if relevant for the specific method.

    Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
    {{
      "parameters": {{
        // method-specific parameters based on the guidelines above
      }},
      "validation": {{
        // validation checks typically needed (e.g., check_balance: true for PSM)
      }}
    }}
    """
    
    # Call LLM with prompt - Assuming analyze_dataset_for_method provides the llm object
    # For now, this internal call still uses the placeholder without passing llm
    # This needs to be updated if analyze_dataset_for_method is intended to use a passed llm
    response = call_llm_with_json_output(None, prompt) # Passing None for llm temporarily
    
    # Process and validate response
    # This step might involve ensuring the structure is correct,
    # parameters are valid types, etc.
    processed_response = process_llm_response(response, method)
    
    return processed_response 


def llm_identify_temporal_and_unit_vars(
    column_names: List[str], 
    column_dtypes: Dict[str, str],
    dataset_description: str,
    dataset_summary: str,
    heuristic_time_candidates: Optional[List[str]] = None, # These are no longer used in the revised prompt
    heuristic_id_candidates: Optional[List[str]] = None,   # These are no longer used in the revised prompt
    query: str = "No query provided.",
    llm: Optional[BaseChatModel] = None
) -> Dict[str, Optional[str]]:
    """Uses LLM to identify the primary time:

    Args:
        column_names: List of all column names.
        column_dtypes: Dictionary mapping column names to string representation of data types.
        dataset_description: Textual description of the dataset.
        dataset_summary: Summary of the dataset
        heuristic_time_candidates: Optional list of columns identified as time vars by heuristics (currently unused by prompt).
        heuristic_id_candidates: Optional list of columns identified as unit ID vars by heuristics (currently unused by prompt).
        llm: The language model client instance.

    Returns:
        A dictionary with keys 'time_variable' and 'unit_variable', 
        whose values are the identified column names or None.
    """
    if not llm:
        logger.warning("LLM client not provided for temporal/unit identification. Returning None.")
        return {"time_variable": None, "unit_variable": None}

    logger.info("Attempting LLM identification of time and unit variables...")

    prompt = f"""
You are a data analysis expert tasked with determining whether a dataset supports a Difference-in-Differences (DiD) or Two-Way Fixed Effects (TWFE) design to answer the following query:
{query}

You are given the following information:

Dataset Description:
{dataset_description}

Columns and Data Types:
{column_dtypes}

First, based on the above information, check if any columns represent information about the time/periods associated directly with intervention application. It could be either:
1. A variable that represents **time periods associated with the intervention**. This must satisfy one of the following:
   - A binary indicator showing pre/post-intervention status,
   - A discrete or continuous variable that records **when units were observed**, which can be aligned with treatment application periods.

 Do **not** select generic time-related variables that merely describe time as a feature, such as **'date of birth'**, **'year of graduation'**, 'week of sign-up', **'years of schooling'** unless they directly represent **observation times relevant to treatment**.

2. A variable that represents the **unit of observation** (e.g., individual, region, school) — the entity over which we compare treated vs. untreated groups across time.

Return ONLY a valid JSON object with this structure and no surrounding explanation:

{{
  "time_variable": "<column_name_or_null>",
  "unit_variable": "<column_name_or_null>"
}}
"""


    parsed_response = None
    try:
        llm_response_obj = llm.invoke(prompt)
        response_content = ""
        if hasattr(llm_response_obj, 'content'):
            response_content = llm_response_obj.content
        elif isinstance(llm_response_obj, str): # Some LLMs might return str directly
            response_content = llm_response_obj
        else:
            logger.warning(f"LLM response object type not recognized for content extraction: {type(llm_response_obj)}")

        if response_content:
            # Attempt to strip markdown ```json ... ``` if present
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()
            elif response_content.strip().startswith("```"):
                 response_content = response_content.strip()[3:-3].strip()

            parsed_response = json.loads(response_content)
        else:
            logger.warning("LLM invocation returned no content.")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response for time/unit vars: {e}. Response content: '{response_content[:500]}...'") # Log snippet
    except Exception as e:
        logger.error(f"Error during LLM invocation or processing for time/unit vars: {e}", exc_info=True)

    # Process the response
    if parsed_response and isinstance(parsed_response, dict):
        time_var = parsed_response.get("time_variable")
        unit_var = parsed_response.get("unit_variable")
        
        # Basic validation: ensure returned names are actual columns or None
        if time_var is not None and time_var not in column_names:
            logger.warning(f"LLM identified time variable '{time_var}' not found in columns. Setting to None.")
            time_var = None
        if unit_var is not None and unit_var not in column_names:
            logger.warning(f"LLM identified unit variable '{unit_var}' not found in columns. Setting to None.")
            unit_var = None
            
        logger.info(f"LLM identified time='{time_var}', unit='{unit_var}'")
        return {"time_variable": time_var, "unit_variable": unit_var}
    else:
        logger.warning("LLM call failed or returned invalid/unparsable JSON for time/unit identification.")
        return {"time_variable": None, "unit_variable": None} 
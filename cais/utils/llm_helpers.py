"""
Utility functions for LLM interactions within the cais module.
"""

from typing import Dict, Any, Optional, List
import re
import pandas as pd
import logging
import json
from langchain_core.language_models import BaseChatModel

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

    logger.debug(f"Attempting LLM call with {type(llm).__name__} for JSON output.")
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


def llm_identify_temporal_and_unit_vars(column_names: List[str], column_dtypes: Dict[str, str],
                                       dataset_description: str, dataset_summary: str,
                                       heuristic_time_candidates: Optional[List[str]] = None,
                                       heuristic_id_candidates: Optional[List[str]] = None,
                                       query: str = "No query provided.",
                                       llm: Optional[BaseChatModel] = None) -> Dict[str, Optional[str]]:
    """
    Main function to identify temporal and unit variables using Chain-of-Thought reasoning.
    
    Args:
        column_names: List of all column names
        column_dtypes: Dictionary mapping column names to string representation of data types
        dataset_description: Textual description of the dataset
        dataset_summary: Summary of the dataset
        heuristic_time_candidates: Optional list (IGNORED - maintained for backward compatibility)
        heuristic_id_candidates: Optional list (IGNORED - maintained for backward compatibility)
        query: User query for context
        llm: The language model client instance
    
    Returns:
        Dictionary with essential DiD variables: time_variable, unit_variable, did_canonical, did_term, treatment_time, treatment_state
    """
    if not llm:
        logger.warning("LLM client not provided. Returning None values.")
        return {
            "time_variable": None, "unit_variable": None, "did_canonical": None,
            "did_term": None, "treatment_time": None, "treatment_state": None
        }
    
    logger.info("Starting enhanced Chain-of-Thought DiD variable identification...")
    
    prompt = f"""
You need to identify variables for Difference-in-Differences analysis to compute causal effects that answer the user query. DiD will estimate the treatment effect by comparing changes over time between treated and control units, which requires careful identification of the right variables.

User query: {query}

Dataset context:
Description: {dataset_description}
Summary: {dataset_summary}
Available columns: {column_names}
Column types: {column_dtypes}

Think through this systematically:

Step 1: DiD appropriateness assessment
- Does the dataset have multiple time periods when data was collected?
- Are there multiple units (entities) observed across these time periods?
- Is there a clear treatment/intervention mentioned in the query or dataset?
- Can we identify before/after periods relative to the treatment?
If any answer is no, this is not suitable for DiD analysis. All variables should be returned as null.

Step 2: Time variable identification
- Which column represents observation periods (when outcomes were measured)?
- Look for: year, quarter, period, date, time, wave
- Avoid personal characteristics: age, experience, duration_since_X, years_employed
- If multiple time-related columns exist, prefer binary (0/1) over multi-valued ones. Watch out for names like post, after. These usually denote before/after treatment periods and are ideal for canonical DiD.

Step 3: Unit variable identification
- Which column represents the cross-sectional entities that receive or do not receive treatment?
- Look for: state, individual_id, firm_id, country, region, school_id, person_id
- Each unit should be observed across multiple time periods
- As with the time variable, there may be multiple unit-related columns. If multiple exist, prefer binary (0/1) variables. These denote the presence or absence of treatment and can be used for canonical DiD.

Step 4: Treatment structure analysis (critical decision point)
Carefully analyze the treatment timing and structure:

For canonical 2x2 DiD:
- Treatment occurs at one specific time point for some units
- Creates clear before/after periods for all units
- Example: Policy implemented in 2010, so 2008-2009 = before, 2010-2011 = after
- All treated units receive treatment at the same time
- Need to identify: treatment_time (when) and treatment_state (which units)

For staggered TWFE DiD:
- Treatment occurs at different times for different units, OR
- Need to track treatment status varying across unit-time observations
- Requires a binary indicator showing treatment status for each unit at each time
- Need to identify: did_term (binary column indicating treatment status)

Step 5: DiD type determination
Determine which approach applies:
- If treatment occurs at one specific time did_canonical and there is only one entity receiving treatment → did_canonical = true
- If treatment is staggered across time or varies by unit-time and multiple entities are involved → did_canonical = false
- If you cannot determine clear treatment timing → did_canonical = false (safer default)

Step 6: Variable extraction
Based on your analysis:
- time_variable: The temporal observation column
- unit_variable: The cross-sectional entity identifier
- did_canonical: Boolean - true for 2x2 DiD, false for staggered TWFE
- did_term: For TWFE only (did_canonical=false) - binary (0/1) column indicating treatment status at time t for unit i
- treatment_time: For 2x2 only (did_canonical=true) - the specific reference time when treatment occurred. This should be determined from the description. If unsure, return null.
- treatment_state: For 2x2 only (did_canonical=true) - the specific entity that received treatment. This should be determined from the description. If unsure, return null.

Critical requirements:
- Base analysis STRICTLY on provided information - do not speculate
- did_term must be binary (0/1) if specified
- Prefer binary time indicators over multi-valued when available
- Return null for any variable you cannot clearly identify
- Only return variables that actually exist in the dataset

Work through each step methodically, then provide your final analysis. You must return ONLY a valid JSON object with the following structure (no explanations or surrounding text):

{{
    "time_variable": "column_name_or_null",
    "unit_variable": "column_name_or_null", 
    "did_canonical": true_or_false_or_null,
    "did_term": "binary_column_name_or_null",
    "treatment_time": "specific_time_value_or_null",
    "treatment_state": "specific_unit_value_or_null"
}}
"""
    
    result = call_llm_with_json_output(llm, prompt)
    if result:
        result = {k: None if isinstance(v, str) and v.lower() == 'null' else v for k,v in result.items()} # convert all null's to None

        if isinstance(result.get('treatment_time'), str): # should be a float according to the model; fix if it is a string i.e. March 2008
            try:
                result['treatment_time'] = float(re.findall(r'-?\d*\.?\d+', result.get('treatment_time')))
            except:
                logger.warning("treatment_time is string but cannot be casted to float. Setting treatment_time to None.")
                result['treatment_time'] = None

        # Validate that identified variables exist in columns
        for key in ["time_variable", "unit_variable", "did_term"]:
            if result.get(key) and result[key] not in column_names:
                logger.warning(f"{key} '{result[key]}' not found in columns, setting to None")
                result[key] = None
        
        # Validate did_canonical is boolean or None
        did_canonical = result.get("did_canonical")
        try:
            if did_canonical is not None:
                if did_canonical is not isinstance(did_canonical, bool): # might be a string
                        did_canonical = did_canonical.strip()
                        result['did_canonical'] = (True if did_canonical.lower() == 'true' else False) and (result.get('treatment_time') is not None)
        except:
            logger.warning(f"Invalid did_canonical '{did_canonical}', must be boolean. Setting to None")
            result["did_canonical"] = None

        logger.info(f"DiD variables identified - time: {result.get('time_variable')}, unit: {result.get('unit_variable')}, "
                   f"canonical: {result.get('did_canonical')}, did_term: {result.get('did_term')}")
        return result
    else:
        logger.error("Error in DiD variable identification")
        return {
            "time_variable": None, "unit_variable": None, "did_canonical": None,
            "did_term": None, "treatment_time": None, "treatment_state": None
        }
    
def repair_spaced_variables(df: pd.DataFrame) -> Dict[str, str]:
    # returns a dictionary to rename columns
    rename = {}
    for column in df:
        if " " in column:
            rename[column] = column.replace(" ", "_")
    return 


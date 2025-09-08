"""LLM Assist functions for Difference-in-Differences method."""

import pandas as pd
import numpy as np
from typing import Optional, Any, Dict, Union
import logging
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException

from langchain_core.language_models import BaseChatModel


from causal_agent.utils.llm_helpers import call_llm_with_json_output 

logger = logging.getLogger(__name__)

# Placeholder LLM/Helper Functions 

# --- Pydantic model for LLM time variable extraction ---
class LLMTimeVar(BaseModel):
    time_variable_name: Optional[str] = Field(None, description="The column name identified as the primary time variable.")


def identify_time_variable(df: pd.DataFrame, 
                           query: Optional[str] = None, 
                           dataset_description: Optional[str] = None,
                           llm: Optional[BaseChatModel] = None) -> Optional[str]:
    '''Identifies the most likely time variable.
    
    Current Implementation: Heuristic based on column names, with LLM fallback.
    Future: Refine LLM prompt and parsing.
    '''
    # 1. Heuristic based on common time-related keywords
    time_patterns = ['time', 'year', 'date', 'period', 'month', 'day']
    columns = df.columns.tolist()
    for col in columns:
        if any(pattern in col.lower() for pattern in time_patterns):
            logger.info(f"Identified '{col}' as time variable (heuristic).")
            return col
            
    # 2. LLM Fallback if heuristic fails and LLM is provided
    if llm and query:
        logger.warning("Heuristic failed for time variable. Trying LLM fallback...")
        # --- Example: Add dataset description context --- 
        context_str = ""
        if dataset_description:
            # col_types = dataset_description.get('column_types', {}) # Description is now a string
            context_str += f"\nDataset Description: {dataset_description}"
            # Add other relevant info like sample values if available
        # ------------------------------------------------
        prompt = f"""Given the user query and the available data columns, identify the single most likely column representing the primary time dimension (e.g., year, date, period).

User Query: "{query}"
Available Columns: {columns}{context_str}

Respond ONLY with a JSON object containing the identified column name using the key 'time_variable_name'. If no suitable time variable is found, return null for the value.
Example: {{"time_variable_name": "Year"}} or {{"time_variable_name": null}}"""
        
        messages = [HumanMessage(content=prompt)]
        structured_llm = llm.with_structured_output(LLMTimeVar)
        
        try:
            parsed_result = structured_llm.invoke(messages)
            llm_identified_col = parsed_result.time_variable_name
            
            if llm_identified_col and llm_identified_col in columns:
                logger.info(f"Identified '{llm_identified_col}' as time variable (LLM fallback).")
                return llm_identified_col
            elif llm_identified_col:
                logger.warning(f"LLM fallback identified '{llm_identified_col}' but it's not in the columns. Ignoring.")
            else:
                logger.info("LLM fallback did not identify a time variable.")
                
        except (OutputParserException, ValidationError) as e:
            logger.error(f"LLM fallback for time variable failed parsing/validation: {e}")
        except Exception as e:
             logger.error(f"LLM fallback for time variable failed unexpectedly: {e}", exc_info=True)

    logger.warning("Could not identify time variable using heuristics or LLM fallback.")
    return None

# --- Pydantic model for LLM treatment period extraction ---
class LLMTreatmentPeriod(BaseModel):
    treatment_start_period: Optional[Union[str, int, float]] = Field(None, description="The time period value (as string) when treatment is believed to start based on the query.")

def determine_treatment_period(df: pd.DataFrame, time_var: str, treatment: str, 
                              query: Optional[str] = None, 
                              dataset_description: Optional[str] = None,
                              llm: Optional[BaseChatModel] = None) -> Any:
    '''Determines the period when treatment starts.
    
    Tries LLM first if available, then falls back to heuristic.
    '''
    if time_var not in df.columns:
         raise ValueError(f"Time variable '{time_var}' not found in DataFrame.")
         
    unique_times_sorted = np.sort(df[time_var].dropna().unique())
    if len(unique_times_sorted) < 2:
        raise ValueError("Need at least two time periods for DiD")

    # --- Try LLM First (if available) --- 
    llm_period = None
    if llm and query:
        logger.info("Attempting LLM call to determine treatment period start...")
        # Provide sorted unique times for context
        times_str = ", ".join(map(str, unique_times_sorted)) if len(unique_times_sorted) < 20 else f"{unique_times_sorted[0]}...{unique_times_sorted[-1]}"
        # --- Example: Add dataset description context --- 
        context_str = ""
        if dataset_description:
            # Example: Show summary stats for time var if helpful
            # time_stats = dataset_description.get('summary_stats', {}).get(time_var) # Cannot get from string
            context_str += f"\nDataset Description: {dataset_description}"
        # ------------------------------------------------
        prompt = f"""Based on the user query and the observed time periods, determine the specific period value when the treatment ('{treatment}') likely started.

User Query: "{query}"
Time Variable Name: '{time_var}'
Observed Time Periods (sorted): [{times_str}]{context_str}

Respond ONLY with a JSON object containing the identified start period using the key 'treatment_start_period'. The value should be one of the observed periods if possible. If the query doesn't specify a start period, return null.
Example: {{"treatment_start_period": 2015}} or {{"treatment_start_period": null}}"""
        
        messages = [HumanMessage(content=prompt)]
        structured_llm = llm.with_structured_output(LLMTreatmentPeriod)
        
        try:
            parsed_result = structured_llm.invoke(messages)
            potential_period = parsed_result.treatment_start_period
            
            # Validate if the period exists in the data (might need type conversion)
            if potential_period is not None:
                # Try converting LLM output type to match data type if needed
                try:
                    series_dtype = df[time_var].dtype
                    converted_period = pd.Series([potential_period]).astype(series_dtype).iloc[0]
                except Exception:
                     converted_period = potential_period # Use raw if conversion fails
                     
                if converted_period in unique_times_sorted:
                    llm_period = converted_period
                    logger.info(f"LLM identified treatment period start: {llm_period}")
                else:
                     logger.warning(f"LLM identified period '{potential_period}' (converted: '{converted_period}'), but it's not in the observed time periods. Ignoring LLM result.")
            else:
                 logger.info("LLM did not identify a specific treatment start period from the query.")
                 
        except (OutputParserException, ValidationError) as e:
            logger.error(f"LLM fallback for treatment period failed parsing/validation: {e}")
        except Exception as e:
             logger.error(f"LLM fallback for treatment period failed unexpectedly: {e}", exc_info=True)
             
    if llm_period is not None:
        return llm_period
        
    # --- Fallback to Heuristic --- 
    logger.warning("Using heuristic (median time) to determine treatment period start.")
    treatment_period_start = None
    try:
        if pd.api.types.is_numeric_dtype(df[time_var]):
            median_time = np.median(unique_times_sorted)
            possible_starts = unique_times_sorted[unique_times_sorted > median_time]
            if len(possible_starts) > 0:
                treatment_period_start = possible_starts[0]
            else:
                treatment_period_start = unique_times_sorted[-1]
                logger.warning(f"Could not determine treatment start > median time. Defaulting to last period: {treatment_period_start}")
        else: # Assume sortable categories or dates
            median_idx = len(unique_times_sorted) // 2
            if median_idx < len(unique_times_sorted):
                treatment_period_start = unique_times_sorted[median_idx] 
            else: 
                 treatment_period_start = unique_times_sorted[0]
                 
        if treatment_period_start is not None:
             logger.info(f"Determined treatment period start: {treatment_period_start} (heuristic: median time).")
             return treatment_period_start
        else:
             raise ValueError("Could not determine treatment start period using heuristic.")
             
    except Exception as e:
         logger.error(f"Error in heuristic for treatment period: {e}")
         raise ValueError(f"Could not determine treatment start period using heuristic: {e}")

# --- Pydantic model for LLM group variable extraction ---
class LLMGroupVar(BaseModel):
    group_variable_name: Optional[str] = Field(None, description="The column name identifying the panel unit (e.g., state, individual, firm).")

def identify_treatment_group(df: pd.DataFrame, treatment_var: str, 
                             query: Optional[str] = None, 
                             dataset_description: Optional[str] = None,
                             llm: Optional[BaseChatModel] = None) -> Optional[str]:
    '''Identifies the variable indicating the treated group/unit ID.
    
    Tries heuristic check for non-binary treatment_var first, then LLM, 
    then falls back to assuming treatment_var is the group/unit identifier.
    '''
    columns = df.columns.tolist()
    if treatment_var not in columns:
        logger.error(f"Treatment variable '{treatment_var}' provided to identify_treatment_group not found in DataFrame.")
        # Fallback: Look for common ID names if specified treatment is missing
        id_keywords = ['id', 'unit', 'group', 'entity', 'state', 'firm']
        for col in columns:
             if any(keyword in col.lower() for keyword in id_keywords):
                 logger.warning(f"Specified treatment '{treatment_var}' not found. Falling back to potential ID column '{col}' as group identifier.")
                 return col
        return None # Give up if no likely ID column found

    # --- Heuristic: Check if treatment_var is non-binary, if so, look for ID columns --- 
    is_potentially_binary = False
    if pd.api.types.is_numeric_dtype(df[treatment_var]):
         unique_vals = set(df[treatment_var].dropna().unique())
         if unique_vals.issubset({0, 1}):
              is_potentially_binary = True
              
    if not is_potentially_binary:
        logger.info(f"Provided treatment variable '{treatment_var}' is not binary (0/1). Searching for a separate group/unit ID column heuristically.")
        id_keywords = ['id', 'unit', 'group', 'entity', 'state', 'firm']
        # Prioritize 'group' or 'unit' if available
        for keyword in ['group', 'unit']:
            for col in columns:
                if keyword == col.lower():
                    logger.info(f"Heuristically identified '{col}' as group/unit ID (treatment '{treatment_var}' was non-binary)." )
                    return col
        # Then check other keywords
        for col in columns:
            if col != treatment_var and any(keyword in col.lower() for keyword in id_keywords):
                logger.info(f"Heuristically identified '{col}' as group/unit ID (treatment '{treatment_var}' was non-binary)." )
                return col
        logger.warning("Heuristic search for group/unit ID failed when treatment was non-binary.")
        
    # --- LLM Attempt (if heuristic didn't find an alternative or wasn't needed) ---
    # Useful if query context helps disambiguate (e.g., "effect across states")
    if llm and query:
        logger.info("Attempting LLM call to identify group/unit variable...")
        # --- Example: Add dataset description context --- 
        context_str = ""
        if dataset_description:
            # col_types = dataset_description.get('column_types', {}) # Description is now a string
            context_str += f"\nDataset Description: {dataset_description}"
        # ------------------------------------------------
        prompt = f"""Given the user query and data columns, identify the single column that most likely represents the unique identifier for the panel units (e.g., state, individual, firm, unit ID), distinct from the treatment status indicator ('{treatment_var}').

User Query: "{query}"
Treatment Variable Mentioned: '{treatment_var}'
Available Columns: {columns}{context_str}

Respond ONLY with a JSON object containing the identified unit identifier column name using the key 'group_variable_name'. If the best identifier seems to be the treatment variable itself or none is suitable, return null.
Example: {{"group_variable_name": "state_id"}} or {{"group_variable_name": null}}"""
        
        messages = [HumanMessage(content=prompt)]
        structured_llm = llm.with_structured_output(LLMGroupVar)
        
        try:
            parsed_result = structured_llm.invoke(messages)
            llm_identified_col = parsed_result.group_variable_name
            
            if llm_identified_col and llm_identified_col in columns:
                logger.info(f"Identified '{llm_identified_col}' as group/unit variable (LLM).")
                return llm_identified_col
            elif llm_identified_col:
                logger.warning(f"LLM identified '{llm_identified_col}' but it's not in the columns. Ignoring.")
            else:
                 logger.info("LLM did not identify a separate group/unit variable.")
                 
        except (OutputParserException, ValidationError) as e:
            logger.error(f"LLM call for group/unit variable failed parsing/validation: {e}")
        except Exception as e:
             logger.error(f"LLM call for group/unit variable failed unexpectedly: {e}", exc_info=True)

    # --- Final Fallback --- 
    logger.info(f"Defaulting to using provided treatment variable '{treatment_var}' as the group/unit identifier.")
    return treatment_var

# --- Add interpret_did_results function ---

def interpret_did_results(
    results: Dict[str, Any], 
    diagnostics: Optional[Dict[str, Any]],
    dataset_description: Optional[str] = None,
    llm: Optional[BaseChatModel] = None
) -> str:
    """Use LLM to interpret Difference-in-Differences results."""
    default_interpretation = "LLM interpretation not available for DiD."
    if llm is None:
        logger.info("LLM not provided for DiD interpretation.")
        return default_interpretation
        
    try:
        # --- Prepare summary for LLM --- 
        results_summary = {}
        params = results.get('parameters', {})
        diag_details = diagnostics.get('details', {}) if diagnostics else {}
        parallel_trends = diag_details.get('parallel_trends', {})
        
        effect = results.get('effect_estimate')
        pval = results.get('p_value')
        ci = results.get('confidence_interval')
        
        results_summary['Method Used'] = results.get('method_details', 'Difference-in-Differences')
        results_summary['Effect Estimate'] = f"{effect:.3f}" if isinstance(effect, (int, float)) else str(effect)
        results_summary['P-value'] = f"{pval:.3f}" if isinstance(pval, (int, float)) else str(pval)
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
             results_summary['Confidence Interval'] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
             results_summary['Confidence Interval'] = str(ci) if ci is not None else "N/A"

        results_summary['Time Variable'] = params.get('time_var', 'N/A')
        results_summary['Group/Unit Variable'] = params.get('group_var', 'N/A')
        results_summary['Treatment Indicator Used'] = params.get('treatment_indicator', 'N/A')
        results_summary['Treatment Start Period'] = params.get('treatment_period_start', 'N/A')
        results_summary['Covariates Included'] = params.get('covariates', [])

        diag_summary = {}
        diag_summary['Parallel Trends Assumption Status'] = "Passed (Placeholder)" if parallel_trends.get('valid', False) else "Failed/Unknown (Placeholder)"
        if not parallel_trends.get('valid', False) and parallel_trends.get('details') != "Placeholder validation":
             diag_summary['Parallel Trends Details'] = parallel_trends.get('details', 'N/A')
             
        # --- Example: Add dataset description context --- 
        context_str = ""
        if dataset_description:
            # context_str += f"\nDataset Context: {dataset_description.get('summary', 'N/A')}" # Use string directly
            context_str += f"\n\nDataset Context Provided:\n{dataset_description}"
        # ------------------------------------------------

        # --- Construct Prompt --- 
        prompt = f"""
        You are assisting with interpreting Difference-in-Differences (DiD) results.
        {context_str} # Add context here
        
        Estimation Results Summary:
        {results_summary}
        
        Diagnostics Summary:
        {diag_summary}
        
        Explain these DiD results in 2-4 concise sentences. Focus on:
        1. The estimated average treatment effect on the treated (magnitude, direction, statistical significance based on p-value < 0.05).
        2. The status of the parallel trends assumption (mentioning it's a key assumption for DiD).
        3. Note that the estimation controlled for unit and time fixed effects, and potentially covariates {results_summary['Covariates Included']} 
        
        Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
        {{
          "interpretation": "<your concise interpretation text>"
        }}
        """
        
        # --- Call LLM --- 
        response = call_llm_with_json_output(llm, prompt)
        
        # --- Process Response --- 
        if response and isinstance(response, dict) and \
            "interpretation" in response and isinstance(response["interpretation"], str):
            return response["interpretation"]
        else:
            logger.warning(f"Failed to get valid interpretation from LLM for DiD. Response: {response}")
            return default_interpretation
            
    except Exception as e:
        logger.error(f"Error during LLM interpretation for DiD: {e}")
        return f"Error generating interpretation: {e}" 
"""
Dataset analyzer component for causal inference.

This module provides functionality to analyze datasets to detect characteristics
relevant for causal inference methods, including temporal structure, potential
instrumental variables, discontinuities, and variable relationships.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import logging
import json
from langchain_core.language_models import BaseChatModel
from causal_agent.utils.llm_helpers import llm_identify_temporal_and_unit_vars

logger = logging.getLogger(__name__)

def _calculate_per_group_stats(df: pd.DataFrame, potential_treatments: List[str]) -> Dict[str, Dict]:
    """Calculates summary stats for numeric covariates grouped by potential binary treatments."""
    stats_dict = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    for treat_var in potential_treatments:
        if treat_var not in df.columns:
            logger.warning(f"Potential treatment '{treat_var}' not found in DataFrame columns.")
            continue
            
        # Ensure treatment is binary (0/1 or similar)
        unique_vals = df[treat_var].dropna().unique()
        if len(unique_vals) != 2:
            logger.info(f"Skipping stats for potential treatment '{treat_var}' as it is not binary ({len(unique_vals)} unique values).")
            continue
            
        # Attempt to map values to 0 and 1 if possible
        try:
             # Ensure boolean is converted to int
            if df[treat_var].dtype == 'bool':
                 df[treat_var] = df[treat_var].astype(int)
                 unique_vals = df[treat_var].dropna().unique()
                 
            # Basic check if values are interpretable as 0/1
            if not set(unique_vals).issubset({0, 1}): 
                 # Attempt conversion if possible (e.g., True/False strings?)
                 logger.warning(f"Potential treatment '{treat_var}' has values {unique_vals}, not {0, 1}. Cannot calculate group stats reliably.")
                 continue 
        except Exception as e:
            logger.warning(f"Could not process potential treatment '{treat_var}' values ({unique_vals}): {e}")
            continue
            
        logger.info(f"Calculating group stats for treatment: '{treat_var}'")
        treat_stats = {'group_sizes': {}, 'covariate_stats': {}}
        
        try:
            grouped = df.groupby(treat_var)
            sizes = grouped.size()
            treat_stats['group_sizes']['treated'] = int(sizes.get(1, 0))
            treat_stats['group_sizes']['control'] = int(sizes.get(0, 0))
            
            if treat_stats['group_sizes']['treated'] == 0 or treat_stats['group_sizes']['control'] == 0:
                logger.warning(f"Treatment '{treat_var}' has zero samples in one group. Skipping covariate stats.")
                stats_dict[treat_var] = treat_stats
                continue

            # Calculate mean and std for numeric covariates
            cov_stats = grouped[numeric_cols].agg(['mean', 'std']).unstack()
            
            for cov in numeric_cols:
                if cov == treat_var: continue # Skip treatment variable itself
                
                mean_control = cov_stats.get(('mean', 0, cov), np.nan)
                std_control = cov_stats.get(('std', 0, cov), np.nan)
                mean_treated = cov_stats.get(('mean', 1, cov), np.nan)
                std_treated = cov_stats.get(('std', 1, cov), np.nan)
                
                treat_stats['covariate_stats'][cov] = {
                    'mean_control': float(mean_control) if pd.notna(mean_control) else None,
                    'std_control': float(std_control) if pd.notna(std_control) else None,
                    'mean_treat': float(mean_treated) if pd.notna(mean_treated) else None,
                    'std_treat': float(std_treated) if pd.notna(std_treated) else None,
                }
            stats_dict[treat_var] = treat_stats
        except Exception as e:
            logger.error(f"Error calculating stats for treatment '{treat_var}': {e}", exc_info=True)
            # Store partial info if possible
            if treat_var not in stats_dict: 
                 stats_dict[treat_var] = {'error': str(e)}
            elif 'error' not in stats_dict[treat_var]:
                 stats_dict[treat_var]['error'] = str(e)
                 
    return stats_dict

def analyze_dataset(
    dataset_path: str, 
    llm_client: Optional[BaseChatModel] = None,
    dataset_description: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a dataset to identify important characteristics for causal inference.
    
    Args:
        dataset_path: Path to the dataset file
        llm_client: Optional LLM client for enhanced analysis
        dataset_description: Optional description of the dataset for context
        
    Returns:
        Dict containing dataset analysis results:
            - dataset_info: Basic information about the dataset
            - columns: List of column names
            - potential_treatments: List of potential treatment variables (possibly LLM augmented)
            - potential_outcomes: List of potential outcome variables (possibly LLM augmented)
            - temporal_structure_detected: Whether temporal structure was detected
            - panel_data_detected: Whether panel data structure was detected
            - potential_instruments_detected: Whether potential instruments were detected
            - discontinuities_detected: Whether discontinuities were detected
            - llm_augmentation: Status of LLM augmentation if used
    """
    llm_augmentation = "Not used" if not llm_client else "Initialized"
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return {"error": f"Dataset file not found at {dataset_path}"}
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Basic dataset information
        sample_size = len(df)
        columns_list = df.columns.tolist()
        num_covariates = len(columns_list) - 2 # Rough estimate (total - T - Y)
        dataset_info = {
            "num_rows": sample_size,
            "num_columns": len(columns_list),
            "file_path": dataset_path,
            "file_name": os.path.basename(dataset_path)
        }
        
        # --- Detailed Analysis (Keep internal) ---
        column_types_detailed = {col: str(df[col].dtype) for col in df.columns}
        missing_values_detailed = df.isnull().sum().to_dict()
        column_categories_detailed = _categorize_columns(df)
        column_nunique_counts_detailed = {col: df[col].nunique() for col in df.columns} # Calculate nunique
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        correlations_detailed = df[numeric_cols].corr() if numeric_cols else pd.DataFrame()
        temporal_structure_detailed = detect_temporal_structure(df, llm_client, dataset_description, original_query)
        
        # First, identify potential treatment and outcome variables
        potential_variables = _identify_potential_variables(
            df, 
            column_categories_detailed,
            llm_client=llm_client,
            dataset_description=dataset_description
        )
        
        if llm_client:
            llm_augmentation = "Used for variable identification"
        
        # Then use that info to help find potential instrumental variables
        potential_instruments_detailed = find_potential_instruments(
            df,
            llm_client=llm_client,
            potential_treatments=potential_variables.get("potential_treatments", []),
            potential_outcomes=potential_variables.get("potential_outcomes", []),
            dataset_description=dataset_description
        )
        
        # Other analyses
        discontinuities_detailed = detect_discontinuities(df)
        variable_relationships_detailed = assess_variable_relationships(df, correlations_detailed)
        
        # Calculate per-group stats for potential binary treatments
        potential_binary_treatments = [ 
            t for t in potential_variables["potential_treatments"] 
            if column_categories_detailed.get(t) == 'binary' 
            or column_categories_detailed.get(t) == 'binary_categorical'
        ]
        per_group_stats = _calculate_per_group_stats(df.copy(), potential_binary_treatments)

        # --- Summarized Analysis (For Output) ---
        
        # Get boolean flags and essential lists
        has_temporal = temporal_structure_detailed.get("has_temporal_structure", False)
        is_panel = temporal_structure_detailed.get("is_panel_data", False)
        logger.info(f"iv is {potential_instruments_detailed}")
        has_instruments = len(potential_instruments_detailed) > 0
        has_discontinuities = discontinuities_detailed.get("has_discontinuities", False)
        
        # --- Extract only instrument names for the final output ---
        potential_instrument_names = [
            inst_dict.get('variable') 
            for inst_dict in potential_instruments_detailed 
            if isinstance(inst_dict, dict) and 'variable' in inst_dict
        ]
        logger.info(f"iv is {potential_instrument_names}")
        # --- Final Output Dictionary (Highly Summarized) ---
        return {
            "dataset_info": dataset_info, # Keep basic info
            "columns": columns_list,
            "potential_treatments": potential_variables["potential_treatments"],
            "potential_outcomes": potential_variables["potential_outcomes"],
            # Return concise flags instead of detailed dicts/lists
            "temporal_structure_detected": has_temporal, 
            "panel_data_detected": is_panel,
            "potential_instruments_detected": has_instruments,
            "discontinuities_detected": has_discontinuities,
            # Use the extracted list of names here
            "potential_instruments": potential_instrument_names,
            "discontinuities": discontinuities_detailed,
            "temporal_structure": temporal_structure_detailed,
            "column_categories": column_categories_detailed,
            "column_nunique_counts": column_nunique_counts_detailed, # Add nunique counts to output
            "sample_size": sample_size,
            "num_covariates_estimate": num_covariates,
            "llm_augmentation": llm_augmentation
        }
    
    except Exception as e:
        logger.error(f"Error analyzing dataset '{dataset_path}': {e}", exc_info=True)
        return {
            "error": f"Error analyzing dataset: {str(e)}",
            "llm_augmentation": llm_augmentation
        }


def _categorize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Categorize columns into types relevant for causal inference.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict mapping column names to their types
    """
    result = {}
    
    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Count number of unique values
            n_unique = df[col].nunique()
            
            # Binary numeric variable
            if n_unique == 2:
                result[col] = "binary"
            # Likely categorical represented as numeric
            elif n_unique < 10:
                result[col] = "categorical_numeric"
            # Discrete numeric (integers)
            elif pd.api.types.is_integer_dtype(df[col]):
                result[col] = "discrete_numeric"
            # Continuous numeric
            else:
                result[col] = "continuous_numeric"
        
        # Check for datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or _is_date_string(df, col):
            result[col] = "datetime"
        
        # Check for categorical
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
            if df[col].nunique() == 2:
                result[col] = "binary_categorical"
            else:
                result[col] = "categorical"
        
        # Must be text or other
        else:
            result[col] = "text_or_other"
    
    return result


def _is_date_string(df: pd.DataFrame, col: str) -> bool:
    """
    Check if a column contains date strings.
    
    Args:
        df: DataFrame to check
        col: Column name to check
        
    Returns:
        True if the column appears to contain date strings
    """
    # Try to convert to datetime
    if not pd.api.types.is_string_dtype(df[col]):
        return False
    
    # Check sample of values
    sample = df[col].dropna().sample(min(10, len(df[col].dropna()))).tolist()
    
    try:
        for val in sample:
            pd.to_datetime(val)
        return True
    except:
        return False


def _identify_potential_variables(
    df: pd.DataFrame, 
    column_categories: Dict[str, str],
    llm_client: Optional[BaseChatModel] = None,
    dataset_description: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Identify potential treatment and outcome variables in the dataset, using LLM if available.
    Falls back to heuristic method if LLM fails or is not available.
    
    Args:
        df: DataFrame to analyze
        column_categories: Dictionary mapping column names to their types
        llm_client: Optional LLM client for enhanced identification
        dataset_description: Optional description of the dataset for context
        
    Returns:
        Dict with potential treatment and outcome variables
    """
    # Try LLM approach if client is provided
    if llm_client:
        try:
            logger.info("Using LLM to identify potential treatment and outcome variables")
            
            # Create a concise prompt with just column information
            columns_list = df.columns.tolist()
            column_types = {col: str(df[col].dtype) for col in columns_list}
            
            # Get binary columns for extra context
            binary_cols = [col for col in columns_list 
                          if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() == 2]
            
            # Add dataset description if available
            description_text = f"\nDataset Description: {dataset_description}" if dataset_description else ""
            
            prompt = f"""
You are an expert causal inference data scientist. Identify potential treatment and outcome variables from this dataset.{description_text}

Dataset columns:
{columns_list}

Column types:
{column_types}

Binary columns (good treatment candidates):
{binary_cols}

Instructions:
1. Identify TREATMENT variables: interventions, treatments, programs, policies, or binary state changes.
   Look for binary variables or names with 'treatment', 'intervention', 'program', 'policy', etc.

2. Identify OUTCOME variables: results, effects, or responses to treatments.
   Look for numeric variables (especially non-binary) or names with 'outcome', 'result', 'effect', 'score', etc.

Return ONLY a valid JSON object with two lists: "potential_treatments" and "potential_outcomes".
Example: {{"potential_treatments": ["treatment_a", "program_b"], "potential_outcomes": ["result_score", "outcome_measure"]}}
"""
            
            # Call the LLM and parse the response
            response = llm_client.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from the response text
            import re
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Validate the response
                if (isinstance(result, dict) and 
                    "potential_treatments" in result and 
                    "potential_outcomes" in result and
                    isinstance(result["potential_treatments"], list) and
                    isinstance(result["potential_outcomes"], list)):
                    
                    # Ensure all suggestions are valid columns
                    valid_treatments = [col for col in result["potential_treatments"] if col in df.columns]
                    valid_outcomes = [col for col in result["potential_outcomes"] if col in df.columns]
                    
                    if valid_treatments and valid_outcomes:
                        logger.info(f"LLM identified {len(valid_treatments)} treatments and {len(valid_outcomes)} outcomes")
                        return {
                            "potential_treatments": valid_treatments,
                            "potential_outcomes": valid_outcomes
                        }
                    else:
                        logger.warning("LLM suggested invalid columns, falling back to heuristic method")
                else:
                    logger.warning("Invalid LLM response format, falling back to heuristic method")
            else:
                logger.warning("Could not extract JSON from LLM response, falling back to heuristic method")
                
        except Exception as e:
            logger.error(f"Error in LLM identification: {e}", exc_info=True)
            logger.info("Falling back to heuristic method")
    
    # Fallback to heuristic method
    logger.info("Using heuristic method to identify potential treatment and outcome variables")
    
    # Identify potential treatment variables
    potential_treatments = []
    
    # Look for binary variables (good treatment candidates)
    binary_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() == 2]
    
    # Look for variables with names suggesting treatment
    treatment_keywords = ['treatment', 'treat', 'intervention', 'program', 'policy', 
                         'exposed', 'assigned', 'received', 'participated']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in treatment_keywords):
            potential_treatments.append(col)
    
    # Add binary variables if we don't have enough candidates
    if len(potential_treatments) < 3:
        for col in binary_cols:
            if col not in potential_treatments:
                potential_treatments.append(col)
                if len(potential_treatments) >= 3:
                    break
    
    # Identify potential outcome variables
    potential_outcomes = []
    
    # Look for numeric variables that aren't binary
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_binary_numeric = [col for col in numeric_cols if col not in binary_cols]
    
    # Look for variables with names suggesting outcomes
    outcome_keywords = ['outcome', 'result', 'effect', 'response', 'score', 'performance',
                       'achievement', 'success', 'failure', 'improvement']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in outcome_keywords):
            potential_outcomes.append(col)
    
    # Add numeric non-binary variables if we don't have enough candidates
    if len(potential_outcomes) < 3:
        for col in non_binary_numeric:
            if col not in potential_outcomes and col not in potential_treatments:
                potential_outcomes.append(col)
                if len(potential_outcomes) >= 3:
                    break
    
    return {
        "potential_treatments": potential_treatments,
        "potential_outcomes": potential_outcomes
    }


def detect_temporal_structure(
    df: pd.DataFrame, 
    llm_client: Optional[BaseChatModel] = None, 
    dataset_description: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect temporal structure in the dataset, using LLM for enhanced identification.
    
    Args:
        df: DataFrame to analyze
        llm_client: Optional LLM client for enhanced identification
        dataset_description: Optional description of the dataset for context
        
    Returns:
        Dict with information about temporal structure:
            - has_temporal_structure: Whether temporal structure exists
            - temporal_columns: Primary time column identified (or list if multiple from heuristic)
            - is_panel_data: Whether data is in panel format
            - time_column: Primary time column identified for panel data
            - id_column: Primary unit ID column identified for panel data
            - time_periods: Number of time periods (if panel data)
            - units: Number of unique units (if panel data)
            - identification_method: How time/unit vars were identified ('LLM', 'Heuristic', 'None')
    """
    result = {
        "has_temporal_structure": False,
        "temporal_columns": [], # Will store primary time column or heuristic list
        "is_panel_data": False,
        "time_column": None,
        "id_column": None,
        "time_periods": None,
        "units": None,
        "identification_method": "None"
    }
    
    # --- Step 1: Heuristic identification (as before) ---
    #heuristic_datetime_cols = []
    #for col in df.columns:
    #    if pd.api.types.is_datetime64_any_dtype(df[col]):
    #        heuristic_datetime_cols.append(col)
    #    elif pd.api.types.is_string_dtype(df[col]):
    #        try:
    #            if pd.to_datetime(df[col], errors='coerce').notna().any():
    #                heuristic_datetime_cols.append(col)
    #        except:
    #            pass # Ignore conversion errors
    
    #time_keywords = ['year', 'month', 'day', 'date', 'time', 'period', 'quarter', 'week']
    #for col in df.columns:
    #    col_lower = col.lower()
    #    if any(keyword in col_lower for keyword in time_keywords) and col not in heuristic_datetime_cols:
    #        heuristic_datetime_cols.append(col)

    #id_keywords = ['id', 'individual', 'person', 'unit', 'entity', 'firm', 'company', 'state', 'country']
    #heuristic_potential_id_cols = []
    #for col in df.columns:
    #    col_lower = col.lower()
    #    # Exclude columns already identified as time-related by heuristics
    #    if any(keyword in col_lower for keyword in id_keywords) and col not in heuristic_datetime_cols:
    #        heuristic_potential_id_cols.append(col)

    # --- Step 2: LLM-assisted identification --- 
    llm_identified_time_var = None
    llm_identified_unit_var = None
    heuristic_datetime_cols = []
    heuristic_potential_id_cols = []
    dataset_summary = df.describe(include='all')

    if llm_client:
        logger.info("Attempting LLM-assisted identification of temporal/unit variables.")
        column_names = df.columns.tolist()
        column_dtypes_dict = {col: str(df[col].dtype) for col in column_names}
        
        try:
            llm_suggestions = llm_identify_temporal_and_unit_vars(
                column_names=column_names,
                column_dtypes=column_dtypes_dict,
                dataset_description=dataset_description if dataset_description else "No dataset description provided.",
                dataset_summary=dataset_summary,
                heuristic_time_candidates=heuristic_datetime_cols,
                heuristic_id_candidates=heuristic_potential_id_cols,
                query=original_query if original_query else "No query provided.",
                llm=llm_client
            )
            llm_identified_time_var = llm_suggestions.get("time_variable")
            llm_identified_unit_var = llm_suggestions.get("unit_variable")
            result["identification_method"] = "LLM"
            
            if not llm_identified_time_var and not llm_identified_unit_var:
                result["identification_method"] = "LLM_NoIdentification"
        except Exception as e:
            logger.warning(f"LLM call for temporal/unit vars failed: {e}. Falling back to heuristics.")
            result["identification_method"] = "Heuristic_LLM_Error"
    else:
        result["identification_method"] = "Heuristic_NoLLM"

    # --- Step 3: Combine LLM and Heuristic Results --- 
    final_time_var = None
    final_unit_var = None

    if llm_identified_time_var:
        final_time_var = llm_identified_time_var
        logger.info(f"Prioritizing LLM identified time variable: {final_time_var}")
    elif heuristic_datetime_cols:
        final_time_var = heuristic_datetime_cols[0] # Fallback to first heuristic time col
        logger.info(f"Using heuristic time variable: {final_time_var}")
    
    if llm_identified_unit_var:
        final_unit_var = llm_identified_unit_var
        logger.info(f"Prioritizing LLM identified unit variable: {final_unit_var}")
    elif heuristic_potential_id_cols:
        final_unit_var = heuristic_potential_id_cols[0] # Fallback to first heuristic ID col
        logger.info(f"Using heuristic unit variable: {final_unit_var}")

    # Update results based on final selections
    if final_time_var:
        result["has_temporal_structure"] = True
        result["temporal_columns"] = [final_time_var] # Store as a list with the primary time var
        result["time_column"] = final_time_var
    else: # If no time var found by LLM or heuristic, use original heuristic list for temporal_columns
        if heuristic_datetime_cols:
            result["has_temporal_structure"] = True
            result["temporal_columns"] = heuristic_datetime_cols
        # time_column remains None

    if final_unit_var:
        result["id_column"] = final_unit_var

    # --- Step 4: Update Panel Data Logic (based on final_time_var and final_unit_var) ---
    if final_time_var and final_unit_var:
        # Check if there are multiple time periods per unit using the identified variables
        try:
            # Ensure columns exist before groupby
            if final_time_var in df.columns and final_unit_var in df.columns:
                if df.groupby(final_unit_var)[final_time_var].nunique().mean() > 1.0:
                    result["is_panel_data"] = True
                    result["time_periods"] = df[final_time_var].nunique()
                    result["units"] = df[final_unit_var].nunique()
                    logger.info(f"Panel data detected: Time='{final_time_var}', Unit='{final_unit_var}', Periods={result['time_periods']}, Units={result['units']}")
                else:
                    logger.info("Not panel data: Each unit does not have multiple time periods.")
            else:
                logger.warning(f"Final time ('{final_time_var}') or unit ('{final_unit_var}') var not in DataFrame. Cannot confirm panel structure.")
        except Exception as e:
            logger.error(f"Error checking panel data structure with time='{final_time_var}', unit='{final_unit_var}': {e}")
            result["is_panel_data"] = False # Default to false on error
    else:
        logger.info("Not panel data: Missing either time or unit variable for panel structure.")

    logger.debug(f"Final temporal structure detection result: {result}")
    return result


def find_potential_instruments(
    df: pd.DataFrame, 
    llm_client: Optional[BaseChatModel] = None,
    potential_treatments: List[str] = None,
    potential_outcomes: List[str] = None,
    dataset_description: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find potential instrumental variables in the dataset, using LLM if available.
    Falls back to heuristic method if LLM fails or is not available.
    
    Args:
        df: DataFrame to analyze
        llm_client: Optional LLM client for enhanced identification
        potential_treatments: Optional list of potential treatment variables
        potential_outcomes: Optional list of potential outcome variables
        dataset_description: Optional description of the dataset for context
        
    Returns:
        List of potential instrumental variables with their properties
    """
    # Try LLM approach if client is provided
    if llm_client:
        try:
            logger.info("Using LLM to identify potential instrumental variables")
            
            # Create a concise prompt with just column information
            columns_list = df.columns.tolist()
            
            # Exclude known treatment and outcome variables from consideration
            excluded_columns = []
            if potential_treatments:
                excluded_columns.extend(potential_treatments)
            if potential_outcomes:
                excluded_columns.extend(potential_outcomes)
                
            # Filter columns to exclude treatments and outcomes
            candidate_columns = [col for col in columns_list if col not in excluded_columns]
            
            if not candidate_columns:
                logger.warning("No eligible columns for instrumental variables after filtering treatments and outcomes")
                return []
                
            # Get column types for context
            column_types = {col: str(df[col].dtype) for col in candidate_columns}
            
            # Add dataset description if available
            description_text = f"\nDataset Description: {dataset_description}" if dataset_description else ""
            
            prompt = f"""
You are an expert causal inference data scientist. Identify potential instrumental variables from this dataset.{description_text}

DEFINITION: Instrumental variables must:
1. Be correlated with the treatment variable (relevance)
2. Only affect the outcome through the treatment (exclusion restriction)
3. Not be correlated with unmeasured confounders (exogeneity)

Treatment variables: {potential_treatments if potential_treatments else "Unknown"}
Outcome variables: {potential_outcomes if potential_outcomes else "Unknown"}

Available columns (excluding treatments and outcomes):
{candidate_columns}

Column types:
{column_types}

Look for variables likely to be:
- Random assignments
- Policy changes
- Geographic or temporal variations
- Variables with names containing: 'instrument', 'iv', 'assigned', 'random', 'lottery', 'exogenous'

Return ONLY a JSON array of objects, each with "variable", "reason", and "data_type" fields.
Example: 
[
  {{"variable": "random_assignment", "reason": "Random assignment variable", "data_type": "int64"}},
  {{"variable": "distance_to_facility", "reason": "Geographic variation", "data_type": "float64"}}
]
"""
            
            # Call the LLM and parse the response
            response = llm_client.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from the response text
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Validate the response
                if isinstance(result, list) and len(result) > 0:
                    # Filter for valid entries
                    valid_instruments = []
                    for item in result:
                        if not isinstance(item, dict) or "variable" not in item:
                            continue
                            
                        if item["variable"] not in df.columns:
                            continue
                            
                        # Ensure all required fields are present
                        if "reason" not in item:
                            item["reason"] = "Identified by LLM"
                        if "data_type" not in item:
                            item["data_type"] = str(df[item["variable"]].dtype)
                            
                        valid_instruments.append(item)
                    
                    if valid_instruments:
                        logger.info(f"LLM identified {len(valid_instruments)} potential instrumental variables {valid_instruments}")
                        return valid_instruments
                    else:
                        logger.warning("No valid instruments found by LLM, falling back to heuristic method")
                else:
                    logger.warning("Invalid LLM response format, falling back to heuristic method")
            else:
                logger.warning("Could not extract JSON from LLM response, falling back to heuristic method")
                
        except Exception as e:
            logger.error(f"Error in LLM identification of instruments: {e}", exc_info=True)
            logger.info("Falling back to heuristic method")
    
    # Fallback to heuristic method
    logger.info("Using heuristic method to identify potential instrumental variables")
    potential_instruments = []
    
    # Look for variables with instrumental-related names
    instrument_keywords = ['instrument', 'iv', 'assigned', 'random', 'lottery', 'exogenous']
    
    for col in df.columns:
        # Skip treatment and outcome variables
        if potential_treatments and col in potential_treatments:
            continue
        if potential_outcomes and col in potential_outcomes:
            continue
            
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in instrument_keywords):
            instrument_info = {
                "variable": col,
                "reason": f"Name contains instrument-related keyword",
                "data_type": str(df[col].dtype)
            }
            potential_instruments.append(instrument_info)
    
    return potential_instruments


def detect_discontinuities(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify discontinuities in continuous variables (for RDD).
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict with information about detected discontinuities
    """
    discontinuities = []
    
    # For each numeric column, check for potential discontinuities
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in numeric_cols:
        # Skip columns with too many unique values
        if df[col].nunique() > 100:
            continue
        
        values = df[col].dropna().sort_values().values
        
        # Calculate gaps between consecutive values
        if len(values) > 10:
            gaps = np.diff(values)
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            
            # Look for unusually large gaps (potential discontinuities)
            large_gaps = np.where(gaps > mean_gap + 2*std_gap)[0]
            
            if len(large_gaps) > 0:
                for idx in large_gaps:
                    cutpoint = (values[idx] + values[idx+1]) / 2
                    discontinuities.append({
                        "variable": col,
                        "cutpoint": float(cutpoint),
                        "gap_size": float(gaps[idx]),
                        "mean_gap": float(mean_gap)
                    })
    
    return {
        "has_discontinuities": len(discontinuities) > 0,
        "discontinuities": discontinuities
    }


def assess_variable_relationships(df: pd.DataFrame, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess relationships between variables in the dataset.
    
    Args:
        df: DataFrame to analyze
        corr_matrix: Precomputed correlation matrix for numeric columns
        
    Returns:
        Dict with information about variable relationships:
            - strongly_correlated_pairs: Pairs of strongly correlated variables
            - potential_confounders: Variables that might be confounders
    """
    result = {"strongly_correlated_pairs": [], "potential_confounders": []}
    
    numeric_cols = corr_matrix.columns.tolist()
    if len(numeric_cols) < 2:
        return result
    
    # Use the precomputed correlation matrix
    corr_matrix_abs = corr_matrix.abs()
    
    # Find strongly correlated variable pairs
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(corr_matrix_abs.iloc[i, j]) > 0.7:  # Correlation threshold
                result["strongly_correlated_pairs"].append({
                    "variables": [numeric_cols[i], numeric_cols[j]],
                    "correlation": float(corr_matrix.iloc[i, j])
                })
    
    # Identify potential confounders (variables correlated with multiple others)
    confounder_counts = {col: 0 for col in numeric_cols}
    
    for pair in result["strongly_correlated_pairs"]:
        confounder_counts[pair["variables"][0]] += 1
        confounder_counts[pair["variables"][1]] += 1
    
    # Variables correlated with multiple others are potential confounders
    for col, count in confounder_counts.items():
        if count >= 2:
            result["potential_confounders"].append({"variable": col, "num_correlations": count})
    
    return result 
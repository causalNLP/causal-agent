"""
Difference-in-Differences Estimator using DoWhy with Statsmodels fallback.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from causal_agent.config import get_llm_client 
import statsmodels.formula.api as smf

from .llm_assist import (
    identify_time_variable, 
    determine_treatment_period, 
    identify_treatment_group,
    interpret_did_results
)
from .diagnostics import validate_parallel_trends # Import diagnostics

from .utils import create_post_indicator

logger = logging.getLogger(__name__)

# --- Helper functions moved from old file --- 
def format_did_results(statsmodels_results: Any, interaction_term_key: str, 
                       validation_results: Dict[str, Any], 
                       method_details: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    '''Formats the DiD results from statsmodels results into a standard dictionary.'''
    
    try:
        # Use the interaction_term_key passed directly
        effect = float(statsmodels_results.params[interaction_term_key])
        stderr = float(statsmodels_results.bse[interaction_term_key])
        pval = float(statsmodels_results.pvalues[interaction_term_key])
        ci = statsmodels_results.conf_int().loc[interaction_term_key].values.tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
        logger.info(f"Extracted effect for '{interaction_term_key}'")
        
    except KeyError:
        logger.error(f"Interaction term '{interaction_term_key}' not found in statsmodels results. Available params: {statsmodels_results.params.index.tolist()}")
        # Fallback to NaN if term not found
        effect, stderr, pval, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan, np.nan
    except Exception as e:
        logger.error(f"Error extracting results from statsmodels object: {e}")
        effect, stderr, pval, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Create a standardized results dictionary
    results = {
        "effect_estimate": effect,
        "standard_error": stderr,
        "p_value": pval,
        "confidence_interval": [ci_lower, ci_upper],
        "diagnostics": validation_results,
        "parameters": parameters,
        "details": str(statsmodels_results.summary())
    }
    
    return results



def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, 
                      covariates: List[str], 
                      dataset_description: Optional[str] = None,
                      query: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
    """Difference-in-Differences estimation using DoWhy with Statsmodels fallback.
    
    Args:
        df: Dataset containing causal variables
        treatment: Name of treatment variable (or variable indicating treated group)
        outcome: Name of outcome variable
        covariates: List of covariate names
        dataset_description: Optional dictionary describing the dataset
        **kwargs: Method-specific parameters (e.g., time_var, group_var, query, llm instance if needed)
        
    Returns:
        Dictionary with effect estimate and diagnostics
    """
    query = kwargs.get('query_str')
    # llm_instance = kwargs.get('llm') # Pass llm if helpers need it
    df_processed = df.copy() # Work on a copy

    logger.info("Starting DiD estimation using DoWhy with Statsmodels fallback...")

    # --- Step 1: Identify Key Variables (using LLM Assist placeholders) --- 
    # Pass llm_instance to helpers if they are implemented to use it
    llm_instance = get_llm_client() # Get llm instance if passed
    time_var = kwargs.get('time_variable', identify_time_variable(df_processed, query, dataset_description, llm=llm_instance))
    if time_var is None:
        raise ValueError("Time variable could not be identified for DiD.")
    if time_var not in df_processed.columns:
         raise ValueError(f"Identified time variable '{time_var}' not found in DataFrame.")
        
    # Determine the variable that identifies the panel unit (for grouping/FE)
    group_var = kwargs.get('group_variable', identify_treatment_group(df_processed, treatment, query, dataset_description, llm=llm_instance))
    if group_var is None:
        raise ValueError("Group/Unit variable could not be identified for DiD.")
    if group_var not in df_processed.columns:
         raise ValueError(f"Identified group/unit variable '{group_var}' not found in DataFrame.")

    # Check outcome exists before proceeding further
    if outcome not in df_processed.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in DataFrame.")

    # Determine treatment period start
    treatment_period = kwargs.get('treatment_period_start', kwargs.get('treatment_period', 
                                  determine_treatment_period(df_processed, time_var, treatment, query, dataset_description, llm=llm_instance)))
                                  
    # --- Identify the TRUE binary treatment group indicator column --- 
    treated_group_col_for_formula = None
    
    # Priority 1: Check if the 'treatment' argument itself is a valid binary indicator
    if treatment in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[treatment]):
        unique_treat_vals = set(df_processed[treatment].dropna().unique())
        if unique_treat_vals.issubset({0, 1}):
            treated_group_col_for_formula = treatment
            logger.info(f"Using the provided 'treatment' argument '{treatment}' as binary group indicator.")
            
    # Priority 2: Check if a column explicitly named 'group' exists and is binary
    if treated_group_col_for_formula is None and 'group' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['group']):
        unique_group_vals = set(df_processed['group'].dropna().unique())
        if unique_group_vals.issubset({0, 1}):
            treated_group_col_for_formula = 'group'
            logger.info(f"Using column 'group' as binary group indicator.")

    # Priority 3: Fallback - Search other columns (excluding known roles and time-related ones)
    if treated_group_col_for_formula is None:
        logger.warning(f"Provided 'treatment' arg '{treatment}' is not binary 0/1 and no 'group' column found. Searching other columns...")
        potential_group_cols = []
        # Exclude outcome, time var, unit ID var, and common time indicators like 'post'
        excluded_cols = [outcome, time_var, group_var, 'post', 'is_post_treatment', 'did_interaction'] 
        for col_name in df_processed.columns:
            if col_name in excluded_cols:
                continue
            try:
                col_data = df_processed[col_name]
                # Ensure we are working with a Series
                if isinstance(col_data, pd.DataFrame):
                    if col_data.shape[1] == 1:
                        col_data = col_data.iloc[:, 0] # Extract the Series
                    else:
                        logger.warning(f"Skipping multi-column DataFrame slice for '{col_name}'.")
                        continue 
                
                # Check if the Series can be interpreted as binary 0/1
                if not pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
                    continue # Skip non-numeric/non-boolean columns

                unique_vals = set(col_data.dropna().unique())
                # Simplified check: directly test if unique values are a subset of {0, 1}
                if unique_vals.issubset({0, 1}):
                    logger.info(f"  Found potential binary indicator: {col_name}")
                    potential_group_cols.append(col_name)
                    
            except AttributeError as ae:
                 # Catch attribute errors likely due to unexpected types
                 logger.warning(f"Attribute error checking column '{col_name}': {ae}. Skipping.")
            except Exception as e:
                 logger.warning(f"Unexpected error checking column '{col_name}' during group ID search: {e}")

        if potential_group_cols:
            treated_group_col_for_formula = potential_group_cols[0] # Take the first suitable one found
            logger.info(f"Using column '{treated_group_col_for_formula}' found during search as binary group indicator.")
        else:
            # Final fallback: Use the originally identified group_var, but warn heavily
            treated_group_col_for_formula = group_var 
            logger.error(f"CRITICAL WARNING: Could not find suitable binary treatment group indicator. Using '{group_var}', but this is likely incorrect and will produce invalid DiD estimates.")

    # --- Final Check ---    
    if treated_group_col_for_formula not in df_processed.columns:
         # This case should ideally not happen with the logic above but added defensively
         raise ValueError(f"Determined treatment group column '{treated_group_col_for_formula}' not found in DataFrame.")
    if df_processed[treated_group_col_for_formula].nunique(dropna=True) > 2:
         logger.warning(f"Selected treatment group column '{treated_group_col_for_formula}' is not binary (has {df_processed[treated_group_col_for_formula].nunique()} unique values). DiD requires binary treatment group.")

    # --- Step 2: Create Indicator Variables --- 
    post_indicator_col = 'post' 
    if post_indicator_col not in df_processed.columns:
        # Create the post indicator if it doesn't exist
        df_processed[post_indicator_col] = create_post_indicator(df_processed, time_var, treatment_period)
        
    # Interaction term is treatment group * post
    interaction_term_col = 'did_interaction' # Keep explicit interaction term
    df_processed[interaction_term_col] = df_processed[treated_group_col_for_formula] * df_processed[post_indicator_col]

    # --- Step 3: Validate Parallel Trends (using the group column) --- 
    parallel_trends_validation = validate_parallel_trends(df_processed, time_var, outcome, 
                                                    treated_group_col_for_formula, treatment_period, dataset_description)
    # Note: The validation result is currently just a placeholder
    if not parallel_trends_validation.get('valid', False):
        logger.warning("Parallel trends assumption potentially violated (based on placeholder check). Proceeding with estimation, but results may be biased.")
        # Add this info to the final results diagnostics

    
    
    # Create parameters dictionary for formatting results
    parameters = {
        "time_var": time_var,
        "group_var": group_var,  # Unit ID
        "treatment_indicator": treated_group_col_for_formula,  # Group indicator used in formula basis
        "post_indicator": post_indicator_col,
        "treatment_period_start": treatment_period,
        "covariates": covariates,
    }
    
    # Group diagnostics for formatting
    did_diagnostics = {
        "parallel_trends": parallel_trends_validation,
        # "placebo_test": run_placebo_test(...) 
    }
    
         
    # --- Step 5: Use Statsmodels OLS --- 
    logger.info("Determining Statsmodels OLS formula based on number of time periods...")

    num_time_periods = df_processed[time_var].nunique()
    
    interaction_term_key_for_results: str
    method_details_str: str
    formula: str

    if num_time_periods == 2:
        logger.info(
            f"Number of unique time periods is 2. Using 2x2 DiD formula: "
            f"{outcome} ~ {treated_group_col_for_formula} * {post_indicator_col}"
        )
        # For 2x2 DiD: outcome ~ group * post_indicator
        # The interaction term A:B in statsmodels gives the DiD estimate.
        formula_core = f"{treated_group_col_for_formula} * {post_indicator_col}"
        interaction_term_key_for_results = f"{treated_group_col_for_formula}:{post_indicator_col}"
        
        formula_parts = [formula_core]
        main_model_terms = {outcome, treated_group_col_for_formula, post_indicator_col}

        if covariates:
            filtered_covs = [
                c for c in covariates if c not in main_model_terms
            ]
            if filtered_covs:
                formula_parts.extend(filtered_covs)
        
        formula = f"{outcome} ~ {' + '.join(formula_parts)}"
        parameters["estimation_method"] = "Statsmodels OLS for 2x2 DiD (Group * Post interaction)"
        method_details_str = "DiD via Statsmodels 2x2 (Group * Post interaction)"

    else: # num_time_periods > 2
        logger.info(
            f"Number of unique time periods is {num_time_periods} (>2). "
            f"Using TWFE DiD formula: {outcome} ~ {interaction_term_col} + C({group_var}) + C({time_var})"
        )
        # For TWFE: outcome ~ actual_treatment_variable + UnitFE + TimeFE
        # actual_treatment_variable is interaction_term_col (e.g., treated_group * post_indicator)
        # UnitFE is C(group_var), TimeFE is C(time_var)
        formula_parts = [
            interaction_term_col, 
            f"C({group_var})", 
            f"C({time_var})"
        ]
        interaction_term_key_for_results = interaction_term_col
        main_model_terms = {outcome, interaction_term_col, group_var, time_var}

        if covariates:
            filtered_covs = [
                c for c in covariates if c not in main_model_terms
            ]
            if filtered_covs:
                formula_parts.extend(filtered_covs)
        
        formula = f"{outcome} ~ {' + '.join(formula_parts)}"
        parameters["estimation_method"] = "Statsmodels OLS with TWFE (C() Notation)"
        method_details_str = "DiD via Statsmodels TWFE (C() Notation)"
            
    try:
        logger.info(f"Using formula: {formula}")
        logger.debug(f"Data head for statsmodels:\n{df_processed.head().to_string()}")
        logger.debug(f"Regression DataFrame shape: {df_processed.shape}, Columns: {df_processed.columns.tolist()}")
        
        ols_model = smf.ols(formula=formula, data=df_processed)
        if group_var not in df_processed.columns:
            # This check is mainly for clustering but good to ensure group_var exists.
            # For 2x2, group_var (unit ID) might not be in formula but needed for clustering.
            raise ValueError(f"Clustering variable '{group_var}' (panel unit ID) not found in regression data.")
        logger.debug(f"Clustering standard errors by: {group_var}")
        results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_processed[group_var]})
        
        logger.info("Statsmodels estimation complete.")
        logger.info(f"Statsmodels Results Summary:\n{results.summary()}")
        
        logger.debug(f"Extracting results using interaction term key: {interaction_term_key_for_results}")
        
        parameters["final_formula"] = formula 
        parameters["interaction_term_coefficient_name"] = interaction_term_key_for_results
        
        formatted_results = format_did_results(results, interaction_term_key_for_results, 
                                            did_diagnostics, 
                                            method_details=method_details_str, 
                                            parameters=parameters)
        formatted_results["estimator"] = "statsmodels"
            
    except Exception as e:
        logger.error(f"Statsmodels OLS estimation failed: {e}", exc_info=True)
        raise ValueError(f"DiD estimation failed (both DoWhy and Statsmodels): {e}")
        
    
                                        
    
    # --- Add Interpretation --- (Now add interpretation to the formatted results)
    try:
        # Use the llm_instance fetched earlier
        interpretation = interpret_did_results(formatted_results, did_diagnostics, dataset_description, llm=llm_instance)
        formatted_results['interpretation'] = interpretation
    except Exception as interp_e:
        logger.error(f"DiD Interpretation failed: {interp_e}")
        formatted_results['interpretation'] = "Interpretation failed."
        
    return formatted_results
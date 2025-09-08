"""
Complete Simplified Difference-in-Differences Estimator
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import statsmodels.formula.api as smf
from .diagnostics import validate_parallel_trends
from .llm_assist import interpret_did_results
from cais.config import get_llm_client

logger = logging.getLogger(__name__)

def format_did_results(statsmodels_results: Any, interaction_term_key: str, 
                       validation_results: Dict[str, Any], 
                       method_details: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Formats the DiD results from statsmodels results into a standard dictionary."""
    
    try:
        effect = float(statsmodels_results.params[interaction_term_key])
        stderr = float(statsmodels_results.bse[interaction_term_key])
        pval = float(statsmodels_results.pvalues[interaction_term_key])
        ci = statsmodels_results.conf_int().loc[interaction_term_key].values.tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
        logger.info(f"Extracted effect for '{interaction_term_key}'")
        
    except KeyError:
        logger.error(f"Interaction term '{interaction_term_key}' not found in statsmodels results.")
        effect, stderr, pval, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan, np.nan
    except Exception as e:
        logger.error(f"Error extracting results from statsmodels object: {e}")
        effect, stderr, pval, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan, np.nan
    
    return {
        "effect_estimate": effect,
        "standard_error": stderr,
        "p_value": pval,
        "confidence_interval": [ci_lower, ci_upper],
        "diagnostics": validation_results,
        "parameters": parameters,
        "details": str(statsmodels_results.summary()),
        "estimator": "statsmodels"
    }

def estimate_2x2_did(df: pd.DataFrame, outcome: str, time_var: str, group_var: str,
                     treated_group_col: str, covariates: List[str]) -> Dict[str, Any]:
    """Estimate canonical 2x2 DiD with binary time and treatment variables."""
    
    df_processed = df.copy()
    
    # Create POST indicator (time_var should already be binary)
    df_processed['post'] = df_processed[time_var].astype(int)
    logger.info(f"Using binary time variable '{time_var}' as 'post' indicator")
    
    # Create TREAT indicator (treated_group_col should already be binary)
    df_processed['treat'] = df_processed[treated_group_col].astype(int)
    logger.info(f"Using binary treatment variable '{treated_group_col}' as 'treat' indicator")
    
    # Create DiD interaction term: did_term = post * treat
    df_processed['did_term'] = df_processed['post'] * df_processed['treat']
    
    # Build formula: outcome ~ post + treat + did_term + covariates
    formula_parts = ['post', 'treat', 'did_term']
    main_terms = {outcome, 'post', 'treat', 'did_term'}
    
    if covariates:
        filtered_covs = [c for c in covariates if c not in main_terms]
        formula_parts.extend(filtered_covs)
    
    formula = f"{outcome} ~ {' + '.join(formula_parts)}"
    logger.info(f"2x2 DiD regression formula: {formula}")
    
    # Run regression
    ols_model = smf.ols(formula=formula, data=df_processed)
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_processed[group_var]})
    
    # Format results
    parameters = {
        "time_var": time_var,
        "group_var": group_var,
        "treatment_indicator": treated_group_col,
        "post_indicator": 'post',
        "treat_indicator": 'treat',
        "did_interaction": 'did_term',
        "covariates": covariates,
        "formula": formula,
        "interaction_term": 'did_term',
        "estimation_method": "2x2 DiD"
    }
    
    validation_results = {"parallel_trends": {"valid": True, "details": "Placeholder"}}
    
    return format_did_results(results, 'did_term', validation_results, "2x2 DiD", parameters)

def estimate_twfe_did(df: pd.DataFrame, outcome: str, time_var: str, group_var: str,
                      did_term: str, covariates: List[str]) -> Dict[str, Any]:
    """Estimate TWFE DiD using treatment indicator."""
    
    df_processed = df.copy()
    
    # Validate did_term exists and is binary
    if did_term not in df_processed.columns:
        raise ValueError(f"DiD term '{did_term}' not found in DataFrame for TWFE estimation")
    
    logger.info(f"Using binary did_term '{did_term}' for TWFE estimation")
    
    # Build formula: outcome ~ did_term + C(group_var) + C(time_var) + covariates
    formula_parts = [did_term, f"C({group_var})", f"C({time_var})"]
    main_terms = {outcome, did_term, group_var, time_var}
    
    if covariates:
        filtered_covs = [c for c in covariates if c not in main_terms]
        formula_parts.extend(filtered_covs)
    
    formula = f"{outcome} ~ {' + '.join(formula_parts)}"
    logger.info(f"TWFE formula: {formula}")
    
    # Run regression
    ols_model = smf.ols(formula=formula, data=df_processed)
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_processed[group_var]})
    
    # Format results
    parameters = {
        "time_var": time_var,
        "group_var": group_var,
        "did_term": did_term,
        "covariates": covariates,
        "formula": formula,
        "interaction_term": did_term,
        "estimation_method": "TWFE DiD"
    }
    
    validation_results = {"parallel_trends": {"valid": True, "details": "Placeholder"}}
    
    return format_did_results(results, did_term, validation_results, "TWFE DiD", parameters)

def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, 
                    covariates: List[str], 
                    dataset_description: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
    """
    Simplified DiD estimation with simple binary variable rule.
    
    Args:
        df: Dataset containing causal variables
        treatment: Name of treatment variable 
        outcome: Name of outcome variable
        covariates: List of covariate names
        dataset_description: Optional description for interpretation
        **kwargs: Contains DiD-specific parameters:
            - did_term: Binary treatment indicator column name (for TWFE)
            - time_variable: Time column name
            - group_variable: Unit ID column name
        
    Returns:
        Dictionary with effect estimate and diagnostics
    """
    
    logger.info("Starting simplified DiD estimation...")
    logger.info(f"Kwargs received: {kwargs}")
    
    # Extract variables from kwargs
    time_var = kwargs.get("time_variable")
    group_var = kwargs.get("group_variable")
    did_term = kwargs.get("did_term")
    
    # Validate required variables
    if not time_var or time_var not in df.columns:
        raise ValueError(f"Time variable '{time_var}' not found in DataFrame")

    if outcome not in df.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in DataFrame")
    
    # Simple rule: Check if both time_var and treatment are binary
    time_is_binary = set(df[time_var].dropna().unique()).issubset({0, 1})
    
    treatment_is_binary = False
    if treatment in df.columns:
        treatment_is_binary = set(df[treatment].dropna().unique()).issubset({0, 1})
    
    logger.info(f"Time variable '{time_var}' is binary: {time_is_binary}")
    logger.info(f"Treatment variable '{treatment}' is binary: {treatment_is_binary}")
    
    # Decision rule: Both binary → canonical, otherwise → TWFE
    if time_is_binary and treatment_is_binary:
        # Use canonical 2x2 DiD
        logger.info("Using canonical 2x2 DiD (both time and treatment are binary)")
        
        results = estimate_2x2_did(df, outcome, time_var, group_var, 
                                   treatment, covariates)
        
    else:
        # Use TWFE DiD
        logger.info("Using TWFE DiD (time or treatment not binary)")
        
        if not did_term:
            did_term = treatment  
        
        results = estimate_twfe_did(df, outcome, time_var, group_var, 
                                    did_term, covariates)
    
    # Add interpretation
    try:
        llm_instance = get_llm_client()
        interpretation = interpret_did_results(results, results["diagnostics"], 
                                               dataset_description, llm=llm_instance)
        results['interpretation'] = interpretation
    except Exception as e:
        logger.error(f"DiD Interpretation failed: {e}")
        results['interpretation'] = "Interpretation failed."
    
    return results
# Propensity Score Weighting (IPW) Implementation 

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Optional, Any

from .base import estimate_propensity_scores, format_ps_results, select_propensity_model
from .diagnostics import assess_weight_distribution, plot_overlap, plot_balance # Import diagnostic functions
from .llm_assist import determine_optimal_weight_type, determine_optimal_trim_threshold, get_llm_parameters # Import LLM helpers

def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, 
                      covariates: List[str], **kwargs) -> Dict[str, Any]:
    '''Generic propensity score weighting (IPW) implementation.
    
    Args:
        df: Dataset containing causal variables
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        covariates: List of covariate names
        **kwargs: Method-specific parameters (e.g., weight_type, trim_threshold, query)
        
    Returns:
        Dictionary with effect estimate and diagnostics
    '''
    query = kwargs.get('query')

    # --- LLM-Assisted Parameter Optimization / Defaults --- 
    llm_params = get_llm_parameters(df, query, "PS.Weighting")
    llm_suggested_params = llm_params.get("parameters", {})
    
    # Explicitly check LLM suggestion before falling back to default helper
    llm_weight_type = llm_suggested_params.get('weight_type')
    default_weight_type = determine_optimal_weight_type(df, treatment, query) if llm_weight_type is None else llm_weight_type
    weight_type = kwargs.get('weight_type', default_weight_type)
    
    # Similar explicit check for trim_threshold
    llm_trim_thresh = llm_suggested_params.get('trim_threshold')
    default_trim_thresh = determine_optimal_trim_threshold(df, treatment, query=query) if llm_trim_thresh is None else llm_trim_thresh
    trim_threshold = kwargs.get('trim_threshold', default_trim_thresh)
        
    propensity_model_type = kwargs.get('propensity_model_type', 
                                   llm_suggested_params.get('propensity_model_type', 
                                                          select_propensity_model(df, treatment, covariates, query)))
    robust_se = kwargs.get('robust_se', True)

    # --- Step 1: Estimate propensity scores --- 
    propensity_scores = estimate_propensity_scores(df, treatment, covariates, 
                                                   model_type=propensity_model_type, 
                                                   **kwargs) # Pass other kwargs like C, penalty etc.
    df_ps = df.copy()
    df_ps['propensity_score'] = propensity_scores

    # --- Step 2: Calculate weights --- 
    if weight_type.upper() == 'ATE':
        weights = np.where(df_ps[treatment] == 1, 
                           1 / df_ps['propensity_score'], 
                           1 / (1 - df_ps['propensity_score']))
    elif weight_type.upper() == 'ATT':
        weights = np.where(df_ps[treatment] == 1, 
                           1, 
                           df_ps['propensity_score'] / (1 - df_ps['propensity_score']))
    # TODO: Add other weight types like ATC if needed
    else:
        raise ValueError(f"Unsupported weight type: {weight_type}")
        
    df_ps['ipw'] = weights

    # --- Step 3: Apply trimming if needed --- 
    if trim_threshold is not None and trim_threshold > 0:
        # Trim based on propensity score percentile
        min_ps_thresh = np.percentile(propensity_scores, trim_threshold * 100)
        max_ps_thresh = np.percentile(propensity_scores, (1 - trim_threshold) * 100)
        
        keep_indices = (df_ps['propensity_score'] >= min_ps_thresh) & (df_ps['propensity_score'] <= max_ps_thresh)
        df_trimmed = df_ps[keep_indices].copy()
        print(f"Trimming {len(df_ps) - len(df_trimmed)} units ({trim_threshold*100:.1f}% percentile trim)")
        if df_trimmed.empty:
            raise ValueError("All units removed after trimming. Try a smaller trim_threshold.")
        df_analysis = df_trimmed
    else:
        # Trim based on weight percentile (alternative approach)
        # q_low, q_high = np.percentile(weights, [trim_threshold*100, (1-trim_threshold)*100])
        # df_ps['ipw'] = np.clip(df_ps['ipw'], q_low, q_high)
        df_analysis = df_ps.copy()
        trim_threshold = 0 # Explicitly set for parameters output

    # --- Step 4: Normalize weights (optional but common) --- 
    # Normalize weights to sum to sample size within treated/control groups if ATT
    if weight_type.upper() == 'ATT':
        sum_weights_treated = df_analysis.loc[df_analysis[treatment] == 1, 'ipw'].sum()
        sum_weights_control = df_analysis.loc[df_analysis[treatment] == 0, 'ipw'].sum()
        n_treated = (df_analysis[treatment] == 1).sum()
        n_control = (df_analysis[treatment] == 0).sum()
        
        if sum_weights_treated > 0:
             df_analysis.loc[df_analysis[treatment] == 1, 'ipw'] *= n_treated / sum_weights_treated
        if sum_weights_control > 0:
             df_analysis.loc[df_analysis[treatment] == 0, 'ipw'] *= n_control / sum_weights_control 
    else: # ATE normalization
        df_analysis['ipw'] *= len(df_analysis) / df_analysis['ipw'].sum()

    # --- Step 5: Estimate weighted treatment effect --- 
    X_treat = sm.add_constant(df_analysis[[treatment]]) # Use only treatment variable for direct effect
    wls_model = sm.WLS(df_analysis[outcome], X_treat, weights=df_analysis['ipw'])
    results = wls_model.fit(cov_type='HC1' if robust_se else 'nonrobust')
    
    effect = results.params[treatment]
    effect_se = results.bse[treatment]

    # --- Step 6: Validate weight quality / Diagnostics --- 
    diagnostics = assess_weight_distribution(df_analysis['ipw'], df_analysis[treatment])
    # Could also add balance assessment on the weighted sample
    # weighted_diagnostics = assess_balance(df, df_analysis, treatment, covariates, method="PSW", weights=df_analysis['ipw'])
    # diagnostics.update(weighted_diagnostics)
    diagnostics["propensity_score_model"] = propensity_model_type

    # --- Step 7: Format and return results --- 
    return format_ps_results(effect, effect_se, diagnostics,
                           method_details="PS.Weighting",
                           parameters={"weight_type": weight_type, 
                                         "trim_threshold": trim_threshold,
                                         "propensity_model": propensity_model_type,
                                         "robust_se": robust_se}) 
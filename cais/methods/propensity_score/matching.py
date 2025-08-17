# Propensity Score Matching Implementation 
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm 
import logging 
from typing import Dict, List, Optional, Any

# Import DoWhy
from dowhy import CausalModel

from .base import estimate_propensity_scores, format_ps_results, select_propensity_model
from .diagnostics import assess_balance 
from .llm_assist import get_llm_parameters # Import LLM helpers

logger = logging.getLogger(__name__)

def _calculate_logit(pscore):
    """Calculate logit of propensity score, clipping to avoid inf."""
    # Clip pscore to prevent log(0) or log(1) issues which lead to inf
    epsilon = 1e-7
    pscore_clipped = np.clip(pscore, epsilon, 1 - epsilon)
    return np.log(pscore_clipped / (1 - pscore_clipped))

def _perform_matching_and_get_att(
    df_sample: pd.DataFrame, 
    treatment: str, 
    outcome: str, 
    covariates: List[str],
    propensity_model_type: str,
    n_neighbors: int,
    caliper: float,
    perform_bias_adjustment: bool,
    **kwargs
) -> float:
    """
    Helper to perform Custom KNN PSM and calculate ATT, potentially with bias adjustment.
    Returns the ATT estimate.
    """
    df_ps = df_sample.copy()
    try:
        propensity_scores = estimate_propensity_scores(
            df_ps, treatment, covariates, model_type=propensity_model_type, **kwargs
        )
    except Exception as e:
        logger.warning(f"Propensity score estimation failed in helper: {e}")
        return np.nan # Cannot proceed without propensity scores
        
    df_ps['propensity_score'] = propensity_scores
    
    treated = df_ps[df_ps[treatment] == 1]
    control = df_ps[df_ps[treatment] == 0]
    
    if treated.empty or control.empty:
        return np.nan 

    nn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper if caliper is not None else np.inf, metric='minkowski', p=2)
    try:
        # Ensure control PS are valid before fitting
        control_ps_values = control[['propensity_score']].values
        if np.isnan(control_ps_values).any():
            logger.warning("NaN values found in control propensity scores before NN fitting.")
            return np.nan
        nn.fit(control_ps_values)

        # Ensure treated PS are valid before querying
        treated_ps_values = treated[['propensity_score']].values
        if np.isnan(treated_ps_values).any():
             logger.warning("NaN values found in treated propensity scores before NN query.")
             return np.nan
        distances, indices = nn.kneighbors(treated_ps_values)

    except ValueError as e:
        # Handles case where control group might be too small or have NaN PS scores
        logger.warning(f"NearestNeighbors fitting/query failed: {e}")
        return np.nan

    matched_outcomes_treated = []
    matched_outcomes_control_means = []
    propensity_diffs = []

    for i in range(len(treated)):
        treated_unit = treated.iloc[[i]]
        valid_neighbors_mask = distances[i] <= (caliper if caliper is not None else np.inf)
        valid_neighbors_idx = indices[i][valid_neighbors_mask]
        
        if len(valid_neighbors_idx) > 0:
            matched_controls_for_this_treated = control.iloc[valid_neighbors_idx]
            if matched_controls_for_this_treated.empty:
                continue # Should not happen with valid_neighbors_idx check, but safety

            matched_outcomes_treated.append(treated_unit[outcome].values[0])
            matched_outcomes_control_means.append(matched_controls_for_this_treated[outcome].mean())
            
            if perform_bias_adjustment:
                # Ensure PS scores are valid before calculating difference
                treated_ps = treated_unit['propensity_score'].values[0]
                control_ps_mean = matched_controls_for_this_treated['propensity_score'].mean()
                if np.isnan(treated_ps) or np.isnan(control_ps_mean):
                    logger.warning("NaN propensity score encountered during bias adjustment calculation.")
                    # Cannot perform bias adjustment for this unit, potentially skip or handle
                    # For now, let's skip adding to propensity_diffs if NaN found
                    continue 
                propensity_diff = treated_ps - control_ps_mean
                propensity_diffs.append(propensity_diff)

    if not matched_outcomes_treated:
        return np.nan

    raw_att_components = np.array(matched_outcomes_treated) - np.array(matched_outcomes_control_means)
    
    if perform_bias_adjustment:
        # Ensure lengths match *after* potential skips due to NaNs
        if not propensity_diffs or len(raw_att_components) != len(propensity_diffs):
            logger.warning("Bias adjustment skipped due to inconsistent data lengths after NaN checks.")
            return np.mean(raw_att_components)

        try:
            X_bias_adj = sm.add_constant(np.array(propensity_diffs))
            y_bias_adj = raw_att_components
            # Add check for NaNs/Infs in inputs to OLS
            if np.isnan(X_bias_adj).any() or np.isnan(y_bias_adj).any() or \
               np.isinf(X_bias_adj).any() or np.isinf(y_bias_adj).any():
                logger.warning("NaN/Inf values detected in OLS inputs for bias adjustment. Falling back.")
                return np.mean(raw_att_components)
                
            bias_model = sm.OLS(y_bias_adj, X_bias_adj).fit()
            bias_adjusted_att = bias_model.params[0]
            return bias_adjusted_att
        except Exception as e:
            logger.warning(f"OLS for bias adjustment failed: {e}. Falling back to raw ATT.")
            return np.mean(raw_att_components)
    else:
        return np.mean(raw_att_components)

def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, 
                      covariates: List[str], **kwargs) -> Dict[str, Any]:
    '''Estimate ATT using Propensity Score Matching. 
    Tries DoWhy's PSM first, falls back to custom implementation if DoWhy fails.
    Uses bootstrap SE based on the custom implementation regardless.
    '''
    query = kwargs.get('query')
    n_bootstraps = kwargs.get('n_bootstraps', 100) 
    
    # --- Parameter Setup (as before) ---
    llm_params = get_llm_parameters(df, query, "PS.Matching")
    llm_suggested_params = llm_params.get("parameters", {})
    
    caliper = kwargs.get('caliper', llm_suggested_params.get('caliper'))
    temp_propensity_scores_for_caliper = None
    try:
        temp_propensity_scores_for_caliper = estimate_propensity_scores(
            df, treatment, covariates, 
            model_type=llm_suggested_params.get('propensity_model_type', 'logistic'), 
            **kwargs
        )
        if caliper is None and temp_propensity_scores_for_caliper is not None:
            logit_ps = _calculate_logit(temp_propensity_scores_for_caliper)
            if not np.isnan(logit_ps).all(): # Check if logit calculation was successful
                 caliper = 0.2 * np.nanstd(logit_ps) # Use nanstd for robustness
            else:
                 logger.warning("Logit of propensity scores resulted in NaNs, cannot calculate heuristic caliper.")
                 caliper = None
        elif caliper is None:
             logger.warning("Could not estimate propensity scores for caliper heuristic.")
             caliper = None

    except Exception as e:
        logger.warning(f"Failed to estimate initial propensity scores for caliper heuristic: {e}. Caliper set to None.")
        caliper = None # Proceed without caliper if heuristic fails
        
    n_neighbors = kwargs.get('n_neighbors', llm_suggested_params.get('n_neighbors', 1))
    propensity_model_type = kwargs.get('propensity_model_type', 
                                   llm_suggested_params.get('propensity_model_type', 
                                                          select_propensity_model(df, treatment, covariates, query)))

    # --- Attempt DoWhy PSM for Point Estimate ---
    att_estimate = np.nan
    method_used_for_att = "Fallback Custom PSM"
    dowhy_model = None
    identified_estimand = None
    
    try:
        logger.info("Attempting estimation using DoWhy Propensity Score Matching...")
        dowhy_model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=covariates,
             estimand_type='nonparametric-ate' # Provide list of names directly
        )
        # Identify estimand (optional step, but good practice)
        identified_estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
        logger.info(f"DoWhy identified estimand: {identified_estimand}")
        
        # Estimate effect using DoWhy's PSM
        estimate = dowhy_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching",
            target_units="att",
            method_params={}
        )
        att_estimate = estimate.value
        method_used_for_att = "DoWhy PSM"
        logger.info(f"DoWhy PSM successful. ATT Estimate: {att_estimate}")
        
    except Exception as e:
        logger.warning(f"DoWhy PSM failed: {e}. Falling back to custom PSM implementation.")
        # Fallback is triggered implicitly if att_estimate remains NaN

    # --- Fallback or if DoWhy failed ---
    if np.isnan(att_estimate):
        logger.info("Calculating ATT estimate using fallback custom PSM...")
        att_estimate = _perform_matching_and_get_att(
            df, treatment, outcome, covariates,
            propensity_model_type, n_neighbors, caliper,
            perform_bias_adjustment=True, **kwargs # Bias adjust the fallback
        )
        method_used_for_att = "Fallback Custom PSM" # Confirm it's fallback
        if np.isnan(att_estimate):
             raise ValueError("Fallback custom PSM estimation also failed. Cannot proceed.")
        logger.info(f"Fallback Custom PSM successful. ATT Estimate: {att_estimate}")

    # --- Bootstrap SE (using custom helper for consistency) ---
    logger.info(f"Calculating Bootstrap SE using custom helper ({n_bootstraps} iterations)...")
    bootstrap_atts = []
    for i in range(n_bootstraps):
        try:
            # Ensure bootstrap samples are drawn correctly
            df_boot = df.sample(n=len(df), replace=True, random_state=np.random.randint(1000000) + i)
            # Bias adjustment in bootstrap can be slow, optionally disable it
            boot_att = _perform_matching_and_get_att(
                df_boot, treatment, outcome, covariates,
                propensity_model_type, n_neighbors, caliper,
                perform_bias_adjustment=False, **kwargs # Set bias adjustment to False for speed in bootstrap
            )
            if not np.isnan(boot_att):
                bootstrap_atts.append(boot_att)
        except Exception as boot_e:
            logger.warning(f"Bootstrap iteration {i+1} failed: {boot_e}")
            continue # Skip failed bootstrap iteration
    
    att_se = np.nanstd(bootstrap_atts) if bootstrap_atts else np.nan # Use nanstd
    actual_bootstrap_iterations = len(bootstrap_atts)
    logger.info(f"Bootstrap SE calculated: {att_se} from {actual_bootstrap_iterations} successful iterations.")

    # --- Diagnostics (using custom matching logic for consistency) ---
    logger.info("Performing diagnostic checks using custom matching logic...")
    diagnostics = {"error": "Diagnostics failed to run."}
    propensity_scores_orig = temp_propensity_scores_for_caliper # Reuse if available and not None
    
    if propensity_scores_orig is None:
        try:
             propensity_scores_orig = estimate_propensity_scores(
                 df, treatment, covariates, model_type=propensity_model_type, **kwargs
             )
        except Exception as e:
             logger.error(f"Failed to estimate propensity scores for diagnostics: {e}")
             propensity_scores_orig = None

    if propensity_scores_orig is not None and not np.isnan(propensity_scores_orig).all():
        df_ps_orig = df.copy()
        df_ps_orig['propensity_score'] = propensity_scores_orig
        treated_orig = df_ps_orig[df_ps_orig[treatment] == 1]
        control_orig = df_ps_orig[df_ps_orig[treatment] == 0]
        unmatched_treated_count = 0

        # Drop rows with NaN propensity scores before diagnostics
        treated_orig = treated_orig.dropna(subset=['propensity_score'])
        control_orig = control_orig.dropna(subset=['propensity_score'])

        if not treated_orig.empty and not control_orig.empty:
            try:
                nn_diag = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper if caliper is not None else np.inf, metric='minkowski', p=2)
                nn_diag.fit(control_orig[['propensity_score']].values)
                distances_diag, indices_diag = nn_diag.kneighbors(treated_orig[['propensity_score']].values)
                
                matched_treated_indices_diag = []
                matched_control_indices_diag = []

                for i in range(len(treated_orig)):
                    valid_neighbors_mask_diag = distances_diag[i] <= (caliper if caliper is not None else np.inf)
                    valid_neighbors_idx_diag = indices_diag[i][valid_neighbors_mask_diag]
                    if len(valid_neighbors_idx_diag) > 0:
                        # Get original DataFrame indices from control_orig based on iloc indices
                        selected_control_original_indices = control_orig.index[valid_neighbors_idx_diag]
                        matched_treated_indices_diag.extend([treated_orig.index[i]] * len(selected_control_original_indices))
                        matched_control_indices_diag.extend(selected_control_original_indices)
                    else:
                        unmatched_treated_count += 1
                
                if matched_control_indices_diag:
                    # Use unique indices for creating the diagnostic dataframe
                    unique_matched_control_indices = list(set(matched_control_indices_diag))
                    unique_matched_treated_indices = list(set(matched_treated_indices_diag))
                    
                    matched_control_df_diag = df.loc[unique_matched_control_indices]
                    matched_treated_df_for_diag = df.loc[unique_matched_treated_indices] 
                    matched_df_diag = pd.concat([matched_treated_df_for_diag, matched_control_df_diag]).drop_duplicates()
                    
                    # Retrieve propensity scores for the specific units in matched_df_diag
                    ps_matched_for_diag = propensity_scores_orig.loc[matched_df_diag.index]

                    diagnostics = assess_balance(df, matched_df_diag, treatment, covariates, 
                                           method="PSM", 
                                           propensity_scores_original=propensity_scores_orig,
                                           propensity_scores_matched=ps_matched_for_diag)
                else: 
                    diagnostics = {"message": "No units could be matched for diagnostic assessment."}
                    # If no controls were matched, all treated were unmatched
                    unmatched_treated_count = len(treated_orig) if not treated_orig.empty else 0 
            except Exception as diag_e:
                 logger.error(f"Error during diagnostic matching/balance assessment: {diag_e}")
                 diagnostics = {"error": f"Diagnostics failed: {diag_e}"}
        else:
            diagnostics = {"message": "Treatment or control group empty after dropping NaN PS, diagnostics skipped."}
            unmatched_treated_count = len(treated_orig) if not treated_orig.empty else 0

        # Ensure unmatched count calculation is safe
        if 'unmatched_treated_count' not in locals():
            unmatched_treated_count = 0 # Initialize if loop didn't run
        diagnostics["unmatched_treated_count"] = unmatched_treated_count
        diagnostics["percent_treated_matched"] = (len(treated_orig) - unmatched_treated_count) / len(treated_orig) * 100 if len(treated_orig) > 0 else 0
    else:
        diagnostics = {"error": "Propensity scores could not be estimated for diagnostics."}

    # Add final details to diagnostics
    diagnostics["att_estimation_method"] = method_used_for_att
    diagnostics["propensity_score_model"] = propensity_model_type
    diagnostics["bootstrap_iterations_for_se"] = actual_bootstrap_iterations
    diagnostics["final_caliper_used"] = caliper

    # --- Format and return results --- 
    logger.info(f"Formatting results. ATT Estimate: {att_estimate}, SE: {att_se}, Method: {method_used_for_att}")
    return format_ps_results(att_estimate, att_se, diagnostics,
                           method_details=f"PSM ({method_used_for_att})",
                           parameters={"caliper": caliper, 
                                         "n_neighbors": n_neighbors, # n_neighbors used in fallback/bootstrap/diag
                                         "propensity_model": propensity_model_type,
                                         "n_bootstraps_config": n_bootstraps}) 
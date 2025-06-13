"""
Core estimation logic for the Generalized Propensity Score (GPS) method.
"""
from typing import Dict, List, Any
import pandas as pd
import logging
import numpy as np
import statsmodels.api as sm

from .diagnostics import assess_gps_balance # Import for balance check

logger = logging.getLogger(__name__)

def estimate_effect_gps(
    df: pd.DataFrame, 
    treatment: str,
    outcome: str,
    covariates: List[str],
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Estimates the causal effect using the Generalized Propensity Score method
    for continuous treatments.

    This function will be called by the method_executor_tool.

    Args:
        df: The input DataFrame.
        treatment: The name of the continuous treatment variable column.
        outcome: The name of the outcome variable column.
        covariates: A list of covariate column names.
        **kwargs: Additional arguments for controlling the estimation, including:
            - gps_model_spec (dict): Specification for the GPS model (T ~ X).
            - outcome_model_spec (dict): Specification for the outcome model (Y ~ T, GPS).
            - t_values_range (list or dict): Specification for treatment levels for ADRF.
            - n_bootstraps (int): Number of bootstrap replications for SEs.

    Returns:
        A dictionary containing the estimation results, including:
            - "effect_estimate": Typically the ADRF or a specific contrast.
            - "standard_error": Standard error for the primary effect estimate.
            - "confidence_interval": Confidence interval for the primary estimate.
            - "adrf_curve": Data representing the Average Dose-Response Function.
            - "specific_contrasts": Any calculated specific contrasts.
            - "diagnostics": Results from diagnostic checks (e.g., balance).
            - "method_details": Description of the method and models used.
            - "parameters_used": Dictionary of parameters used.
    """
    logger.info(f"Starting GPS estimation for treatment '{treatment}', outcome '{outcome}'.")

    # --- Parameter Extraction and Defaults ---
    gps_model_spec = kwargs.get('gps_model_spec', {"type": "linear"})
    outcome_model_spec = kwargs.get('outcome_model_spec', {"type": "polynomial", "degree": 2, "interaction": True})
    
    # Get t_values for ADRF from llm_assist or kwargs, default to 10 points over observed range
    # For simplicity, we'll use a simple range here. In a full impl, this might call llm_assist.
    t_values_for_adrf = kwargs.get('t_values_for_adrf')
    if t_values_for_adrf is None:
        min_t_obs = df[treatment].min()
        max_t_obs = df[treatment].max()
        if pd.isna(min_t_obs) or pd.isna(max_t_obs) or min_t_obs == max_t_obs:
            logger.warning(f"Cannot determine a valid range for treatment '{treatment}' for ADRF. Using limited points.")
            t_values_for_adrf = sorted(list(df[treatment].dropna().unique()))[:10] # Fallback
        else:
            t_values_for_adrf = np.linspace(min_t_obs, max_t_obs, 10).tolist()
    
    n_bootstraps = kwargs.get('n_bootstraps', 0) # Default to 0, meaning no bootstrap for now

    logger.info(f"Using GPS model spec: {gps_model_spec}")
    logger.info(f"Using outcome model spec: {outcome_model_spec}")
    logger.info(f"Evaluating ADRF at t-values: {t_values_for_adrf}")

    try:
        # 2. Estimate GPS Values
        df_with_gps, gps_estimation_diagnostics = _estimate_gps_values(
            df.copy(), treatment, covariates, gps_model_spec
        )
        if 'gps_score' not in df_with_gps.columns or df_with_gps['gps_score'].isnull().all():
            logger.error("GPS estimation failed or resulted in all NaNs.")
            return {
                "error": "GPS estimation failed.",
                "diagnostics": gps_estimation_diagnostics,
                "method_details": "GPS (Failed)",
                "parameters_used": kwargs
            }
        
        # Drop rows where GPS or outcome or necessary modeling variables are NaN before proceeding
        modeling_cols = [outcome, treatment, 'gps_score'] + covariates
        df_with_gps.dropna(subset=modeling_cols, inplace=True)
        if df_with_gps.empty:
            logger.error("DataFrame is empty after GPS estimation and NaN removal.")
            return {"error": "No data available after GPS estimation and NaN removal.", "method_details": "GPS (Failed)", "parameters_used": kwargs}


        # 3. Assess GPS Balance (call diagnostics.assess_gps_balance)
        balance_diagnostics = assess_gps_balance(
            df_with_gps, treatment, covariates, 'gps_score' # kwargs for assess_gps_balance can be passed if needed
        )

        # 4. Estimate Outcome Model
        fitted_outcome_model = _estimate_outcome_model(
            df_with_gps, outcome, treatment, 'gps_score', outcome_model_spec
        )

        # 5. Generate Dose-Response Function
        adrf_results = _generate_dose_response_function(
            df_with_gps, fitted_outcome_model, treatment, 'gps_score', outcome_model_spec, t_values_for_adrf
        )
        adrf_curve_data = {"t_levels": t_values_for_adrf, "expected_outcomes": adrf_results}

        # 6. Calculate specific contrasts if requested (Placeholder)
        specific_contrasts = {"info": "Specific contrasts not implemented in this version."}

        # 7. Perform bootstrapping for SEs if requested (Placeholder for now)
        standard_error_info = {"info": "Bootstrap SEs not implemented in this version."}
        confidence_interval_info = {"info": "Bootstrap CIs not implemented in this version."}
        if n_bootstraps > 0:
            logger.info(f"Bootstrapping with {n_bootstraps} replications (placeholder).")
            # Actual bootstrapping logic would go here.
            # For now, we'll just note that it's not implemented.

        logger.info("GPS estimation steps completed.")

        # Consolidate diagnostics
        all_diagnostics = {
            "gps_estimation_diagnostics": gps_estimation_diagnostics,
            "balance_check": balance_diagnostics, # Now using the actual balance check results
            "outcome_model_summary": str(fitted_outcome_model.summary()) if fitted_outcome_model else "Outcome model not fitted.",
            "warnings": [], # Populate with any warnings during the process
            "summary": "GPS estimation complete."
        }

        return {
            "effect_estimate": adrf_curve_data, # The ADRF is the primary "effect"
            "standard_error_info": standard_error_info, # Placeholder
            "confidence_interval_info": confidence_interval_info, # Placeholder
            "adrf_curve": adrf_curve_data,
            "specific_contrasts": specific_contrasts, # Placeholder
            "diagnostics": all_diagnostics,
            "method_details": f"Generalized Propensity Score (GPS) with {gps_model_spec.get('type', 'N/A')} GPS model and {outcome_model_spec.get('type', 'N/A')} outcome model.",
            "parameters_used": {
                "treatment_var": treatment,
                "outcome_var": outcome,
                "covariate_vars": covariates,
                "gps_model_spec": gps_model_spec,
                "outcome_model_spec": outcome_model_spec,
                "t_values_for_adrf": t_values_for_adrf,
                "n_bootstraps": n_bootstraps,
                **kwargs
            }
        }
    except Exception as e:
        logger.error(f"Error during GPS estimation pipeline: {e}", exc_info=True)
        return {
            "error": f"Pipeline failed: {str(e)}",
            "method_details": "GPS (Failed)",
            "diagnostics": {"error": f"Pipeline failed during GPS estimation: {str(e)}"}, # Add diagnostics here too
            "parameters_used": kwargs
        }


# Placeholder for internal helper functions
def _estimate_gps_values(
    df: pd.DataFrame, 
    treatment: str,
    covariates: List[str],
    gps_model_spec: Dict
) -> tuple[pd.DataFrame, Dict]:
    """
    Estimates Generalized Propensity Scores.
    Assumes T | X ~ N(X*beta, sigma^2), so GPS is the conditional density.
    """
    logger.info(f"Estimating GPS for treatment '{treatment}' using covariates: {covariates}")
    diagnostics = {}

    if not covariates:
        logger.error("No covariates provided for GPS estimation.")
        diagnostics["error"] = "No covariates provided."
        df['gps_score'] = np.nan # Ensure gps_score column is added
        return df, diagnostics

    X_df = df[covariates]
    T_series = df[treatment]

    # Handle potential NaN values in covariates or treatment before modeling
    valid_indices = X_df.dropna().index.intersection(T_series.dropna().index)
    if len(valid_indices) < len(df):
        logger.warning(f"Dropped {len(df) - len(valid_indices)} rows due to NaNs in treatment/covariates before GPS estimation.")
        diagnostics["pre_estimation_nan_rows_dropped"] = len(df) - len(valid_indices)
    
    X = X_df.loc[valid_indices]
    T = T_series.loc[valid_indices]

    if X.empty or T.empty:
        logger.error("Covariate or treatment data is empty after NaN handling.")
        diagnostics["error"] = "Covariate or treatment data is empty after NaN handling."
        return df, diagnostics
    
    X_sm = sm.add_constant(X, has_constant='add')
    
    try:
        if gps_model_spec.get("type") == 'linear':
            model = sm.OLS(T, X_sm).fit()
            t_hat = model.predict(X_sm)
            residuals = T - t_hat
            # MSE: sum of squared residuals / (n - k) where k is number of regressors (including const)
            if len(T) <= X_sm.shape[1]:
                 logger.error("Not enough degrees of freedom to estimate sigma_sq_hat.")
                 diagnostics["error"] = "Not enough degrees of freedom for GPS variance."
                 df['gps_score'] = np.nan
                 return df, diagnostics

            sigma_sq_hat = np.sum(residuals**2) / (len(T) - X_sm.shape[1]) 
            
            if sigma_sq_hat <= 1e-9: # Check for effectively zero or very small variance
                logger.warning(f"Estimated residual variance (sigma_sq_hat) is very close to zero ({sigma_sq_hat}). GPS will be set to NaN.")
                diagnostics["warning_sigma_sq_hat_near_zero"] = sigma_sq_hat
                df['gps_score'] = np.nan # Set GPS to NaN as density is ill-defined
                if sigma_sq_hat == 0: # if it is exactly zero, add specific error
                     diagnostics["error_sigma_sq_hat_is_zero"] = "Residual variance is exactly zero."
                return df, diagnostics


            # Calculate GPS: (1 / sqrt(2*pi*sigma_hat^2)) * exp(-(T_i - T_hat_i)^2 / (2*sigma_hat^2))
            # Ensure calculation is done on the original T values (T_series.loc[valid_indices])
            # and corresponding t_hat for those valid_indices
            gps_values_calculated = (1 / np.sqrt(2 * np.pi * sigma_sq_hat)) * np.exp(-((T - t_hat)**2) / (2 * sigma_sq_hat))
            
            # Assign back to the original DataFrame using .loc to ensure alignment
            df['gps_score'] = np.nan # Initialize column
            df.loc[valid_indices, 'gps_score'] = gps_values_calculated
            
            diagnostics["gps_model_type"] = "linear_ols"
            diagnostics["gps_model_rsquared"] = model.rsquared
            diagnostics["gps_residual_variance_mse"] = sigma_sq_hat
            diagnostics["num_observations_for_gps_model"] = len(T)

        else:
            logger.error(f"GPS model type '{gps_model_spec.get('type')}' not implemented.")
            diagnostics["error"] = f"GPS model type '{gps_model_spec.get('type')}' not implemented."
            df['gps_score'] = np.nan
            
    except Exception as e:
        logger.error(f"Error during GPS model estimation: {e}", exc_info=True)
        diagnostics["error"] = f"Exception during GPS estimation: {str(e)}"
        df['gps_score'] = np.nan

    # Ensure the original df is not modified if no valid indices for GPS estimation
    if 'gps_score' not in df.columns:
        df['gps_score'] = np.nan

    return df, diagnostics

def _estimate_outcome_model(
    df_with_gps: pd.DataFrame, 
    outcome: str,
    treatment: str,
    gps_col_name: str, 
    outcome_model_spec: Dict
) -> Any: # Returns a fitted statsmodels model
    """
    Estimates the outcome model Y ~ f(T, GPS).
    """
    logger.info(f"Estimating outcome model for '{outcome}' using T='{treatment}', GPS='{gps_col_name}'")
    
    Y = df_with_gps[outcome]
    T_val = pd.Series(df_with_gps[treatment].values, index=df_with_gps.index)
    GPS_val = pd.Series(df_with_gps[gps_col_name].values, index=df_with_gps.index)
    
    X_outcome_dict = {'intercept': np.ones(len(df_with_gps))}

    model_type = outcome_model_spec.get("type", "polynomial")
    degree = outcome_model_spec.get("degree", 2)
    interaction = outcome_model_spec.get("interaction", True)

    if model_type == "polynomial":
        X_outcome_dict['T'] = T_val
        X_outcome_dict['GPS'] = GPS_val
        if degree >= 2:
            X_outcome_dict['T_sq'] = T_val**2
            X_outcome_dict['GPS_sq'] = GPS_val**2
        if degree >=3: # Example for higher order, can be made more general
            X_outcome_dict['T_cub'] = T_val**3
            X_outcome_dict['GPS_cub'] = GPS_val**3
        if interaction:
            X_outcome_dict['T_x_GPS'] = T_val * GPS_val
            if degree >=2: # Interaction with squared terms if degree allows
                 X_outcome_dict['T_sq_x_GPS'] = (T_val**2) * GPS_val
                 X_outcome_dict['T_x_GPS_sq'] = T_val * (GPS_val**2)

    # Add more model types as needed (e.g., splines)
    else:
        logger.warning(f"Outcome model type '{model_type}' not fully recognized. Defaulting to T + GPS.")
        X_outcome_dict['T'] = T_val
        X_outcome_dict['GPS'] = GPS_val
        # Fallback to linear if spec is unknown or simple

    X_outcome_df = pd.DataFrame(X_outcome_dict, index=df_with_gps.index)
    
    # Drop rows with NaNs that might have been introduced by transformations if T or GPS were NaN
    # (though earlier dropna should handle most of this for input T/GPS)
    valid_outcome_model_indices = Y.dropna().index.intersection(X_outcome_df.dropna().index)
    if len(valid_outcome_model_indices) < len(df_with_gps):
        logger.warning(f"Dropped {len(df_with_gps) - len(valid_outcome_model_indices)} rows due to NaNs before outcome model fitting.")

    Y_fit = Y.loc[valid_outcome_model_indices]
    X_outcome_df_fit = X_outcome_df.loc[valid_outcome_model_indices]

    if Y_fit.empty or X_outcome_df_fit.empty:
        logger.error("Not enough data to fit outcome model after NaN handling.")
        raise ValueError("Empty data for outcome model fitting.")

    try:
        model = sm.OLS(Y_fit, X_outcome_df_fit).fit()
        logger.info("Outcome model estimated successfully.")
        return model
    except Exception as e:
        logger.error(f"Error during outcome model estimation: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the main try-except block

def _generate_dose_response_function(
    df_with_gps: pd.DataFrame, 
    fitted_outcome_model: Any, 
    treatment: str,
    gps_col_name: str, 
    outcome_model_spec: Dict, # To know how to construct X_pred features
    t_values_to_evaluate: List[float]
) -> List[float]:
    """
    Calculates the Average Dose-Response Function (ADRF).
    E[Y(t)] = integral over E[Y | T=t, GPS=g] * f(g) dg
            ~= (1/N) * sum_i E[Y | T=t, GPS=g_i] (using observed GPS values)
    """
    logger.info(f"Calculating ADRF for treatment levels: {t_values_to_evaluate}")
    adrf_estimates = []
    
    if not t_values_to_evaluate: # Handle empty list case
        logger.warning("t_values_to_evaluate is empty. ADRF calculation will be skipped.")
        return []

    model_exog_names = fitted_outcome_model.model.exog_names
    
    # Original GPS values from the dataframe
    original_gps_values = pd.Series(df_with_gps[gps_col_name].values, index=df_with_gps.index)

    for t_level in t_values_to_evaluate:
        # Create a new DataFrame for prediction at this t_level
        # Each row corresponds to an original observation's GPS, but with T set to t_level
        X_pred_dict = {'intercept': np.ones(len(df_with_gps))}
        
        # Reconstruct features based on outcome_model_spec and model_exog_names
        # This mirrors the construction in _estimate_outcome_model
        degree = outcome_model_spec.get("degree", 2)
        interaction = outcome_model_spec.get("interaction", True)

        if 'T' in model_exog_names: X_pred_dict['T'] = t_level
        if 'GPS' in model_exog_names: X_pred_dict['GPS'] = original_gps_values
        
        if 'T_sq' in model_exog_names: X_pred_dict['T_sq'] = t_level**2
        if 'GPS_sq' in model_exog_names: X_pred_dict['GPS_sq'] = original_gps_values**2
            
        if 'T_cub' in model_exog_names: X_pred_dict['T_cub'] = t_level**3 # Example
        if 'GPS_cub' in model_exog_names: X_pred_dict['GPS_cub'] = original_gps_values**3 # Example
            
        if 'T_x_GPS' in model_exog_names and interaction:
            X_pred_dict['T_x_GPS'] = t_level * original_gps_values
        if 'T_sq_x_GPS' in model_exog_names and interaction and degree >=2:
            X_pred_dict['T_sq_x_GPS'] = (t_level**2) * original_gps_values
        if 'T_x_GPS_sq' in model_exog_names and interaction and degree >=2:
            X_pred_dict['T_x_GPS_sq'] = t_level * (original_gps_values**2)

        X_pred_df = pd.DataFrame(X_pred_dict, index=df_with_gps.index)
        
        # Ensure all required columns are present and in the correct order
        # Drop any rows that might have NaNs if original_gps_values had NaNs (though they should be filtered before this)
        X_pred_df_fit = X_pred_df[model_exog_names].dropna()
        
        if X_pred_df_fit.empty:
            logger.warning(f"Prediction data for t_level={t_level} is empty after NaN drop. Assigning NaN to ADRF point.")
            adrf_estimates.append(np.nan)
            continue
            
        predicted_outcomes_at_t = fitted_outcome_model.predict(X_pred_df_fit)
        adrf_estimates.append(np.mean(predicted_outcomes_at_t))
        
    return adrf_estimates 
"""
Core estimation logic for the Generalized Propensity Score (GPS) method.
"""
from typing import Dict, List, Any, Optional, Sequence, Tuple
import pandas as pd
import logging
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

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
        adrf_results, support_diagnostics = _generate_dose_response_function(
            df_with_gps, fitted_outcome_model, treatment, 'gps_score', outcome_model_spec,
            t_values_for_adrf, gps_model_diagnostics=gps_estimation_diagnostics,
            return_support=True,
        )
        adrf_curve_data = {"t_levels": t_values_for_adrf, "expected_outcomes": adrf_results}

        requested_contrasts = kwargs.get("contrasts") or []
        curve_lookup = {float(t): float(mu) for t, mu in zip(t_values_for_adrf, adrf_results)}
        specific_contrasts = []
        for pair in requested_contrasts:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each GPS contrast must be a pair (t1, t0).")
            t1, t0 = float(pair[0]), float(pair[1])
            if t1 not in curve_lookup or t0 not in curve_lookup:
                raise ValueError("GPS contrasts must use doses present in t_values_for_adrf.")
            specific_contrasts.append({
                "t1": t1, "t0": t0,
                "difference": curve_lookup[t1] - curve_lookup[t0],
            })

        standard_error_info = None
        confidence_interval_info = None
        if n_bootstraps > 0:
            bootstrap_curves = _bootstrap_adrf(
                df=df, treatment=treatment, outcome=outcome, covariates=covariates,
                gps_model_spec=gps_model_spec, outcome_model_spec=outcome_model_spec,
                t_values=t_values_for_adrf, n_bootstraps=int(n_bootstraps),
                random_state=int(kwargs.get("random_state", 0)),
            )
            if len(bootstrap_curves) < max(20, int(n_bootstraps) // 2):
                raise ValueError(
                    f"Too few successful GPS bootstrap replications "
                    f"({len(bootstrap_curves)}/{n_bootstraps})."
                )
            boot = np.asarray(bootstrap_curves, dtype=float)
            standard_error_info = np.std(boot, axis=0, ddof=1).tolist()
            confidence_interval_info = {
                "lower": np.percentile(boot, 2.5, axis=0).tolist(),
                "upper": np.percentile(boot, 97.5, axis=0).tolist(),
                "successful_replications": int(len(boot)),
            }

        logger.info("GPS estimation steps completed.")

        # Consolidate diagnostics
        all_diagnostics = {
            "gps_estimation_diagnostics": gps_estimation_diagnostics,
            "balance_check": balance_diagnostics, # Now using the actual balance check results
            "dose_support": support_diagnostics,
            "outcome_model_summary": str(fitted_outcome_model.summary()) if fitted_outcome_model else "Outcome model not fitted.",
            "warnings": [
                "Weak unconfoundedness and correct model specification are substantive "
                "assumptions; diagnostics are advisory and do not prove them."
            ] + support_diagnostics.get("warnings", []),
            "summary": "GPS estimation complete."
        }

        return {
            "effect_estimate": adrf_curve_data, # The ADRF is the primary "effect"
            "standard_error": standard_error_info,
            "confidence_interval": confidence_interval_info,
            "standard_error_info": standard_error_info,
            "confidence_interval_info": confidence_interval_info,
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
            diagnostics["gps_model_columns"] = list(X_sm.columns)
            diagnostics["gps_model_coefficients"] = {
                str(name): float(value) for name, value in model.params.items()
            }

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
    t_values_to_evaluate: List[float],
    gps_model_diagnostics: Optional[Dict[str, Any]] = None,
    return_support: bool = False,
) -> Any:
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
    
    support = {"by_dose": [], "warnings": []}
    observed_gps = pd.Series(df_with_gps[gps_col_name].values, index=df_with_gps.index)
    lower_density = float(observed_gps.quantile(0.01))

    for t_level in t_values_to_evaluate:
        # Hirano--Imbens requires r(t, X_i), not the GPS evaluated at each
        # unit's observed dose r(T_i, X_i).
        if gps_model_diagnostics and gps_model_diagnostics.get("gps_model_coefficients"):
            counterfactual_gps = _gps_at_dose(
                df_with_gps, float(t_level), gps_model_diagnostics
            )
        else:
            # Backward-compatible helper use. The public estimator always passes
            # model diagnostics; direct legacy calls retain their old behavior.
            logger.warning("GPS model diagnostics absent; using observed GPS values.")
            counterfactual_gps = observed_gps
        supported = np.isfinite(counterfactual_gps) & (counterfactual_gps >= lower_density)
        supported_fraction = float(np.mean(supported))
        support["by_dose"].append({
            "dose": float(t_level), "supported_fraction": supported_fraction,
            "density_threshold": lower_density,
        })
        if supported_fraction < 0.8:
            support["warnings"].append(
                f"Dose {float(t_level):g} has weak conditional support "
                f"({supported_fraction:.1%} of units above the observed 1% GPS-density threshold)."
            )

        X_pred_dict = {'intercept': np.ones(len(df_with_gps))}
        
        # Reconstruct features based on outcome_model_spec and model_exog_names
        # This mirrors the construction in _estimate_outcome_model
        degree = outcome_model_spec.get("degree", 2)
        interaction = outcome_model_spec.get("interaction", True)

        if 'T' in model_exog_names: X_pred_dict['T'] = t_level
        if 'GPS' in model_exog_names: X_pred_dict['GPS'] = counterfactual_gps
        
        if 'T_sq' in model_exog_names: X_pred_dict['T_sq'] = t_level**2
        if 'GPS_sq' in model_exog_names: X_pred_dict['GPS_sq'] = counterfactual_gps**2
            
        if 'T_cub' in model_exog_names: X_pred_dict['T_cub'] = t_level**3 # Example
        if 'GPS_cub' in model_exog_names: X_pred_dict['GPS_cub'] = counterfactual_gps**3 # Example
            
        if 'T_x_GPS' in model_exog_names and interaction:
            X_pred_dict['T_x_GPS'] = t_level * counterfactual_gps
        if 'T_sq_x_GPS' in model_exog_names and interaction and degree >=2:
            X_pred_dict['T_sq_x_GPS'] = (t_level**2) * counterfactual_gps
        if 'T_x_GPS_sq' in model_exog_names and interaction and degree >=2:
            X_pred_dict['T_x_GPS_sq'] = t_level * (counterfactual_gps**2)

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
        
    return (adrf_estimates, support) if return_support else adrf_estimates


def _gps_at_dose(
    df: pd.DataFrame, dose: float, gps_model_diagnostics: Dict[str, Any]
) -> pd.Series:
    """Evaluate the fitted conditional treatment density r(dose, X_i)."""
    coefficients = gps_model_diagnostics.get("gps_model_coefficients") or {}
    sigma_sq = float(gps_model_diagnostics.get("gps_residual_variance_mse", np.nan))
    if not coefficients or not np.isfinite(sigma_sq) or sigma_sq <= 0:
        raise ValueError("Fitted GPS model parameters are unavailable for ADRF prediction.")
    mean = np.full(len(df), float(coefficients.get("const", 0.0)))
    for name, coefficient in coefficients.items():
        if name == "const":
            continue
        if name not in df.columns:
            raise ValueError(f"GPS prediction covariate {name!r} is missing.")
        mean += float(coefficient) * pd.to_numeric(df[name], errors="coerce").to_numpy()
    values = norm.pdf(float(dose), loc=mean, scale=np.sqrt(sigma_sq))
    return pd.Series(values, index=df.index, dtype=float)


def _bootstrap_adrf(
    df: pd.DataFrame, treatment: str, outcome: str, covariates: Sequence[str],
    gps_model_spec: Dict[str, Any], outcome_model_spec: Dict[str, Any],
    t_values: Sequence[float], n_bootstraps: int, random_state: int,
) -> List[List[float]]:
    """Row-bootstrap the complete GPS and outcome-model estimation procedure."""
    rng = np.random.default_rng(random_state)
    curves: List[List[float]] = []
    for _ in range(n_bootstraps):
        sample = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        fitted, diagnostics = _estimate_gps_values(
            sample.copy(), treatment, list(covariates), gps_model_spec
        )
        fitted = fitted.dropna(subset=[treatment, outcome, "gps_score", *covariates])
        if fitted.empty or diagnostics.get("error"):
            continue
        try:
            outcome_fit = _estimate_outcome_model(
                fitted, outcome, treatment, "gps_score", outcome_model_spec
            )
            curve = _generate_dose_response_function(
                fitted, outcome_fit, treatment, "gps_score", outcome_model_spec,
                list(t_values), gps_model_diagnostics=diagnostics,
            )
            if np.isfinite(curve).all():
                curves.append([float(value) for value in curve])
        except (ValueError, np.linalg.LinAlgError):
            continue
    return curves

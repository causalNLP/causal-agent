"""
Diagnostic checks for the Generalized Propensity Score (GPS) method.
"""
from typing import Dict, List, Any
import pandas as pd
import logging
import numpy as np
import statsmodels.api as sm

logger = logging.getLogger(__name__)

def assess_gps_balance(
    df_with_gps: pd.DataFrame, 
    treatment_var: str, 
    covariate_vars: List[str], 
    gps_col_name: str, 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Assesses the balance of covariates conditional on the estimated GPS.

    This function is typically called after GPS estimation to validate the 
    assumption that covariates are independent of treatment conditional on GPS.

    Args:
        df_with_gps: DataFrame containing the original data plus the estimated GPS column.
        treatment_var: The name of the continuous treatment variable column.
        covariate_vars: A list of covariate column names to check for balance.
        gps_col_name: The name of the column containing the estimated GPS values.
        **kwargs: Additional arguments (e.g., number of strata for checking balance).

    Returns:
        A dictionary containing balance statistics and summaries. For example:
            {
                "overall_balance_metric": 0.05, 
                "covariate_balance": {
                    "cov1": {"statistic": 0.03, "p_value": 0.5, "balanced": True},
                    "cov2": {"statistic": 0.12, "p_value": 0.02, "balanced": False}
                },
                "summary": "Balance assessment complete."
            }
    """
    logger.info(f"Assessing GPS balance for covariates: {covariate_vars}")

    # Default to 5 strata (quintiles) if not specified
    num_strata = kwargs.get('num_strata', 5)
    if not isinstance(num_strata, int) or num_strata <= 1:
        logger.warning(f"Invalid num_strata ({num_strata}), defaulting to 5.")
        num_strata = 5

    balance_results = {}
    overall_summary = {
        "num_strata_used": num_strata,
        "covariates_tested": len(covariate_vars),
        "warnings": [],
        "all_strata_coefficients": {cov: [] for cov in covariate_vars},
        "all_strata_p_values": {cov: [] for cov in covariate_vars}
    }

    if df_with_gps[gps_col_name].isnull().all():
        logger.error(f"All GPS scores in column '{gps_col_name}' are NaN. Cannot perform balance assessment.")
        overall_summary["error"] = "All GPS scores are NaN."
        return {
            "error": "All GPS scores are NaN.",
            "summary": "Balance assessment failed."
        }
    
    try:
        # Create GPS strata (e.g., quintiles)
        # Ensure unique bin edges for qcut, duplicates='drop' will handle cases with sparse GPS values
        # but might result in fewer than num_strata if GPS distribution is highly skewed or has few unique values.
        try:
            df_with_gps['gps_stratum'] = pd.qcut(df_with_gps[gps_col_name], num_strata, labels=False, duplicates='drop')
            actual_num_strata = df_with_gps['gps_stratum'].nunique()
            if actual_num_strata < num_strata and actual_num_strata > 0:
                logger.warning(f"Requested {num_strata} strata, but due to GPS distribution, only {actual_num_strata} could be formed.")
                overall_summary["warnings"].append(f"Only {actual_num_strata} strata formed out of {num_strata} requested.")
            overall_summary["actual_num_strata_formed"] = actual_num_strata
        except ValueError as ve:
            logger.error(f"Could not create strata using pd.qcut due to: {ve}. This might happen if GPS has too few unique values.")
            logger.info("Attempting to use unique GPS values as strata if count is low.")
            unique_gps_count = df_with_gps[gps_col_name].nunique()
            if unique_gps_count <= num_strata * 2 and unique_gps_count > 1: # Arbitrary threshold to try unique values as strata
                strata_map = {val: i for i, val in enumerate(df_with_gps[gps_col_name].unique())}
                df_with_gps['gps_stratum'] = df_with_gps[gps_col_name].map(strata_map)
                actual_num_strata = df_with_gps['gps_stratum'].nunique()
                overall_summary["actual_num_strata_formed"] = actual_num_strata
                overall_summary["warnings"].append(f"Used {actual_num_strata} unique GPS values as strata due to qcut error.")
            else:
                overall_summary["error"] = f"Failed to create GPS strata: {ve}. GPS may have too few unique values."
                return {
                    "error": overall_summary["error"],
                    "summary": "Balance assessment failed due to strata creation issues."
                }

        if df_with_gps['gps_stratum'].isnull().all():
            logger.error("Stratum assignment resulted in all NaNs.")
            overall_summary["error"] = "Stratum assignment resulted in all NaNs."
            return {"error": overall_summary["error"], "summary": "Balance assessment failed."}


        for cov in covariate_vars:
            balance_results[cov] = {
                "strata_details": [],
                "mean_abs_coefficient": None,
                "num_significant_strata_p005": 0,
                "balanced_heuristic": True # Assume balanced until proven otherwise
            }
            coeffs_for_cov = []
            p_values_for_cov = []

            for stratum_idx in sorted(df_with_gps['gps_stratum'].dropna().unique()):
                stratum_data = df_with_gps[df_with_gps['gps_stratum'] == stratum_idx]
                stratum_detail = {"stratum_index": int(stratum_idx), "n_obs": len(stratum_data)}

                if len(stratum_data) < 10: # Need a minimum number of observations for stable regression
                    stratum_detail["status"] = "Skipped (too few observations)"
                    stratum_detail["coefficient_on_treatment"] = np.nan
                    stratum_detail["p_value_on_treatment"] = np.nan
                    balance_results[cov]["strata_details"].append(stratum_detail)
                    continue
                
                # Ensure covariate and treatment have variance within the stratum
                if stratum_data[cov].nunique() < 2 or stratum_data[treatment_var].nunique() < 2:
                    stratum_detail["status"] = "Skipped (no variance in cov or treatment)"
                    stratum_detail["coefficient_on_treatment"] = np.nan
                    stratum_detail["p_value_on_treatment"] = np.nan
                    balance_results[cov]["strata_details"].append(stratum_detail)
                    continue

                try:
                    X_balance = sm.add_constant(stratum_data[[treatment_var]])
                    y_balance = stratum_data[cov]
                    
                    # Drop NaNs for this specific regression within stratum
                    temp_df = pd.concat([y_balance, X_balance], axis=1).dropna()
                    if len(temp_df) < X_balance.shape[1] +1: # Check for enough data points after NaNs for regression
                         stratum_detail["status"] = "Skipped (too few non-NaN obs for regression)"
                         stratum_detail["coefficient_on_treatment"] = np.nan
                         stratum_detail["p_value_on_treatment"] = np.nan
                         balance_results[cov]["strata_details"].append(stratum_detail)
                         continue

                    y_balance_fit = temp_df[cov]
                    X_balance_fit = temp_df[[col for col in temp_df.columns if col != cov]]

                    balance_model = sm.OLS(y_balance_fit, X_balance_fit).fit()
                    coeff = balance_model.params.get(treatment_var, np.nan)
                    p_value = balance_model.pvalues.get(treatment_var, np.nan)
                    
                    coeffs_for_cov.append(coeff)
                    p_values_for_cov.append(p_value)
                    overall_summary["all_strata_coefficients"][cov].append(coeff)
                    overall_summary["all_strata_p_values"][cov].append(p_value)

                    stratum_detail["status"] = "Analyzed"
                    stratum_detail["coefficient_on_treatment"] = coeff
                    stratum_detail["p_value_on_treatment"] = p_value
                    if not pd.isna(p_value) and p_value < 0.05:
                        balance_results[cov]["num_significant_strata_p005"] += 1
                        balance_results[cov]["balanced_heuristic"] = False # If any stratum is unbalanced

                except Exception as e_bal:
                    logger.debug(f"Balance check regression failed for {cov} in stratum {stratum_idx}: {e_bal}")
                    stratum_detail["status"] = f"Error: {str(e_bal)}"
                    stratum_detail["coefficient_on_treatment"] = np.nan
                    stratum_detail["p_value_on_treatment"] = np.nan
                
                balance_results[cov]["strata_details"].append(stratum_detail)

            if coeffs_for_cov:
                balance_results[cov]["mean_abs_coefficient"] = np.nanmean(np.abs(coeffs_for_cov))
            else:
                 balance_results[cov]["mean_abs_coefficient"] = np.nan # No strata were analyzable
        
        overall_summary["num_covariates_potentially_imbalanced_p005"] = sum(
            1 for cov_data in balance_results.values() if not cov_data["balanced_heuristic"]
        )

    except Exception as e:
        logger.error(f"Error during GPS balance assessment: {e}", exc_info=True)
        overall_summary["error"] = f"Overall assessment error: {str(e)}"
        return {
            "error": str(e),
            "balance_results": balance_results,
            "summary_stats": overall_summary,
            "summary": "Balance assessment failed due to an unexpected error."
        }
    
    logger.info("GPS balance assessment complete.")
    
    return {
        "balance_results_per_covariate": balance_results,
        "summary_stats": overall_summary,
        "summary": "GPS balance assessment finished. Review strata details and mean absolute coefficients."
    } 
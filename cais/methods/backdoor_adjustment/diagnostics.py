"""
Diagnostic checks for Backdoor Adjustment models (typically OLS).
"""

from typing import Dict, Any, List
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_backdoor_diagnostics(results: RegressionResultsWrapper, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs diagnostic checks on a fitted OLS model used for backdoor adjustment.

    Args:
        results: A fitted statsmodels OLS results object.
        X: The design matrix (including constant and all predictors) used.

    Returns:
        Dictionary containing diagnostic metrics.
    """
    diagnostics = {}
    details = {}

    try:
        details['r_squared'] = results.rsquared
        details['adj_r_squared'] = results.rsquared_adj
        details['f_statistic'] = results.fvalue
        details['f_p_value'] = results.f_pvalue
        details['n_observations'] = int(results.nobs)
        details['degrees_of_freedom_resid'] = int(results.df_resid)
        details['durbin_watson'] = durbin_watson(results.resid) if results.nobs > 5 else 'N/A (Too few obs)' # Autocorrelation

        # --- Normality of Residuals (Jarque-Bera) ---
        try:
            if results.nobs >= 2:
                jb_value, jb_p_value, skew, kurtosis = jarque_bera(results.resid)
                details['residuals_normality_jb_stat'] = jb_value
                details['residuals_normality_jb_p_value'] = jb_p_value
                details['residuals_skewness'] = skew
                details['residuals_kurtosis'] = kurtosis
                details['residuals_normality_status'] = "Normal" if jb_p_value > 0.05 else "Non-Normal"
            else:
                 details['residuals_normality_status'] = "N/A (Too few obs)"
        except Exception as e:
            logger.warning(f"Could not run Jarque-Bera test: {e}")
            details['residuals_normality_status'] = "Test Failed"

        # --- Homoscedasticity (Breusch-Pagan) ---
        try:
            if X.shape[0] > X.shape[1]: # Needs more observations than predictors
                lm_stat, lm_p_value, f_stat, f_p_value = het_breuschpagan(results.resid, X)
                details['homoscedasticity_bp_lm_stat'] = lm_stat
                details['homoscedasticity_bp_lm_p_value'] = lm_p_value
                details['homoscedasticity_status'] = "Homoscedastic" if lm_p_value > 0.05 else "Heteroscedastic"
            else:
                details['homoscedasticity_status'] = "N/A (Too few obs or too many predictors)"
        except Exception as e:
            logger.warning(f"Could not run Breusch-Pagan test: {e}")
            details['homoscedasticity_status'] = "Test Failed"

        # --- Multicollinearity (VIF - Placeholder/Basic) ---
        # Full VIF requires calculating for each predictor vs others.
        # Providing a basic status based on condition number as a proxy.
        try:
            cond_no = np.linalg.cond(results.model.exog)
            details['model_condition_number'] = cond_no
            if cond_no > 30:
                details['multicollinearity_status'] = "High (Cond. No. > 30)"
            elif cond_no > 10:
                 details['multicollinearity_status'] = "Moderate (Cond. No. > 10)"
            else:
                 details['multicollinearity_status'] = "Low"
        except Exception as e:
             logger.warning(f"Could not calculate condition number: {e}")
             details['multicollinearity_status'] = "Check Failed"
        # details['VIF'] = "Not Fully Implemented" 

        # --- Linearity (Still requires visual inspection) ---
        details['linearity_check'] = "Requires visual inspection (e.g., residual vs fitted plot)"

        return {"status": "Success", "details": details}

    except Exception as e:
        logger.error(f"Error running Backdoor Adjustment diagnostics: {e}")
        return {"status": "Failed", "error": str(e), "details": details}

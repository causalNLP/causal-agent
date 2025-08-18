"""
Diagnostic checks for Linear Regression models.
"""

from typing import Dict, Any
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from statsmodels.stats.stattools import jarque_bera
from statsmodels.regression.linear_model import RegressionResultsWrapper
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def run_lr_diagnostics(results: RegressionResultsWrapper, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs diagnostic checks on a fitted OLS model.

    Args:
        results: A fitted statsmodels OLS results object.
        X: The design matrix (including constant) used for the regression.
           Needed for heteroskedasticity tests.

    Returns:
        Dictionary containing diagnostic metrics.
    """

    diagnostics = {}
    
    try:
        diagnostics['r_squared'] = results.rsquared
        diagnostics['adj_r_squared'] = results.rsquared_adj
        diagnostics['f_statistic'] = results.fvalue
        diagnostics['f_p_value'] = results.f_pvalue
        diagnostics['n_observations'] = int(results.nobs)
        diagnostics['degrees_of_freedom_resid'] = int(results.df_resid)
        
        # --- Normality of Residuals (Jarque-Bera) ---
        try:
            jb_value, jb_p_value, skew, kurtosis = jarque_bera(results.resid)
            diagnostics['residuals_normality_jb_stat'] = jb_value
            diagnostics['residuals_normality_jb_p_value'] = jb_p_value
            diagnostics['residuals_skewness'] = skew
            diagnostics['residuals_kurtosis'] = kurtosis
            diagnostics['residuals_normality_status'] = "Normal" if jb_p_value > 0.05 else "Non-Normal"
        except Exception as e:
            logger.warning(f"Could not run Jarque-Bera test: {e}")
            diagnostics['residuals_normality_status'] = "Test Failed"

        # --- Homoscedasticity (Breusch-Pagan) ---
        # Requires the design matrix X used in the model fitting
        try:
            lm_stat, lm_p_value, f_stat, f_p_value = het_breuschpagan(results.resid, X)
            diagnostics['homoscedasticity_bp_lm_stat'] = lm_stat
            diagnostics['homoscedasticity_bp_lm_p_value'] = lm_p_value
            diagnostics['homoscedasticity_bp_f_stat'] = f_stat
            diagnostics['homoscedasticity_bp_f_p_value'] = f_p_value
            diagnostics['homoscedasticity_status'] = "Homoscedastic" if lm_p_value > 0.05 else "Heteroscedastic"
        except Exception as e:
            logger.warning(f"Could not run Breusch-Pagan test: {e}")
            diagnostics['homoscedasticity_status'] = "Test Failed"
            
        # --- Linearity (Basic check - often requires visual inspection) ---
        # No standard quantitative test implemented here. Usually assessed via residual plots.
        diagnostics['linearity_check'] = "Requires visual inspection (e.g., residual vs fitted plot)"
        
        # --- Multicollinearity (Placeholder - requires VIF calculation) ---
        # VIF requires iterating through predictors, more involved
        diagnostics['multicollinearity_check'] = "Not Implemented (Requires VIF)"

        return {"status": "Success", "details": diagnostics}
        
    except Exception as e:
        logger.error(f"Error running LR diagnostics: {e}")
        return {"status": "Failed", "error": str(e), "details": diagnostics} # Return partial results if possible


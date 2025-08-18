"""
Diagnostic checks for Regression Discontinuity Design (RDD).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def run_rdd_diagnostics(
    df: pd.DataFrame,
    outcome: str,
    running_variable: str,
    cutoff: float,
    covariates: Optional[List[str]] = None,
    bandwidth: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs diagnostic checks for RDD analysis.

    Currently includes:
    - Covariate Balance Check (t-tests)
    Placeholders for:
    - Density Test (McCrary)
    - Placebo Cutoff Tests
    - Bandwidth Sensitivity

    Args:
        df: Input DataFrame.
        outcome: Name of the outcome variable.
        running_variable: Name of the running variable.
        cutoff: The threshold value.
        covariates: Optional list of covariate names to check for balance.
        bandwidth: Optional bandwidth to restrict the analysis. If None, a default is used.

    Returns:
        Dictionary containing diagnostic results.
    """
    diagnostics = {}
    details = {}

    if bandwidth is None:
        # Use the same default as estimator for consistency
        range_rv = df[running_variable].max() - df[running_variable].min()
        bandwidth = 0.1 * range_rv
        logger.warning(f"No bandwidth provided for diagnostics, using basic default: {bandwidth:.3f}")

    # --- Filter data within bandwidth --- 
    df_bw = df[(df[running_variable] >= cutoff - bandwidth) & (df[running_variable] <= cutoff + bandwidth)].copy()
    if df_bw.empty:
        logger.warning("No data within bandwidth for diagnostics.")
        return {"status": "Skipped", "reason": "No data in bandwidth", "details": details}

    df_below = df_bw[df_bw[running_variable] < cutoff]
    df_above = df_bw[df_bw[running_variable] >= cutoff]

    if df_below.empty or df_above.empty:
        logger.warning("Insufficient data above or below cutoff within bandwidth for diagnostics.")
        return {"status": "Skipped", "reason": "Insufficient data near cutoff", "details": details}

    # --- Covariate Balance Check --- 
    if covariates:
        balance_results = {}
        details['covariate_balance'] = balance_results
        for cov in covariates:
            if cov in df_bw.columns:
                try:
                    # Perform t-test for difference in means
                    t_stat, p_val = stats.ttest_ind(
                        df_below[cov].dropna(), 
                        df_above[cov].dropna(), 
                        equal_var=False # Welch's t-test
                    )
                    balance_results[cov] = {
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'balanced': "Yes" if p_val > 0.05 else "No (p <= 0.05)"
                    }
                except Exception as e:
                    logger.warning(f"Could not perform t-test for covariate '{cov}': {e}")
                    balance_results[cov] = {"status": "Test Failed", "error": str(e)}
            else:
                balance_results[cov] = {"status": "Column Not Found"}
    else:
         details['covariate_balance'] = "No covariates provided to check."

    # --- Placeholders for other common RDD diagnostics --- 
    details['continuity_density_test'] = "Not Implemented (Requires specialized libraries like rdd)"
    details['placebo_cutoff_test'] = "Not Implemented (Requires re-running estimation)"
    details['bandwidth_sensitivity'] = "Not Implemented (Requires re-running estimation)"
    details['visual_inspection'] = "Recommended (Plot outcome vs running variable with fits)"

    return {"status": "Success (Partial Implementation)", "details": details}


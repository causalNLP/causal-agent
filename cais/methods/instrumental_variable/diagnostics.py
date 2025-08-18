import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np 

# Configure logger
logger = logging.getLogger(__name__)

def calculate_first_stage_f_statistic(df: pd.DataFrame, treatment: str, instruments: List[str], covariates: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the F-statistic for instrument relevance in the first stage regression.

    Regresses treatment ~ instruments + covariates.
    Tests the joint significance of the instrument coefficients.

    Args:
        df: Input DataFrame.
        treatment: Name of the treatment variable.
        instruments: List of instrument variable names.
        covariates: List of covariate names.

    Returns:
        A tuple containing (F-statistic, p-value). Returns (None, None) on error.
    """
    logger.info("Diagnostics: Calculating First-Stage F-statistic...")
    try:
        df_copy = df.copy()
        df_copy['intercept'] = 1
        exog_vars = ['intercept'] + covariates
        all_first_stage_exog = list(dict.fromkeys(exog_vars + instruments)) # Ensure unique columns

        endog = df_copy[treatment]
        exog = df_copy[all_first_stage_exog]

        # Check for perfect multicollinearity before fitting
        if exog.shape[1] > 1:
            corr_matrix = exog.corr()
            # Check if correlation matrix calculation failed (e.g., constant columns) or high correlation
            if corr_matrix.isnull().values.any() or (corr_matrix.abs() > 0.9999).sum().sum() > exog.shape[1]: # Check off-diagonal elements
                 logger.warning("High multicollinearity or constant column detected in first stage exogenous variables.")
                 # Note: statsmodels OLS might handle perfect collinearity by dropping columns, but F-test might be unreliable.

        first_stage_model = OLS(endog, exog).fit()

        # Construct the restriction matrix (R) to test H0: instrument coeffs = 0
        num_instruments = len(instruments)
        if num_instruments == 0:
            logger.warning("No instruments provided for F-statistic calculation.")
            return None, None
        num_exog_total = len(all_first_stage_exog)

        # Ensure instruments are actually in the fitted model's exog names (in case statsmodels dropped some)
        fitted_exog_names = first_stage_model.model.exog_names
        valid_instruments = [inst for inst in instruments if inst in fitted_exog_names]
        if not valid_instruments:
             logger.error("None of the provided instruments were included in the first-stage regression model (possibly due to collinearity).")
             return None, None
        if len(valid_instruments) < len(instruments):
            logger.warning(f"Instruments dropped by OLS: {set(instruments) - set(valid_instruments)}")

        instrument_indices = [fitted_exog_names.index(inst) for inst in valid_instruments]

        # Need to adjust R matrix size based on fitted model's exog
        R = np.zeros((len(valid_instruments), len(fitted_exog_names)))
        for i, idx in enumerate(instrument_indices):
            R[i, idx] = 1

        # Perform F-test
        f_test_result = first_stage_model.f_test(R)

        f_statistic = float(f_test_result.fvalue)
        p_value = float(f_test_result.pvalue)

        logger.info(f"  F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")
        return f_statistic, p_value

    except Exception as e:
        logger.error(f"Error calculating first-stage F-statistic: {e}", exc_info=True)
        return None, None

def run_overidentification_test(sm_results: Optional[Any], df: pd.DataFrame, treatment: str, outcome: str, instruments: List[str], covariates: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Runs an overidentification test (Sargan-Hansen) if applicable.

    This test is only valid if the number of instruments exceeds the number
    of endogenous regressors (typically 1, the treatment variable).

    Requires results from a statsmodels IV estimation.

    Args:
        sm_results: The fitted results object from statsmodels IV2SLS.fit().
        df: Input DataFrame.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        instruments: List of instrument variable names.
        covariates: List of covariate names.

    Returns:
        Tuple: (test_statistic, p_value, status_message) or (None, None, error_message)
    """
    logger.info("Diagnostics: Running Overidentification Test...")
    num_instruments = len(instruments)
    num_endog = 1 # Assuming only one treatment variable is endogenous

    if num_instruments <= num_endog:
        logger.info("  Over-ID test not applicable (model is exactly identified or underidentified).")
        return None, None, "Test not applicable (Need more instruments than endogenous regressors)"

    if sm_results is None or not hasattr(sm_results, 'resid'):
        logger.warning("  Over-ID test requires valid statsmodels results object with residuals.")
        return None, None, "Statsmodels results object not available or invalid for test."

    try:
        # Statsmodels IV2SLSResults does not seem to have a direct method for this test (as of common versions).
        # We need to calculate it manually using residuals and instruments.
        # Formula: N * R^2 from regressing residuals (u_hat) on all exogenous variables (instruments + covariates).
        # Degrees of freedom = num_instruments - num_endogenous_vars

        residuals = sm_results.resid
        df_copy = df.copy()
        df_copy['intercept'] = 1
        exog_vars = ['intercept'] + covariates
        all_exog_instruments = list(dict.fromkeys(exog_vars + instruments))

        # Ensure columns exist in the dataframe before selecting
        missing_cols = [col for col in all_exog_instruments if col not in df_copy.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required for Over-ID test: {missing_cols}")

        exog_for_test = df_copy[all_exog_instruments]

        # Check shapes match after potential NA handling in main estimator
        if len(residuals) != exog_for_test.shape[0]:
             # Attempt to align based on index if lengths differ (might happen if NAs were dropped)
            logger.warning(f"Residual length ({len(residuals)}) differs from exog_for_test rows ({exog_for_test.shape[0]}). Trying to align indices.")
            common_index = residuals.index.intersection(exog_for_test.index)
            if len(common_index) == 0:
                 raise ValueError("Cannot align residuals and exogenous variables for Over-ID test after NA handling.")
            residuals = residuals.loc[common_index]
            exog_for_test = exog_for_test.loc[common_index]
            logger.warning(f"Aligned to {len(common_index)} common observations.")


        # Regress residuals on all exogenous instruments
        aux_model = OLS(residuals, exog_for_test).fit()
        r_squared = aux_model.rsquared
        n_obs = len(residuals) # Use length of residuals after potential alignment

        test_statistic = n_obs * r_squared

        # Calculate p-value from Chi-squared distribution
        from scipy.stats import chi2
        degrees_of_freedom = num_instruments - num_endog
        if degrees_of_freedom < 0:
            # This shouldn't happen if the initial check passed, but as a safeguard
            raise ValueError("Degrees of freedom for Sargan test are negative.")
        elif degrees_of_freedom == 0:
            # R-squared should be 0 if exactly identified, but handle edge case
            p_value = 1.0 if np.isclose(test_statistic, 0) else 0.0
        else:
            p_value = chi2.sf(test_statistic, degrees_of_freedom)

        logger.info(f"  Sargan Test Statistic: {test_statistic:.4f}, p-value: {p_value:.4f}, df: {degrees_of_freedom}")
        return test_statistic, p_value, "Test successful"

    except Exception as e:
        logger.error(f"Error running overidentification test: {e}", exc_info=True)
        return None, None, f"Error during test: {e}"

def run_iv_diagnostics(df: pd.DataFrame, treatment: str, outcome: str, instruments: List[str], covariates: List[str], sm_results: Optional[Any] = None, dw_results: Optional[Any] = None) -> Dict[str, Any]:
    """
    Runs standard IV diagnostic checks.

    Args:
        df: Input DataFrame.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        instruments: List of instrument variable names.
        covariates: List of covariate names.
        sm_results: Optional fitted results object from statsmodels IV2SLS.fit().
        dw_results: Optional results object from DoWhy (structure may vary).

    Returns:
        Dictionary containing diagnostic results.
    """
    diagnostics = {}

    # 1. Instrument Relevance / Weak Instrument Test (First-Stage F-statistic)
    f_stat, f_p_val = calculate_first_stage_f_statistic(df, treatment, instruments, covariates)
    diagnostics['first_stage_f_statistic'] = f_stat
    diagnostics['first_stage_p_value'] = f_p_val
    diagnostics['is_instrument_weak'] = (f_stat < 10) if f_stat is not None else None # Common rule of thumb
    if f_stat is None:
        diagnostics['weak_instrument_test_status'] = "Error during calculation"
    elif diagnostics['is_instrument_weak']:
        diagnostics['weak_instrument_test_status'] = "Warning: Instrument(s) may be weak (F < 10)"
    else:
        diagnostics['weak_instrument_test_status'] = "Instrument(s) appear sufficiently strong (F >= 10)"


    # 2. Overidentification Test (e.g., Sargan-Hansen)
    overid_stat, overid_p_val, overid_status = run_overidentification_test(sm_results, df, treatment, outcome, instruments, covariates)
    diagnostics['overid_test_statistic'] = overid_stat
    diagnostics['overid_test_p_value'] = overid_p_val
    diagnostics['overid_test_status'] = overid_status
    diagnostics['overid_test_applicable'] = not ("not applicable" in overid_status.lower() if overid_status else True)

    # 3. Exogeneity/Exclusion Restriction (Conceptual Check)
    diagnostics['exclusion_restriction_assumption'] = "Assumed based on graph/input; cannot be statistically tested directly. Qualitative LLM check recommended."

    # Potential future additions:
    # - Endogeneity tests (e.g., Hausman test - requires comparing OLS and IV estimates)

    return diagnostics 
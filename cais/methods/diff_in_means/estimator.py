"""
Difference in Means / Simple Linear Regression Estimator.

Estimates the Average Treatment Effect (ATE) by comparing the mean outcome
between the treated and control groups. This is equivalent to a simple OLS
regression of the outcome on the treatment indicator.

Assumes no confounding (e.g., suitable for RCT data).
"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
from typing import Dict, Any, Optional
import logging
from langchain.chat_models.base import BaseChatModel 

from .diagnostics import run_dim_diagnostics
from .llm_assist import interpret_dim_results

logger = logging.getLogger(__name__)

def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    query: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    **kwargs 
) -> Dict[str, Any]:
    """
    Estimates the causal effect using Difference in Means (via OLS).

    Ignores any provided covariates.

    Args:
        df: Input DataFrame.
        treatment: Name of the binary treatment variable column (should be 0 or 1).
        outcome: Name of the outcome variable column.
        query: Optional user query for context.
        llm: Optional Language Model instance.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Dictionary containing estimation results:
        - 'effect_estimate': The difference in means (treatment coefficient).
        - 'p_value': The p-value associated with the difference.
        - 'confidence_interval': The 95% confidence interval for the difference.
        - 'standard_error': The standard error of the difference.
        - 'formula': The regression formula used.
        - 'model_summary': Summary object from statsmodels.
        - 'diagnostics': Basic group statistics.
        - 'interpretation': LLM interpretation.
    """
    required_cols = [treatment, outcome]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate treatment is binary (or close to it)
    treat_vals = df[treatment].dropna().unique()
    if not np.all(np.isin(treat_vals, [0, 1])):
        warnings.warn(f"Treatment column '{treatment}' contains values other than 0 and 1: {treat_vals}. Proceeding, but results may be unreliable.", UserWarning)
        # Optional: could raise ValueError here if strict binary is required
        
    # Prepare data for statsmodels (add constant, handle potential NaNs)
    df_analysis = df[required_cols].dropna()
    if df_analysis.empty:
        raise ValueError("No data remaining after dropping NaNs for required columns.")
        
    X = df_analysis[[treatment]]
    X = sm.add_constant(X) # Add intercept
    y = df_analysis[outcome]

    formula = f"{outcome} ~ {treatment} + const"
    logger.info(f"Running Difference in Means regression: {formula}")

    try:
        model = sm.OLS(y, X)
        results = model.fit()

        effect_estimate = results.params[treatment]
        p_value = results.pvalues[treatment]
        conf_int = results.conf_int(alpha=0.05).loc[treatment].tolist()
        std_err = results.bse[treatment]

        # Run basic diagnostics (group means, stds, counts)
        diag_results = run_dim_diagnostics(df_analysis, treatment, outcome)
        
        # Get interpretation
        interpretation = interpret_dim_results(results, diag_results, treatment, llm=llm)

        return {
            'effect_estimate': effect_estimate,
            'p_value': p_value,
            'confidence_interval': conf_int,
            'standard_error': std_err,
            'formula': formula,
            'model_summary': results.summary(), 
            'diagnostics': diag_results,
            'interpretation': interpretation,
            'method_used': 'Difference in Means (OLS)'
        }

    except Exception as e:
        logger.error(f"Difference in Means failed: {e}")
        raise

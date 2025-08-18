"""
Backdoor Adjustment Estimator using Regression.

Estimates the Average Treatment Effect (ATE) by regressing the outcome on the
treatment and a set of covariates assumed to satisfy the backdoor criterion.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, List, Optional
import logging
from langchain.chat_models.base import BaseChatModel 

from .diagnostics import run_backdoor_diagnostics
from .llm_assist import interpret_backdoor_results, identify_backdoor_set

logger = logging.getLogger(__name__)

def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str], 
    query: Optional[str] = None,
    llm: Optional[BaseChatModel] = None, 
    **kwargs 
) -> Dict[str, Any]:
    """
    Estimates the causal effect using Backdoor Adjustment (via OLS regression).

    Assumes the provided `covariates` list satisfies the backdoor criterion.

    Args:
        df: Input DataFrame.
        treatment: Name of the treatment variable column.
        outcome: Name of the outcome variable column.
        covariates: List of covariate names forming the backdoor adjustment set.
        query: Optional user query for context (e.g., for LLM).
        llm: Optional Language Model instance.
        **kwargs: Additional keyword arguments.

    Returns:
        Dictionary containing estimation results:
        - 'effect_estimate': The estimated coefficient for the treatment variable.
        - 'p_value': The p-value associated with the treatment coefficient.
        - 'confidence_interval': The 95% confidence interval for the effect.
        - 'standard_error': The standard error of the treatment coefficient.
        - 'formula': The regression formula used.
        - 'model_summary': Summary object from statsmodels.
        - 'diagnostics': Placeholder for diagnostic results.
        - 'interpretation': LLM interpretation.
    """
    if not covariates: # Check if the list is empty or None
        raise ValueError("Backdoor Adjustment requires a non-empty list of covariates (adjustment set).")

    required_cols = [treatment, outcome] + covariates
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for Backdoor Adjustment: {missing_cols}")

    # Prepare data for statsmodels (add constant, handle potential NaNs)
    df_analysis = df[required_cols].dropna()
    if df_analysis.empty:
        raise ValueError("No data remaining after dropping NaNs for required columns.")
        
    X = df_analysis[[treatment] + covariates]
    X = sm.add_constant(X) # Add intercept
    y = df_analysis[outcome]

    # Build the formula string for reporting
    formula = f"{outcome} ~ {treatment} + " + " + ".join(covariates) + " + const"
    logger.info(f"Running Backdoor Adjustment regression: {formula}")

    try:
        model = sm.OLS(y, X)
        results = model.fit()

        effect_estimate = results.params[treatment]
        p_value = results.pvalues[treatment]
        conf_int = results.conf_int(alpha=0.05).loc[treatment].tolist()
        std_err = results.bse[treatment]

        # Run diagnostics (Placeholders)
        # Pass the full design matrix X for potential VIF checks etc.
        diag_results = run_backdoor_diagnostics(results, X) 
        
        # Get interpretation
        interpretation = interpret_backdoor_results(results, diag_results, treatment, covariates, llm=llm)

        return {
            'effect_estimate': effect_estimate,
            'p_value': p_value,
            'confidence_interval': conf_int,
            'standard_error': std_err,
            'formula': formula,
            'model_summary': results.summary(), 
            'diagnostics': diag_results,
            'interpretation': interpretation,
            'method_used': 'Backdoor Adjustment (OLS)'
        }

    except Exception as e:
        logger.error(f"Backdoor Adjustment failed: {e}")
        raise

"""
LLM assistance functions for Linear Regression analysis.
"""

from typing import List, Dict, Any, Optional
import logging

# Imported for type hinting
from langchain.chat_models.base import BaseChatModel
from statsmodels.regression.linear_model import RegressionResultsWrapper

# Import shared LLM helpers
from causal_agent.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)

def suggest_lr_covariates(
    df_cols: List[str],
    treatment: str,
    outcome: str,
    query: str,
    llm: Optional[BaseChatModel] = None
) -> List[str]:
    """
    (Placeholder) Use LLM to suggest relevant covariates for linear regression.
    
    Args:
        df_cols: List of available column names.
        treatment: Treatment variable name.
        outcome: Outcome variable name.
        query: User's causal query text.
        llm: Optional LLM model instance.
        
    Returns:
        List of suggested covariate names.
    """
    logger.info("LLM covariate suggestion for LR is not implemented yet.")
    # TODO: Implement actual call to LLM
    if llm:
        # Placeholder: Call LLM here in future
        pass
    return []

def interpret_lr_results(
    results: RegressionResultsWrapper,
    diagnostics: Dict[str, Any], 
    treatment_var: str, # Need treatment variable name to extract coefficient
    llm: Optional[BaseChatModel] = None
) -> str:
    """
    Use LLM to interpret Linear Regression results.
    
    Args:
        results: Fitted statsmodels OLS results object.
        diagnostics: Dictionary of diagnostic test results.
        treatment_var: Name of the treatment variable.
        llm: Optional LLM model instance.
        
    Returns:
        String containing natural language interpretation.
    """
    default_interpretation = "LLM interpretation not available for Linear Regression."
    if llm is None:
        logger.info("LLM not provided for LR interpretation.")
        return default_interpretation
        
    try:
        # --- Prepare summary for LLM --- 
        results_summary = {}
        treatment_val = results.params.get(treatment_var)
        pval_val = results.pvalues.get(treatment_var)
        
        if treatment_val is not None:
            results_summary['Treatment Effect Estimate'] = f"{treatment_val:.3f}"
        else:
            logger.warning(f"Treatment variable '{treatment_var}' not found in regression parameters.")
            results_summary['Treatment Effect Estimate'] = "Not Found"

        if pval_val is not None:
            results_summary['Treatment P-value'] = f"{pval_val:.3f}"
        else:
             logger.warning(f"P-value for treatment variable '{treatment_var}' not found in regression results.")
             results_summary['Treatment P-value'] = "Not Found"

        try:
            conf_int = results.conf_int().loc[treatment_var]
            results_summary['Treatment 95% CI'] = f"[{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
        except KeyError:
            logger.warning(f"Confidence interval for treatment variable '{treatment_var}' not found.")
            results_summary['Treatment 95% CI'] = "Not Found"
        except Exception as ci_e:
             logger.warning(f"Could not extract confidence interval for '{treatment_var}': {ci_e}")
             results_summary['Treatment 95% CI'] = "Error"
            
        results_summary['R-squared'] = f"{results.rsquared:.3f}"
        results_summary['Adj. R-squared'] = f"{results.rsquared_adj:.3f}"
        
        diag_summary = {}
        if diagnostics.get("status") == "Success":
            diag_details = diagnostics.get("details", {})
            # Format p-values only if they are numbers
            jb_p = diag_details.get('residuals_normality_jb_p_value')
            bp_p = diag_details.get('homoscedasticity_bp_lm_p_value')
            diag_summary['Residuals Normality (Jarque-Bera P-value)'] = f"{jb_p:.3f}" if isinstance(jb_p, (int, float)) else str(jb_p)
            diag_summary['Homoscedasticity (Breusch-Pagan P-value)'] = f"{bp_p:.3f}" if isinstance(bp_p, (int, float)) else str(bp_p)
            diag_summary['Homoscedasticity Status'] = diag_details.get('homoscedasticity_status', 'N/A')
            diag_summary['Residuals Normality Status'] = diag_details.get('residuals_normality_status', 'N/A')
        else:
             diag_summary['Status'] = diagnostics.get("status", "Unknown")
             if "error" in diagnostics:
                 diag_summary['Error'] = diagnostics["error"]

        # --- Construct Prompt --- 
        prompt = f"""
        You are assisting with interpreting Linear Regression (OLS) results for causal inference.
        
        Model Results Summary:
        {results_summary}
        
        Model Diagnostics Summary:
        {diag_summary}
        
        Explain these results in 2-4 concise sentences. Focus on:
        1. The estimated causal effect of the treatment variable '{treatment_var}' (magnitude, direction, statistical significance based on p-value < 0.05).
        2. Overall model fit (using R-squared as a rough guide).
        3. Key diagnostic findings (specifically, mention if residuals are non-normal or if heteroscedasticity is detected, as these violate OLS assumptions and can affect inference).
        
        Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
        {{
          "interpretation": "<your concise interpretation text>"
        }}
        """
        
        # --- Call LLM --- 
        response = call_llm_with_json_output(llm, prompt)
        
        # --- Process Response --- 
        if response and isinstance(response, dict) and \
           "interpretation" in response and isinstance(response["interpretation"], str):
            return response["interpretation"]
        else:
            logger.warning(f"Failed to get valid interpretation from LLM. Response: {response}")
            return default_interpretation
            
    except Exception as e:
        logger.error(f"Error during LLM interpretation for LR: {e}")
        return f"Error generating interpretation: {e}"

"""
LLM assistance functions for Backdoor Adjustment analysis.
"""

from typing import List, Dict, Any, Optional
import logging

from langchain.chat_models.base import BaseChatModel
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_agent.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)

def identify_backdoor_set(
    df_cols: List[str],
    treatment: str,
    outcome: str,
    query: Optional[str] = None,
    existing_covariates: Optional[List[str]] = None, # Allow user to provide some
    llm: Optional[BaseChatModel] = None
) -> List[str]:
    """
    Use LLM to suggest a potential backdoor adjustment set (confounders).

    Tries to identify variables that affect both treatment and outcome.
    
    Args:
        df_cols: List of available column names in the dataset.
        treatment: Treatment variable name.
        outcome: Outcome variable name.
        query: User's causal query text (provides context).
        existing_covariates: Covariates already considered/provided by user.
        llm: Optional LLM model instance.
        
    Returns:
        List of suggested variable names for the backdoor adjustment set.
    """
    if llm is None:
        logger.warning("No LLM provided for backdoor set identification.")
        return existing_covariates or []

    # Exclude treatment and outcome from potential confounders
    potential_confounders = [c for c in df_cols if c not in [treatment, outcome]]
    if not potential_confounders:
        return existing_covariates or []
        
    prompt = f"""
    You are assisting with identifying a backdoor adjustment set for causal inference.
    The goal is to find observed variables that confound the relationship between the treatment and outcome.
    Assume the causal effect of '{treatment}' on '{outcome}' is of interest.
    
    User query context (optional): {query}
    Available variables in the dataset (excluding treatment and outcome): {potential_confounders}
    Variables already specified as covariates by user (if any): {existing_covariates}
    
    Based *only* on the variable names and the query context, identify which of the available variables are likely to be common causes (confounders) of both '{treatment}' and '{outcome}'. 
    These variables should be included in the backdoor adjustment set.
    Consider variables that likely occurred *before* or *at the same time as* the treatment.
    
    Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
    {{
      "suggested_backdoor_set": ["confounder1", "confounder2", ...] 
    }}
    Include variables from the user-provided list if they seem appropriate as confounders.
    If no plausible confounders are identified among the available variables, return an empty list.
    """
    
    response = call_llm_with_json_output(llm, prompt)
    
    suggested_set = []
    if response and "suggested_backdoor_set" in response and isinstance(response["suggested_backdoor_set"], list):
        # Basic validation
        valid_vars = [item for item in response["suggested_backdoor_set"] if isinstance(item, str)]
        if len(valid_vars) != len(response["suggested_backdoor_set"]):
            logger.warning("LLM returned non-string items in suggested_backdoor_set list.")
        suggested_set = valid_vars
    else:
         logger.warning(f"Failed to get valid backdoor set recommendations from LLM. Response: {response}")

    # Combine with existing covariates, removing duplicates
    final_set = list(dict.fromkeys((existing_covariates or []) + suggested_set))
    return final_set

def interpret_backdoor_results(
    results: RegressionResultsWrapper, 
    diagnostics: Dict[str, Any],
    treatment_var: str, 
    covariates: List[str],
    llm: Optional[BaseChatModel] = None
) -> str:
    """
    Use LLM to interpret Backdoor Adjustment results.
    
    Args:
        results: Fitted statsmodels OLS results object.
        diagnostics: Dictionary of diagnostic results.
        treatment_var: Name of the treatment variable.
        covariates: List of covariates used in the adjustment set.
        llm: Optional LLM model instance.
        
    Returns:
        String containing natural language interpretation.
    """
    default_interpretation = "LLM interpretation not available for Backdoor Adjustment."
    if llm is None:
        logger.info("LLM not provided for Backdoor Adjustment interpretation.")
        return default_interpretation
        
    try:
        # --- Prepare summary for LLM --- 
        results_summary = {}
        diag_details = diagnostics.get('details', {})
        
        effect = results.params.get(treatment_var)
        pval = results.pvalues.get(treatment_var)
        
        results_summary['Treatment Effect Estimate'] = f"{effect:.3f}" if isinstance(effect, (int, float)) else str(effect)
        results_summary['P-value'] = f"{pval:.3f}" if isinstance(pval, (int, float)) else str(pval)
        try:
            conf_int = results.conf_int().loc[treatment_var]
            results_summary['95% Confidence Interval'] = f"[{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
        except KeyError:
             results_summary['95% Confidence Interval'] = "Not Found"
        except Exception as ci_e:
             results_summary['95% Confidence Interval'] = f"Error ({ci_e})"
        
        results_summary['Adjustment Set (Covariates Used)'] = covariates
        results_summary['Model R-squared'] = f"{diagnostics.get('details', {}).get('r_squared', 'N/A'):.3f}" if isinstance(diagnostics.get('details', {}).get('r_squared'), (int, float)) else "N/A"

        diag_summary = {}
        if diagnostics.get("status") == "Success":
             diag_summary['Residuals Normality Status'] = diag_details.get('residuals_normality_status', 'N/A')
             diag_summary['Homoscedasticity Status'] = diag_details.get('homoscedasticity_status', 'N/A')
             diag_summary['Multicollinearity Status'] = diag_details.get('multicollinearity_status', 'N/A')
        else:
             diag_summary['Status'] = diagnostics.get("status", "Unknown")
        
        # --- Construct Prompt --- 
        prompt = f"""
        You are assisting with interpreting Backdoor Adjustment (Regression) results.
        The key assumption is that the specified adjustment set (covariates) blocks all confounding paths between the treatment ('{treatment_var}') and outcome.
        
        Results Summary:
        {results_summary}
        
        Diagnostics Summary (OLS model checks):
        {diag_summary}
        
        Explain these results in 2-4 concise sentences. Focus on:
        1. The estimated average treatment effect after adjusting for the specified covariates (magnitude, direction, statistical significance based on p-value < 0.05).
        2. **Crucially, mention that this estimate relies heavily on the assumption that the included covariates ('{str(covariates)[:100]}...') are sufficient to control for confounding (i.e., satisfy the backdoor criterion).**
        3. Briefly mention any major OLS diagnostic issues noted (e.g., non-normal residuals, heteroscedasticity, high multicollinearity).
        
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
            logger.warning(f"Failed to get valid interpretation from LLM for Backdoor Adj. Response: {response}")
            return default_interpretation
            
    except Exception as e:
        logger.error(f"Error during LLM interpretation for Backdoor Adj: {e}")
        return f"Error generating interpretation: {e}"

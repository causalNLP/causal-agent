"""
LLM assistance functions for Regression Discontinuity Design (RDD).
"""

from typing import List, Dict, Any, Optional
import logging

from langchain.chat_models.base import BaseChatModel

from causal_agent.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)

def suggest_rdd_parameters(
    df_cols: List[str],
    query: str,
    llm: Optional[BaseChatModel] = None
) -> Dict[str, Any]:
    """
    (Placeholder) Use LLM to suggest RDD parameters (running variable, cutoff).
    
    Args:
        df_cols: List of available column names.
        query: User's causal query text.
        llm: Optional LLM model instance.
        
    Returns:
        Dictionary containing suggested 'running_variable' and 'cutoff', or empty.
    """
    logger.info("LLM RDD parameter suggestion is not implemented yet.")
    # TODO: Implement actual call to LLM
    if llm:
        # Placeholder: Analyze columns, distributions, query for potential
        # running variables (e.g., 'score', 'age') and cutoffs (e.g., 50, 65).
        pass
    return {}

def interpret_rdd_results(
    results: Dict[str, Any], 
    diagnostics: Optional[Dict[str, Any]],
    llm: Optional[BaseChatModel] = None
) -> str:
    """
    Use LLM to interpret Regression Discontinuity Design (RDD) results.
    
    Args:
        results: Dictionary of estimation results from the RDD estimator.
        diagnostics: Dictionary of diagnostic test results.
        llm: Optional LLM model instance.
        
    Returns:
        String containing natural language interpretation.
    """
    default_interpretation = "LLM interpretation not available for RDD."
    if llm is None:
        logger.info("LLM not provided for RDD interpretation.")
        return default_interpretation
        
    try:
        # --- Prepare summary for LLM --- 
        results_summary = {}
        effect = results.get('effect_estimate')
        p_val = results.get('p_value')
        ci = results.get('confidence_interval')
        
        results_summary['Method Used'] = results.get('method_used', 'RDD')
        results_summary['Effect Estimate'] = f"{effect:.3f}" if isinstance(effect, (int, float)) else str(effect)
        results_summary['P-value'] = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
             results_summary['Confidence Interval'] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
             results_summary['Confidence Interval'] = str(ci) if ci is not None else "N/A"

        diag_summary = {}
        if diagnostics and diagnostics.get("status", "").startswith("Success"):
            diag_details = diagnostics.get("details", {})
            diag_summary['Covariate Balance Status'] = "Checked" if 'covariate_balance' in diag_details else "Not Checked"
            if isinstance(diag_details.get('covariate_balance'), dict):
                num_unbalanced = sum(1 for cov, res in diag_details['covariate_balance'].items() if isinstance(res, dict) and res.get('balanced', '').startswith("No"))
                diag_summary['Number of Unbalanced Covariates (p<=0.05)'] = num_unbalanced
            
            diag_summary['Density Continuity Test'] = diag_details.get('continuity_density_test', 'N/A')
            diag_summary['Visual Inspection Recommended'] = "Yes" if 'visual_inspection' in diag_details else "No"
        elif diagnostics:
             diag_summary['Status'] = diagnostics.get("status", "Unknown")
             if "error" in diagnostics:
                 diag_summary['Error'] = diagnostics["error"]
        else:
            diag_summary['Status'] = "Diagnostics not available or failed."

        # --- Construct Prompt --- 
        prompt = f"""
        You are assisting with interpreting Regression Discontinuity Design (RDD) results.
        
        Estimation Results Summary:
        {results_summary}
        
        Diagnostics Summary:
        {diag_summary}
        
        Explain these RDD results in 2-4 concise sentences. Focus on:
        1. The estimated causal effect at the cutoff (magnitude, direction, statistical significance based on p-value < 0.05, if available).
        2. Key diagnostic findings (specifically mention covariate balance issues if present, and note that other checks like density continuity were not performed).
        3. Mention that visual inspection of the running variable vs outcome is recommended.
        
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
            logger.warning(f"Failed to get valid interpretation from LLM for RDD. Response: {response}")
            return default_interpretation
            
    except Exception as e:
        logger.error(f"Error during LLM interpretation for RDD: {e}")
        return f"Error generating interpretation: {e}"


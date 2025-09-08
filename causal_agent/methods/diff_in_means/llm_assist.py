"""
LLM assistance functions for Difference in Means analysis.
"""

from typing import Dict, Any, Optional
import logging


from langchain.chat_models.base import BaseChatModel
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_agent.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)

def interpret_dim_results(
    results: RegressionResultsWrapper, 
    diagnostics: Dict[str, Any],
    treatment_var: str, 
    llm: Optional[BaseChatModel] = None
) -> str:
    """
    Use LLM to interpret Difference in Means results.
    
    Args:
        results: Fitted statsmodels OLS results object (from outcome ~ treatment).
        diagnostics: Dictionary of diagnostic results (group stats).
        treatment_var: Name of the treatment variable.
        llm: Optional LLM model instance.
        
    Returns:
        String containing natural language interpretation.
    """
    default_interpretation = "LLM interpretation not available for Difference in Means."
    if llm is None:
        logger.info("LLM not provided for Difference in Means interpretation.")
        return default_interpretation
        
    try:
        # --- Prepare summary for LLM --- 
        results_summary = {}
        diag_details = diagnostics.get('details', {})
        control_stats = diag_details.get('control_group_stats', {})
        treated_stats = diag_details.get('treated_group_stats', {})
        
        effect = results.params.get(treatment_var)
        pval = results.pvalues.get(treatment_var)
        
        results_summary['Effect Estimate (Difference in Means)'] = f"{effect:.3f}" if isinstance(effect, (int, float)) else str(effect)
        results_summary['P-value'] = f"{pval:.3f}" if isinstance(pval, (int, float)) else str(pval)
        try:
            conf_int = results.conf_int().loc[treatment_var]
            results_summary['95% Confidence Interval'] = f"[{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
        except KeyError:
             results_summary['95% Confidence Interval'] = "Not Found"
        except Exception as ci_e:
             results_summary['95% Confidence Interval'] = f"Error ({ci_e})"

        results_summary['Control Group Mean Outcome'] = f"{control_stats.get('mean', 'N/A'):.3f}" if isinstance(control_stats.get('mean'), (int, float)) else str(control_stats.get('mean'))
        results_summary['Treated Group Mean Outcome'] = f"{treated_stats.get('mean', 'N/A'):.3f}" if isinstance(treated_stats.get('mean'), (int, float)) else str(treated_stats.get('mean'))
        results_summary['Control Group Size'] = control_stats.get('count', 'N/A')
        results_summary['Treated Group Size'] = treated_stats.get('count', 'N/A')
        
        # --- Construct Prompt --- 
        prompt = f"""
        You are assisting with interpreting Difference in Means results, likely from an RCT.
        
        Results Summary:
        {results_summary}
        
        Explain these results in 1-3 concise sentences. Focus on:
        1. The estimated average treatment effect (magnitude, direction, statistical significance based on p-value < 0.05).
        2. Compare the mean outcomes between the treated and control groups.
        
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
            logger.warning(f"Failed to get valid interpretation from LLM for Difference in Means. Response: {response}")
            return default_interpretation
            
    except Exception as e:
        logger.error(f"Error during LLM interpretation for Difference in Means: {e}")
        return f"Error generating interpretation: {e}"

"""
Output formatter component for causal inference results.

This module formats the results of causal analysis into a clear, 
structured output for presentation to the user.
"""

from typing import Dict, List, Any, Optional
import json 

from causal_agent.models import FormattedOutput

CURRENT_OUTPUT_LOG_FILE = None

def format_output(
    query: str,
    method: str,
    results: Dict[str, Any],
    explanation: Dict[str, Any],
    dataset_analysis: Optional[Dict[str, Any]] = None,
    dataset_description: Optional[str] = None
) -> FormattedOutput:
    """
    Format final results including numerical estimates and explanations.
    
    Args:
        query: Original user query
        method: Causal inference method used (string name)
        results: Numerical results from method_executor_tool
        explanation: Structured explanation object from explainer_tool
        dataset_analysis: Optional dictionary of dataset analysis results
        dataset_description: Optional string description of the dataset
        
    Returns:
        Dict with formatted output fields ready for presentation.
    """ 
    # Extract numerical results
    effect_estimate = results.get("effect_estimate")
    confidence_interval = results.get("confidence_interval")
    p_value = results.get("p_value")
    effect_se = results.get("standard_error") # Get SE if available
    
    # Format method name for readability
    method_name_formatted = _format_method_name(method)
    
    # Extract explanation components (assuming explainer returns structured dict again)
    # If explainer returns single string, adjust this
    method_explanation_text = explanation.get("method_explanation", "")
    interpretation_guide = explanation.get("interpretation_guide", "") 
    limitations = explanation.get("limitations", [])
    assumptions_discussion = explanation.get("assumptions", "") # Assuming key is 'assumptions'
    practical_implications = explanation.get("practical_implications", "")
    # Add back final_explanation_text if explainer provides it
    # final_explanation_text = explanation.get("final_explanation_text")

    # Create summary using numerical results
    ci_text = ""
    if confidence_interval and confidence_interval[0] is not None and confidence_interval[1] is not None:
        ci_text = f" (95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])"
        
    p_value_text = f", p={p_value:.4f}" if p_value is not None else ""
    effect_text = f"{effect_estimate:.4f}" if effect_estimate is not None else "N/A"
    
    summary = (
        f"Based on {method_name_formatted}, the estimated causal effect is {effect_text}"
        f"{ci_text}{p_value_text}. {_create_effect_interpretation(effect_estimate, p_value)}"
        f" See details below regarding assumptions and limitations."
    )
    
    # Assemble formatted output dictionary
    results_dict = {
        "query": query,
        "method_used": method_name_formatted,
        "causal_effect": effect_estimate,
        "standard_error": effect_se,
        "confidence_interval": confidence_interval,
        "p_value": p_value,
        "summary": summary,
        "method_explanation": method_explanation_text,
        "interpretation_guide": interpretation_guide,
        "limitations": limitations,
        "assumptions": assumptions_discussion,
        "practical_implications": practical_implications,
        # "full_explanation_text": final_explanation_text # Optionally include combined text
    }
    final_results_dict = {key : results_dict[key] for key in {"query", "method_used", "causal_effect", "standard_error", "confidence_interval"}}
    # print(final_results_dict)

    # Validate and instantiate the Pydantic model
    try:
        formatted_output_model = FormattedOutput(**results_dict)
    except Exception as e: # Catch validation errors specifically if needed
        # Handle validation error - perhaps log and return a default or raise
        print(f"Error creating FormattedOutput model: {e}") # Or use logger
        # Decide on error handling: raise, return None, return default? 
        # For now, re-raising might be simplest if the structure is expected
        raise ValueError(f"Failed to create FormattedOutput from results: {e}")

    return formatted_output_model # Return the Pydantic model instance


def _format_method_name(method: str) -> str:
    """Format method name for readability."""
    method_names = {
        "propensity_score_matching": "Propensity Score Matching",
        "regression_adjustment": "Regression Adjustment",
        "instrumental_variable": "Instrumental Variable Analysis",
        "difference_in_differences": "Difference-in-Differences",
        "regression_discontinuity": "Regression Discontinuity Design",
        "backdoor_adjustment": "Backdoor Adjustment",
        "propensity_score_weighting": "Propensity Score Weighting"
    }
    return method_names.get(method, method.replace("_", " ").title())

# Reinstate helper function for interpretation
def _create_effect_interpretation(effect: Optional[float], p_value: Optional[float] = None) -> str:
    """Create a basic interpretation of the effect."""
    if effect is None:
        return "Effect estimate not available."
        
    significance = ""
    if p_value is not None:
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    
    magnitude = ""
    if abs(effect) < 0.01:
        magnitude = "no practical effect"
    elif abs(effect) < 0.1:
        magnitude = "a small effect"
    elif abs(effect) < 0.5:
        magnitude = "a moderate effect"
    else:
        magnitude = "a substantial effect"
        
    return f"This suggests {magnitude}{f' and is {significance}' if significance else ''}." 
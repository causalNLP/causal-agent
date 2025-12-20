"""
Output formatter component for causal inference results.

This module formats the results of causal analysis into a clear, 
structured output for presentation to the user.
"""

from typing import Dict, List, Any, Optional
import numbers
import json 

from cais.models import FormattedOutput

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

    def _normalize_single_value(value: Any) -> Any:
        if isinstance(value, dict) and len(value) == 1:
            return next(iter(value.values()))
        return value

    def _format_stat_value(value: Any, precision: int = 4) -> str:
        value = _normalize_single_value(value)
        if isinstance(value, numbers.Number):
            return f"{float(value):.{precision}f}"
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, numbers.Number) for v in value):
            return f"[{float(value[0]):.{precision}f}, {float(value[1]):.{precision}f}]"
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                parts.append(f"{key}: {_format_stat_value(item, precision)}")
            return "; ".join(parts)
        return "N/A"
    
    # Format method name for readability
    method_name_formatted = _format_method_name(method)
    
    # Extract explanation components (assuming explainer returns structured dict again)
    # If explainer returns single string, adjust this
    method_explanation_text = explanation.get("method_explanation", "")
    interpretation_guide = explanation.get("interpretation_guide", "") 
    limitations = explanation.get("limitations", [])
    assumptions_discussion = explanation.get("assumptions", "") # Assuming key is 'assumptions'
    practical_implications = explanation.get("practical_implications", "")
    interpretation_text = explanation.get("interpretation_text", "")
    # Add back final_explanation_text if explainer provides it
    # final_explanation_text = explanation.get("final_explanation_text")

    # Create summary using numerical results
    normalized_ci = _normalize_single_value(confidence_interval)
    if isinstance(normalized_ci, (list, tuple)) and len(normalized_ci) == 2:
        ci_text = f" (95% CI: {_format_stat_value(normalized_ci)})"
    elif isinstance(confidence_interval, dict) and confidence_interval:
        ci_text = f" (95% CI by level: {_format_stat_value(confidence_interval)})"
    else:
        ci_text = ""

    normalized_p_value = _normalize_single_value(p_value)
    if isinstance(normalized_p_value, numbers.Number):
        p_value_text = f", p={float(normalized_p_value):.4f}"
    elif isinstance(p_value, dict) and p_value:
        p_value_text = f", p-values by level: {_format_stat_value(p_value)}"
    else:
        p_value_text = ""

    effect_text = _format_stat_value(effect_estimate) if effect_estimate is not None else "N/A"
    
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
        "interpretation_text": interpretation_text,
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

    if isinstance(effect, dict):
        return "Effect estimates are provided by level; see details below."

    significance = ""
    if isinstance(p_value, dict):
        return "Statistical significance varies by level; see details below."
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

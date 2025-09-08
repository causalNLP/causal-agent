"""
LLM assistance functions for Instrumental Variable (IV) analysis.

This module provides functions for LLM-based assistance in instrumental variable analysis,
including identifying potential instruments, validating IV assumptions, and interpreting results.
"""

from typing import List, Dict, Any, Optional
import logging

from langchain.chat_models.base import BaseChatModel

from causal_agent.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)

def identify_instrument_variable(
    df_cols: List[str],
    query: str,
    llm: Optional[BaseChatModel] = None
) -> List[str]:
    """
    Use LLM to identify potential instrumental variables from available columns.
    
    Args:
        df_cols: List of column names from the dataset
        query: User's causal query text
        llm: Optional LLM model instance
        
    Returns:
        List of column names identified as potential instruments
    """
    if llm is None:
        logger.warning("No LLM provided for instrument identification")
        return []
    
    prompt = f"""
    You are assisting with an instrumental variable analysis.
    
    Available columns in the dataset: {df_cols}
    User query: {query}
    
    Identify potential instrumental variable(s) from the available columns based on the query.
    The treatment and outcome should NOT be included as instruments.
    
    Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
    {{
      "potential_instruments": ["column_name1", "column_name2", ...] 
    }}
    """
    
    response = call_llm_with_json_output(llm, prompt)
    
    if response and "potential_instruments" in response and isinstance(response["potential_instruments"], list):
        # Basic validation: ensure items are strings (column names)
        valid_instruments = [item for item in response["potential_instruments"] if isinstance(item, str)]
        if len(valid_instruments) != len(response["potential_instruments"]):
            logger.warning("LLM returned non-string items in potential_instruments list.")
        return valid_instruments
    
    logger.warning(f"Failed to get valid instrument recommendations from LLM. Response: {response}")
    return []

def validate_instrument_assumptions_qualitative(
    treatment: str,
    outcome: str,
    instrument: List[str],
    covariates: List[str],
    query: str,
    llm: Optional[BaseChatModel] = None
) -> Dict[str, str]:
    """
    Use LLM to provide qualitative assessment of IV assumptions.
    
    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        instrument: List of instrumental variable names
        covariates: List of covariate variable names
        query: User's causal query text
        llm: Optional LLM model instance
        
    Returns:
        Dictionary with qualitative assessments of exclusion and exogeneity assumptions
    """
    default_fail = {
        "exclusion_assessment": "LLM Check Failed",
        "exogeneity_assessment": "LLM Check Failed"
    }
    
    if llm is None:
        return {
            "exclusion_assessment": "LLM Not Provided",
            "exogeneity_assessment": "LLM Not Provided"
        }
    
    prompt = f"""
    You are assisting with assessing the validity of instrumental variable assumptions.
    
    Treatment variable: {treatment}
    Outcome variable: {outcome}
    Instrumental variable(s): {instrument}
    Covariates: {covariates}
    User query: {query}
    
    Assess the core Instrumental Variable (IV) assumptions based *only* on the provided variable names and query context:
    1. Exclusion restriction: Plausibility that the instrument(s) affect the outcome ONLY through the treatment.
    2. Exogeneity (also called Independence): Plausibility that the instrument(s) are not correlated with unobserved confounders that also affect the outcome.
    
    Provide a brief, qualitative assessment (e.g., 'Plausible', 'Unlikely', 'Requires Domain Knowledge', 'Potentially Violated').
    
    Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
    {{
      "exclusion_assessment": "<brief assessment of exclusion restriction>",
      "exogeneity_assessment": "<brief assessment of exogeneity assumption>"
    }}
    """
    
    response = call_llm_with_json_output(llm, prompt)
    
    if response and isinstance(response, dict) and \
       "exclusion_assessment" in response and isinstance(response["exclusion_assessment"], str) and \
       "exogeneity_assessment" in response and isinstance(response["exogeneity_assessment"], str):
        return response
    
    logger.warning(f"Failed to get valid assumption assessment from LLM. Response: {response}")
    return default_fail

def interpret_iv_results(
    results: Dict[str, Any],
    diagnostics: Dict[str, Any],
    llm: Optional[BaseChatModel] = None
) -> str:
    """
    Use LLM to interpret IV results in natural language.
    
    Args:
        results: Dictionary of estimation results (e.g., effect_estimate, p_value, confidence_interval)
        diagnostics: Dictionary of diagnostic test results (e.g., first_stage_f_statistic, overid_test)
        llm: Optional LLM model instance
        
    Returns:
        String containing natural language interpretation of results
    """
    if llm is None:
        return "LLM was not available to provide interpretation. Please review the numeric results manually."
    
    # Construct a concise summary of inputs for the prompt
    results_summary = {}
    
    effect = results.get('effect_estimate')
    if effect is not None:
        try:
            results_summary['Effect Estimate'] = f"{float(effect):.3f}"
        except (ValueError, TypeError):
            results_summary['Effect Estimate'] = 'N/A (Invalid Format)'
    else:
        results_summary['Effect Estimate'] = 'N/A'

    p_value = results.get('p_value')
    if p_value is not None:
        try:
            results_summary['P-value'] = f"{float(p_value):.3f}"
        except (ValueError, TypeError):
            results_summary['P-value'] = 'N/A (Invalid Format)'
    else:
        results_summary['P-value'] = 'N/A'

    ci = results.get('confidence_interval')
    if ci is not None and isinstance(ci, (list, tuple)) and len(ci) == 2:
        try:
            results_summary['Confidence Interval'] = f"[{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
        except (ValueError, TypeError):
            results_summary['Confidence Interval'] = 'N/A (Invalid Format)'
    else:
        # Handle cases where CI is None or not a 2-element list/tuple
        results_summary['Confidence Interval'] = str(ci) if ci is not None else 'N/A'

    if 'treatment_variable' in results:
         results_summary['Treatment'] = results['treatment_variable']
    if 'outcome_variable' in results:
         results_summary['Outcome'] = results['outcome_variable']

    diagnostics_summary = {}
    f_stat = diagnostics.get('first_stage_f_statistic')
    if f_stat is not None:
        try:
            diagnostics_summary['First-Stage F-statistic'] = f"{float(f_stat):.2f}"
        except (ValueError, TypeError):
             diagnostics_summary['First-Stage F-statistic'] = 'N/A (Invalid Format)'
    else:
         diagnostics_summary['First-Stage F-statistic'] = 'N/A'
         
    if 'weak_instrument_test_status' in diagnostics:
        diagnostics_summary['Weak Instrument Test'] = diagnostics['weak_instrument_test_status']
        
    overid_p = diagnostics.get('overid_test_p_value')
    if overid_p is not None:
        try:
             diagnostics_summary['Overidentification Test P-value'] = f"{float(overid_p):.3f}"
             diagnostics_summary['Overidentification Test Applicable'] = diagnostics.get('overid_test_applicable', 'N/A')
        except (ValueError, TypeError):
             diagnostics_summary['Overidentification Test P-value'] = 'N/A (Invalid Format)'
             diagnostics_summary['Overidentification Test Applicable'] = diagnostics.get('overid_test_applicable', 'N/A')
    else:
        # Explicitly state if not applicable or not available
        if diagnostics.get('overid_test_applicable') == False:
             diagnostics_summary['Overidentification Test'] = 'Not Applicable'
        else:
             diagnostics_summary['Overidentification Test P-value'] = 'N/A'
             diagnostics_summary['Overidentification Test Applicable'] = diagnostics.get('overid_test_applicable', 'N/A')

    prompt = f"""
    You are assisting with interpreting instrumental variable (IV) analysis results.
    
    Estimation results summary: {results_summary}
    Diagnostic test results summary: {diagnostics_summary}
    
    Explain these Instrumental Variable (IV) results in clear, concise language (2-4 sentences).
    Focus on:
    1. The estimated causal effect (magnitude, direction, statistical significance based on p-value < 0.05).
    2. The strength of the instrument(s) (based on F-statistic, typically > 10 indicates strength).
    3. Any implications from other diagnostic tests (e.g., overidentification test suggesting instrument validity issues if p < 0.05).
    
    Return ONLY a valid JSON object with the following structure (no explanations or surrounding text):
    {{
      "interpretation": "<your concise interpretation text>"
    }}
    """
    
    response = call_llm_with_json_output(llm, prompt)
    
    if response and isinstance(response, dict) and \
       "interpretation" in response and isinstance(response["interpretation"], str):
        return response["interpretation"]
    
    logger.warning(f"Failed to get valid interpretation from LLM. Response: {response}")
    return "LLM interpretation could not be generated. Please review the numeric results manually." 
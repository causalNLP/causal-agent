"""
LLM assistance functions for Instrumental Variable (IV) analysis.

This module provides functions for LLM-based assistance in instrumental variable analysis,
including identifying potential instruments, validating IV assumptions, and interpreting results.

Uses the IV-LLM pipeline for robust instrument discovery and validation with
Hypothesizer, ExclusionCritic, and IndependenceCritic agents.
"""

from typing import List, Dict, Any, Optional
import logging

from langchain.chat_models.base import BaseChatModel

from cais.utils.llm_helpers import call_llm_with_json_output

# Import IV-LLM components for instrument discovery
from cais.iv_llm.src.agents.hypothesizer import Hypothesizer
from cais.iv_llm.src.agents.confounder_miner import ConfounderMiner
from cais.iv_llm.src.critics.exclusion_critic import ExclusionCritic
from cais.iv_llm.src.critics.independence_critic import IndependenceCritic

logger = logging.getLogger(__name__)


class _LangChainLLMAdapter:
    """Adapter that wraps a LangChain BaseChatModel for IV-LLM components."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM given a prompt string."""
        response = self.llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        return str(response)


def discover_instruments_with_validation(
    treatment: str,
    outcome: str,
    df_cols: List[str],
    query: str,
    confounders: Optional[List[str]] = None,
    llm: Optional[BaseChatModel] = None,
    k_ivs: int = 5,
) -> Dict[str, Any]:
    """
    Use IV-LLM pipeline to discover and validate instrumental variables.
    
    This is the main entry point for IV discovery using the full IV-LLM pipeline
    with Hypothesizer, ExclusionCritic, and IndependenceCritic agents.
    
    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name  
        df_cols: List of column names from the dataset
        query: User's causal query text
        confounders: Optional list of known confounders
        llm: Optional LLM model instance
        k_ivs: Number of instrument candidates to propose
        
    Returns:
        Dict containing:
            - proposed_ivs: All proposed instruments
            - valid_ivs: Instruments that passed validation
            - validation_results: Detailed results per instrument
            - confounders: Identified or provided confounders
    """
    if llm is None:
        logger.warning("No LLM provided for IV discovery")
        return {
            "proposed_ivs": [],
            "valid_ivs": [],
            "validation_results": [],
            "confounders": confounders or [],
        }
    
    # Wrap LLM for IV-LLM components
    llm_adapter = _LangChainLLMAdapter(llm)
    
    # Build context string with available columns
    context = f"User query: {query}\nAvailable columns in dataset: {', '.join(df_cols)}"
    
    # Step 1: Hypothesize potential IVs
    hypothesizer = Hypothesizer(llm_adapter, k=k_ivs)
    proposed_ivs = hypothesizer.propose_ivs(treatment, outcome, context=context)
    logger.info("IV-LLM Hypothesizer proposed IVs: %s", proposed_ivs)
    
    if not proposed_ivs:
        return {
            "proposed_ivs": [],
            "valid_ivs": [],
            "validation_results": [],
            "confounders": confounders or [],
        }
    
    # Step 2: Identify confounders if not provided
    if confounders is None or len(confounders) == 0:
        confounder_miner = ConfounderMiner(llm_adapter, j=5)
        confounders = confounder_miner.identify_confounders(treatment, outcome, context=context)
        logger.info("IV-LLM ConfounderMiner identified confounders: %s", confounders)
    
    # Step 3: Validate IVs with critics
    exclusion_critic = ExclusionCritic(llm_adapter)
    independence_critic = IndependenceCritic(llm_adapter)
    
    validation_results = []
    valid_ivs = []
    
    for iv in proposed_ivs:
        # Check exclusion restriction
        exclusion_valid = exclusion_critic.validate_exclusion(
            iv, treatment, outcome, confounders
        )
        
        # Check independence assumption
        independence_valid = independence_critic.validate_independence(
            iv, treatment, outcome, confounders
        )
        
        result = {
            "iv": iv,
            "exclusion_valid": exclusion_valid,
            "independence_valid": independence_valid,
            "overall_valid": exclusion_valid and independence_valid,
        }
        validation_results.append(result)
        
        if exclusion_valid and independence_valid:
            valid_ivs.append(iv)
            logger.info("IV '%s' passed all validation checks", iv)
        else:
            logger.info(
                "IV '%s' failed validation - exclusion: %s, independence: %s",
                iv, exclusion_valid, independence_valid
            )
    
    logger.info("IV-LLM validation complete. Valid IVs: %s", valid_ivs)
    
    return {
        "proposed_ivs": proposed_ivs,
        "valid_ivs": valid_ivs,
        "validation_results": validation_results,
        "confounders": confounders,
    }


def identify_confounders(
    treatment: str,
    outcome: str,
    df_cols: List[str],
    query: str,
    llm: Optional[BaseChatModel] = None,
    j_confounders: int = 5,
) -> List[str]:
    """
    Use IV-LLM ConfounderMiner to identify potential confounders.
    
    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        df_cols: List of column names from the dataset
        query: User's causal query text
        llm: Optional LLM model instance
        j_confounders: Number of confounders to identify
        
    Returns:
        List of identified confounder variable names
    """
    if llm is None:
        logger.warning("No LLM provided for confounder identification")
        return []
    
    llm_adapter = _LangChainLLMAdapter(llm)
    context = f"User query: {query}\nAvailable columns in dataset: {', '.join(df_cols)}"
    
    confounder_miner = ConfounderMiner(llm_adapter, j=j_confounders)
    confounders = confounder_miner.identify_confounders(treatment, outcome, context=context)
    
    logger.info("IV-LLM ConfounderMiner identified confounders: %s", confounders)
    return confounders


def identify_instrument_variable(
    df_cols: List[str],
    query: str,
    llm: Optional[BaseChatModel] = None,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    confounders: Optional[List[str]] = None,
    validate: bool = True,
) -> List[str]:
    """
    Use IV-LLM pipeline to identify and optionally validate instrumental variables.
    
    This is a drop-in replacement for the previous simple LLM-based identification.
    When treatment and outcome are provided, uses the full IV-LLM pipeline with
    validation. Otherwise falls back to simple LLM-based identification.
    
    Args:
        df_cols: List of column names from the dataset
        query: User's causal query text
        llm: Optional LLM model instance
        treatment: Treatment variable name (enables full IV-LLM pipeline)
        outcome: Outcome variable name (enables full IV-LLM pipeline)
        confounders: Optional list of known confounders
        validate: Whether to validate instruments (requires treatment/outcome)
        
    Returns:
        List of valid instrument variable names
    """
    if llm is None:
        logger.warning("No LLM provided for instrument identification")
        return []
    
    # Use full IV-LLM pipeline if treatment and outcome are provided
    if treatment and outcome and validate:
        discovery_results = discover_instruments_with_validation(
            treatment=treatment,
            outcome=outcome,
            df_cols=df_cols,
            query=query,
            confounders=confounders,
            llm=llm,
        )
        # Return valid IVs if any, otherwise return proposed IVs
        valid_ivs = discovery_results.get("valid_ivs", [])
        if valid_ivs:
            return valid_ivs
        # Fall back to proposed IVs if none passed validation
        proposed = discovery_results.get("proposed_ivs", [])
        if proposed:
            logger.warning("No IVs passed validation, returning proposed IVs: %s", proposed)
            return proposed[:1]  # Return first proposed as fallback
        return []
    
    # Fallback: Simple LLM-based identification (legacy behavior)
    llm_adapter = _LangChainLLMAdapter(llm)
    context = f"User query: {query}\nAvailable columns in dataset: {', '.join(df_cols)}"
    
    # Use hypothesizer without validation
    hypothesizer = Hypothesizer(llm_adapter, k=5)
    # Need treatment/outcome for hypothesizer - try to extract from query
    if treatment and outcome:
        proposed_ivs = hypothesizer.propose_ivs(treatment, outcome, context=context)
    else:
        # Cannot use hypothesizer without treatment/outcome, use simple prompt
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
            return [item for item in response["potential_instruments"] if isinstance(item, str)]
        return []
    
    return proposed_ivs


def validate_instrument_assumptions_qualitative(
    treatment: str,
    outcome: str,
    instrument: List[str],
    covariates: List[str],
    query: str,
    llm: Optional[BaseChatModel] = None
) -> Dict[str, Any]:
    """
    Use IV-LLM critics to validate IV assumptions (exclusion and independence).
    
    This uses the ExclusionCritic and IndependenceCritic from the IV-LLM pipeline
    for rigorous validation of instrumental variable assumptions.
    
    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        instrument: List of instrumental variable names
        covariates: List of covariate variable names (treated as confounders)
        query: User's causal query text
        llm: Optional LLM model instance
        
    Returns:
        Dictionary with validation results for each instrument and overall assessment
    """
    default_fail = {
        "exclusion_assessment": "LLM Check Failed",
        "exogeneity_assessment": "LLM Check Failed",
        "validation_details": [],
    }
    
    if llm is None:
        return {
            "exclusion_assessment": "LLM Not Provided",
            "exogeneity_assessment": "LLM Not Provided",
            "validation_details": [],
        }
    
    # Wrap LLM for IV-LLM components
    llm_adapter = _LangChainLLMAdapter(llm)
    
    # Initialize critics
    exclusion_critic = ExclusionCritic(llm_adapter)
    independence_critic = IndependenceCritic(llm_adapter)
    
    # Validate each instrument
    validation_details = []
    all_exclusion_valid = True
    all_independence_valid = True
    
    instrument_list = instrument if isinstance(instrument, list) else [instrument]
    confounders = covariates if covariates else []
    
    for iv in instrument_list:
        # Check exclusion restriction
        exclusion_valid = exclusion_critic.validate_exclusion(
            iv, treatment, outcome, confounders
        )
        
        # Check independence assumption  
        independence_valid = independence_critic.validate_independence(
            iv, treatment, outcome, confounders
        )
        
        validation_details.append({
            "instrument": iv,
            "exclusion_valid": exclusion_valid,
            "independence_valid": independence_valid,
            "overall_valid": exclusion_valid and independence_valid,
        })
        
        if not exclusion_valid:
            all_exclusion_valid = False
        if not independence_valid:
            all_independence_valid = False
        
        logger.info(
            "IV '%s' validation - exclusion: %s, independence: %s",
            iv, exclusion_valid, independence_valid
        )
    
    # Generate overall assessment strings
    if all_exclusion_valid:
        exclusion_assessment = "Valid - Instruments appear to satisfy exclusion restriction"
    else:
        exclusion_assessment = "Potentially Violated - Some instruments may affect outcome directly"
    
    if all_independence_valid:
        exogeneity_assessment = "Valid - Instruments appear independent of confounders"
    else:
        exogeneity_assessment = "Potentially Violated - Some instruments may be correlated with confounders"
    
    return {
        "exclusion_assessment": exclusion_assessment,
        "exogeneity_assessment": exogeneity_assessment,
        "validation_details": validation_details,
        "all_valid": all_exclusion_valid and all_independence_valid,
    }

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
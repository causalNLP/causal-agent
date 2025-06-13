"""
Query interpreter component for causal inference.

This module provides functionality to match query concepts to actual dataset variables,
identifying treatment, outcome, and covariate variables for causal inference analysis.
"""

import re
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import logging
import numpy as np
from auto_causal.config import get_llm_client
# Import LLM and message types
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException
# Import base Pydantic models needed directly
from pydantic import BaseModel, ValidationError

# Import shared Pydantic models from the central location
from auto_causal.models import (
    LLMSelectedVariable,
    LLMSelectedCovariates,
    LLMIVars,
    LLMRDDVars,
    LLMRCTCheck,
    LLMTreatmentReferenceLevel,
    LLMInteractionSuggestion
)

# Import the new prompt templates
from auto_causal.prompts.method_identification_prompts import (
    IV_IDENTIFICATION_PROMPT_TEMPLATE,
    RDD_IDENTIFICATION_PROMPT_TEMPLATE,
    RCT_IDENTIFICATION_PROMPT_TEMPLATE,
    TREATMENT_REFERENCE_IDENTIFICATION_PROMPT_TEMPLATE,
    INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE
)

# Assume central models are defined elsewhere or keep local definitions for now
# from ..models import ... 

# --- Pydantic models for LLM structured output --- 
# REMOVED - Now defined in causalscientist/auto_causal/models.py
# class LLMSelectedVariable(BaseModel): ...
# class LLMSelectedCovariates(BaseModel): ...
# class LLMIVars(BaseModel): ...
# class LLMRDDVars(BaseModel): ...
# class LLMRCTCheck(BaseModel): ...


logger = logging.getLogger(__name__)

def interpret_query(
    query_info: Dict[str, Any], 
    dataset_analysis: Dict[str, Any],
    dataset_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Interpret query using hybrid heuristic/LLM approach to identify variables.
    
    Args:
        query_info: Information extracted from the user's query (text, hints).
        dataset_analysis: Information about the dataset structure (columns, types, etc.).
        dataset_description: Optional textual description of the dataset.
        llm: Optional language model instance.
        
    Returns:
        Dict containing identified variables (treatment, outcome, covariates, etc., and is_rct).
    """
    logger.info("Interpreting query with hybrid approach...")
    llm = get_llm_client()
    
    query_text = query_info.get("query_text", "")
    columns = dataset_analysis.get("columns", [])
    column_categories = dataset_analysis.get("column_categories", {})
    file_path = dataset_analysis["dataset_info"]["file_path"]
    
    # --- Identify Treatment --- 
    treatment_hints = query_info.get("potential_treatments", [])
    dataset_treatments = dataset_analysis.get("potential_treatments", [])
    treatment_variable = _identify_variable_hybrid(
        role="treatment",
        query_hints=treatment_hints,
        dataset_suggestions=dataset_treatments,
        columns=columns,
        column_categories=column_categories,
        prioritize_types=["binary", "binary_categorical", "discrete_numeric","continuous_numeric"], # Prioritize binary/discrete
        query_text=query_text,
        dataset_description=dataset_description,
        llm=llm
    )
    logger.info(f"Identified Treatment: {treatment_variable}")
    #.exit()

    # --- Determine Treatment Variable Type ---
    treatment_variable_type = "unknown" # Default
    if treatment_variable and treatment_variable in column_categories:
        category = column_categories[treatment_variable]
        logger.info(f"Category for treatment '{treatment_variable}' is '{category}'.")

        if category == "continuous_numeric":
            treatment_variable_type = "continuous"
        elif category == "discrete_numeric":
            # Check for number of unique values if available in dataset_analysis
            num_unique_values = -1
            if "column_nunique_counts" in dataset_analysis and treatment_variable in dataset_analysis["column_nunique_counts"]:
                num_unique_values = dataset_analysis["column_nunique_counts"][treatment_variable]
            else:
                logger.warning(f"'column_nunique_counts' not found in dataset_analysis or treatment variable '{treatment_variable}' missing from it.")

            if num_unique_values > 10: # Heuristic threshold for treating discrete as continuous for GPS
                logger.info(f"Treatment '{treatment_variable}' is discrete_numeric with {num_unique_values} unique values, treating as continuous-like.")
                treatment_variable_type = "continuous" 
            elif num_unique_values == 2: # A discrete numeric with 2 unique values is binary
                 logger.info(f"Treatment '{treatment_variable}' is discrete_numeric with 2 unique values, treating as binary.")
                 treatment_variable_type = "binary"
            elif num_unique_values > 0 : # Discrete with few values
                logger.info(f"Treatment '{treatment_variable}' is discrete_numeric with {num_unique_values} unique values, treating as discrete_multi_value.")
                treatment_variable_type = "discrete_multi_value"
            else:
                logger.info(f"Treatment '{treatment_variable}' is discrete_numeric, but unique value count not available or too few. Defaulting type.")
                treatment_variable_type = "discrete_numeric_unknown_cardinality" # Or some other default

        elif category == "binary" or category == "binary_categorical":
            treatment_variable_type = "binary"
        elif category == "categorical_numeric" or category == "categorical":
            # Could also check nunique here if needed to distinguish from binary
            num_unique_values = -1
            if "column_nunique_counts" in dataset_analysis and treatment_variable in dataset_analysis["column_nunique_counts"]:
                num_unique_values = dataset_analysis["column_nunique_counts"][treatment_variable]
            else:
                logger.warning(f"For categorical, 'column_nunique_counts' not found or var '{treatment_variable}' missing.")

            if num_unique_values == 2:
                 treatment_variable_type = "binary"
            elif num_unique_values > 0:
                 treatment_variable_type = "categorical_multi_value"
            else:
                 treatment_variable_type = "categorical_unknown_cardinality"
        else:
            logger.warning(f"Treatment variable '{treatment_variable}' has category '{category}' which is not explicitly mapped to a treatment type. Setting to 'other'.")
            treatment_variable_type = "other" # For types like datetime, text_or_other
    elif treatment_variable:
        logger.warning(f"Treatment variable '{treatment_variable}' not found in column_categories. Cannot determine type.")
    else:
        logger.info("No treatment variable identified. Treatment type remains 'unknown'.")
    
    logger.info(f"Final Determined Treatment Variable Type: {treatment_variable_type}")

    # --- Identify Outcome --- 
    outcome_hints = query_info.get("outcome_hints", [])
    dataset_outcomes = dataset_analysis.get("potential_outcomes", [])
    outcome_variable = _identify_variable_hybrid(
        role="outcome",
        query_hints=outcome_hints,
        dataset_suggestions=dataset_outcomes,
        columns=columns,
        column_categories=column_categories,
        prioritize_types=["continuous_numeric", "discrete_numeric"], # Prioritize numeric
        exclude_vars=[treatment_variable], # Exclude treatment
        query_text=query_text,
        dataset_description=dataset_description,
        llm=llm
    )
    logger.info(f"Identified Outcome: {outcome_variable}")

    # --- Identify Covariates --- 
    covariate_hints = query_info.get("covariates_hints", [])
    covariates = _identify_covariates_hybrid(
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        columns=columns,
        column_categories=column_categories,
        query_hints=covariate_hints,
        query_text=query_text,
        dataset_description=dataset_description,
        llm=llm
    )
    logger.info(f"Identified Covariates: {covariates}")

    # --- Identify Time/Group (from dataset analysis) --- 
    time_variable = None
    group_variable = None
    temporal_structure = dataset_analysis.get("temporal_structure", {})
    if temporal_structure.get("has_temporal_structure", False):
        time_variable = temporal_structure.get("time_column") or temporal_structure.get("temporal_columns", [None])[0]
        if temporal_structure.get("is_panel_data", False):
            group_variable = temporal_structure.get("id_column")
    logger.info(f"Identified Time Var: {time_variable}, Group Var: {group_variable}")

    # --- Identify IV/RDD/RCT using LLM --- 
    instrument_variable = None
    running_variable = None
    cutoff_value = None
    is_rct = None

    if llm:
        try:
            # Check for RCT
            prompt_rct = _create_identify_prompt("whether data is from RCT", query_text, dataset_description, columns, column_categories, treatment_variable, outcome_variable)
            rct_result = _call_llm_for_var(llm, prompt_rct, LLMRCTCheck)
            is_rct = rct_result.is_rct if rct_result else None
            logger.info(f"LLM identified RCT: {is_rct}")

            # Check for IV
            prompt_iv = _create_identify_prompt("instrumental variable", query_text, dataset_description, columns, column_categories, treatment_variable, outcome_variable)
            iv_result = _call_llm_for_var(llm, prompt_iv, LLMIVars)
            instrument_variable = None#iv_result.instrument_variable if iv_result else None
            logger.info(f"LLM identified IV: {instrument_variable}")

            # Check for RDD
            prompt_rdd = _create_identify_prompt("regression discontinuity (running variable and cutoff)", query_text, dataset_description, columns, column_categories, treatment_variable, outcome_variable)
            rdd_result = _call_llm_for_var(llm, prompt_rdd, LLMRDDVars)
            if rdd_result:
                running_variable = rdd_result.running_variable
                cutoff_value = rdd_result.cutoff_value
            logger.info(f"LLM identified RDD: Running={running_variable}, Cutoff={cutoff_value}")


        except Exception as e:
            logger.error(f"Error during LLM checks for IV/RDD/RCT: {e}")
            
    # If LLM didn't explicitly say RCT, default to False or keep None?
    # Let's default to False if LLM didn't provide a boolean value.
    if is_rct is None:
        is_rct = False

    # --- Identify Treatment Reference Level --- 
    treatment_reference_level = None
    if llm and treatment_variable and treatment_variable in columns:
        treatment_values_sample = [] # Initialize here

        # Attempt to get treatment values from dataset_path first
        if file_path: # dataset_path is available from interpret_query args
            try:
                df = pd.read_csv(file_path)
                if treatment_variable in df.columns:
                    unique_vals = df[treatment_variable].unique()
                    # Convert numpy types to standard Python types if necessary, then to list
                    treatment_values_sample = [item.item() if hasattr(item, 'item') else item for item in unique_vals][:10]
                    if treatment_values_sample:
                        logger.info(f"Successfully read treatment values sample from dataset at '{file_path}' for variable '{treatment_variable}'.")
                    else:
                        logger.info(f"Treatment variable '{treatment_variable}' in '{file_path}' has no unique values or is empty. Will try fallback.")
                else:
                    logger.warning(f"Treatment variable '{treatment_variable}' not found in columns of dataset at '{file_path}'. Will try fallback.")
            except FileNotFoundError:
                logger.warning(f"Dataset file not found at path: {file_path}. Will try fallback for treatment values.")
            except pd.errors.EmptyDataError:
                logger.warning(f"Dataset file at '{file_path}' is empty. Will try fallback for treatment values.")
            except Exception as e: # Catch other potential pandas errors
                logger.warning(f"Error reading or processing dataset from '{file_path}' to get treatment values: {e}. Will try fallback.")


        if not treatment_values_sample:
            logger.warning(f"Could not retrieve unique values for treatment variable '{treatment_variable}' for reference level identification from any source.")
            # LLM prompt will receive an empty list for treatment_variable_values
        else:
            logger.info(f"Final treatment values sample for '{treatment_variable}' to be used in prompt: {treatment_values_sample}")
        
        try:
            # Fetch unique values for the treatment variable for the prompt
            # This part assumes dataset_analysis might have value_counts or similar
            # or we might need to pass the df or a way to access it.
            # For now, let's assume dataset_analysis.get("value_counts", {}).get(treatment_variable, {}).get("values") exists and is a list.
            # A more robust way would be to ensure this info is consistently available.
            # The old logic for populating treatment_values_sample and related logging has been moved and enhanced above.

            prompt_ref_level = TREATMENT_REFERENCE_IDENTIFICATION_PROMPT_TEMPLATE.format(
                query=query_text,
                description=dataset_description or 'N/A',
                treatment_variable=treatment_variable,
                treatment_variable_values=treatment_values_sample
            )
            ref_level_result = _call_llm_for_var(llm, prompt_ref_level, LLMTreatmentReferenceLevel)
            if ref_level_result and ref_level_result.reference_level:
                # Validate if the reference level is actually one of the unique values if possible
                # For now, we trust the LLM if it returns a value and it was in the prompt's sample.
                if treatment_values_sample and ref_level_result.reference_level not in treatment_values_sample:
                    logger.warning(f"LLM identified reference level '{ref_level_result.reference_level}' which was not in the provided sample values for '{treatment_variable}'. Using it cautiously.")
                treatment_reference_level = ref_level_result.reference_level
                logger.info(f"LLM identified Treatment Reference Level: {treatment_reference_level} (Reason: {ref_level_result.reasoning})")
            elif ref_level_result:
                 logger.info(f"LLM did not identify a specific treatment reference level (Reason: {ref_level_result.reasoning}).")

        except Exception as e:
            logger.error(f"Error during LLM check for Treatment Reference Level: {e}")

    # --- Identify Interaction Term Suggestion --- 
    interaction_term_suggested = False
    interaction_variable_candidate = None

    if llm and treatment_variable and covariates:
        try:
            # Prepare covariates list with types for the prompt
            covariates_with_types_list = []
            for cov in covariates:
                cov_type = column_categories.get(cov, "Unknown")
                covariates_with_types_list.append(f"{cov}: {cov_type}")
            covariates_list_str = "\n".join([f"- {cwt}" for cwt in covariates_with_types_list])
            if not covariates_list_str:
                covariates_list_str = "No covariates identified or available."

            prompt_interaction = INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE.format(
                query=query_text,
                description=dataset_description or 'N/A',
                treatment_variable=treatment_variable,
                covariates_list_with_types=covariates_list_str
            )
            interaction_result = _call_llm_for_var(llm, prompt_interaction, LLMInteractionSuggestion)

            if interaction_result:
                # Explicitly default to False if LLM omits the field or returns null for interaction_needed
                interaction_term_suggested = interaction_result.interaction_needed if interaction_result.interaction_needed is not None else False
                
                if interaction_term_suggested and interaction_result.interaction_variable:
                    # Validate if the suggested interaction variable is one of the identified covariates
                    if interaction_result.interaction_variable in covariates:
                        interaction_variable_candidate = interaction_result.interaction_variable
                        logger.info(f"LLM suggested interaction: needed={interaction_term_suggested}, variable='{interaction_variable_candidate}' (Reason: {interaction_result.reasoning})")
                    else:
                        logger.warning(f"LLM suggested interaction variable '{interaction_result.interaction_variable}' which is not in the identified covariates list {covariates}. Ignoring suggestion.")
                        interaction_term_suggested = False # Reset if variable is invalid
                elif interaction_term_suggested and not interaction_result.interaction_variable:
                    logger.info(f"LLM suggested interaction is needed but did not specify a variable. (Reason: {interaction_result.reasoning})")
                    # interaction_term_suggested remains True, but interaction_variable_candidate is None
                else:
                    logger.info(f"LLM suggested no interaction is needed. (Reason: {interaction_result.reasoning})") # This covers if interaction_needed was False or None
            else:
                logger.warning("LLM call for interaction term suggestion returned no result.")
                interaction_term_suggested = False # Ensure it's False if call fails

        except Exception as e:
            logger.error(f"Error during LLM check for Interaction Term: {e}")
            interaction_term_suggested = False # Ensure it's False on exception

    # --- Consolidate --- 
    return {
        "treatment_variable": treatment_variable,
        "treatment_variable_type": treatment_variable_type,
        "outcome_variable": outcome_variable,
        "covariates": covariates,
        "time_variable": time_variable,
        "group_variable": group_variable,
        "instrument_variable": instrument_variable,
        "running_variable": running_variable,
        "cutoff_value": cutoff_value,
        "is_rct": is_rct,
        "treatment_reference_level": treatment_reference_level,
        "interaction_term_suggested": interaction_term_suggested,
        "interaction_variable_candidate": interaction_variable_candidate
    }

# --- Helper Functions for Hybrid Identification --- 

def _identify_variable_hybrid(
    role: str, # e.g., "treatment", "outcome"
    query_hints: List[str],
    dataset_suggestions: List[str],
    columns: List[str],
    column_categories: Dict[str, str],
    prioritize_types: List[str], # e.g., ["binary", "continuous_numeric"]
    query_text: str,
    dataset_description: Optional[str],
    llm: Optional[BaseChatModel],
    exclude_vars: Optional[List[str]] = None
) -> Optional[str]:
    """Hybrid logic to identify a single variable (T or O)."""
    
    candidates = set()
    available_columns = [c for c in columns if c not in (exclude_vars or [])]
    if not available_columns:
        return None
        
    # 1. Exact matches from hints
    for hint in query_hints:
        if hint in available_columns:
            candidates.add(hint)
            
    # 2. Add dataset suggestions
    for sugg in dataset_suggestions:
         if sugg in available_columns:
             candidates.add(sugg)
             
    # 3. Programmatic Filtering based on type
    plausible_candidates = [
        c for c in candidates 
        if column_categories.get(c) in prioritize_types
    ]
    
    logger.debug(f"Identifying {role}: Candidates={candidates}, Plausible (by type)={plausible_candidates}")
    
    # 4. Selection Logic
    if len(plausible_candidates) == 1:
        logger.info(f"Selected {role} '{plausible_candidates[0]}' (single plausible candidate)." )
        return plausible_candidates[0]
        
    elif len(plausible_candidates) > 1 and llm:
        # 5. LLM Selection from plausible list
        logger.info(f"Multiple plausible {role} candidates: {plausible_candidates}. Using LLM to select best.")
        prompt = _create_select_best_prompt(role, query_text, dataset_description, plausible_candidates, column_categories)
        llm_choice = _call_llm_for_var(llm, prompt, LLMSelectedVariable)
        if llm_choice and llm_choice.variable_name in plausible_candidates:
             logger.info(f"LLM selected {role}: '{llm_choice.variable_name}' from plausible list.")
             return llm_choice.variable_name
        else:
             logger.warning(f"LLM failed to select valid {role} from {plausible_candidates}. Falling back to first plausible.")
             return plausible_candidates[0]
             
    elif candidates: # No plausible types, but had candidates
        logger.warning(f"No candidates for {role} match priority types {prioritize_types}. Using first candidate '{list(candidates)[0]}'.")
        return list(candidates)[0]
        
    else: # No candidates found initially
        logger.warning(f"No candidates found for {role} from hints or suggestions.")
        # Final fallback: pick first column matching type, excluding others
        fallback_candidates = [
            col for col in available_columns 
            if column_categories.get(col) in prioritize_types
        ]
        if fallback_candidates:
            logger.warning(f"Using first column matching priority type as fallback {role}: '{fallback_candidates[0]}'")
            return fallback_candidates[0]
        else:
             logger.error(f"Could not identify any suitable {role} variable.")
             return None

def _identify_covariates_hybrid(
    treatment_variable: Optional[str],
    outcome_variable: Optional[str],
    columns: List[str],
    column_categories: Dict[str, str],
    query_hints: List[str],
    query_text: str,
    dataset_description: Optional[str],
    llm: Optional[BaseChatModel]
) -> List[str]:
    """Hybrid logic to identify covariates."""
    
    # 1. Initial Programmatic Filtering
    exclude_cols = [treatment_variable, outcome_variable]
    potential_covariates = [col for col in columns if col not in exclude_cols and col is not None]
    
    # Filter out unusable types
    usable_covariates = [
        col for col in potential_covariates 
        if column_categories.get(col) not in ["text_or_other"] # Add other unusable types?
    ]
    logger.debug(f"Initial usable covariates: {usable_covariates}")

    # 2. LLM Refinement (if LLM available)
    if llm:
        logger.info("Using LLM to refine covariate list...")
        prompt = _create_refine_covariates_prompt(query_text, dataset_description, treatment_variable, outcome_variable, usable_covariates, column_categories)
        llm_selection = _call_llm_for_var(llm, prompt, LLMSelectedCovariates)
        
        if llm_selection and llm_selection.covariates:
            # Validate LLM output against available columns
            valid_llm_covs = [c for c in llm_selection.covariates if c in usable_covariates]
            if len(valid_llm_covs) < len(llm_selection.covariates):
                 logger.warning("LLM suggested covariates not found in initial usable list.")
            if valid_llm_covs: # Use LLM selection if it's valid and non-empty
                 logger.info(f"LLM refined covariates to: {valid_llm_covs}")
                 return valid_llm_covs[:10] # Cap at 10
            else:
                 logger.warning("LLM refinement failed or returned empty/invalid list. Falling back.")
        else:
             logger.warning("LLM refinement call failed or returned no covariates. Falling back.")

    # 3. Fallback to Programmatic List (Capped)
    logger.info(f"Using programmatically determined covariates (capped at 10): {usable_covariates[:10]}")
    return usable_covariates[:10]

# --- Helper Functions for LLM Calls & Prompts --- 

def _call_llm_for_var(llm: BaseChatModel, prompt: str, pydantic_model: BaseModel) -> Optional[BaseModel]:
    """Helper to call LLM with structured output and handle errors."""
    try:
        messages = [HumanMessage(content=prompt)]
        structured_llm = llm.with_structured_output(pydantic_model)
        parsed_result = structured_llm.invoke(messages)
        return parsed_result
    except (OutputParserException, ValidationError) as e:
        logger.error(f"LLM call failed parsing/validation for {pydantic_model.__name__}: {e}")
    except Exception as e:
         logger.error(f"LLM call failed unexpectedly for {pydantic_model.__name__}: {e}", exc_info=True)
    return None

def _create_select_best_prompt(role: str, query: str, description: Optional[str], candidates: List[str], categories: Dict[str, str]) -> str:
    """Creates a prompt to ask LLM to select the best variable for a role from a list."""
    candidate_info = "\n".join([f"- '{c}' (Type: {categories.get(c, 'Unknown')})" for c in candidates])
    
    # --- Enhanced Prompt ---
    prompt = f"""
You are an expert causal inference assistant.
User Query: "{query}"
Dataset Description: {description or 'N/A'}

Based on the user query and dataset description which of the following columns is the single most likely **{role} variable**?

Candidate Columns:
{candidate_info}

Respond ONLY with a JSON object containing the selected column name using the key 'variable_name'. If none seem appropriate, return null.
Example: {{\"variable_name\": \"column_name\"}} or {{\"variable_name\": null}}
"""
    return prompt

def _create_refine_covariates_prompt(query: str, description: Optional[str], treatment: str, outcome: str, candidates: List[str], categories: Dict[str, str]) -> str:
    """Creates a prompt for LLM to refine a list of potential covariates."""
    candidate_info = "\n".join([f"- '{c}' (Type: {categories.get(c, 'Unknown')})" for c in candidates])
    
    prompt = f"""
You are a causal inference expert tasked with identifying valid covariates to adjust for confounding in estimating the causal effect of '{treatment}' on '{outcome}'.

User Query: "{query}"
Dataset Description: {description or 'N/A'}

Available Variables (Potential Covariates):
{candidate_info}

Causal Role Definitions:
- **Treatment ('{treatment}')**: The intervention or exposure being studied.
- **Outcome ('{outcome}')**: The target variable affected by the treatment.
- **Covariates / Confounders**: Variables that affect both the treatment and the outcome, measured **before** the treatment. These are critical for adjustment to reduce bias.
- **Effect Modifiers / Interaction Terms**: Variables that are necessary for estimating heterogeneous treatment effects. These should also be included if referenced in the causal query.


Identify the columns that could serve as covariates for a causal inference analysis. Covariates should be variables that might influence the outcome or the treatment assignment, 
excluding the treatment and outcome variables themselves.

Return ONLY a valid JSON object with the selected covariates using this format:
{{ "covariates": ["var1", "var2", ...] }}
"""
    return prompt


def _create_identify_prompt(target: str, query: str, description: Optional[str], columns: List[str], categories: Dict[str,str], treatment: Optional[str], outcome: Optional[str]) -> str:
    """Creates a prompt to ask LLM to identify specific roles like IV, RDD, or RCT by selecting and formatting a specific template."""
    column_info = "\n".join([f"- '{c}' (Type: {categories.get(c, 'Unknown')})" for c in columns])
    
    # Select the appropriate detailed prompt template based on the target
    if "instrumental variable" in target.lower():
        template = IV_IDENTIFICATION_PROMPT_TEMPLATE
    elif "regression discontinuity" in target.lower():
        template = RDD_IDENTIFICATION_PROMPT_TEMPLATE
    elif "rct" in target.lower():
        template = RCT_IDENTIFICATION_PROMPT_TEMPLATE
    else:
        # Fallback or error? For now, let's raise an error if target is unexpected.
        logger.error(f"Unsupported target for _create_identify_prompt: {target}")
        raise ValueError(f"Unsupported target for specific identification prompt: {target}")

    # Format the selected template with the provided context
    prompt = template.format(
        query=query,
        description=description or 'N/A',
        column_info=column_info,
        treatment=treatment or 'N/A',
        outcome=outcome or 'N/A'
    )
    return prompt



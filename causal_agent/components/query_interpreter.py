"""
Query interpreter component for causal inference.

This module provides functionality to match query concepts to actual dataset variables,
identifying treatment, outcome, and covariate variables for causal inference analysis.
"""

import re
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import logging
import numpy as np
from causal_agent.config import get_llm_client

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException

from pydantic import BaseModel, ValidationError
from dowhy import CausalModel
import json

from causal_agent.models import (
    LLMSelectedVariable,
    LLMSelectedCovariates,
    LLMIVars,
    LLMRDDVars,
    LLMRCTCheck,
    LLMTreatmentReferenceLevel,
    LLMInteractionSuggestion, 
    LLMEstimand,
)

from causal_agent.prompts.method_identification_prompts import (
    IV_IDENTIFICATION_PROMPT_TEMPLATE,
    RDD_IDENTIFICATION_PROMPT_TEMPLATE,
    RCT_IDENTIFICATION_PROMPT_TEMPLATE,
    TREATMENT_REFERENCE_IDENTIFICATION_PROMPT_TEMPLATE,
    INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE,
    TREATMENT_VAR_IDENTIFICATION_PROMPT_TEMPLATE, 
    OUTCOME_VAR_IDENTIFICATION_PROMPT_TEMPLATE,
    COVARIATES_IDENTIFICATION_PROMPT_TEMPLATE, 
    ESTIMAND_PROMPT_TEMPLATE,
    CONFOUNDER_IDENTIFICATION_PROMPT_TEMPLATE,
    DID_TERM_IDENTIFICATION_PROMPT_TEMPLATE)





logger = logging.getLogger(__name__)

def infer_treatment_variable_type(treatment_variable: str, column_categories: Dict[str, str],
                                  dataset_analysis: Dict[str, Any]) -> str:
    """
    Determine treatment variable type from column category and unique value count
    Args:
        treatment_variable: name of the treatment variable
        column_categories: mapping of column names to their categories
        dataset_analysis: exploratory analysis results

    Returns:
        str: type of the treatment variable (e.g., "binary", "continuous", etc
    """

    treatment_variable_type = "unknown"
    if treatment_variable and treatment_variable in column_categories:
        category = column_categories[treatment_variable]
        logger.info(f"Category for treatment '{treatment_variable}' is '{category}'.")

        if category == "continuous_numeric":
            treatment_variable_type = "continuous"

        elif category == "discrete_numeric":
            num_unique = dataset_analysis.get("column_nunique_counts", {}).get(treatment_variable, -1)
            if num_unique > 10:
                logger.info(f"'{treatment_variable}' has {num_unique} unique values, treating as continuous.")
                treatment_variable_type = "continuous"
            elif num_unique == 2:
                logger.info(f"'{treatment_variable}' has 2 unique values, treating as binary.")
                treatment_variable_type = "binary"
            elif num_unique > 0:
                logger.info(f"'{treatment_variable}' has {num_unique} unique values, treating as discrete_multi_value.")
                treatment_variable_type = "discrete_multi_value"
            else:
                logger.info(f"'{treatment_variable}' unique value count unknown or too few.")
                treatment_variable_type = "discrete_numeric_unknown_cardinality"

        elif category in ["binary", "binary_categorical"]:
            treatment_variable_type = "binary"

        elif category in ["categorical", "categorical_numeric"]:
            num_unique = dataset_analysis.get("column_nunique_counts", {}).get(treatment_variable, -1)
            if num_unique == 2:
                treatment_variable_type = "binary"
            elif num_unique > 0:
                treatment_variable_type = "categorical_multi_value"
            else:
                treatment_variable_type = "categorical_unknown_cardinality"

        else:
            logger.warning(f"Unmapped category '{category}' for '{treatment_variable}', setting as 'other'.")
            treatment_variable_type = "other"

    elif treatment_variable:
        logger.warning(f"'{treatment_variable}' not found in column_categories.")
    else:
        logger.info("No treatment variable identified.")

    logger.info(f"Final Determined Treatment Variable Type: {treatment_variable_type}")
    return treatment_variable_type

def determine_treatment_reference_level(is_rct: Optional[bool], llm: Optional[BaseChatModel], treatment_variable: Optional[str], 
                                      query_text: str, dataset_description: Optional[str], file_path: Optional[str], 
                                      columns: List[str]) -> Optional[str]:
    """
    Determines the treatment reference level
    """

    
    if is_rct is None: is_rct = False
    treatment_reference_level = None

    if llm and treatment_variable and treatment_variable in columns:
        treatment_values_sample = []
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if treatment_variable in df.columns:
                    unique_vals = df[treatment_variable].unique()
                    treatment_values_sample = [item.item() if hasattr(item, 'item') else item for item in unique_vals][:10]
                    if treatment_values_sample:
                        logger.info(f"Successfully read treatment values sample from dataset at '{file_path}' for variable '{treatment_variable}'.")
                    else:
                        logger.info(f"'{treatment_variable}' in '{file_path}' has no unique values or is empty.")
                else:
                    logger.warning(f"'{treatment_variable}' not found in dataset columns at '{file_path}'.")
            except FileNotFoundError:
                logger.warning(f"File not found at: {file_path}")
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty file at: {file_path}")
            except Exception as e:
                logger.warning(f"Error reading dataset at '{file_path}' for '{treatment_variable}': {e}")

        if not treatment_values_sample:
            logger.warning(f"No unique values found for treatment '{treatment_variable}'. LLM prompt will receive empty list.")
        else:
            logger.info(f"Final treatment values sample: {treatment_values_sample}")

        try:
            prompt = TREATMENT_REFERENCE_IDENTIFICATION_PROMPT_TEMPLATE.format(query=query_text, description=dataset_description or 'N/A', treatment_variable=treatment_variable, treatment_variable_values=treatment_values_sample)
            ref_result = _call_llm_for_var(llm, prompt, LLMTreatmentReferenceLevel)
            if ref_result and ref_result.reference_level:
                if treatment_values_sample and ref_result.reference_level not in treatment_values_sample:
                    logger.warning(f"LLM reference level '{ref_result.reference_level}' not in sampled values for '{treatment_variable}'.")
                treatment_reference_level = ref_result.reference_level
                logger.info(f"LLM identified reference level: {treatment_reference_level} (Reason: {ref_result.reasoning})")
            elif ref_result:
                logger.info(f"LLM returned no reference level (Reason: {ref_result.reasoning})")
        except Exception as e:
            logger.error(f"LLM error for treatment reference level: {e}")

    return treatment_reference_level

def identify_interaction_term(llm: Optional[BaseChatModel], treatment_variable: Optional[str], covariates: List[str],
                              column_categories: Dict[str, str], query_text: str, 
                              dataset_description: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Identifies the interaction term based on the query and the dataset information
    """

    interaction_term_suggested, interaction_variable_candidate = False, None
    
    if llm and treatment_variable and covariates:
        try:
            covariates_list_str = "\n".join([f"- {cov}: {column_categories.get(cov, 'Unknown')}" for cov in covariates]) or "No covariates identified or available."
            prompt = INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE.format(query=query_text, description=dataset_description or 'N/A', treatment_variable=treatment_variable, covariates_list_with_types=covariates_list_str)
            result = _call_llm_for_var(llm, prompt, LLMInteractionSuggestion)
            if result:
                interaction_term_suggested = result.interaction_needed if result.interaction_needed is not None else False
                if interaction_term_suggested and result.interaction_variable:
                    if result.interaction_variable in covariates:
                        interaction_variable_candidate = result.interaction_variable
                        logger.info(f"LLM suggested interaction: needed={interaction_term_suggested}, variable='{interaction_variable_candidate}' (Reason: {result.reasoning})")
                    else:
                        logger.warning(f"LLM suggested variable '{result.interaction_variable}' not in covariates {covariates}. Ignoring.")
                        interaction_term_suggested = False
                elif interaction_term_suggested:
                    logger.info(f"LLM suggested interaction is needed but no variable provided (Reason: {result.reasoning})")
                else:
                    logger.info(f"LLM suggested no interaction is needed (Reason: {result.reasoning})")
            else:
                logger.warning("LLM returned no result for interaction term suggestion.")
        except Exception as e:
            logger.error(f"LLM error during interaction term check: {e}")

    return interaction_term_suggested, interaction_variable_candidate


def interpret_query(query_info: Dict[str, Any], dataset_analysis: Dict[str, Any],
                    dataset_description: Optional[str] = None) -> Dict[str, Any]:
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
    treatment_variable = _identify_variable_hybrid(role="treatment", query_hints=treatment_hints, 
                                                   dataset_suggestions=dataset_treatments, columns=columns,
                                                   column_categories=column_categories,
                                                   prioritize_types=["binary", "binary_categorical", "discrete_numeric","continuous_numeric"], # Prioritize binary/discrete
                                                   query_text=query_text, dataset_description=dataset_description,llm=llm)
    logger.info(f"Identified Treatment: {treatment_variable}")
    treatment_variable_type = infer_treatment_variable_type(treatment_variable, column_categories, dataset_analysis)

    
    # --- Identify Outcome --- 
    outcome_hints = query_info.get("outcome_hints", [])
    dataset_outcomes = dataset_analysis.get("potential_outcomes", [])
    outcome_variable = _identify_variable_hybrid(role="outcome", query_hints=outcome_hints, dataset_suggestions=dataset_outcomes,
                                                 columns=columns, column_categories=column_categories,
                                                 prioritize_types=["continuous_numeric", "discrete_numeric"], # Prioritize numeric
                                                 exclude_vars=[treatment_variable], # Exclude treatment
                                                 query_text=query_text, dataset_description=dataset_description, llm=llm)
    logger.info(f"Identified Outcome: {outcome_variable}")

    # --- Identify Covariates --- 
    covariate_hints = query_info.get("covariates_hints", [])
    covariates = _identify_covariates_hybrid("covars", treatment_variable=treatment_variable, outcome_variable=outcome_variable,
                                             columns=columns, column_categories=column_categories, query_hints=covariate_hints,
                                             query_text=query_text, dataset_description=dataset_description, llm=llm)
    logger.info(f"Identified Covariates: {covariates}")

    # --- Identify Confounders ---
    confounder_hints = query_info.get("covariates_hints", [])
    confounders = _identify_covariates_hybrid("confounders", treatment_variable=treatment_variable, outcome_variable=outcome_variable,
                                              columns=columns, column_categories=column_categories, query_hints=confounder_hints,
                                              query_text=query_text, dataset_description=dataset_description, llm=llm)
    logger.info(f"Identified Confounders: {confounders}")

    # --- Identify Time/Group (from dataset analysis) --- 
    time_variable = None
    group_variable = None
    has_temporal = dataset_analysis.get("temporal_structure", {}).get("has_temporal_structure", False)
    temporal_structure = dataset_analysis.get("temporal_structure", {})
    if temporal_structure.get("has_temporal_structure", False):
        time_variable = temporal_structure.get("time_column") or temporal_structure.get("temporal_columns", [None])[0]
        if temporal_structure.get("is_panel_data", False):
            group_variable = temporal_structure.get("id_column")
    logger.info(f"Identified Time Var: {time_variable}, Group Var: {group_variable}, temporal structure: {temporal_structure}")

    # --- Identify IV/RDD/RCT using LLM --- 
    instrument_variable = None
    running_variable = None
    cutoff_value = None
    is_rct = None
    smd_score = None

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
            instrument_variable = iv_result.instrument_variable if iv_result else None
            if instrument_variable not in columns:
                instrument_variable = None  
            logger.info(f"LLM identified IV: {instrument_variable}")

            # Check for RDD
            prompt_rdd = _create_identify_prompt("regression discontinuity (running variable and cutoff)", query_text, dataset_description, columns, column_categories, treatment_variable, outcome_variable)
            rdd_result = _call_llm_for_var(llm, prompt_rdd, LLMRDDVars)
            if rdd_result:
                running_variable = rdd_result.running_variable
                cutoff_value = rdd_result.cutoff_value
            if running_variable not in columns or cutoff_value is None:
                running_variable = None
                cutoff_value = None
            logger.info(f"LLM identified RDD: Running={running_variable}, Cutoff={cutoff_value}")

            ## For graph based methods 
            exclude_cols = [treatment_variable, outcome_variable]
            potential_covariates = [col for col in columns if col not in exclude_cols and col is not None]
            usable_covariates = [col for col in potential_covariates if column_categories.get(col) not in ["text_or_other"]]
            logger.info(f"Usable covariates for graph: {usable_covariates}")
  
            estimand_prompt = ESTIMAND_PROMPT_TEMPLATE.format(query=query_text,dataset_description=dataset_description,
                                                               dataset_columns=usable_covariates,
                                                               treatment=treatment_variable, outcome=outcome_variable)

            estimand_result = _call_llm_for_var(llm, estimand_prompt, LLMEstimand)
            estimand = "ate" if "ate" in estimand_result.estimand.strip().lower() else "att"
            logger.info(f"LLM identified estimand: {estimand}")

            ## Did Term  
            did_term_prompt = DID_TERM_IDENTIFICATION_PROMPT_TEMPLATE.format(query=query_text, description=dataset_description,
                                                                             column_info=columns, time_variable=time_variable,
                                                                             group_variable=group_variable, column_types=column_categories)
            did_term_result = _call_llm_for_var(llm, did_term_prompt, LLMRDDVars)
            did_term_result = did_term_result.did_term if did_term_result in columns else None
            logger.info(f"LLM identified DiD term: {did_term_result}")



            #smd_score_all = compute_smd(dataset_analysis.get("data", pd.DataFrame()), treatment_variable, usable_covariates)
            #smd_score = smd_score_all.get("ate", 0.0) if smd_score_all else 0.0
            #logger.info(f"Computed SMD score: {smd_score}")

            #logger.debug(f"Computed SMD score for {estimand}: {smd_score}")


        except Exception as e:
            logger.error(f"Error during LLM checks for IV/RDD/RCT: {e}")
            


    # --- Identify Treatment Reference Level --- 
    treatment_reference_level = determine_treatment_reference_level(is_rct=is_rct, llm=llm, treatment_variable=treatment_variable,
                                                                    query_text=query_text, dataset_description=dataset_description, 
                                                                    file_path=file_path, columns=columns)

    # --- Identify Interaction Term Suggestion --- 
    interaction_term_suggested, interaction_variable_candidate = identify_interaction_term(llm=llm, treatment_variable=treatment_variable, 
                                                                                           covariates=covariates,
                                                                                           column_categories=column_categories, query_text=query_text, 
                                                                                           dataset_description=dataset_description)
    

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
        "interaction_variable_candidate": interaction_variable_candidate, 
        "confounders": confounders,
        "did_term": did_term_result
    }

def compute_smd(df: pd.DataFrame, treat, covars_list) -> Dict[str, float]:
    """
    Computed the standardized mean differences (SMD) for the treatment variable
    Args:
        df (pd.DataFrame): The dataset.
        treat (str): Name of the binary treatment column (0/1).
        covars_list (List[str]): List of covariate names to consider for SMD calculation

    Returns:
        Dict{str ->float}: the standardized mean difference (SMD)
    """
    logger.info(f"Computing SMD for treatment variable '{treat}' with covariates: {covars_list}")
    df_t = df[df[treat] == 1]
    df_c = df[df[treat] == 0]

    covariates = covars_list if covars_list else df.columns.tolist()
    smd_ate = np.zeros(len(covariates))
    smd_att = np.zeros(len(covariates))

    for i, col in enumerate(covariates):
        try:
            m_t, m_c = df_t[col].mean(), df_c[col].mean()
            s_t, s_c = df_t[col].std(ddof=0), df_c[col].std(ddof=0)
            pooled = np.sqrt((s_t**2 + s_c**2) / 2)

            ate_val = 0.0 if pooled == 0 else (m_t - m_c) / pooled
            att_val = 0.0 if s_t == 0 else (m_t - m_c) / s_t

            smd_ate.append(ate_val)
            smd_att.append(att_val)
        except Exception as e:
            logger.warning(f"SMD computation failed for column '{col}': {e}")
            continue

    avg_ate = np.nanmean(np.abs(smd_ate))
    avg_att = np.nanmean(np.abs(smd_att))

    return {"ate":avg_ate, "att":avg_att}



# --- Helper Functions for Hybrid Identification --- 
def _identify_variable_hybrid(role: str, query_hints: List[str], dataset_suggestions: List[str],
                               columns: List[str], column_categories: Dict[str, str],
                               prioritize_types: List[str], query_text: str,
                               dataset_description: Optional[str],llm: Optional[BaseChatModel],
                               exclude_vars: Optional[List[str]] = None) -> Optional[str]:
    """
    Used to identify a variable from the avaiable information by prompting the LLM. In case of failure, 
    it will fallback to a programmatic selection (heuristics)

    Args:
        role: variable type (treatment or outcome)
        query_hints: hints from the query for this variable
        dataset_suggestions: dataset-specific suggestions for this variable
        columns: list of available columns in the dataset
        column_categories: mapping of column names to their categories
        prioritize_types: types to prioritize for this variable
        query_text: the original query text
        dataset_description: description of the dataset   
        llm: language model 
        exclude_vars: list of variables to exclude from selection (e.g., treatment for outcome)
    Returns:
        str: name of the identified variable, or None if not found
    """

    candidates = set()
    available_columns = [c for c in columns if c not in (exclude_vars or [])]
    if not available_columns: return None

    # 1. Exact matches from hints
    for hint in query_hints:
        if hint in available_columns:
            candidates.add(hint)
    # 2. Add dataset suggestions
    for sugg in dataset_suggestions:
        if sugg in available_columns:
            candidates.add(sugg)

    # 3. Programmatic Filtering based on type
    plausible_candidates = [c for c in candidates if column_categories.get(c) in prioritize_types]

    if llm:
        if role == "treatment":
            prompt_template = TREATMENT_VAR_IDENTIFICATION_PROMPT_TEMPLATE
        elif role == "outcome":
            prompt_template = OUTCOME_VAR_IDENTIFICATION_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Unsupported role for LLM variable identification: {role}")

        prompt = prompt_template.format(query=query_text, description=dataset_description,
                                        column_info=available_columns)
        llm_choice = _call_llm_for_var(llm, prompt, LLMSelectedVariable)

        if llm_choice and llm_choice.variable_name in available_columns:
            logger.info(f"LLM selected {role}: {llm_choice.variable_name}")
            return llm_choice.variable_name
        else:
            fallback = plausible_candidates[0] if plausible_candidates else None
            logger.warning(f"LLM failed to select valid {role}. Falling back to: {fallback}")
            return fallback

    if plausible_candidates:
        logger.info(f"No LLM provided. Using first plausible {role}: {plausible_candidates[0]}")
        return plausible_candidates[0]

    logger.warning(f"No plausible candidates for {role}. Cannot identify variable.")
    return None


def _identify_covariates_hybrid(role, treatment_variable: Optional[str], outcome_variable: Optional[str],
                                columns: List[str], column_categories: Dict[str, str], query_hints: List[str], 
                                query_text: str, dataset_description: Optional[str], llm: Optional[BaseChatModel]) -> List[str]:
    """
    Prompts an LLM to identify the covariates
    """
    
    # 1. Initial Programmatic Filtering
    exclude_cols = [treatment_variable, outcome_variable]
    potential_covariates = [col for col in columns if col not in exclude_cols and col is not None]
    
    # Filter out unusable types
    usable_covariates = [col for col in potential_covariates if column_categories.get(col) not in ["text_or_other"]]
    logger.debug(f"Initial usable covariates: {usable_covariates}")

    # 2. LLM Refinement (if LLM available)
    if llm:
        logger.info("Using LLM to refine covariate list...")
        prompt = ""
        if role == "covars":
            prompt = COVARIATES_IDENTIFICATION_PROMPT_TEMPLATE.format("covars", query=query_text, description=dataset_description, 
                                                                 column_info=", ".join(usable_covariates), 
                                                                 treatment=treatment_variable, outcome=outcome_variable)
        elif role == "confounders":
            prompt = CONFOUNDER_IDENTIFICATION_PROMPT_TEMPLATE.format(query=query_text, description=dataset_description, 
                                                       column_info=", ".join(usable_covariates), 
                                                       treatment=treatment_variable, outcome=outcome_variable)
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

def _create_identify_prompt(target: str, query: str, description: Optional[str], columns: List[str], 
                            categories: Dict[str,str], treatment: Optional[str], outcome: Optional[str]) -> str:
    """
    Creates a prompt to ask LLM to identify specific roles like IV, RDD, or RCT by selecting and formatting a specific template
    """
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
    prompt = template.format(query=query, description=description or 'N/A', column_info=column_info,
                             treatment=treatment or 'N/A', outcome=outcome or 'N/A')
    return prompt

def _call_llm_for_var(llm: BaseChatModel, prompt: str, pydantic_model: BaseModel) -> Optional[BaseModel]:
    """Helper to call LLM with structured output and handle errors."""
    try:
        messages = [HumanMessage(content=prompt)]
        # Use function_calling method to avoid json_schema compatibility issues with older models
        structured_llm = llm.with_structured_output(pydantic_model, method='function_calling')
        parsed_result = structured_llm.invoke(messages)
        return parsed_result
    except (OutputParserException, ValidationError) as e:
        logger.error(f"LLM call failed parsing/validation for {pydantic_model.__name__}: {e}")
    except Exception as e:
         logger.error(f"LLM call failed unexpectedly for {pydantic_model.__name__}: {e}", exc_info=True)
    return None

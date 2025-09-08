"""
Linear Regression Estimator for Causal Inference.

Uses Ordinary Least Squares (OLS) to estimate the treatment effect, potentially
adjusting for covariates.
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Any, List, Optional, Union
import logging
from langchain.chat_models.base import BaseChatModel
import re
import json
from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException


from causal_agent.models import LLMIdentifiedRelevantParams
from causal_agent.prompts.regression_prompts import STATSMODELS_PARAMS_IDENTIFICATION_PROMPT_TEMPLATE
from causal_agent.config import get_llm_client

# Placeholder for potential future LLM assistance integration
# from .llm_assist import interpret_lr_results, suggest_lr_covariates
# Placeholder for potential future diagnostics integration
# from .diagnostics import run_lr_diagnostics

logger = logging.getLogger(__name__)

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

# Define module-level helper function
def _clean_variable_name_for_patsy_local(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = 'var_' + name
    return name


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[List[str]] = None,
    query_str: Optional[str] = None, # For potential LLM use
    llm: Optional[BaseChatModel] = None, # For potential LLM use
    **kwargs # To capture any other potential arguments
) -> Dict[str, Any]:
    """
    Estimates the causal effect using Linear Regression (OLS).

    Args:
        df: Input DataFrame.
        treatment: Name of the treatment variable column.
        outcome: Name of the outcome variable column.
        covariates: Optional list of covariate names.
        query_str: Optional user query for context (e.g., for LLM).
        llm: Optional Language Model instance.
        **kwargs: Additional keyword arguments.

    Returns:
        Dictionary containing estimation results:
        - 'effect_estimate': The estimated coefficient for the treatment variable.
        - 'p_value': The p-value associated with the treatment coefficient.
        - 'confidence_interval': The 95% confidence interval for the effect.
        - 'standard_error': The standard error of the treatment coefficient.
        - 'formula': The regression formula used.
        - 'model_summary': Summary object from statsmodels.
        - 'diagnostics': Placeholder for diagnostic results.
        - 'interpretation': Placeholder for LLM interpretation.
    """
    if covariates is None:
        covariates = []

    # Retrieve additional args from kwargs
    interaction_term_suggested = kwargs.get('interaction_term_suggested', False)
    # interaction_variable_candidate is the *original* name from query_interpreter
    interaction_variable_candidate_orig_name = kwargs.get('interaction_variable_candidate')
    treatment_reference_level = kwargs.get('treatment_reference_level')
    column_mappings = kwargs.get('column_mappings', {})

    required_cols = [treatment, outcome] + covariates
    # If interaction variable is suggested, ensure it (or its processed form) is in df for analysis
    # This check is complex here as interaction_variable_candidate_orig_name needs mapping to processed column(s)
    # We'll rely on df_analysis.dropna() and formula construction to handle missing interaction var columns later

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare data for statsmodels (add constant, handle potential NaNs)
    df_analysis = df[required_cols].dropna()
    if df_analysis.empty:
        raise ValueError("No data remaining after dropping NaNs for required columns.")
        
    X = df_analysis[[treatment] + covariates]
    X = sm.add_constant(X) # Add intercept
    y = df_analysis[outcome]

    # --- Formula Construction --- 
    outcome_col_name = outcome # Name in processed df
    treatment_col_name = treatment # Name in processed df
    processed_covariate_col_names = covariates # List of names in processed df

    rhs_terms = []

    # 1. Treatment Term
    treatment_patsy_term = treatment_col_name # Default
    original_treatment_info = column_mappings.get(treatment_col_name, {}) # Info from preprocess_data

    is_binary_encoded = original_treatment_info.get('transformed_as') == 'label_encoded_binary'
    is_still_categorical_in_df = df_analysis[treatment_col_name].dtype.name in ['object', 'category'] 

    if is_still_categorical_in_df and not is_binary_encoded: # Covers multi-level and binary categoricals not yet numeric
        if treatment_reference_level:
            treatment_patsy_term = f"C({treatment_col_name}, Treatment(reference='{treatment_reference_level}'))"
            logger.info(f"Treating '{treatment_col_name}' as multi-level categorical with reference '{treatment_reference_level}'.")
        else:
            # Default C() wrapping for categoricals if no specific reference is given.
            # This applies to multi-level or binary categoricals that were not label_encoded to 0/1 by preprocess_data.
            treatment_patsy_term = f"C({treatment_col_name})"
            logger.info(f"Treating '{treatment_col_name}' as categorical (Patsy will pick reference).")
    elif is_binary_encoded: # Was binary and explicitly label encoded to 0/1 by preprocess_data
        # Even if it's now numeric 0/1, C() ensures Patsy treats it categorically for parameter naming consistency.
        treatment_patsy_term = f"C({treatment_col_name})"
        logger.info(f"Treating label-encoded binary '{treatment_col_name}' as categorical for Patsy.")
    else: # Assumed to be already numeric (continuous or discrete numeric not needing C() for main effect)
        # treatment_patsy_term remains treatment_col_name (default)
        logger.info(f"Treating '{treatment_col_name}' as numeric for Patsy formula.")
    
    rhs_terms.append(treatment_patsy_term)

    # 2. Covariate Terms
    for cov_col_name in processed_covariate_col_names:
        if cov_col_name == treatment_col_name: # Should not happen if covariates list is clean
            continue 
        # Assume covariates are already numeric/dummy. If one was object/category in df_analysis (unlikely), C() it.
        if df_analysis[cov_col_name].dtype.name in ['object', 'category']:
            rhs_terms.append(f"C({cov_col_name})")
        else:
            rhs_terms.append(cov_col_name)
    
    # 3. Interaction Term (Simplified: interaction_variable_candidate_orig_name must map to a single column in df_analysis)
    actual_interaction_term_added_to_formula = None
    if interaction_term_suggested and interaction_variable_candidate_orig_name:
        processed_interaction_col_name = None
        interaction_var_info = column_mappings.get(interaction_variable_candidate_orig_name, {})

        if interaction_var_info.get('transformed_as') == 'one_hot_encoded':
            logger.warning(f"Interaction with one-hot encoded variable '{interaction_variable_candidate_orig_name}' is complex. Currently skipping this interaction for Linear Regression.")
        elif interaction_var_info.get('new_column_name') and interaction_var_info['new_column_name'] in df_analysis.columns:
            processed_interaction_col_name = interaction_var_info['new_column_name']
        elif interaction_variable_candidate_orig_name in df_analysis.columns: # Was not in mappings, or mapping didn't change name (e.g. numeric)
            processed_interaction_col_name = interaction_variable_candidate_orig_name
        
        if processed_interaction_col_name:
            interaction_var_patsy_term = processed_interaction_col_name
            # If the processed interaction column itself is categorical (e.g. label encoded binary)
            if df_analysis[processed_interaction_col_name].dtype.name in ['object', 'category', 'bool'] or \
               interaction_var_info.get('original_dtype') in ['bool', 'category']:
                interaction_var_patsy_term = f"C({processed_interaction_col_name})"
            
            actual_interaction_term_added_to_formula = f"{treatment_patsy_term}:{interaction_var_patsy_term}"
            rhs_terms.append(actual_interaction_term_added_to_formula)
            logger.info(f"Adding interaction term to formula: {actual_interaction_term_added_to_formula}")
        elif interaction_variable_candidate_orig_name: # Log if it was suggested but couldn't be mapped/found
            logger.warning(f"Could not resolve interaction variable candidate '{interaction_variable_candidate_orig_name}' to a single usable column in processed data. Skipping interaction term.")

    # Build the formula string for reporting and fitting
    if not rhs_terms: # Should always have at least treatment
        formula = f"{outcome_col_name} ~ 1"
    else:
        formula = f"{outcome_col_name} ~ {' + '.join(rhs_terms)}"
    logger.info(f"Using formula for Linear Regression: {formula}")

    try:
        model = smf.ols(formula=formula, data=df_analysis)
        results = model.fit()
        logger.info("OLS model fitted successfully.")
        logger.info(results.summary()) # Changed to debug level for less verbose default logging

        # --- Result Extraction: LLM attempt first, then Regex fallback ---
        effect_estimates_by_level = {}
        all_params_extracted = False # Default to False
        llm_extraction_successful = False

        # Attempt LLM-based extraction if llm client and query are available
        llm = get_llm_client()
        if llm and query_str:
            logger.info(f"Attempting LLM-based result extraction (informed by query: '{query_str[:50]}...').")
            try:
                param_names_list = results.params.index.tolist()
                param_estimates_list = results.params.tolist()
                param_p_values_list = results.pvalues.tolist()
                param_std_errs_list = results.bse.tolist()
                
                conf_int_df = results.conf_int(alpha=0.05)
                param_conf_ints_low_list = []
                param_conf_ints_high_list = []

                if not conf_int_df.empty and len(conf_int_df.columns) == 2:
                    aligned_conf_int_df = conf_int_df.reindex(results.params.index)
                    param_conf_ints_low_list = aligned_conf_int_df.iloc[:, 0].fillna(float('nan')).tolist()
                    param_conf_ints_high_list = aligned_conf_int_df.iloc[:, 1].fillna(float('nan')).tolist()
                else:
                    nan_list_ci = [float('nan')] * len(param_names_list)
                    param_conf_ints_low_list = nan_list_ci
                    param_conf_ints_high_list = nan_list_ci

                # Placeholder for the new prompt template tailored for this extraction task
                # MOVED TO causalscientist/cais/prompts/regression_prompts.py

                is_multilevel_case_for_prompt = bool(treatment_reference_level and is_still_categorical_in_df and not is_binary_encoded)
                reference_level_for_prompt_str = str(treatment_reference_level) if is_multilevel_case_for_prompt else "N/A"
                
                indexed_param_names_for_prompt = [f"{idx}: '{name}'" for idx, name in enumerate(param_names_list)]
                indexed_param_names_str_for_prompt = "\n".join(indexed_param_names_for_prompt)

                prompt_text_for_identification = STATSMODELS_PARAMS_IDENTIFICATION_PROMPT_TEMPLATE.format(
                    user_query=query_str,
                    treatment_patsy_term=treatment_patsy_term,
                    treatment_col_name=treatment_col_name,
                    is_multilevel_case=is_multilevel_case_for_prompt,
                    reference_level_for_prompt=reference_level_for_prompt_str,
                    indexed_param_names_str=indexed_param_names_str_for_prompt, # Pass the indexed list as a string
                    llm_response_schema_json=json.dumps(LLMIdentifiedRelevantParams.model_json_schema(), indent=2)
                )
                
                llm_identification_response = _call_llm_for_var(llm, prompt_text_for_identification, LLMIdentifiedRelevantParams)

                if llm_identification_response and llm_identification_response.identified_params:
                    logger.info("LLM identified relevant parameters. Proceeding with programmatic extraction.")
                    for item in llm_identification_response.identified_params:
                        param_idx = item.param_index
                        # Validate index against actual list length
                        if 0 <= param_idx < len(results.params.index):
                            actual_param_name = results.params.index[param_idx]
                            # Sanity check if LLM returned name matches actual name at index
                            if item.param_name != actual_param_name:
                                logger.warning(f"LLM returned param_name '{item.param_name}' but name at index {param_idx} is '{actual_param_name}'. Using actual name from results.")
                            
                            current_effect_stats = {
                                'estimate': results.params.iloc[param_idx],
                                'p_value': results.pvalues.iloc[param_idx],
                                'conf_int': results.conf_int(alpha=0.05).iloc[param_idx].tolist(),
                                'std_err': results.bse.iloc[param_idx]
                            }

                            key_for_effect_dict = 'treatment_effect' # Default for single/binary
                            if is_multilevel_case_for_prompt: # If it was a multi-level case
                                match = re.search(r'\[T\.([^]]+)]', actual_param_name) # Use actual_param_name
                                if match:
                                    level = match.group(1)
                                    if level != reference_level_for_prompt_str: # Ensure it's not the ref level itself
                                        key_for_effect_dict = level
                                else:
                                    logger.warning(f"Could not parse level from LLM-identified param: {actual_param_name}. Storing with raw name.")
                                    key_for_effect_dict = actual_param_name # Fallback key
                            
                            effect_estimates_by_level[key_for_effect_dict] = current_effect_stats
                        else:
                            logger.warning(f"LLM returned an invalid parameter index: {param_idx}. Skipping.")
                    
                    if effect_estimates_by_level: # If any effects were successfully processed
                        all_params_extracted = llm_identification_response.all_parameters_successfully_identified
                        llm_extraction_successful = True
                        logger.info(f"Successfully processed LLM-identified parameters. all_parameters_successfully_identified={all_params_extracted}")
                        print(f"effect_estimates_by_level: {effect_estimates_by_level}")
                    else:
                        logger.warning("LLM identified parameters, but none could be processed into effects_estimates_by_level. Falling back to regex.")
                else:
                    logger.warning("LLM parameter identification did not yield usable parameters. Falling back to regex.")
            
            except Exception as e_llm:
                logger.warning(f"LLM-based result extraction failed: {e_llm}. Falling back to regex.", exc_info=True)
        
        
            # --- End of Existing Regex Logic Block ---

        # Primary effect_estimate for simple reporting (e.g. first level or the only one)
        # For multi-level, this is ambiguous. For now, let's report None or the first one.
        # The full details are in effect_estimates_by_level.
        main_effect_estimate = None
        main_p_value = None
        main_conf_int = [None, None] # Default for single or if no effects
        main_std_err = None

        if effect_estimates_by_level:
            if 'treatment_effect' in effect_estimates_by_level: # Single effect case
                single_effect_data = effect_estimates_by_level['treatment_effect']
                main_effect_estimate = single_effect_data['estimate']
                main_p_value = single_effect_data['p_value']
                main_conf_int = single_effect_data['conf_int']
                main_std_err = single_effect_data['std_err']
            else: # Multi-level case
                logger.info("Multi-level treatment effects extracted. Populating dicts for main estimate fields.")
                effect_estimate_dict = {}
                p_value_dict = {}
                conf_int_dict = {}
                std_err_dict = {}
                for level, stats in effect_estimates_by_level.items():
                    effect_estimate_dict[level] = stats.get('estimate')
                    p_value_dict[level] = stats.get('p_value')
                    conf_int_dict[level] = stats.get('conf_int') # This is already a list [low, high]
                    std_err_dict[level] = stats.get('std_err')
                
                main_effect_estimate = effect_estimate_dict
                main_p_value = p_value_dict
                main_conf_int = conf_int_dict
                main_std_err = std_err_dict

        interpretation_details = {}
        if actual_interaction_term_added_to_formula and actual_interaction_term_added_to_formula in results.params.index:
            interpretation_details['interaction_term_coefficient'] = results.params[actual_interaction_term_added_to_formula]
            interpretation_details['interaction_term_p_value'] = results.pvalues[actual_interaction_term_added_to_formula]
            logger.info(f"Interaction term '{actual_interaction_term_added_to_formula}' coeff: {interpretation_details['interaction_term_coefficient']}")

        diag_results = {} 
        interpretation = "Interpretation not available." 

        output_dict = {
            'effect_estimate': main_effect_estimate,
            'p_value': main_p_value,
            'confidence_interval': main_conf_int,
            'standard_error': main_std_err,
            'estimated_effects_by_level': effect_estimates_by_level if (treatment_reference_level and is_still_categorical_in_df and not is_binary_encoded and effect_estimates_by_level) else None,
            'reference_level_used': treatment_reference_level if (treatment_reference_level and is_still_categorical_in_df and not is_binary_encoded) else None,
            'formula': formula,
            'model_summary_text': results.summary().as_text(), # Store as text for easier serialization
            'diagnostics': diag_results,
            'interpretation_details': interpretation_details, # Added interaction details
            'interpretation': interpretation,
            'method_used': 'Linear Regression (OLS)'
        }
        if not all_params_extracted:
            output_dict['warnings'] = ["Could not reliably extract all requested parameters from model results. Please check model_summary_text."]
        return output_dict

    except Exception as e:
        logger.error(f"Linear Regression failed: {e}")
        raise # Re-raise the exception after logging 
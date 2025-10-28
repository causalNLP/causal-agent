## This file contains the program selecting the control variables for the causal estimation model. 
## This should be added after the method selection step

from cais.components.query_interpreter import _identify_covariates_hybrid 
from cais.prompts.method_identification_prompts import CONTROLS_IDENTIFICATION_PROMPT_TEMPLATE
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import OutputParserException

from cais.models import LLMSelectedCovariates

import logging

logger = logging.getLogger(__name__)

def _call_llm_for_var(llm: BaseChatModel, prompt: str, pydantic_model: BaseModel) -> Optional[BaseModel]:
    """Helper to call LLM with structured output and handle errors."""
    try:
        messages = [HumanMessage(content=prompt)]
        structured_llm = llm.with_structured_output(pydantic_model)
        parsed_result = structured_llm.invoke(messages)
        print(f"LLM parsed result: {parsed_result}")
        return parsed_result
    except (OutputParserException, ValidationError) as e:
        logger.error(f"LLM call failed parsing/validation for {pydantic_model.__name__}: {e}")
    except Exception as e:
         logger.error(f"LLM call failed unexpectedly for {pydantic_model.__name__}: {e}", exc_info=True)
    return None


## based on the _identify_covariates_hybrid function in query_interpreter.py. 
## This needs to happen after method selection
def identify_controls(treatment_variable: str, outcome_variable:str, method_name: str,
                      columns: List[str], column_categories: Dict[str, str], instrument: Optional[str], 
                      running_variable: Optional[str], time_variable: Optional[str], state_variable: Optional[str],
                      query_text: str, dataset_description: Optional[str], 
                      llm: Optional[BaseChatModel]) -> List[str]:
    """
    Identify the control variables for the causal estimation model. 
    Preference is given to LLM-based selection, but if that fails, we use heuristic selection.

    Args:
        treatment_variable (str): the treatment variable
        outcome_variable (str): the outcome variable
        method_name (str): the causal inference method selected
        columns (List[str]): list of the column candidates
        column_categories (Dict[str, str]): dictionary of column categories
        instrument (Optional[str]): the instrument variable, if selected method is IV
        running_variable (Optional[str]): the running variable, if selected method is RDD
        time_variable (Optional[str]): the time variable, if selected method is DiD
        state_variable (Optional[str]): the state variable, if selected method is DiD
        query_text (str): the original user query
        dataset_description (Optional[str]): the dataset description
        llm (Optional[BaseChatModel]): the language model to use for selection
    """
    
    ## exclude pre-selected variables, including treatment, outcome, and model-specific variables
    exclude_cols = [treatment_variable, outcome_variable, instrument, running_variable, time_variable, state_variable]
    potential_controls = [col for col in columns if col not in exclude_cols and col is not None]
    
    # Filter out unusable types
    usable_controls = [col for col in potential_controls if column_categories.get(col) not in ["text_or_other"]]
    logger.debug(f"Initial usable covariates: {usable_controls}")

    usable_control_categories = {col: column_categories.get(col, "") for col in usable_controls}

    # 2. Use LLM 
    if llm:
        print("Using LLM to refine covariate list for controls selection")
        print("Method Name: ", method_name)
        prompt = CONTROLS_IDENTIFICATION_PROMPT_TEMPLATE.format(query=query_text, description=dataset_description, 
                                                                 column_info=", ".join(usable_controls), 
                                                                 treatment=treatment_variable, outcome=outcome_variable, 
                                                                 method=method_name)
  
        llm_selection = _call_llm_for_var(llm, prompt, LLMSelectedCovariates)
        
        if llm_selection and llm_selection.covariates:
            # Validate LLM output against available columns
            valid_llm_controls = [c for c in llm_selection.covariates if c in usable_controls]
            if len(valid_llm_controls) < len(llm_selection.covariates):
                 logger.warning("LLM suggested controls not found in initial usable list.")
            if valid_llm_controls: # Use LLM selection if it's valid and non-empty
                 print(f"LLM refined controls to: {valid_llm_controls}")
                 return valid_llm_controls
            else:
                 logger.warning("LLM refinement failed or returned empty/invalid list. Using heuristically chosen controls.")
        else:
             logger.warning("LLM refinement call failed or returned no controls. Using heuristically chosen controls")

    # 3. Choose heuristically if LLM not used or failed
    print(f"Using heuristically determined controls: {usable_controls}")

    return usable_controls

def select_controls(method_name, variables, columns, column_categories, query, description, llm):
    """
    Selects the control variables based on the method and identified model-specific variables. C
    Controls are used in DiD, RDD, IV, and regression to improve the precision of the causal effect estimates. 
    They are not the same as confounders. 
    If controls are not included, the causal effect estimates should not vary significantly. However, including them can 
    reduce standard error. 

    Args:
        method_name (str): the causal inference method selected 
        variables (dict): dictionary of key identified variables 
        columns (list): list of the column candidates
        column_categories (dict): dictionary of column categories
        query (str): the original user query
        description (str): the dataset description
        llm (BaseChatModel): the language model to use for selection

    Returns:
        list (List[str]): list of selected control variables
    """

    treatment = variables.get("treatment_variable")
    outcome = variables.get("outcome_variable")
    
    instrument = variables.get("instrument_variable")
    running_var = variables.get("running_variable")
    time_var = variables.get("time_variable")
    state_var = variables.get("state_variable")



    controls = []
    # For propensity-score based methods and matching, we need to use confounders not controls
    if method_name in ["difference_in_differences", "regression_discontinuity_design", "instrumental_variables", 
                       "linear_regression"]:
        controls = identify_controls(treatment, outcome, method_name, columns, column_categories, instrument, 
                                     running_var, time_var, state_var, query_text=query, dataset_description=description, 
                                     llm=llm)
                                     

    return controls






"""
LLM-based Decision tree component for selecting causal inference methods.

This module implements the decision tree logic via an LLM prompt
to select the most appropriate causal inference method based on
dataset characteristics and available variables.
"""

import logging
import json
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

# Import constants and assumptions from the original decision_tree module
from .decision_tree import (
    METHOD_ASSUMPTIONS,
    BACKDOOR_ADJUSTMENT,
    LINEAR_REGRESSION,
    DIFF_IN_MEANS,
    DIFF_IN_DIFF,
    REGRESSION_DISCONTINUITY,
    PROPENSITY_SCORE_MATCHING,
    INSTRUMENTAL_VARIABLE,
    CORRELATION_ANALYSIS,
    PROPENSITY_SCORE_WEIGHTING,
    GENERALIZED_PROPENSITY_SCORE
)

# Configure logging
logger = logging.getLogger(__name__)

# Define a list of all known methods for the LLM prompt
ALL_METHODS = [
    DIFF_IN_MEANS,
    LINEAR_REGRESSION,
    DIFF_IN_DIFF,
    REGRESSION_DISCONTINUITY,
    INSTRUMENTAL_VARIABLE,
    PROPENSITY_SCORE_MATCHING,
    PROPENSITY_SCORE_WEIGHTING,
    GENERALIZED_PROPENSITY_SCORE,
    BACKDOOR_ADJUSTMENT, # Often a general approach rather than a specific model.
    CORRELATION_ANALYSIS,
]

METHOD_DESCRIPTIONS_FOR_LLM = {
    DIFF_IN_MEANS: "Appropriate for Randomized Controlled Trials (RCTs) with no covariates. Compares the average outcome between treated and control groups.",
    LINEAR_REGRESSION: "Can be used for RCTs with covariates to increase precision, or for observational data assuming linear relationships and no unmeasured confounders. Models the outcome as a linear function of treatment and covariates.",
    DIFF_IN_DIFF: "Suitable for observational data with a temporal structure (e.g., panel data with pre/post treatment periods). Requires the 'parallel trends' assumption: treatment and control groups would have followed similar trends in the outcome in the absence of treatment.",
    REGRESSION_DISCONTINUITY: "Applicable when treatment assignment is determined by whether an observed 'running variable' crosses a specific cutoff point. Assumes individuals cannot precisely manipulate the running variable.",
    INSTRUMENTAL_VARIABLE: "Used when there's an 'instrument' variable that is correlated with the treatment, affects the outcome only through the treatment, and is not confounded with the outcome. Useful for handling unobserved confounding.",
    PROPENSITY_SCORE_MATCHING: "For observational data with covariates. Estimates the probability of receiving treatment (propensity score) for each unit and then matches treated and control units with similar scores. Aims to create balanced groups.",
    PROPENSITY_SCORE_WEIGHTING: "Similar to PSM, for observational data with covariates. Uses propensity scores to weight units to create a pseudo-population where confounders are balanced. Can estimate ATE, ATT, or ATC.",
    GENERALIZED_PROPENSITY_SCORE: "An extension of propensity scores for continuous treatment variables. Aims to estimate the dose-response function, assuming unconfoundedness given covariates.",
    BACKDOOR_ADJUSTMENT: "A general strategy for causal inference in observational studies that involves statistically controlling for all common causes (confounders) of the treatment and outcome. Specific methods like regression or matching implement this.",
    CORRELATION_ANALYSIS: "A fallback method when causal inference is not feasible due to data limitations (e.g., no clear design, no covariates for adjustment). Measures the statistical association between variables, but does not imply causation."
}


class DecisionTreeLLMEngine:
    """
    Engine for applying an LLM-based decision tree to select appropriate causal methods.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the LLM decision tree engine.

        Args:
            verbose: Whether to print verbose information.
        """
        self.verbose = verbose

    def _construct_prompt(self, dataset_analysis: Dict[str, Any], variables: Dict[str, Any], is_rct: bool, excluded_methods: Optional[List[str]] = None) -> str:
        """
        Constructs the detailed prompt for the LLM.
        """
        # Filter out excluded methods
        excluded_methods = excluded_methods or []
        available_methods = [method for method in ALL_METHODS if method not in excluded_methods]
        methods_list_str = "\n".join([f"- {method}: {METHOD_DESCRIPTIONS_FOR_LLM[method]}" for method in available_methods if method in METHOD_DESCRIPTIONS_FOR_LLM])

        excluded_info = ""
        if excluded_methods:
            excluded_info = f"\nEXCLUDED METHODS (do not select these): {', '.join(excluded_methods)}\nReason: These methods failed validation in previous attempts.\n"

        prompt = f"""You are an expert in causal inference. Your task is to select the most appropriate causal inference method based on the provided dataset analysis and variable information.

Dataset Analysis:
{json.dumps(dataset_analysis, indent=2)}

Identified Variables:
{json.dumps(variables, indent=2)}

Is the data from a Randomized Controlled Trial (RCT)? {'Yes' if is_rct else 'No'}{excluded_info}

Available Causal Inference Methods and their descriptions:
{methods_list_str}

Instructions:
1. Carefully review all the provided information: dataset analysis, variables, and RCT status.
2. Reason step-by-step to determine the most suitable method. Consider the hierarchy of methods (e.g., specific designs like DiD, RDD, IV before general adjustment methods).
3. Explain your reasoning for selecting a particular method.
4. Identify any potential alternative methods if applicable.
5. State the key assumptions for your *selected* method by referring to the general list of assumptions for all methods that will be provided to you separately (you don't need to list them here, just be aware that you need to select a method for which assumptions are known).

Output your final decision as a JSON object with the following exact keys:
- "selected_method": string (must be one of {', '.join(available_methods)})
- "method_justification": string (your detailed reasoning)
- "alternative_methods": list of strings (alternative method names, can be empty)

Example JSON output format:
{{
  "selected_method": "difference_in_differences",
  "method_justification": "The dataset has a clear time variable and group variable, indicating a panel structure suitable for DiD. The parallel trends assumption will need to be checked.",
  "alternative_methods": ["instrumental_variable"]
}}

Please provide only the JSON object in your response.
"""
        return prompt

    def select_method_llm(self, dataset_analysis: Dict[str, Any], variables: Dict[str, Any], is_rct: bool = False, llm: Optional[BaseChatModel] = None, excluded_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply LLM-based decision tree to select appropriate causal method.

        Args:
            dataset_analysis: Dataset analysis results.
            variables: Identified variables from query_interpreter.
            is_rct: Boolean indicating if the data comes from an RCT.
            llm: Langchain BaseChatModel instance for making the call.
            excluded_methods: Optional list of method names to exclude from selection.

        Returns:
            Dict with selected method, justification, and assumptions.
            Example:
            {{
                "selected_method": "difference_in_differences",
                "method_justification": "Reasoning...",
                "method_assumptions": ["Assumption 1", ...],
                "alternative_methods": ["instrumental_variable"]
            }}
        """
        if not llm:
            logger.error("LLM client not provided to DecisionTreeLLMEngine. Cannot select method.")
            return {
                "selected_method": CORRELATION_ANALYSIS,
                "method_justification": "LLM client not provided. Defaulting to Correlation Analysis as causal inference method selection is not possible. This indicates association, not causation.",
                "method_assumptions": METHOD_ASSUMPTIONS.get(CORRELATION_ANALYSIS, []),
                "alternative_methods": []
            }

        prompt = self._construct_prompt(dataset_analysis, variables, is_rct, excluded_methods)
        if self.verbose:
            logger.info("LLM Prompt for method selection:")
            logger.info(prompt)

        messages = [HumanMessage(content=prompt)]
        
        llm_output_str = ""  # Initialize llm_output_str here
        try:
            response = llm.invoke(messages)
            llm_output_str = response.content.strip()
            
            if self.verbose:
                logger.info(f"LLM Raw Output: {llm_output_str}")

            # Attempt to parse the JSON output
            # The LLM might sometimes include explanations outside the JSON block.
            # Try to extract JSON from within ```json ... ``` if present.
            if "```json" in llm_output_str:
                json_str = llm_output_str.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output_str and llm_output_str.startswith("{") == False : # if it doesn't start with { then likely ```{}```
                 json_str = llm_output_str.split("```")[1].strip()
            else: # Assume the entire string is the JSON if no triple backticks
                json_str = llm_output_str
            
            parsed_response = json.loads(json_str)
            
            selected_method = parsed_response.get("selected_method")
            justification = parsed_response.get("method_justification", "No justification provided by LLM.")
            alternatives = parsed_response.get("alternative_methods", [])

            if selected_method and selected_method in METHOD_ASSUMPTIONS:
                logger.info(f"LLM selected method: {selected_method}")
                return {
                    "selected_method": selected_method,
                    "method_justification": justification,
                    "method_assumptions": METHOD_ASSUMPTIONS[selected_method],
                    "alternative_methods": alternatives
                }
            else:
                logger.warning(f"LLM selected an invalid or unknown method: '{selected_method}'. Or method not in METHOD_ASSUMPTIONS. Raw response: {llm_output_str}")
                fallback_justification = f"LLM output was problematic (selected: {selected_method}). Defaulting to Correlation Analysis. LLM Raw Response: {llm_output_str}"
                selected_method = CORRELATION_ANALYSIS
                justification = fallback_justification
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: {e}. Raw response: {llm_output_str}", exc_info=True)
            fallback_justification = f"LLM response was not valid JSON. Defaulting to Correlation Analysis. Error: {e}. LLM Raw Response: {llm_output_str}"
            selected_method = CORRELATION_ANALYSIS
            justification = fallback_justification
            alternatives = []
        except Exception as e:
            logger.error(f"Error during LLM call for method selection: {e}. Raw response: {llm_output_str}", exc_info=True)
            fallback_justification = f"An unexpected error occurred during LLM method selection. Defaulting to Correlation Analysis. Error: {e}. LLM Raw Response: {llm_output_str}"
            selected_method = CORRELATION_ANALYSIS
            justification = fallback_justification
            alternatives = []

        return {
            "selected_method": selected_method,
            "method_justification": justification,
            "method_assumptions": METHOD_ASSUMPTIONS.get(selected_method, []),
            "alternative_methods": alternatives
        } 
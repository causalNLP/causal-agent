"""
decision tree component for selecting causal inference methods

this module implements the decision tree logic to select the most appropriate
causal inference method based on dataset characteristics and available variables
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

# define method names
BACKDOOR_ADJUSTMENT = "backdoor_adjustment"
LINEAR_REGRESSION = "linear_regression"
DIFF_IN_MEANS = "diff_in_means"
DIFF_IN_DIFF = "difference_in_differences"
REGRESSION_DISCONTINUITY = "regression_discontinuity_design"
PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
INSTRUMENTAL_VARIABLE = "instrumental_variable"
CORRELATION_ANALYSIS = "correlation_analysis"
PROPENSITY_SCORE_WEIGHTING = "propensity_score_weighting"
GENERALIZED_PROPENSITY_SCORE = "generalized_propensity_score"
FRONTDOOR_ADJUSTMENT = "frontdoor_adjustment"


logger = logging.getLogger(__name__)

# method assumptions mapping
METHOD_ASSUMPTIONS = {
    BACKDOOR_ADJUSTMENT: [
        "no unmeasured confounders (conditional ignorability given covariates)",
        "correct model specification for outcome conditional on treatment and covariates",
        "positivity/overlap (for all covariate values, units could potentially receive either treatment level)"
    ],
    LINEAR_REGRESSION: [
        "linear relationship between treatment, covariates, and outcome",
        "no unmeasured confounders (if observational)",
        "correct model specification",
        "homoscedasticity of errors",
        "normally distributed errors (for inference)"
    ],
    DIFF_IN_MEANS: [
        "treatment is randomly assigned (or as-if random)",
        "no spillover effects",
        "stable unit treatment value assumption (SUTVA)"
    ],
    DIFF_IN_DIFF: [
        "parallel trends between treatment and control groups before treatment",
        "no spillover effects between groups",
        "no anticipation effects before treatment",
        "stable composition of treatment and control groups",
        "treatment timing is exogenous"
    ],
    REGRESSION_DISCONTINUITY: [
        "units cannot precisely manipulate the running variable around the cutoff",
        "continuity of conditional expectation functions of potential outcomes at the cutoff",
        "no other changes occurring precisely at the cutoff"
    ],
    PROPENSITY_SCORE_MATCHING: [
        "no unmeasured confounders (conditional ignorability)",
        "sufficient overlap (common support) between treatment and control groups",
        "correct propensity score model specification"
    ],
    INSTRUMENTAL_VARIABLE: [
        "instrument is correlated with treatment (relevance)",
        "instrument affects outcome only through treatment (exclusion restriction)",
        "instrument is independent of unmeasured confounders (exogeneity/independence)"
    ],
    CORRELATION_ANALYSIS: [
        "data represents a sample from the population of interest",
        "variables are measured appropriately"
    ],
    PROPENSITY_SCORE_WEIGHTING: [
        "no unmeasured confounders (conditional ignorability)",
        "sufficient overlap (common support) between treatment and control groups",
        "correct propensity score model specification",
        "weights correctly specified (e.g., ATE, ATT)"
    ],
    GENERALIZED_PROPENSITY_SCORE: [
        "conditional mean independence",
        "positivity/common support for GPS",
        "correct specification of the GPS model",
        "correct specification of the outcome model",
        "no unmeasured confounders affecting both treatment and outcome, given X",
        "treatment variable is continuous"
    ],
    FRONTDOOR_ADJUSTMENT: [
        "mediator is affected by treatment and affects outcome",
        "mediator is not affected by any confounders of the treatment-outcome relationship"
    ]
}


def select_method(dataset_properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selects the most appropriate causal inference method using decision tree logic.
    The logic is the following:
    If RCT:
        If instrument_var and instrument_var != treatment:
             this is an RCT encouragement design -> Instrumental Variable
        If covariates:
            Pure RCT with covariates -> OLS for precision
        Else:
            Pure RCT without covariates -> Difference-in-Means
    If Observational:
        If treatment_variable_type is binary:
            If has_temporal and time_var:
                Difference-in-Differences
            If running_var:
                Regression Discontinuity
            If instrument_var:
                Instrumental Variable
            If propensity score method is IPW:
                Propensity Score Weighting
            Else:
                Propensity Score Matching
            If frontdoor criterion:
                Front-door Adjustment
        Else:
            If instrument_var:
                Instrumental Variable
            If frontdoor criterion:
                Front-door Adjustment
            Else:
                Linear Regression (fallback for non-binary treatment)
    Args:
        dataset_properties (Dict[str, Any]): Dictionary containing dataset properties such as treatment variable
    
    Returns:
        Dict[str, Any]: Dictionary containing the selected method, justification, assumptions, and alternatives.
    """
    
    treatment = dataset_properties.get("treatment_variable")
    outcome = dataset_properties.get("outcome_variable")

    if not treatment or not outcome:
        raise ValueError("Both treatment and outcome variables must be specified")

    instrument_var = dataset_properties.get("instrument_variable")
    running_var = dataset_properties.get("running_variable")
    cutoff_val = dataset_properties.get("cutoff_value")
    time_var = dataset_properties.get("time_variable")
    is_rct = dataset_properties.get("is_rct", False)
    has_temporal = dataset_properties.get("has_temporal_structure", False)
    frontdoor = dataset_properties.get("frontdoor_criterion", False)
    covariate_overlap_result = dataset_properties.get("covariate_overlap_score")
    covariates = dataset_properties.get("covariates", [])
    treatment_variable_type = dataset_properties.get("treatment_variable_type", "binary")

    if is_rct:
       
        if instrument_var and instrument_var != treatment:
            logger.info(f"Encouragement design detected with instrument '{instrument_var}' differing from treatment '{treatment}'")
            return {
                "selected_method": INSTRUMENTAL_VARIABLE,
                "method_justification": f"rct with instrument '{instrument_var}' differing from treatment '{treatment}'. instrumental variable method selected as this is encouragement design",
                "method_assumptions": METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE],
                "alternatives": []
            }
        logger.info("Dataset is from a randomized controlled trial (RCT)")
        if covariates:
            return {
                "selected_method": LINEAR_REGRESSION,
                "method_justification": "rct with covariates supplied—using linear regression for precision",
                "method_assumptions": METHOD_ASSUMPTIONS[LINEAR_REGRESSION],
                "alternatives": []
            }
        return {
            "selected_method": DIFF_IN_MEANS,
            "method_justification": "pure rct without covariates—difference-in-means selected",
            "method_assumptions": METHOD_ASSUMPTIONS[DIFF_IN_MEANS],
            "alternatives": []
        }

    valid_methods = []
    justifications = {}
    assumptions = {}

    if treatment_variable_type == "binary":
        logger.info("Binary")
        if has_temporal and time_var:
            logger.info("Difference-in-Differences method selected due to temporal structure")
            return {
                "selected_method": DIFF_IN_DIFF,
                "method_justification": f"temporal structure via '{time_var}'. assuming parallel trends; using difference-in-differences",
                "method_assumptions": METHOD_ASSUMPTIONS[DIFF_IN_DIFF],
                "alternatives": []
            }
        if running_var and cutoff_val is not None:
            logger.info(f"Regression Discontinuity Design method selected with running variable '{running_var}' and cutoff {cutoff_val}")
            return {
                "selected_method": REGRESSION_DISCONTINUITY,
                "method_justification": f"running variable '{running_var}' around cutoff {cutoff_val}. regression discontinuity design selected",
                "method_assumptions": METHOD_ASSUMPTIONS[REGRESSION_DISCONTINUITY],
                "alternatives": []
            }
        if instrument_var:
            logger.info("Instrument var is selected, binary, {}".format(instrument_var))
            valid_methods.append(INSTRUMENTAL_VARIABLE)
            justifications[INSTRUMENTAL_VARIABLE] = f"instrumental variable '{instrument_var}' available"
            assumptions[INSTRUMENTAL_VARIABLE] = METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE]

        # Default to matching if overlap not provided
        if covariate_overlap_result is not None:
            ps_method = PROPENSITY_SCORE_WEIGHTING if covariate_overlap_result < 0.1 else PROPENSITY_SCORE_MATCHING
        else:
            ps_method = PROPENSITY_SCORE_MATCHING

        valid_methods.append(ps_method)
        justifications[ps_method] = "covariates observed & back-door open; ps method selected based on overlap"
        assumptions[ps_method] = METHOD_ASSUMPTIONS[ps_method]

        if frontdoor:
            valid_methods.append(FRONTDOOR_ADJUSTMENT)
            justifications[FRONTDOOR_ADJUSTMENT] = "front-door criterion satisfied"
            assumptions[FRONTDOOR_ADJUSTMENT] = METHOD_ASSUMPTIONS[FRONTDOOR_ADJUSTMENT]
        valid_methods.append(LINEAR_REGRESSION)

        priority = [INSTRUMENTAL_VARIABLE, PROPENSITY_SCORE_MATCHING, PROPENSITY_SCORE_WEIGHTING, FRONTDOOR_ADJUSTMENT, LINEAR_REGRESSION]

    else:
        logger.info("Non-binary treatment variable detected,{}".format(treatment_variable_type))
        if instrument_var:
            logger.info(f"Instrumental Variable method selected with instrument {instrument_var}".format(instrument_var))
            valid_methods.append(INSTRUMENTAL_VARIABLE)
            justifications[INSTRUMENTAL_VARIABLE] = f"instrument '{instrument_var}' deemed valid for non-binary treatment"
            assumptions[INSTRUMENTAL_VARIABLE] = METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE]
        if frontdoor:
            valid_methods.append(FRONTDOOR_ADJUSTMENT)
            justifications[FRONTDOOR_ADJUSTMENT] = "front-door criterion satisfied"
            assumptions[FRONTDOOR_ADJUSTMENT] = METHOD_ASSUMPTIONS[FRONTDOOR_ADJUSTMENT]

        valid_methods.append(LINEAR_REGRESSION)
        justifications[LINEAR_REGRESSION] = "fallback for non-binary treatment without iv/front-door—OLS chosen"
        assumptions[LINEAR_REGRESSION] = METHOD_ASSUMPTIONS[LINEAR_REGRESSION]

        priority = [INSTRUMENTAL_VARIABLE, FRONTDOOR_ADJUSTMENT, LINEAR_REGRESSION]
    
    candidate_methods = [m for m in priority if m in valid_methods]
    selected_method  = candidate_methods[0]
    alternatives = [m for m in candidate_methods if m != selected_method]

    logger.info(f"Selected method: {selected_method} alternatives: {alternatives}")
    logger.info(f"Valid methods: {valid_methods}, priorities: {priority}")

    return {
        "selected_method": selected_method,
        "method_justification": justifications[selected_method],
        "method_assumptions": assumptions[selected_method],
        "alternatives": alternatives
    }


def rule_based_select_method(dataset_analysis, variables, is_rct, llm, dataset_description, original_query):
    """
    Wrapped function to select causal method based on dataset properties and query 

    Args:
      dataset_analysis (Dict): results of dataset analysis
      variables (Dict): dictionary of variable names and types
      is_rct (bool): whether the dataset is from a randomized controlled trial
      llm (BaseChatModel): language model instance for generating prompts
      dataset_description (str): description of the dataset
    """

    logger.info("Running rule-based method selection")


    properties = {"treatment_variable": variables.get("treatment_variable"), "instrument_variable":variables.get("instrument_variable"),
                  "covariates": variables.get("covariates", []), "outcome_variable": variables.get("outcome_variable"),
                  "time_variable": variables.get("time_variable"), "running_variable": variables.get("running_variable"),
                  "treatment_variable_type": variables.get("treatment_variable_type", "binary"),
                  "has_temporal_structure": dataset_analysis.get("temporal_structure", False).get("has_temporal_structure", False),
                  "frontdoor_criterion": variables.get("frontdoor_criterion", False),
                  "cutoff_value": variables.get("cutoff_value"),
                  "covariate_overlap_score": variables.get("covariate_overlap_result", 0)}
    
    properties["is_rct"] = is_rct
    logger.info(f"Dataset properties for method selection: {properties}")

    return select_method(properties)



class DecisionTreeEngine:
    """
    Engine for applying decision trees to select appropriate causal methods.
    
    This class wraps the functional decision tree implementation to provide
    an object-oriented interface for method selection.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def select_method(self, df: pd.DataFrame, treatment: str, outcome: str, covariates: List[str],
                      dataset_analysis: Dict[str, Any], query_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply decision tree to select appropriate causal method.
        """

        if self.verbose:
            print(f"Applying decision tree for treatment: {treatment}, outcome: {outcome}")
            print(f"Available covariates: {covariates}")

        treatment_variable_type = query_details.get("treatment_variable_type")
        covariate_overlap_result = query_details.get("covariate_overlap_result")
        info = {"treatment_variable": treatment, "outcome_variable": outcome,
                     "covariates": covariates, "time_variable": query_details.get("time_variable"),
                     "group_variable": query_details.get("group_variable"),
                     "instrument_variable": query_details.get("instrument_variable"),
                     "running_variable": query_details.get("running_variable"),
                     "cutoff_value": query_details.get("cutoff_value"),
                     "is_rct": query_details.get("is_rct", False),
                     "has_temporal_structure": dataset_analysis.get("temporal_structure", False).get("has_temporal_structure", False),
                     "frontdoor_criterion": query_details.get("frontdoor_criterion", False),
                     "covariate_overlap_score": covariate_overlap_result,
                     "treatment_variable_type": treatment_variable_type}
        
        result = select_method(info)

        if self.verbose:
            print(f"Selected method: {result['selected_method']}")
            print(f"Justification: {result['method_justification']}")

        result["decision_path"] = self._get_decision_path(result["selected_method"])
        return result
    
    
    def _get_decision_path(self, method):
        if method == "linear_regression":
            return ["Check if randomized experiment", "Data appears to be from a randomized experiment with covariates"]
        elif method == "propensity_score_matching":
            return ["Check if randomized experiment", "Data is observational", 
                    "Check for sufficient covariate overlap", "Sufficient overlap exists"]
        elif method == "propensity_score_weighting":
            return ["Check if randomized experiment", "Data is observational", 
                "Check for sufficient covariate overlap", "Low overlap—weighting preferred"]
        elif method == "backdoor_adjustment":
            return ["Check if randomized experiment", "Data is observational", 
                "Check for sufficient covariate overlap", "Adjusting for covariates"]
        elif method == "instrumental_variable":
            return ["Check if randomized experiment", "Data is observational", 
                "Check for instrumental variables", "Instrument is available"]
        elif method == "regression_discontinuity_design":
            return ["Check if randomized experiment", "Data is observational", 
                "Check for discontinuity", "Discontinuity exists"]
        elif method == "difference_in_differences":
            return ["Check if randomized experiment", "Data is observational", 
                "Check for temporal structure", "Panel data structure exists"]
        elif method == "frontdoor_adjustment":
            return ["Check if randomized experiment", "Data is observational",
                "Check front-door criterion", "Front-door path identified"]
        elif method == "diff_in_means":
            return ["Check if randomized experiment", "Pure RCT without covariates"]
        else:
            return ["Default method selection"]
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


def select_method(dataset_properties: Dict[str, Any], excluded_methods: Optional[List[str]] = None) -> Dict[str, Any]:
    excluded_methods = set(excluded_methods or [])
    logger.info(f"Excluded methods: {sorted(excluded_methods)}")

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

    # Helpers to collect candidates
    candidates = []  # list of (method, priority_index)
    justifications: Dict[str, str] = {}
    assumptions: Dict[str, List[str]] = {}

    def add(method: str, justification: str, prio_order: List[str]):
        if method in justifications:  # already added
            return
        justifications[method] = justification
        assumptions[method] = METHOD_ASSUMPTIONS[method]
        # priority index from provided order (fallback large if not present)
        try:
            idx = prio_order.index(method)
        except ValueError:
            idx = 10**6
        candidates.append((method, idx))

    # ----- Build candidate set (no returns here) -----

    # RCT branch
    if is_rct:
        logger.info("Dataset is from a randomized controlled trial (RCT)")
        rct_priority = [INSTRUMENTAL_VARIABLE, LINEAR_REGRESSION, DIFF_IN_MEANS]

        if instrument_var and instrument_var != treatment:
            add(INSTRUMENTAL_VARIABLE,
                f"RCT encouragement: instrument '{instrument_var}' differs from treatment '{treatment}'.",
                rct_priority)

        if covariates:
            add(LINEAR_REGRESSION,
                "RCT with covariates—use OLS for precision.",
                rct_priority)
        else:
            add(DIFF_IN_MEANS,
                "Pure RCT without covariates—difference-in-means.",
                rct_priority)

    # Observational branch
    obs_priority_binary = [
        INSTRUMENTAL_VARIABLE,
        PROPENSITY_SCORE_MATCHING,
        PROPENSITY_SCORE_WEIGHTING,
        FRONTDOOR_ADJUSTMENT,
        LINEAR_REGRESSION,
    ]
    obs_priority_nonbinary = [
        INSTRUMENTAL_VARIABLE,
        FRONTDOOR_ADJUSTMENT,
        LINEAR_REGRESSION,
    ]

    # Common early structural signals first (still only add as candidates)
    if has_temporal and time_var:
        add(DIFF_IN_DIFF,
            f"Temporal structure via '{time_var}'—consider Difference-in-Differences (assumes parallel trends).",
            [DIFF_IN_DIFF])  # highest among itself

    if running_var and cutoff_val is not None:
        add(REGRESSION_DISCONTINUITY,
            f"Running variable '{running_var}' with cutoff {cutoff_val}—consider RDD.",
            [REGRESSION_DISCONTINUITY])

    # Binary vs non-binary pathways
    if treatment_variable_type == "binary":
        if instrument_var:
            add(INSTRUMENTAL_VARIABLE,
                f"Instrumental variable '{instrument_var}' available.",
                obs_priority_binary)

        # Propensity score methods only if covariates exist
        if covariates:
            if covariate_overlap_result is not None:
                ps_method = (PROPENSITY_SCORE_WEIGHTING
                             if covariate_overlap_result < 0.1
                             else PROPENSITY_SCORE_MATCHING)
            else:
                ps_method = PROPENSITY_SCORE_MATCHING
            add(ps_method,
                "Covariates observed; PS method chosen based on overlap.",
                obs_priority_binary)

        if frontdoor:
            add(FRONTDOOR_ADJUSTMENT,
                "Front-door criterion satisfied.",
                obs_priority_binary)

        add(LINEAR_REGRESSION,
            "OLS as a fallback specification.",
            obs_priority_binary)

    else:
        logger.info(f"Non-binary treatment variable detected: {treatment_variable_type}")
        if instrument_var:
            add(INSTRUMENTAL_VARIABLE,
                f"Instrument '{instrument_var}' candidate for non-binary treatment.",
                obs_priority_nonbinary)
        if frontdoor:
            add(FRONTDOOR_ADJUSTMENT,
                "Front-door criterion satisfied.",
                obs_priority_nonbinary)
        add(LINEAR_REGRESSION,
            "Fallback for non-binary treatment without stronger identification.",
            obs_priority_nonbinary)

    # ----- Centralized exclusion handling -----
    # Remove excluded
    filtered = [(m, p) for (m, p) in candidates if m not in excluded_methods]

    # If nothing survives, attempt a safe fallback not excluded
    if not filtered:
        logger.warning(f"All candidates excluded. Candidates were: {[m for m,_ in candidates]}. Excluded: {sorted(excluded_methods)}")
        fallback_order = [
            LINEAR_REGRESSION,
            DIFF_IN_MEANS,
            PROPENSITY_SCORE_MATCHING,
            PROPENSITY_SCORE_WEIGHTING,
            DIFF_IN_DIFF,
            REGRESSION_DISCONTINUITY,
            INSTRUMENTAL_VARIABLE,
            FRONTDOOR_ADJUSTMENT,
        ]
        fallback = next((m for m in fallback_order if m in justifications and m not in excluded_methods), None)
        if not fallback:
            # truly nothing left; raise with context
            raise RuntimeError("No viable method remains after exclusions.")
        selected_method = fallback
        alternatives = []
        justifications[selected_method] = justifications.get(selected_method, "Fallback after exclusions.")
    else:
        # Pick by smallest priority index, then stable by insertion
        filtered.sort(key=lambda x: x[1])
        selected_method = filtered[0][0]
        alternatives = [m for (m, _) in filtered[1:] if m != selected_method]

    logger.info(f"Selected method: {selected_method}; alternatives: {alternatives}")

    return {
        "selected_method": selected_method,
        "method_justification": justifications[selected_method],
        "method_assumptions": assumptions[selected_method],
        "alternatives": alternatives,
        "excluded_methods": sorted(excluded_methods),
    }



def rule_based_select_method(dataset_analysis, variables, is_rct, llm, dataset_description, original_query, excluded_methods=None):
    """
    Wrapped function to select causal method based on dataset properties and query 

    Args:
      dataset_analysis (Dict): results of dataset analysis
      variables (Dict): dictionary of variable names and types
      is_rct (bool): whether the dataset is from a randomized controlled trial
      llm (BaseChatModel): language model instance for generating prompts
      dataset_description (str): description of the dataset
      original_query (str): the original user query
      excluded_methods (List[str], optional): list of methods to exclude from selection
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

    return select_method(properties, excluded_methods)



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
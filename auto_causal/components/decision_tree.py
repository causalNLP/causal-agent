"""
Decision tree component for selecting causal inference methods.

This module implements the decision tree logic to select the most appropriate
causal inference method based on dataset characteristics and available variables.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import json
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

# Configure logging
logger = logging.getLogger(__name__)

# Define method names (ensure these match keys/names used elsewhere)
BACKDOOR_ADJUSTMENT = "backdoor_adjustment"
LINEAR_REGRESSION = "linear_regression"
DIFF_IN_MEANS = "diff_in_means"
DIFF_IN_DIFF = "difference_in_differences"
REGRESSION_DISCONTINUITY = "regression_discontinuity_design"
PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
INSTRUMENTAL_VARIABLE = "instrumental_variable"
CORRELATION_ANALYSIS = "correlation_analysis" # Added for fallback
PROPENSITY_SCORE_WEIGHTING = "propensity_score_weighting"
GENERALIZED_PROPENSITY_SCORE = "generalized_propensity_score" # Added constant


# Method Assumptions Mapping
METHOD_ASSUMPTIONS = {
    BACKDOOR_ADJUSTMENT: [
        "No unmeasured confounders (conditional ignorability given covariates)",
        "Correct model specification for outcome conditional on treatment and covariates",
        "Positivity/Overlap (for all covariate values, units could potentially receive either treatment level)"
    ],
    LINEAR_REGRESSION: [
        "Linear relationship between treatment, covariates, and outcome",
        "No unmeasured confounders (if observational)",
        "Correct model specification",
        "Homoscedasticity of errors",
        "Normally distributed errors (for inference)"
    ],
    DIFF_IN_MEANS: [
        "Treatment is randomly assigned (or as-if random)",
        "No spillover effects",
        "Stable Unit Treatment Value Assumption (SUTVA)"
    ],
    DIFF_IN_DIFF: [
        "Parallel trends between treatment and control groups before treatment",
        "No spillover effects between groups",
        "No anticipation effects before treatment",
        "Stable composition of treatment and control groups",
        "Treatment timing is exogenous"
    ],
    REGRESSION_DISCONTINUITY: [
        "Units cannot precisely manipulate the running variable around the cutoff",
        "Continuity of conditional expectation functions of potential outcomes at the cutoff",
        "No other changes occurring precisely at the cutoff"
    ],
    PROPENSITY_SCORE_MATCHING: [
        "No unmeasured confounders (conditional ignorability)",
        "Sufficient overlap (common support) between treatment and control groups",
        "Correct propensity score model specification"
    ],
    INSTRUMENTAL_VARIABLE: [
        "Instrument is correlated with treatment (relevance)",
        "Instrument affects outcome only through treatment (exclusion restriction)",
        "Instrument is independent of unmeasured confounders (exogeneity/independence)"
    ],
    CORRELATION_ANALYSIS: [
        "Data represents a sample from the population of interest",
        "Variables are measured appropriately"
        # Note: Correlation does not imply causation
    ],
    PROPENSITY_SCORE_WEIGHTING: [
        "No unmeasured confounders (conditional ignorability)",
        "Sufficient overlap (common support) between treatment and control groups",
        "Correct propensity score model specification",
        "Weights correctly specified (e.g., ATE, ATT)"
    ],
    GENERALIZED_PROPENSITY_SCORE: [
        "Conditional Mean Independence: E[Y(t) | X, T=t] = E[Y(t) | X, GPS(t,X)] (Unconfoundedness of Y(t) given X for all t)",
        "Positivity/Common Support for GPS: For any value of X, the conditional density f(T=t|X=x) > 0 for all relevant t.",
        "Correct specification of the GPS model (model for T|X, often conditional density).",
        "Correct specification of the outcome model (model for Y|T, GPS).",
        "No unmeasured confounders affecting both treatment and outcome, given X.",
        "Treatment variable is continuous."
    ]
}

class DecisionTreeEngine:
    """
    Engine for applying decision trees to select appropriate causal methods.
    
    This class wraps the functional decision tree implementation to provide
    an object-oriented interface for method selection.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize the decision tree engine.
        
        Args:
            verbose: Whether to print verbose information
        """
        self.verbose = verbose
    
    def select_method(self, df, treatment, outcome, covariates, dataset_analysis, query_details):
        """
        Apply decision tree to select appropriate causal method.
        
        Args:
            df: Pandas DataFrame containing the dataset
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            covariates: List of covariate variable names
            dataset_analysis: Results from dataset analysis
            query_details: Details about the causal query
            
        Returns:
            Dict with selected method and justification
        """
        variables = {
            "treatment_variable": treatment,
            "outcome_variable": outcome,
            "covariates": covariates,
            "time_variable": query_details.get("time_variable"),
            "group_variable": query_details.get("group_variable"),
            "instrument_variable": query_details.get("instrument_variable"),
            "running_variable": query_details.get("running_variable"),
            "cutoff_value": query_details.get("cutoff_value")
        }
        
        if self.verbose:
            print(f"Applying decision tree for treatment: {treatment}, outcome: {outcome}")
            print(f"Available covariates: {covariates}")
        
        result = select_method(dataset_analysis, variables)
        
        if self.verbose:
            print(f"Selected method: {result.get('method')}")
            print(f"Justification: {result.get('justification')}")
        
        # Add the decision path and any alternatives
        result["decision_path"] = self._get_decision_path(result.get("method"))
        result["alternative_methods"] = result.get("alternatives", [])
        
        return result
    
    def analyze_dataset_for_decisions(self, df, treatment, outcome, covariates):
        """
        Perform additional analysis on the dataset to support decision making.
        
        Args:
            df: Pandas DataFrame containing the dataset
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            covariates: List of covariate variable names
            
        Returns:
            Dict with additional analysis results
        """
        analysis = {}
        
        # Add column information
        analysis["columns"] = list(df.columns)
        analysis["column_types"] = {col: str(df[col].dtype) for col in df.columns}
        
        # Categorize columns
        analysis["column_categories"] = {}
        for col in df.columns:
            if df[col].dtype == 'bool':
                analysis["column_categories"][col] = 'binary'
            elif pd.api.types.is_numeric_dtype(df[col]):
                if len(df[col].unique()) <= 2:
                    analysis["column_categories"][col] = 'binary'
                else:
                    analysis["column_categories"][col] = 'continuous'
            else:
                unique_values = df[col].nunique()
                if unique_values <= 2:
                    analysis["column_categories"][col] = 'binary'
                elif unique_values <= 10:
                    analysis["column_categories"][col] = 'categorical'
                else:
                    analysis["column_categories"][col] = 'high_cardinality'
        
        # Check for temporal structure
        date_cols = [col for col in df.columns if 
                    any(keyword in col.lower() for keyword in 
                        ['date', 'time', 'year', 'month', 'day', 'period'])]
        
        analysis["temporal_structure"] = {
            "has_temporal_structure": len(date_cols) > 0,
            "is_panel_data": len(date_cols) > 0 and any(keyword in col.lower() 
                                                      for col in df.columns 
                                                      for keyword in ['id', 'group', 'entity']),
            "time_variables": date_cols
        }
        
        # Identify potential instruments
        analysis["potential_instruments"] = []
        if treatment in df.columns and outcome in df.columns:
            for col in df.columns:
                if (col != treatment and col != outcome and col not in covariates and
                    self._check_potential_instrument(df, col, treatment, outcome)):
                    analysis["potential_instruments"].append(col)
        
        # Identify potential discontinuities
        analysis["discontinuities"] = {"has_discontinuities": False}
        
        # Identify potential confounders
        analysis["variable_relationships"] = {"potential_confounders": []}
        
        if self.verbose:
            print(f"Dataset analysis for decisions completed")
        
        return analysis
    
    def _check_potential_instrument(self, df, instrument, treatment, outcome):
        """
        Check if a variable is a potential instrument.
        
        Args:
            df: Pandas DataFrame
            instrument: Name of potential instrument variable
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            
        Returns:
            Boolean indicating if variable may be an instrument
        """
        # Very simplified check - would need more sophisticated tests in practice
        try:
            # Check correlation with treatment
            treatment_corr = df[instrument].corr(df[treatment])
            # Check correlation with outcome
            outcome_corr = df[instrument].corr(df[outcome])
            
            # If correlated with treatment but less so with outcome
            return abs(treatment_corr) > 0.3 and abs(outcome_corr) < 0.2
        except:
            return False
    
    def _get_decision_path(self, method):
        """
        Get the decision path that led to the selected method.
        
        Args:
            method: Selected method
            
        Returns:
            List of decisions made
        """
        # This would ideally track the actual decision path
        # For now, return a simplified version
        if method == "regression_adjustment":
            return ["Check if randomized experiment", "Data appears to be from a randomized experiment"]
        elif method == "propensity_score_matching":
            return ["Check if randomized experiment", "Data is observational", 
                    "Check for sufficient covariate overlap", "Sufficient overlap exists"]
        elif method == "backdoor_adjustment":
            return ["Check if randomized experiment", "Data is observational", 
                    "Check for sufficient covariate overlap", "Insufficient overlap"]
        elif method == "instrumental_variable":
            return ["Check if randomized experiment", "Data is observational", 
                    "Check for instrumental variables", "Instrument is available"]
        elif method == "regression_discontinuity":
            return ["Check if randomized experiment", "Data is observational", 
                    "Check for discontinuity", "Discontinuity exists"]
        elif method == "difference_in_differences":
            return ["Check if randomized experiment", "Data is observational", 
                    "Check for temporal structure", "Panel data structure exists"]
        else:
            return ["Default method selection"]


def _treatment_timing_is_clear(variables: Dict[str, Any]) -> bool:
    """Helper to check if time variable and treatment context are suitable for DiD."""
    # Simple check: requires a time variable and potentially group variable
    # More robust checks could involve analyzing treatment patterns over time later
    return bool(variables.get("time_variable")) # and bool(variables.get("group_variable")) - Group var check might be too strict?

def _recommend_ps_method(treatment_variable: str, per_group_stats: Dict, llm: Optional[BaseChatModel] = None) -> str:
    """Recommends PSM vs PSW based on per-group summary stats using LLM."""
    default_method = PROPENSITY_SCORE_MATCHING # Default if LLM fails or not provided
    if not llm:
        logger.warning("LLM client not provided for PSM/PSW recommendation. Defaulting to PSM.")
        return default_method
        
    # Extract stats for the specific treatment variable
    specific_stats = per_group_stats.get(treatment_variable)
    if not specific_stats or 'covariate_stats' not in specific_stats or 'group_sizes' not in specific_stats:
         logger.warning(f"Summary stats for treatment '{treatment_variable}' incomplete or missing. Defaulting to PSM.")
         return default_method
         
    # Construct prompt with summary statistics
    prompt = f"""Given the following summary statistics for numeric covariates, grouped by the treatment variable '{treatment_variable}':
Group Sizes: {json.dumps(specific_stats['group_sizes'])}
Covariate Means/Stds (Treat=1, Control=0):
{json.dumps(specific_stats['covariate_stats'], indent=2)}

Recommend either 'Propensity Score Matching' or 'Propensity Score Weighting' as the more suitable method.
Consider these factors:
- Large differences in means or std deviations between groups suggest imbalance, potentially favoring Weighting.
- Very small group sizes (e.g., < 20) in either treated or control group might make Matching difficult.
- Very large overall sample size might favor Weighting.
- If unsure or stats suggest reasonable balance and sample size, Matching is a common default.

Respond ONLY with the chosen method name (either "Propensity Score Matching" or "Propensity Score Weighting")."""
    
    logger.info(f"Asking LLM to recommend PSM vs PSW based on summary stats for '{treatment_variable}'...")
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = llm.invoke(messages)
        recommended = response.content.strip().strip('\'').strip("'") # Get string content and clean quotes
        
        if recommended == PROPENSITY_SCORE_MATCHING:
            return PROPENSITY_SCORE_MATCHING
        elif recommended == PROPENSITY_SCORE_WEIGHTING:
            return PROPENSITY_SCORE_WEIGHTING
        else:
            logger.warning(f"LLM recommendation for PS method was unclear: '{recommended}'. Defaulting to PSM.")
            return default_method
            
    except Exception as e:
        logger.error(f"Error during LLM call for PSM/PSW recommendation: {e}", exc_info=True)
        return default_method

def select_method(dataset_analysis: Dict[str, Any], variables: Dict[str, Any], is_rct: bool = False, llm: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """
    Apply decision tree to select appropriate causal method based on the UPDATED logic.
    
    Args:
        dataset_analysis: Dataset analysis results from dataset_analyzer.
                          Expected keys: 'temporal_structure' (bool).
        variables: Identified variables from query_interpreter.
                   Expected keys: 'treatment_variable', 'outcome_variable', 'covariates' (list),
                                  'time_variable', 'group_variable', 'instrument_variable',
                                  'running_variable', 'cutoff_value'.
        is_rct: Boolean indicating if the data comes from a Randomized Controlled Trial.
                (This should be determined upstream, e.g., by query_interpreter or metadata).
        llm: Language model for recommendation
        
    Returns:
        Dict with selected method, justification, and assumptions:
            - selected_method: String indicating selected method name.
            - method_justification: String explaining why this method was selected.
            - method_assumptions: List of key assumptions for the selected method.
    """
    logger.info("Starting causal method selection based on updated decision tree...")
    logger.debug(f"Inputs: dataset_analysis={dataset_analysis}, variables={variables}, is_rct={is_rct}")

    covariates = variables.get("covariates", [])
    treatment_variable_type = variables.get("treatment_variable_type", "unknown") # Get treatment type
    logger.info(f"Received treatment_variable_type: {treatment_variable_type}")

    # --- Initial Check: If no covariates identified, default differently based on RCT status ---
    # UPDATE: Removed this initial check, handle covariate presence within RCT/Observational logic
    # if not covariates:
    #    logger.info("No covariates identified. Selecting method based on RCT status.")
    #    method = DIFF_IN_MEANS if is_rct else CORRELATION_ANALYSIS # Or maybe Backdoor w/o covs?
    #    justification = ("Data is from RCT with no covariates. Difference in Means is appropriate." if is_rct 
    #                     else "No covariates identified for observational data. Correlation analysis is possible, but causal claims require strong assumptions.")
    #    return {
    #        "selected_method": method,
    #        "method_justification": justification,
    #        "method_assumptions": METHOD_ASSUMPTIONS.get(method, [])
    #    }

    # --- RCT Branch --- 
    if is_rct:
        logger.info("Data identified as RCT.")
        
        # Check for Encouragement Design first in RCT
        instrument_var = variables.get("instrument_variable")
        treatment_var = variables.get("treatment_variable")
        if instrument_var and treatment_var and instrument_var != treatment_var:
            logger.info(f"RCT context with Instrument ('{instrument_var}') differing from Treatment ('{treatment_var}'). Selecting IV for Encouragement Design.")
            return {
                "selected_method": INSTRUMENTAL_VARIABLE,
                "method_justification": f"Data is from an RCT, but the identified instrument (encouragement) '{instrument_var}' differs from the treatment '{treatment_var}'. Selecting Instrumental Variable method for this Encouragement Design.",
                "method_assumptions": METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE],
                "alternatives": [] # Explicitly empty alternatives here
            }
        
        # Standard RCT logic (if not encouragement design)
        if covariates:
            logger.info("RCT with covariates present. Selecting Linear Regression.")
            return {
                "selected_method": LINEAR_REGRESSION,
                "method_justification": "Data is from an RCT and covariates are provided. Linear regression with covariates is used to potentially increase precision.",
                "method_assumptions": METHOD_ASSUMPTIONS[LINEAR_REGRESSION],
                "alternatives": [] # Explicitly empty alternatives here
            }
        else:
            logger.info("RCT with no covariates present. Selecting Difference in Means.")
            return {
                "selected_method": DIFF_IN_MEANS, 
                "method_justification": "Data is from an RCT with no covariates specified. Simple Difference in Means is appropriate.",
                "method_assumptions": METHOD_ASSUMPTIONS[DIFF_IN_MEANS],
                "alternatives": [] # Explicitly empty alternatives here
            }

    # --- Observational Branch --- 
    else:
        logger.info("Data identified as Observational (Non-RCT).")
        alternatives = [] # Initialize alternatives list for observational path
        
        # # === NEW: Check for Continuous Treatment and GPS FIRST in Observational ===
        # if treatment_variable_type == "continuous":
        #     logger.info("Continuous treatment variable identified for observational data. Selecting Generalized Propensity Score.")
        #     # Could add IV as an alternative if an instrument is also identified
        #     if variables.get("instrument_variable"):
        #         alternatives.append(INSTRUMENTAL_VARIABLE)
        #     return {
        #         "selected_method": GENERALIZED_PROPENSITY_SCORE,
        #         "method_justification": "Observational data with a continuous treatment variable. Generalized Propensity Score (GPS) is selected to estimate the dose-response function, assuming unconfoundedness given covariates.",
        #         "method_assumptions": METHOD_ASSUMPTIONS[GENERALIZED_PROPENSITY_SCORE],
        #         "alternatives": alternatives
        #     }
        # # === END NEW GPS CHECK ===

        # Define variables needed multiple times
        time_var = variables.get("time_variable")
        # Using dataset_analysis key which might be more reliable? Check dataset_analyzer output structure.
        has_temporal = dataset_analysis.get('temporal_structure', {}).get('has_temporal_structure', False) 
        running_var = variables.get("running_variable")
        cutoff_val = variables.get("cutoff_value")
        instrument_var = variables.get("instrument_variable")
        treatment_var = variables.get("treatment_variable") # Needed for IV encouragement check
        
        # --- Priority 1: Difference-in-Differences --- 
        if has_temporal and time_var:
            logger.info("Temporal structure found. Selecting Difference-in-Differences.")
            # Check for IV as alternative
            if instrument_var:
                logger.info("Instrument variable also found, adding IV as alternative to DiD.")
                alternatives.append(INSTRUMENTAL_VARIABLE)
            return {
                "selected_method": DIFF_IN_DIFF,
                "method_justification": f"Observational data with temporal structure identified (time variable: '{time_var}'). Difference-in-Differences is selected, assuming parallel trends holds (to be validated).",
                "method_assumptions": METHOD_ASSUMPTIONS[DIFF_IN_DIFF],
                "alternatives": alternatives
            }

        # --- Priority 2: Regression Discontinuity Design --- 
        elif running_var and cutoff_val is not None:
            logger.info("Running variable and cutoff found. Selecting Regression Discontinuity Design.")
            # Could check for alternatives here too if needed (e.g., if IV also present)
            return {
                "selected_method": REGRESSION_DISCONTINUITY,
                "method_justification": f"Observational data where treatment assignment appears determined by a running variable ('{running_var}') around a cutoff ({cutoff_val}). Regression Discontinuity Design is selected.",
                "method_assumptions": METHOD_ASSUMPTIONS[REGRESSION_DISCONTINUITY],
                "alternatives": alternatives # Pass empty list for now
            }
        
        # --- Priority 3: Instrumental Variable (if no DiD, RDD, and no covariates for PSM/PSW) --- 
        elif instrument_var:
            justification = f"Observational data with a potential instrumental variable ('{instrument_var}') identified. Instrumental Variable Regression is selected, assuming the instrument is valid (to be validated)."
            # Check for Encouragement Design
            if treatment_var and instrument_var != treatment_var:
                logger.info(f"Instrument ('{instrument_var}') differs from treatment ('{treatment_var}'). Identifying as Encouragement Design.")
                justification = f"Observational data with treatment '{treatment_var}' potentially influenced by instrument (encouragement) '{instrument_var}'. Selecting Instrumental Variable method for this Encouragement Design, assuming instrument validity (to be validated)."
            else:
                 logger.info("Potential instrumental variable identified. Selecting Instrumental Variable Regression.")
            
            # Check for DiD as alternative (already checked DiD conditions above, but check vars again for consistency)
            if has_temporal and time_var:
                logger.info("Temporal structure also found, adding DiD as alternative to IV.")
                alternatives.append(DIFF_IN_DIFF)
                
            return {
                "selected_method": INSTRUMENTAL_VARIABLE,
                "method_justification": justification,
                "method_assumptions": METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE],
                "alternatives": alternatives
            }


        # --- Priority 4: Propensity Score Methods (if covariates exist) --- 
        elif covariates:
            logger.info("Observed confounders present, no specific design (DiD, RDD) indicated. Considering PSM/PSW.")
            
            # Check for IV as alternative *before* selecting PSM/PSW
            if instrument_var:
                 logger.info("Instrument variable also found, adding IV as alternative to PSM/PSW.")
                 alternatives.append(INSTRUMENTAL_VARIABLE)
                 
            # Use LLM to recommend PSM/PSW based on dataset characteristics.
            metrics = dataset_analysis # Pass the whole analysis dict for now
            recommended_ps_method_name = _recommend_ps_method(treatment_var, metrics, llm)
    
            if recommended_ps_method_name == PROPENSITY_SCORE_WEIGHTING:
                logger.info("LLM recommended PSW based on dataset characteristics.")
                selected_method = PROPENSITY_SCORE_WEIGHTING
                justification = "Observational data with observed confounders. Propensity Score Weighting is selected (potentially based on LLM recommendation considering dataset characteristics like sample size, prevalence, or overlap)."
            else: # Default to PSM
                if recommended_ps_method_name != PROPENSITY_SCORE_MATCHING:
                     logger.warning(f"LLM recommendation for PS method was unclear or failed ('{recommended_ps_method_name}'). Defaulting to PSM.")
                else:
                     logger.info("LLM recommended PSM based on dataset characteristics.")
                selected_method = PROPENSITY_SCORE_MATCHING
                justification = "Observational data with observed confounders. Propensity Score Matching is selected (potentially based on LLM recommendation considering dataset characteristics)."
    
            return {
                "selected_method": selected_method,
                "method_justification": justification,
                "method_assumptions": METHOD_ASSUMPTIONS[selected_method],
                "alternatives": alternatives # Include IV if found
             }
             

        # --- Fallback: Correlation Analysis (Observational, no specific design, no covariates for PSM/PSW, no IV) --- 
        else:
            logger.warning("Observational data with no specific design (DiD, RDD, IV) and no covariates identified for adjustment. Causal effect estimation is likely biased. Suggesting Correlation Analysis.")
            return {
                 "selected_method": CORRELATION_ANALYSIS,
                 "method_justification": "Observational data with no identified conditions for DiD, RDD, IV and no covariates for adjustment. Cannot estimate causal effect reliably. Correlation analysis is suggested, but results are not causal.",
                 "method_assumptions": METHOD_ASSUMPTIONS[CORRELATION_ANALYSIS],
                 "alternatives": alternatives # Likely empty
             }


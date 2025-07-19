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
from auto_causal.config import get_llm_client
import numpy as np
# Configure logging
logger = logging.getLogger(__name__)

CAUSAL_GRAPH_PROMPT = """
You are an expert in causal inference. Your task is to construct a causal graph to help answer a user query.

Here is the user query:
{query}

Dataset Description:
{description}

Here are the treatment and outcome variables:
Treatment: {treatment}
Outcome: {outcome}

Here are the available variables in the dataset:
{column_info}

Based on the query, dataset description, and available variables, construct a causal graph that captures the relationships between the treatment, outcome, and other relevant variables.

Use only variables present in the dataset. Do not invent or assume any variables. However, not all variables need to be included—only those that are relevant to the causal relationships should appear in the graph.

Return the causal graph in DOT format. The DOT format should include:
- Nodes for each included variable.
- Directed edges representing causal relationships among variables.

Also return the list of edges in the format "A -> B", where A and B are variable names.

Here is an example of the DOT format:
digraph G {{
    A -> B;
    B -> C;
    A -> C;
}}

And the corresponding list of edges:
["A -> B", "B -> C", "A -> C"]

Return your response as a valid JSON object in the following format:
{{ 
  "causal_graph": "DOT_FORMAT_STRING",
  "edges": ["EDGE_1", "EDGE_2", ...] 
}}
Do not wrap the response in triple backticks or formatting it as a code block.
"""

ESTIMAND_PROMPT = """
what is the better estimand, ATE or ATT, to answer this query: {query}
given the following information about the dataset
Description: {dataset_description}
Treatment Variable: {treat_var}
Outcome Variable: {outcome_var}

Only return the estimand name: ATT or ATE
"""
def _llm_choose_estimand(
    query:str,
    description:str,
    treat_var:str,
    outcome_var:str,
    llm:BaseChatModel
) -> str:
    """Returns 'ATE' or 'ATT' (fallback ATE) using the user‑supplied prompt."""
    prompt = ESTIMAND_PROMPT.format(
        query=query,
        dataset_description=description,
        treat_var=treat_var,
        outcome_var=outcome_var
    )
    reply = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    return "ATT" if reply == "ATT" else "ATE"

def _llm_build_causal_graph(
    query:str,
    description:str,
    treatment:str,
    outcome:str,
    df:pd.DataFrame,
    llm:BaseChatModel
) -> Dict[str, Any]:
    """Returns {"edges": [...], "dot": "..."} using the user-given prompt."""
    column_info = ", ".join(df.columns.tolist())
    prompt = CAUSAL_GRAPH_PROMPT.format(
        query=query,
        description=description,
        treatment=treatment,
        outcome=outcome,
        column_info=column_info
    )
    resp = llm.invoke([HumanMessage(content=prompt)]).content
    graph = json.loads(resp)                    # will raise if not valid JSON
    return graph                                # keys: causal_graph, edges

def _check_backdoor(edges, treatment, outcome, covariates):
    """Very light-weight back-door test: every back-door path must be blocked by an observed covariate."""
    # Build adjacency for quick look-ups
    children = {}
    parents  = {}
    for e in edges:
        a,b = [x.strip() for x in e.split("->")]
        children.setdefault(a, set()).add(b)
        parents .setdefault(b, set()).add(a)

    # DFS all back-door paths T ← … → Y
    stack = [(par, [treatment, par]) for par in parents.get(treatment, [])]
    while stack:
        node, path = stack.pop()
        if node == outcome:
            return False            # unblocked path found
        for nxt in children.get(node, set()) | parents.get(node, set()):
            if nxt not in path:
                # path continues only if we haven’t conditioned on it
                if nxt not in covariates:
                    stack.append((nxt, path+[nxt]))
    return True                     # every path blocked by covariate set

def _check_frontdoor(edges, treatment, outcome):
    """Looks for a single mediator M satisfying front-door heuristics."""
    # T -> M, M -> Y, no T -> Y edge
    ts, ys = [], []
    for e in edges:
        a,b = [x.strip() for x in e.split("->")]
        if a==treatment: ts.append(b)
        if b==outcome:   ys.append(a)
    mediators = set(ts) & set(ys)
    direct_TY = any((a.strip()==treatment and b.strip()==outcome) 
                    for a,b in (map(str.strip,e.split("->")) for e in edges))
    return bool(mediators) and not direct_TY

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

from textwrap import dedent

def _llm_decide_observational_method(
    analysis: Dict[str, Any],
    variables: Dict[str, Any],
    llm: BaseChatModel
) -> str:
    """
    Ask the LLM to choose ONE label from
      {IV, BACKDOOR_REGRESSION, MATCHING, IPW, FRONTDOOR}
    after walking through the same nodes shown in the paper’s tree.
    """
    prompt = dedent(f"""
    You are a senior econometrician.  Use the following *decision‑tree nodes*
    to pick the single best causal estimator.

    ───────────────────  DECISION NODES  ───────────────────
    1. Is there a **valid instrument**? (relevance + exclusion + independence)
       ➜ If YES  →   choose **IV**.
    2. Is there a **valid back‑door adjustment set** (all confounders observed)?
       ➜ If NO   →   go to node 4.
       ➜ If YES  →   check node 3.
    3. **Do covariates overlap** in treated vs. control groups?
         • Good overlap → **BACKDOOR_REGRESSION** (e.g., OLS with covariates)
         • Moderate     → **MATCHING**
         • Poor         → **IPW**
    4. Is the **front‑door criterion** satisfied?
       ➜ If YES  →   **FRONTDOOR**
       ➜ If NO   →   (no valid choice; default to MATCHING)

    ───────────────────  CONTEXT  ───────────────────
    Dataset diagnostics:
    {json.dumps(analysis, indent=2)}

    Variable summary:
    {json.dumps(variables, indent=2)}

    Allowed labels (return one exactly):
      - IV
      - BACKDOOR_REGRESSION
      - MATCHING
      - IPW
      - FRONTDOOR

    Think through each numbered node above, referencing the JSON keys
    (e.g., `analysis["instrument_strength"]`, `analysis["overlap_quality"]`,
    `analysis["frontdoor_criterion"]`).  On the **last line of your reply**
    output only the chosen label, in UPPER‑CASE, with no extra text.
    """).strip()

    reply = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    if reply in {"IV", "BACKDOOR_REGRESSION", "MATCHING", "IPW", "FRONTDOOR"}:
        return reply
    return "MATCHING"   

def enrich_dataset_analysis_with_graph(
    df, dataset_analysis, variables, llm, dataset_description, query
):
    graph = _llm_build_causal_graph(
        query=query,
        description=dataset_description,
        treatment=variables["treatment_variable"],
        outcome =variables["outcome_variable"],
        df=df,
        llm=llm
    )
    edges   = graph["edges"]
    covars  = variables.get("covariates", [])
    dataset_analysis["backdoor_condition"]   = _check_backdoor(
        edges, variables["treatment_variable"], variables["outcome_variable"], covars
    )
    dataset_analysis["frontdoor_criterion"]  = _check_frontdoor(
        edges, variables["treatment_variable"], variables["outcome_variable"]
    )
    dataset_analysis["causal_graph_edges"]   = edges
    dataset_analysis["causal_graph_dot"]     = graph["causal_graph"]
    return dataset_analysis

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

def decide_ps_method(
    df: pd.DataFrame,
    treat: str,
    estimand: str = "ATE"
) -> str:
    """
    Decide propensity-score method based on standardized mean differences.

    Parameters
    ----------
    df : pd.DataFrame
        Your dataset, including the binary treatment column.
    treat : str
        Name of the binary treatment column (values 0 or 1).
    estimand : str, {"ATE","ATT"}
        Which estimand to check balance for.

    Returns
    -------
    "IPW" or "Matching"
        If average absolute SMD < 0.1 → IPW (inverse‑prob weighting),
        else → Matching.
    """
    # split groups
    df_t = df[df[treat] == 1]
    df_c = df[df[treat] == 0]

    # only numeric covariates
    covariates = (
        df
        .drop(columns=[treat])
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    # preallocate
    smd_ate = np.zeros(len(covariates))
    smd_att = np.zeros(len(covariates))

    for i, col in enumerate(covariates):
        m_t, m_c = df_t[col].mean(), df_c[col].mean()
        s_t, s_c = df_t[col].std(ddof=0), df_c[col].std(ddof=0)
        pooled = np.sqrt((s_t**2 + s_c**2) / 2)

        # avoid zero‑div errors
        if pooled == 0 or s_t == 0:
            smd_ate[i] = 0.0
            smd_att[i] = 0.0
        else:
            smd_ate[i] = (m_t - m_c) / pooled
            smd_att[i] = (m_t - m_c) / s_t

    avg_ate = np.nanmean(np.abs(smd_ate))
    avg_att = np.nanmean(np.abs(smd_att))

    e = estimand.upper()
    if e == "ATE":
        return "IPW" if avg_ate < 0.1 else "Matching"
    elif e == "ATT":
        return "IPW" if avg_att < 0.1 else "Matching"
    else:
        raise ValueError("estimand must be 'ATE' or 'ATT'")

def select_method(
    dataset_analysis: Dict[str, Any],
    variables: Dict[str, Any],
    is_rct: bool = False,
    llm: Optional[BaseChatModel] = None,
    dataset_description: Optional[str] = None, 
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    logger.info("Starting causal method selection based on updated decision tree…")
    logger.debug(f"Inputs → dataset_analysis keys: {list(dataset_analysis.keys())}, "
                 f"variables: {variables}, is_rct: {is_rct}")

    covariates = variables.get("covariates", [])
    treatment_variable_type = variables.get("treatment_variable_type", "unknown")
    logger.info(f"Treatment variable type detected: '{treatment_variable_type}'")

    # ───────────────────────────── RCT BRANCH ─────────────────────────────
    if is_rct:
        logger.info("Entering RCT branch.")
        instrument_var = variables.get("instrument_variable")
        treatment_var = variables.get("treatment_variable")

        if instrument_var and treatment_var and instrument_var != treatment_var:
            logger.info("RCT → Encouragement design detected. Choosing IV.")
            return {
                "selected_method": INSTRUMENTAL_VARIABLE,
                "method_justification": (
                    f"RCT with instrument '{instrument_var}' differing from treatment "
                    f"'{treatment_var}'. Instrumental Variable method selected."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE],
                "alternatives": []
            }

        if covariates:
            logger.info("RCT → Covariates present. Choosing Linear Regression.")
            return {
                "selected_method": LINEAR_REGRESSION,
                "method_justification": (
                    "RCT with covariates supplied — using Linear Regression for precision."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[LINEAR_REGRESSION],
                "alternatives": []
            }

        logger.info("RCT → No covariates. Choosing Difference‑in‑Means.")
        return {
            "selected_method": DIFF_IN_MEANS,
            "method_justification": "Pure RCT without covariates — Difference‑in‑Means selected.",
            "method_assumptions": METHOD_ASSUMPTIONS[DIFF_IN_MEANS],
            "alternatives": []
        }

    # ─────────────────────── OBSERVATIONAL BRANCH ─────────────────────────
    logger.info("Entering Observational (non‑RCT) branch.")
    alternatives: List[str] = []

    time_var          = variables.get("time_variable")
    has_temporal      = dataset_analysis.get("temporal_structure", {}).get("has_temporal_structure", False)
    running_var       = variables.get("running_variable")
    cutoff_val        = variables.get("cutoff_value")
    instrument_var    = variables.get("instrument_variable")
    treatment_var     = variables.get("treatment_variable")
    outcome_variable = variables.get("outcome_variable")
    df = pd.read_csv(dataset_analysis['dataset_info']['file_path'])
    dataset_analysis = enrich_dataset_analysis_with_graph(
                df, dataset_analysis, variables, llm, dataset_description, original_query
            )
    logger.info("Updated backdoor and frontdoor conditions")
    # ---------------------- Binary treatments ----------------------------
    if treatment_variable_type == "binary":
        logger.info("Binary treatment logic activated.")

        if has_temporal:
            logger.info("Temporal structure found → selecting Difference‑in‑Differences.")
            return {
                "selected_method": DIFF_IN_DIFF,
                "method_justification": (
                    f"Temporal structure via '{time_var}'. Assuming parallel trends; using DiD."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[DIFF_IN_DIFF],
                "alternatives": alternatives
            }
        logger.info(f"Running variable '{running_var}' with cutoff {cutoff_val} detected → RDD chosen.")
        if running_var and cutoff_val is not None:
            logger.info(f"Running variable '{running_var}' with cutoff {cutoff_val} detected → RDD chosen.")
            return {
                "selected_method": REGRESSION_DISCONTINUITY,
                "method_justification": (
                    f"Running variable '{running_var}' around cutoff {cutoff_val}. "
                    "Regression Discontinuity Design selected."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[REGRESSION_DISCONTINUITY],
                "alternatives": alternatives
            }

        if instrument_var:
            logger.info(f"Instrument '{instrument_var}' available → choosing IV.")
            return {
                "selected_method": INSTRUMENTAL_VARIABLE,
                "method_justification": (
                    f"Instrument '{instrument_var}' deemed valid for non‑binary treatment."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE],
                "alternatives": alternatives
            }

        if covariates:
            logger.info("Covariates present → assessing overlap for PS method (Matching/IPW).")
            estimand = _llm_choose_estimand(
                query=original_query,
                description=dataset_description,
                treat_var=treatment_var,
                outcome_var=outcome_variable,
                llm=llm
            )
            ps_choice = decide_ps_method(df, treatment_var, estimand=estimand)
            chosen = (PROPENSITY_SCORE_MATCHING
                      if ps_choice == "Matching"
                      else PROPENSITY_SCORE_WEIGHTING)
            logger.info(f"Propensity‑score decision: '{ps_choice}' → selected '{chosen}'.")
            return {
                "selected_method": chosen,
                "method_justification": (
                    "Covariates observed & back‑door open; PS method selected based on overlap."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[chosen],
                "alternatives": []
            }

        if dataset_analysis.get("frontdoor_criterion"):
            logger.info("Front‑door criterion satisfied → choosing Front‑Door estimator.")
            return {
                "selected_method": "frontdoor_estimator",
                "method_justification": "Mediator meets front‑door conditions.",
                "method_assumptions": METHOD_ASSUMPTIONS.get("frontdoor_estimator", []),
                "alternatives": []
            }

    # ------------------ Continuous / Discrete treatments -----------------
    else:
        logger.info("Non‑binary (continuous/discrete) treatment logic activated.")

        if instrument_var:
            logger.info(f"Instrument '{instrument_var}' present → choosing IV.")
            return {
                "selected_method": INSTRUMENTAL_VARIABLE,
                "method_justification": (
                    f"Instrument '{instrument_var}' deemed valid for non‑binary treatment."
                ),
                "method_assumptions": METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE],
                "alternatives": []
            }

        if dataset_analysis.get("frontdoor_criterion"):
            logger.info("Front‑door criterion satisfied → choosing Front‑Door estimator.")
            return {
                "selected_method": "frontdoor_estimator",
                "method_justification": "Mediator meets front‑door conditions.",
                "method_assumptions": METHOD_ASSUMPTIONS.get("frontdoor_estimator", []),
                "alternatives": []
            }

        logger.info("No IV or front‑door mediator; defaulting to OLS.")
        return {
            "selected_method": LINEAR_REGRESSION,
            "method_justification": (
                "Fallback for non‑binary treatment without IV/front‑door — OLS chosen."
            ),
            "method_assumptions": METHOD_ASSUMPTIONS[LINEAR_REGRESSION],
            "alternatives": []
        }

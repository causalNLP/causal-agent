"""
Method validator component for causal inference methods.

This module validates the selected causal inference method against
dataset characteristics and available variables.
"""

from typing import Dict, List, Any, Optional
from cais.components.assumption_checks import *
import pandas as pd
from cais.config import get_llm_client

IV_USER_PROMPT_TEMPLATE = """\
You are evaluating an Instrumental Variables (IV / 2SLS) design. Read the CONTEXT,
accept the STATISTICAL EVIDENCE as already computed, then reason ONLY about the
untestable assumptions at a general level (note that they require domain/design justification).
Finally, return STRICT JSON matching our validation_result schema. No prose outside JSON.

=== CONTEXT ===
Method: Instrumental Variables (IV / 2SLS)
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Instrument(s) (Z): {instrument_variable}
Covariates (X): {covariates}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
First-stage F-statistic: {first_stage_F}
First-stage p-value: {first_stage_F_p}
Weak-IV flag (F < 10): {weak_iv_flag}

=== IV ASSUMPTIONS ===
- Relevance: already checked via first-stage stats above.
- Exclusion Restriction: untestable; requires substantive/domain justification.
- Independence: untestable; requires design/domain justification.
- Monotonicity: untestable; typically argued by design (no defiers).

=== YOUR TASK ===
1) Accept the statistical evidence for Relevance as given.
2) Highlight that Exclusion, Independence, and Monotonicity are untestable and require domain/design justification.
3) If evidence of weak IV appears (F < 10), add a concern and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON (no extra keys, no prose):

{{
  "valid": <true|false>,
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "recommended_method": "<string, e.g., 'instrumental_variable' or a fallback>",
  "assumptions": [
    "Relevance",
    "Exclusion restriction",
    "Independence",
    "Monotonicity"
  ]
}}

Constraints:
- Set "valid" = true only if (a) relevance is adequate per the stats above AND (b) you note that untestables need justification.
- Keep "concerns" terse (e.g., "Weak instrument: F<10", "Exclusion not justified").
- Use "alternative_suggestions" only if IV looks weak/inappropriate (e.g., "propensity_score_matching", "regression_adjustment").
- Return ONLY the JSON object.
"""

PSM_USER_PROMPT_TEMPLATE = """\
You are evaluating a Propensity Score Matching (PSM) design. Read the CONTEXT,
accept the STATISTICAL EVIDENCE as already computed, then reason about the
untestable assumption of conditional ignorability at a general level.
Finally, return STRICT JSON matching our validation_result schema. No prose outside JSON.

=== CONTEXT ===
Method: Propensity Score Matching (PSM)
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Covariates (X): {covariates}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
- Covariate balance (SMDs): {covariate_SMDs}
- Propensity score SMD: {propensity_SMD}
- Propensity score overlap summary: {ps_overlap}

=== PSM ASSUMPTION ===
- Conditional Ignorability: untestable; requires domain justification that X contains all confounders.

=== YOUR TASK ===
1) Accept the statistical evidence above (SMDs, PS overlap).
2) Note that Conditional Ignorability is untestable and must be argued externally.
3) If balance is poor or overlap is weak, add a concern and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:

{{
  "valid": <true|false>,
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "recommended_method": "<string, e.g., 'propensity_score_matching' or a fallback>",
  "assumptions": [
    "Conditional Ignorability"
  ]
}}
"""

DiD_USER_PROMPT_TEMPLATE = """\
You are evaluating a Difference-in-Differences (DiD) design. Read the CONTEXT,
accept the STATISTICAL EVIDENCE as already computed, then reason about the
untestable assumptions at a general level. Finally, return STRICT JSON
matching our validation_result schema. No prose outside JSON.

=== CONTEXT ===
Method: Difference-in-Differences (DiD)
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Group variable (G): {group_variable}
Time variable (T): {time_variable}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
- Adoption time (if inferred): {adoption_time}
- Pre-periods: {pre_periods}
- Post-periods: {post_periods}
- No anticipatory effects (design check): {no_anticipation}
- Pretrend test p-value: {pretrend_pval}
- Pretrend slope difference: {pretrend_slope_diff}

=== DiD ASSUMPTIONS ===
- Parallel Trends: partly testable (visual / pretrend) but largely requires justification if limited pre-periods.
- No Anticipatory Effects: usually assumed by design.

=== YOUR TASK ===
1) Accept the statistical evidence above.
2) Focus on whether parallel trends and no anticipation are plausible.
3) If evidence is weak or assumptions doubtful, add concise concerns and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:

{{
  "valid": <true|false>,
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "recommended_method": "<string, e.g., 'difference_in_differences' or a fallback>",
  "assumptions": [
    "Parallel Trends",
    "No Anticipatory Effects"
  ]
}}
"""

RDD_USER_PROMPT_TEMPLATE = """\
You are evaluating a Regression Discontinuity Design (RDD). Read the CONTEXT,
accept the STATISTICAL EVIDENCE as already computed, then reason about the
untestable continuity assumption at a general level. Finally, return STRICT JSON
matching our validation_result schema. No prose outside JSON.

=== CONTEXT ===
Method: Regression Discontinuity Design (RDD)
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Running variable (R): {running_variable}
Cutoff (c): {cutoff_value}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
- Compliance with cutoff rule (share correctly assigned): {compliance_rate}
- Misclassified share: {misclassified_share}
- Observations near cutoff — left: {n_left}, right: {n_right}
- Mean outcome left: {mean_left}, right: {mean_right}
- Jump (right - left): {jump}

=== RDD ASSUMPTION ===
- Continuity at cutoff: untestable; must be argued by design. Visual inspection for a jump supports this assumption.

=== YOUR TASK ===
1) Accept the statistical evidence above.
2) Focus on whether continuity at cutoff is plausible given the observed jump and compliance.
3) If evidence is weak or sample around cutoff too small, add concise concerns and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:

{{
  "valid": <true|false>,
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "recommended_method": "<string, e.g., 'regression_discontinuity_design' or a fallback>",
  "assumptions": [
    "Continuity at Cutoff"
  ]
}}
"""


def build_iv_user_prompt(validation_result: Dict[str, Any], variables: Dict[str, Any]) -> str:
    """Fill the IV user-only prompt with values from validation_result/variables."""
    fst = (validation_result.get("evidence", {})
                             .get("iv", {})
                             .get("first_stage", {}))
    return IV_USER_PROMPT_TEMPLATE.format(
        outcome_variable    = variables.get("outcome_variable"),
        treatment_variable  = variables.get("treatment_variable"),
        instrument_variable = variables.get("instrument_variable"),
        covariates          = variables.get("covariates", []),
        first_stage_F       = fst.get("first_stage_F"),
        first_stage_F_p     = fst.get("first_stage_F_p"),
        weak_iv_flag        = fst.get("weak_iv_flag"),
    )

def validate_method(method_info: Dict[str, Any], dataset_analysis: Dict[str, Any], 
                    variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the selected causal method against dataset characteristics.
    
    Args:
        method_info: Information about the selected method from decision_tree
        dataset_analysis: Dataset analysis results from dataset_analyzer
        variables: Identified variables from query_interpreter
        
    Returns:
        Dict with validation results:
            - valid: Boolean indicating if method is valid
            - concerns: List of concerns/issues with the selected method
            - alternative_suggestions: Alternative methods if the selected method is problematic
            - recommended_method: Updated method recommendation if issues are found
    """
    method = method_info.get("selected_method")
    assumptions = method_info.get("method_assumptions", [])
    
    # Get required variables
    treatment = variables.get("treatment_variable")
    outcome = variables.get("outcome_variable")
    covariates = variables.get("covariates", [])
    time_variable = variables.get("time_variable")
    group_variable = variables.get("group_variable")
    instrument_variable = variables.get("instrument_variable")
    running_variable = variables.get("running_variable")
    cutoff_value = variables.get("cutoff_value")
    
    # Initialize validation result
    validation_result = {
        "valid": True,
        "concerns": [],
        "alternative_suggestions": [],
        "recommended_method": method,
        "evidence" : {}
    }
    
    # Common validations for all methods
    if treatment is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment variable is not identified")
    
    if outcome is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Outcome variable is not identified")
    
    # Method-specific validations
    if method == "propensity_score_matching":
        validate_propensity_score_matching(validation_result, dataset_analysis, variables)
    
    elif method == "regression_adjustment":
        validate_regression_adjustment(validation_result, dataset_analysis, variables)
    
    elif method == "instrumental_variable":
        validate_instrumental_variable(validation_result, dataset_analysis, variables)
    
    elif method == "difference_in_differences":
        validate_difference_in_differences(validation_result, dataset_analysis, variables)
    
    elif method == "regression_discontinuity_design":
        validate_regression_discontinuity(validation_result, dataset_analysis, variables)
    
    elif method == "backdoor_adjustment":
        validate_backdoor_adjustment(validation_result, dataset_analysis, variables)
    
    # If there are serious concerns, recommend alternatives
    if not validation_result["valid"]:
        validation_result["recommended_method"] = recommend_alternative(
            method, validation_result["concerns"], method_info.get("alternatives", [])
        )
    user_prompt = build_iv_user_prompt(validation_result, variables)
    client = get_llm_client()
    res = client.invoke(user_prompt)
    
    # Make sure assumptions are listed in the validation result
    validation_result["assumptions"] = assumptions
    print("--------------------------")
    print("Validation result:", validation_result)
    print("--------------------------")
    print("--------------------------")
    print("LLM Response:", res)
    print("--------------------------")
    ok
    return validation_result


def validate_propensity_score_matching(validation_result: Dict[str, Any], 
                                      dataset_analysis: Dict[str, Any],
                                      variables: Dict[str, Any]) -> None:
    treatment = variables.get("treatment_variable")
    covariates = variables.get("covariates", [])
    df = pd.read_csv(dataset_analysis['dataset_info']['file_path'])

    if treatment is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment variable is not identified")
        return
    if not covariates:
        validation_result["concerns"].append("Few/No covariates identified; balance may be inadequate")

    try:
        diag = psm_diagnostics(df, treatment, covariates)
        validation_result.setdefault("evidence", {})["psm"] = diag

        # Threshold checks
        bad_covs = [k for k, v in diag["covariate_SMDs"].items() if abs(v) > 0.10]
        if bad_covs:
            validation_result["concerns"].append(f"Imbalanced covariates (|SMD|>0.10): {', '.join(bad_covs)[:300]}")
        if abs(diag["propensity_SMD"]) > 0.10:
            validation_result["concerns"].append("Propensity score distributions differ (|SMD(PS)|>0.10).")
        if not diag["ps_overlap"]["range_overlap"]:
            validation_result["concerns"].append("Poor common support: minimal PS range overlap between groups.")
            validation_result["alternative_suggestions"].append("regression_adjustment")
    except Exception as e:
        validation_result["concerns"].append(f"PSM diagnostics failed: {e}")

    # Explicit note about ignorability
    validation_result.setdefault("evidence", {}).setdefault(
        "psm_notes",
        "Conditional ignorability is untestable; justify covariate set with domain knowledge."
    )


def validate_regression_discontinuity(validation_result: Dict[str, Any], 
                                      dataset_analysis: Dict[str, Any],
                                      variables: Dict[str, Any]) -> None:
    """
    RDD checks per your text:
    - Assumption (local randomization/exclusion) is untestable.
    - Visual inspection around cutoff: plot outcome vs running near cutoff; look for a jump.
    - Enforced by design: treatment determined by a cutoff variable.
    """
    running_variable = variables.get("running_variable")
    cutoff_value = variables.get("cutoff_value")
    treatment = variables.get("treatment_variable")
    outcome = variables.get("outcome_variable")
    df = pd.read_csv(dataset_analysis['dataset_info']['file_path'])

    # Required fields
    if running_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("No running variable identified, required for RDD.")
        validation_result["alternative_suggestions"].append("propensity_score_matching")
        return
    if cutoff_value is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("No cutoff value identified, required for RDD.")
        validation_result["alternative_suggestions"].append("propensity_score_matching")
        return
    if treatment is None or outcome is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment and/or outcome variable missing for RDD.")
        validation_result["alternative_suggestions"].append("regression_adjustment")
        return
    if df is None:
        validation_result["concerns"].append("Dataframe missing in dataset_analysis['df']; cannot run RDD checks.")
        return

    # 1) Enforced-by-design check: is treatment determined by cutoff?
    design = rdd_design_compliance(df, running_variable, treatment, cutoff_value)
    validation_result.setdefault("evidence", {})["rdd_design"] = design
    if not design.get("ok", False):
        validation_result["concerns"].append(
            "Treatment does not closely follow cutoff assignment (low compliance with T ≈ 1{X≥c})."
        )

    # 2) Visual inspection prep: outcome vs running near cutoff
    win = rdd_window_summary(df, running_variable, outcome, cutoff_value, h=None)
    validation_result["evidence"]["rdd_window"] = win
    if (win.get("n_left", 0) < 5) or (win.get("n_right", 0) < 5):
        validation_result["concerns"].append("Too few observations near cutoff for reliable visual inspection.")

    # (Optional) Provide bins for plotting downstream (no plotting here)
    try:
        h = float(win["window_h"]) if win.get("window_h") is not None else None
        if h is not None and h > 0:
            bins = rdd_bins_for_plot(df, running_variable, outcome, cutoff_value, h, bins_per_side=10)
            validation_result["evidence"]["rdd_bins"] = bins
    except Exception as _:
        pass  # plotting prep is best-effort

    # 3) Assumption status (per text)
    validation_result.setdefault("evidence", {})["rdd_notes"] = {
        "assumption_status": "Untestable; assess visually around cutoff for an abrupt jump.",
        "design_enforcement": "Treatment is (or should be) determined by a cutoff variable.",
        "visual_recommendation": "Plot outcome vs running within a symmetric window around the cutoff; inspect for a jump."
    }

    # If no jump in the simple window means, add a soft concern (still visual, not a test)
    jump = win.get("jump_right_minus_left")
    if jump is not None and abs(jump) < 1e-8:
        validation_result["concerns"].append("No visible outcome jump in a narrow window around cutoff (visual).")



def validate_instrumental_variable(validation_result: Dict[str, Any], 
                                   dataset_analysis: Dict[str, Any],
                                   variables: Dict[str, Any]) -> None:
    """
    IV checks per your text:
    - Relevance: first-stage F-statistic (D ~ Z + X)
    - Exclusion restriction: untestable (domain justification)
    - Monotonicity & Independence: usually design/domain-justified
    """
    instrument = variables.get("instrument_variable")
    treatment = variables.get("treatment_variable")
    outcome = variables.get("outcome_variable")
    controls = variables.get("covariates", []) or None
    df = pd.read_csv(dataset_analysis['dataset_info']['file_path'])

    if instrument is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("No instrumental variable identified, required for IV.")
        validation_result["alternative_suggestions"].append("propensity_score_matching")
        return
    if treatment is None or outcome is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment and/or outcome missing for IV.")
        validation_result["alternative_suggestions"].append("regression_adjustment")
        return
    if df is None:
        validation_result["concerns"].append("Dataframe missing in dataset_analysis['df']; cannot compute first-stage F.")
        return

    # --- Relevance: first-stage F ---
    try:
        fs = iv_first_stage_relevance(df, instrument, treatment, controls)
        validation_result.setdefault("evidence", {})["iv"] = {"first_stage": fs}
        if fs.get("weak_iv_flag", False):
            validation_result["concerns"].append("Weak instrument: first-stage F < 10 (potential bias).")
            validation_result["alternative_suggestions"].append("propensity_score_matching")
    except Exception as e:
        validation_result["concerns"].append(f"IV relevance check failed: {e}")

    # --- Untestable/Design-justified assumptions (surface explicitly) ---
    validation_result["evidence"]["iv_notes"] = {
        "exclusion_restriction": "Untestable; justify via substantive/domain knowledge.",
        "monotonicity": "Usually argued by design (e.g., no defiers in encouragement).",
        "independence": "As-if random instrument conditional on controls; justify by design."
    }

def validate_difference_in_differences(validation_result: Dict[str, Any], 
                                       dataset_analysis: Dict[str, Any],
                                       variables: Dict[str, Any]) -> None:
    """
    Validate difference-in-differences method requirements (per your text):
    - Ensure treatment variable exists and time indicates treatment timing.
    - Check no anticipatory effects (design-based).
    - Focus on parallel trends: visual-style pre-trend slopes test when >=3 pre periods,
      else defer to domain knowledge.
    """
    time_variable = variables.get("time_variable")
    group_variable = variables.get("group_variable")
    treatment = variables.get("treatment_variable")
    outcome = variables.get("outcome_variable")
    df = pd.read_csv(dataset_analysis['dataset_info']['file_path'])

    # Hard requirements
    if time_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("No time variable identified, required for DiD.")
        validation_result["alternative_suggestions"].append("propensity_score_matching")
        return
    if group_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("No group variable identified, required for DiD.")
        validation_result["alternative_suggestions"].append("propensity_score_matching")
        return
    if treatment is None or outcome is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment and/or outcome variable missing for DiD.")
        validation_result["alternative_suggestions"].append("regression_adjustment")
        return

    # 1) Does the time variable indicate treatment timing?
    tt = infer_treatment_timing(df, time_variable, group_variable, treatment)
    validation_result.setdefault("evidence", {})["did_timing"] = tt
    if not tt["ok"]:
        validation_result["concerns"].append("Time variable does not clearly encode treatment timing across groups.")
        validation_result["alternative_suggestions"].append("regression_adjustment")

    # 2) No anticipatory effects (design-based; flag if treated group shows treatment pre-adoption)
    if tt.get("ok_no_anticipation") is False:
        validation_result["concerns"].append("Possible anticipatory effects: treated group exhibits treatment before adoption time.")

    # 3) Parallel trends (visual proxy): if >= 3 pre periods, test slopes difference; else defer to domain knowledge
    pre = tt.get("pre_periods", [])
    if len(set(pre)) >= 3:
        pretest = pretrend_parallel_test_with_periods(df, time_variable, group_variable, outcome, pre, tt["treated_group"])
        validation_result["evidence"]["did_pretrend"] = pretest
        if pretest.get("ok") is False:
            validation_result["concerns"].append(
                f"Pre-trend slopes differ (p={pretest.get('pval'):.3g}); parallel trends questionable."
            )
            validation_result["alternative_suggestions"].append("synthetic_control")
    else:
        validation_result.setdefault("evidence", {})["did_pretrend"] = {
            "insufficient_pre_periods": True,
            "n_pre_periods": len(set(pre)),
            "note": "Only two (or fewer) pre periods; justify parallel trends via domain knowledge."
        }

    # Final note reflecting your text
    validation_result.setdefault("evidence", {})["did_notes"] = (
        "No anticipatory effects typically valid by design if timing is exogenous. "
        "Parallel trends is primary; use visual inspection of pre-period slopes."
    )

# def validate_regression_discontinuity(validation_result: Dict[str, Any], 
#                                     dataset_analysis: Dict[str, Any],
#                                     variables: Dict[str, Any]) -> None:
#     """
#     Validate regression discontinuity method requirements.
    
#     Args:
#         validation_result: Current validation result to update
#         dataset_analysis: Dataset analysis results
#         variables: Identified variables
#     """ 
#     running_variable = variables.get("running_variable")
#     cutoff_value = variables.get("cutoff_value")
    
#     if running_variable is None:
#         validation_result["valid"] = False
#         validation_result["concerns"].append(
#             "No running variable identified, which is required for regression discontinuity"
#         )
#         validation_result["alternative_suggestions"].append("propensity_score_matching")
    
#     if cutoff_value is None:
#         validation_result["valid"] = False
#         validation_result["concerns"].append(
#             "No cutoff value identified, which is required for regression discontinuity"
#         )
#         validation_result["alternative_suggestions"].append("propensity_score_matching")
    
#     # Check for discontinuity at threshold
#     discontinuities = dataset_analysis.get("discontinuities", {})
#     has_discontinuity = discontinuities.get("has_discontinuities", False)
    
#     if not has_discontinuity:
#         validation_result["valid"] = False
#         validation_result["concerns"].append(
#             "No clear discontinuity detected at the threshold, which is necessary for this method"
#         )
#         validation_result["alternative_suggestions"].append("regression_adjustment") 

def validate_backdoor_adjustment(validation_result: Dict[str, Any], 
                               dataset_analysis: Dict[str, Any],
                               variables: Dict[str, Any]) -> None:
    """
    Validate backdoor adjustment method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """
    covariates = variables.get("covariates", [])
    
    if len(covariates) == 0:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No covariates identified for backdoor adjustment"
        )
        validation_result["alternative_suggestions"].append("regression_adjustment")


def recommend_alternative(method: str, concerns: List[str], alternatives: List[str]) -> str:
    """
    Recommend an alternative method if the current one has issues.
    
    Args:
        method: Current method
        concerns: List of concerns with the current method
        alternatives: List of alternative methods suggested by the decision tree
        
    Returns:
        String with the recommended method
    """
    # If there are alternatives, recommend the first one
    if alternatives:
        return alternatives[0]
    
    # If no alternatives, use regression adjustment as a fallback
    if method != "regression_adjustment":
        return "regression_adjustment"
    
    # If regression adjustment is also problematic, use propensity score matching
    return "propensity_score_matching" 
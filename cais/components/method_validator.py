"""
Method validator component for causal inference methods.

This module validates the selected causal inference method against
dataset characteristics and available variables.
"""

from typing import Dict, List, Any, Optional
from cais.components.assumption_checks import *
import pandas as pd
from cais.config import get_llm_client

IV_VALIDATION_PROMPT_TEMPLATE = """\
You are evaluating an Instrumental Variables (IV / 2SLS) design. 
Read the CONTEXT, and the STATISTICAL EVIDENCE, then reason ONLY about the ntestable assumptions.
Finally, return STRICT JSON matching our validation_result schema. 

=== CONTEXT ===
Method: Instrumental Variables (IV / 2SLS)
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Instrument(s) (Z): {instrument_variable}
Control Covariates (X): {covariates}
Data Description: {data_description}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
First-stage F-statistic: {first_stage_F}
First-stage p-value: {first_stage_F_p}
Weak-IV flag (F < 10): {weak_iv_flag}

=== IV ASSUMPTIONS ===
- Relevance: This assumption states that the instrument is correlated with treatment, i.e., the instrument affects treatment.
  We can check this assumption using the first-stage F-statistic, i.e., from the regression of treatment on instrument.
- Exclusion Restriction: This is an untestable assumption, which states that the instrument affects the outcome only through the treatment.
  We typically justify this assumption using domain knowledge.
- Independence: This is again an untestable assumption that requires design/domain justification.
  It states that the instrument is not related to unobserved confounders affecting the outcome and treatment.
  If the data comes from a randomized encouragement design, this assumption is easily argued. Otherwise,
  we must justify using domain knowledge or argue the instrument is "as-good-as-random" conditional on the control variables.
- Monotonicity: This is untestable and usually argued by design. Moreover, this assumption is more relevant for LATE interpretation of IV, where
  we assume there are no defiers (people who would do the opposite of the treatment assignment).

=== YOUR TASK ===
1) Critically assess the statistical evidence for Relevance as given.
2) Critically assess whether the untestable assumptions (Exclusion, Independence, Monotonicity (for LATE framework)) are plausible in general terms.
3) If all assumptions are plausible, i.e.,
   a. Selected instrument is relevant
   b. Exclusion restriction is justified in this setting
   c. Independence is justified in this setting
   d. Monotonicity is justified in this setting (if LATE interpretation is desired)
   then set "valid" = true. Otherwise, set "valid" = false, and add alternative suggestions as well as the concerns.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:
{{
  "valid": <true|false>,
  "reference_study": ["<relevant paper/article 1>", "<relevant paper/article 2>", ...],
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "assumptions": ["Relevance", "Exclusion restriction", "Independence", "Monotonicity"]
}}

Constraints:
- Set "valid" = true only if relevance is adequate as per the stats above and the untestable assumptions are plausible.
- In the reference_study field, add the names of 2-3 papers or articles that you rely on to make your assessment.
- Keep "concerns" brief (e.g., "Weak instrument: F<10", "Exclusion not justified"). The concerns should be raised if valid = false. Otherwise, keep it empty.
- Use "alternative_suggestions" only if valid = false, i.e., IV is not the right method. You should suggest alternative methods, e.g., "propensity_score_matching", "regression_adjustment".
- Return ONLY the JSON object
"""

## this should be about validating conditional ignorability + positivity only
PS_VALIDATION_PROMPT_TEMPLATE = """\
You are evaluating whether methods like matching or inverse probability weighting are appropriate. Read the CONTEXT,
and the STATISTICAL EVIDENCE, then reason whether conditional ignorability and positivity are plausible in this setting.
Finally, return STRICT JSON matching our validation_result schema. 

=== CONTEXT ===
Method: Propensity Score Matching (PSM) 
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Confounders (X): {Confounders}
Data Description: {data_description}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
- Covariate balance (SMDs): {covariate_SMDs}
- Propensity score SMD: {propensity_SMD}
- Propensity score overlap summary: {ps_overlap}

=== PSM ASSUMPTION ===
- Conditional Ignorability: This is an untestable assumption. We justify its plausibility by arguing that we have measured all confounders affecting both treatment and outcome. 
                            This means you need to check that there are no unmeasured confounders affecting both treatment and outcome.
- Positivity: This assumption states that every unit has a positive probability of receiving treatment given the confounders.
              We can partially check this by looking at the propensity score distributions. 

=== YOUR TASK ===
1) Critically assess the statistical evidence above (SMDs, PS overlap). 
2) Note that both Conditional Ignorability and Positivity are untestable. We must justify their plausibility using domain knowledge + statistical results presented above.
3) If we cannot justify these assumptions, add concise concerns and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:
{{
  "valid": <true|false>,
  "reference_study": ["<relevant paper/article 1>", "<relevant paper/article 2>", ...],
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "assumptions": ["Conditional Ignorability"]
}}

Constraints:
- Set "valid" = true only if balance is adequate as per the stats above and conditional ignorability is plausible.
- In the reference_study field, add the names of 2-3 papers or articles that you rely on to make your assessment.
- Keep "concerns" brief. This should be raised if valid = false.
- Use "alternative_suggestions" only if valid = false, i.e., PSM is not the right method. You should suggest alternative methods, "instrumental_variable")
- Return ONLY the JSON object
"""

DiD_VALIDATION_PROMPT_TEMPLATE = """\
You are evaluating whether a Difference-in-Differences (DiD) design is appropriate. Read the CONTEXT
and the STATISTICAL EVIDENCE, then reason about whether the assumptions underlying DiD are justified.
Finally, return STRICT JSON matching our validation_result schema.

=== CONTEXT ===
Method: Difference-in-Differences (DiD)
Outcome (Y): {outcome_variable}
Treatment (D): {treatment_variable}
Group variable (G): {group_variable}
Time variable (T): {time_variable}
Data Description: {data_description}

Note that the treatment variable indicates whether an observation is treated or not. Note that DiD has two cases:
1) Staggered adoption: Different groups get treated at different times. In this case, treatment indicates whether a group (also called unit)
is treated at a given time or not.
2) Canonical DiD: All groups get treated at the same time. In this case, treatment indicates whether a group is in the treated group or not.

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
- Specific adoption time (if inferred): {adoption_time}
- Pre-periods: {pre_periods}
- Post-periods: {post_periods}
- Pretrend test p-value: {pretrend_pval}
- Pretrend slope difference: {pretrend_slope_diff}

=== DiD ASSUMPTIONS ===
- Parallel Trends: This assumption states that the treatment and control groups would have followed similar trends over time in the absence of treatment.
 We can test this partially. Usually, we plot the outcomes over time to visually inspect the presence of parallel trends.
 However, in case we do not have data for multiple pre-periods, we cannot test this. Instead, we need to justify this assumption using domain knowledge.
- No Anticipatory Effects: This assumption states that the treatment effects do not occur before the treatment is actually implemented.
 This is usually argued by design.

=== YOUR TASK ===
1) Critically assess the statistical evidence above.
2) Focus on determining whether parallel trends and no anticipation assumptions are plausible in this setting.
3) Also check if the correct set of variables has been selected. For instance, the time variable should give information about treatment timing.
 It cannot be a time-related variable that is not related to treatment timing.
4) If evidence is weak or assumptions are doubtful, add concise concerns and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:
{{
 "valid": <true|false>,
 "reference_study": ["<relevant paper/article 1>", "<relevant paper/article 2>", ...],
 "concerns": ["<short issue 1>", "<short issue 2>", ...],
 "alternative_suggestions": ["<method 1>", "<method 2>", ...],
 "assumptions": ["Parallel Trends", "No Anticipatory Effects"]
}}

Constraints:
- Set "valid" = true only if parallel trends and no anticipatory effects are plausible, and the right set of variables has been selected.
- In the reference_study field, add the names of 2-3 papers or articles that you rely on to make your assessment.
- Keep "concerns" brief. These should be raised if valid = false. Otherwise, keep it empty.
- Use "alternative_suggestions" only if valid = false and DiD is applied incorrectly for this problem.
 You should suggest alternative methods, e.g., "regression_adjustment", "propensity_score_matching".
- Return ONLY the JSON object
"""


## we can add statistical test for RDD. 
RDD_VALIDATION_PROMPT_TEMPLATE = """\
You are evaluating a Regression Discontinuity Design (RDD). Read the CONTEXT and the STATISTICAL EVIDENCE, and then reason about the
untestable continuity assumption at a general level. 
Finally, return STRICT JSON matching our validation_result schema. 

=== CONTEXT ===
Method: Regression Discontinuity Design (RDD)
Outcome (Y): {outcome_variable}
Running variable (R): {running_variable}
Cutoff (c): {cutoff_value}
Description: {data_description}

=== STATISTICAL EVIDENCE (already computed; do NOT recompute) ===
- Compliance with cutoff rule (share correctly assigned): {compliance_rate}
- Misclassified share: {misclassified_share}
- Observations near cutoff — left: {n_left}, right: {n_right}
- Mean outcome left: {mean_left}, right: {mean_right}
- Jump (right - left): {jump}
- McCrary test p-value: {mccrary_pval}

=== RDD ASSUMPTION ===
- Continuity at cutoff: This assumption states that in the absence of treatment, the outcome would have been continuous at the cutoff.
  This is an untestable assumption. We justify its plausibility by arguing how this is implied by design i.e.
   treatment is determined by a cutoff variable. We also recommend inspecting the outcome around the cutoff visually to see if there is an abrupt jump.

=== YOUR TASK ===
1) Critically assess the statistical information above. More over, check strictly if RDD is applied correctly i.e. treatment can be determined by a cutoff variable.
2) Focus on whether continuity at cutoff is plausible given the observed jump and compliance.
3) If evidence is weak or sample around cutoff too small, add concise concerns and suggest alternatives.

=== OUTPUT FORMAT (STRICT JSON ONLY; EXACT KEYS) ===
Return ONLY this JSON:

{{
  "valid": <true|false>,
   "reference_study": ["<relevant paper/article 1>", "<relevant paper/article 2>", ...],
  "concerns": ["<short issue 1>", "<short issue 2>", ...],
  "alternative_suggestions": ["<method 1>", "<method 2>", ...],
  "assumptions": ["Continuity at Cutoff"]
}}

Constraints:
- Set "valid" = true only if the discontinuity assumption around the cutoff is plausible. 
- In the reference_study field, add the names of 2-3 papers or articles that you rely on to make your assessment.
- Keep "concerns" brief. These should be raised if valid = false. Otherwise, keep it empty.
- Use "alternative_suggestions" only if valid = false and RDD is not the right. You should suggest alternative methods. 
- Return ONLY the JSON object
"""


def build_iv_user_prompt(validation_result: Dict[str, Any], variables: Dict[str, Any]) -> str:
    """Fill the IV user-only prompt with values from validation_result/variables."""
    fst = (validation_result.get("evidence", {})
                             .get("iv", {})
                             .get("first_stage", {}))
    return IV_VALIDATION_PROMPT_TEMPLATE.format(
        outcome_variable    = variables.get("outcome_variable"),
        treatment_variable  = variables.get("treatment_variable"),
        instrument_variable = variables.get("instrument_variable"),
        covariates          = variables.get("covariates", []),
        first_stage_F       = fst.get("first_stage_F"),
        first_stage_F_p     = fst.get("first_stage_F_p"),
        weak_iv_flag        = fst.get("weak_iv_flag"),)

def validate_propensity_score_matching(validation_result: Dict[str, Any], 
                                      dataset_analysis: Dict[str, Any],
                                      variables: Dict[str, Any]) -> None:
    """
    This method validates if propensity score matching is appropriate or not
    """

    treatment = variables.get("treatment_variable")
    ## Instead of covariates, this should be confounders. W
    ## We must have changed the prompts earlier to separate out confounders from covariates.

    confounders = variables.get("covariates", [])
    if confounders is None or len(confounders) == 0:
        raise ValueError("No confounders identified for propensity score matching")
    
    df = pd.read_csv(dataset_analysis['dataset_info']['file_path'])

    ## If treatment is None, we cannot do anything. We need to exit right away.
    if treatment is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment variable is not identified")
        return
    if not confounders:
        validation_result["concerns"].append("Few/No confounders identified; balance may be inadequate")

    try:
        diag = psm_diagnostics(df, treatment, confounders)
        validation_result.setdefault("evidence", {})["psm"] = diag

        # Threshold checks
        bad_confounders = [k for k, v in diag["covariate_SMDs"].items() if abs(v) > 0.10]
        if bad_confounders:
            validation_result["concerns"].append(f"Imbalanced confounders (|SMD|>0.10): {', '.join(bad_confounders)[:300]}")
        if abs(diag["propensity_SMD"]) > 0.10:
            validation_result["concerns"].append("Propensity score distributions differ (|SMD(PS)|>0.10).")
        if not diag["ps_overlap"]["range_overlap"]:
            validation_result["concerns"].append("Poor common support: minimal PS range overlap between groups.")
            validation_result["alternative_suggestions"].append("regression_adjustment")
    except Exception as e:
        validation_result["concerns"].append(f"PSM diagnostics failed: {e}")

    # Explicit note about ignorability
    validation_result.setdefault("evidence", {}).setdefault(
        "psm_notes", "Conditional ignorability is untestable; justify covariate set with domain knowledge "
        "+ empirical evidence about covariate balance.")


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
    
    ## If outcome is None, we cannot do anything. We need to exit right away. 
    if outcome is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Outcome is variable missing for RDD.")
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

    ## Adding McCrary test 
    mccrary_p = mccrary_test(df, running_variable, cutoff_value)
    if mccrary_p is np.nan:
        validation_result["concerns"].append("McCrary test could not be computed.")
    else:
        if mccrary_p < 0.05:
            validation_result["concerns"].append(f"McCrary test p={mccrary_p:.3g} suggests manipulation around cutoff.")

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
    ## we need to change this to controls. This should happen after we the method selection step. 
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
        "monotonicity": "Applied towards LATE framework. Usually argued by design (e.g., no defiers in encouragement).",
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
    validation_result = {"valid": True, "reference_study": [], "concerns": [], "alternative_suggestions": [],
                         "evidence" : {}}
    
    # Common validations for all methods; For RDD, we may not have a treatment variable. 
    ## We might have to create one. In RDD, treatment / control is not determined initially, but we create one based on the cutoff. 
    ## Some datasets may have a treatment variable already, but not all.
    if treatment is None and method != "regression_discontinuity_design":
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
    return validation_result



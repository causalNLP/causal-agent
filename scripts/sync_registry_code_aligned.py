"""Synchronise the reviewed registry with the executable CAIS route inventory.

This is deliberately fail-closed: a code route is recorded as implemented, but
never enabled for automatic causal routing until its identification contract,
inference procedure, and expert review are complete.
"""
from copy import copy
from pathlib import Path

import openpyxl
import json


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "CAIS_Registry_Code_Aligned.xlsx"
OUTPUT_PATH = ROOT / "CAIS_Registry_Code_Aligned_updated.xlsx"


def row_dict(ws, row):
    headers = [c.value for c in ws[1]]
    return dict(zip(headers, row))


def append_row(ws, values):
    headers = [c.value for c in ws[1]]
    row = ws.max_row + 1
    ws.cell(row, 1).value = values.get(headers[0])
    for col, name in enumerate(headers[1:], 2):
        ws.cell(row, col).value = values.get(name)
    if row > 2:
        for col in range(1, ws.max_column + 1):
            source = ws.cell(row - 1, col)
            target = ws.cell(row, col)
            if source.has_style:
                target._style = copy(source._style)
            if source.number_format:
                target.number_format = source.number_format
    return row


def upsert(ws, key, value, updates):
    headers = [c.value for c in ws[1]]
    idx = headers.index(key)
    for row in ws.iter_rows(min_row=2):
        if row[idx].value == value:
            for field, new_value in updates.items():
                ws.cell(row[0].row, headers.index(field) + 1).value = new_value
            return row[0].row
    return append_row(ws, {key: value, **updates})


def write_preference_graph(wb):
    """Create the machine-readable conditional preference graph.

    These are comparisons, not a total ranking.  Each row is inactive unless
    its explicit evidence conditions are satisfied.  In particular, IV is not
    preferred merely because its first stage is strong: exclusion and
    independence evidence are separate conditions.
    """
    title = "Preference_Graph"
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)
    ws.append([
        "Edge_ID", "Preferred_Strategy_ID", "Fallback_Strategy_ID", "Estimand_IDs",
        "Conditions_JSON", "Unknown_Policy", "Edge_Type", "Rationale",
    ])
    rows = [
        ("P_RANDOMIZATION_OVER_ADJUSTMENT", "S1", "S7", "E01,E06",
         {"all": [{"field": "randomization_protocol", "equals": "supported"}]},
         "inactive", "preference", "Design-based assignment is preferred when the randomization protocol is documented."),
        ("P_IV_LATE_OVER_ADJUSTMENT", "S16", "S7", "E07",
         {"all": [
             {"field": "instrument_relevance", "equals": "supported"},
             {"field": "iv_exclusion", "equals": "supported"},
             {"field": "iv_independence", "equals": "supported"},
             {"field": "iv_monotonicity", "equals": "supported"},
         ]}, "inactive", "preference",
         "A validated local-IV contract can be preferred for LATE; a strong first stage alone is insufficient."),
        ("P_IV_ATE_HOMOGENEITY_OVERRIDE", "S16", "S7", "E01",
         {"all": [
             {"field": "instrument_relevance", "equals": "supported"},
             {"field": "iv_exclusion", "equals": "supported"},
             {"field": "iv_independence", "equals": "supported"},
             {"field": "iv_homogeneity", "equals": "supported"},
         ]}, "inactive", "override",
         "ATE interpretation requires the additional homogeneity advisory; otherwise no IV ATE preference is activated."),
        ("P_SHARP_RDD_OVER_ADJUSTMENT", "S29", "S7", "E01,E02",
         {"all": [{"field": "sharp_rdd_design", "equals": "supported"}, {"field": "rdd_no_manipulation", "equals": "supported"}]},
         "inactive", "preference", "A supported sharp-RDD design is local to the cutoff and is not a population ATE."),
        ("P_FUZZY_RDD_OVER_ADJUSTMENT", "S30", "S7", "E07",
         {"all": [{"field": "fuzzy_rdd_design", "equals": "supported"}, {"field": "local_first_stage", "equals": "supported"}]},
         "inactive", "preference", "Fuzzy RDD is preferred for a local complier effect only when a local first stage exists."),
        ("P_DID_OVER_ADJUSTMENT", "S34", "S7", "E02",
         {"all": [{"field": "did_2x2_design", "equals": "supported"}, {"field": "parallel_trends", "equals": "supported"}]},
         "inactive", "preference", "The 2x2 DiD contract is preferred only for its ATT target and supported panel structure."),
        ("P_SYNTHETIC_CONTROL_OVER_ADJUSTMENT", "S41", "S7", "E01,E02",
         {"all": [{"field": "synthetic_control_design", "equals": "supported"}, {"field": "pre_treatment_fit", "equals": "supported"}]},
         "inactive", "preference", "Synthetic control is model-dependent and remains conditional on donor-pool and fit diagnostics."),
        ("P_SURVIVAL_G_METHOD_OVER_GENERIC_ADJUSTMENT", "S46", "S7", "E33",
         {"all": [{"field": "right_censored_survival", "equals": "supported"}, {"field": "rmst_horizon_support", "equals": "supported"}]},
         "inactive", "preference", "Survival routes are compared only for the survival/RMST target and adequate follow-up support."),
    ]
    for edge_id, preferred, fallback, estimands, conditions, unknown, edge_type, rationale in rows:
        ws.append([edge_id, preferred, fallback, estimands, json.dumps(conditions), unknown, edge_type, rationale])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = copy(wb["Estimator_Routes"][1][0].font)


def main():
    wb = openpyxl.load_workbook(PATH)

    # The source route was present but incorrectly marked as absent.  Keep its
    # historical Route_ID while aligning the estimator ID with METHOD_MAPPING.
    routes = wb["Estimator_Routes"]
    upsert(routes, "Route_ID", "R_S30_fuzzy_RD_local_Wald", {
        "Estimator_ID": "fuzzy_regression_discontinuity_design",
        "Estimator_Family": "Fuzzy regression discontinuity / local Wald",
        "Target_Identified_Functional": "Local Wald ratio of the outcome discontinuity to the treatment-receipt discontinuity at the cutoff; LATE for compliers under IV/RD assumptions.",
        "Supported_Estimand_IDs": "C24 (conditional)",
        "Treatment_Type": "Binary treatment receipt",
        "Outcome_Type": "Numeric",
        "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
        "Test_Coverage": "tests/cais/methods/fuzzy_regression_discontinuity/",
        "Eligibility": "Running variable, finite cutoff, binary treatment receipt, local first stage and compatible bandwidth.",
        "Relation_Status": "code_verified_conditional_mapping",
        "Explanation": "The implementation computes a local Wald ratio; it does not identify a population ATE.",
        "Safe_For_Auto_Routing": "No",
    })
    upsert(routes, "Route_ID", "R_S18_2SLS_with_homogeneity", {
        "Implemented_in_CAIS": "Yes — IV/2SLS callable; homogeneity interpretation is advisory LLM-only",
        "Relation_Status": "code_verified_partial_mapping",
        "Explanation": "The coefficient may be computed, but population-ATE interpretation requires a favorable advisory homogeneity argument; otherwise route to LATE interpretation or report no causal ATE.",
        "Safe_For_Auto_Routing": "No",
    })

    new_routes = [
        {
            "Route_ID": "R_S7_AIPW", "Strategy_ID": "S7", "Estimator_ID": "aipw",
            "Estimator_Family": "Cross-fitted augmented inverse probability weighting",
            "Target_Identified_Functional": "ATE, ATT or ATC under the declared binary-treatment backdoor contract.",
            "Supported_Estimand_IDs": "E01,E02,E03", "Treatment_Type": "Binary 0/1", "Outcome_Type": "Numeric",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
            "Test_Coverage": "tests/cais/methods/doubly_robust/test_estimator.py",
            "Eligibility": "Binary treatment, numeric outcome, pre-treatment covariates and explicit target.",
            "Relation_Status": "code_verified_restricted_mapping",
            "Explanation": "Cross-fitted AIPW; clipping is reported and does not silently trim the target population.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S7_DML", "Strategy_ID": "S7", "Estimator_ID": "dml",
            "Estimator_Family": "Cross-fitted orthogonal-score DML",
            "Target_Identified_Functional": "ATE under the binary-treatment orthogonal-score contract.",
            "Supported_Estimand_IDs": "E01", "Treatment_Type": "Binary 0/1", "Outcome_Type": "Numeric",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
            "Test_Coverage": "tests/cais/methods/doubly_robust/test_estimator.py",
            "Eligibility": "Binary treatment, numeric outcome and pre-treatment covariates.",
            "Relation_Status": "code_verified_restricted_mapping",
            "Explanation": "ATT/ATC and longitudinal DML are outside this route.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S7_TMLE", "Strategy_ID": "S7", "Estimator_ID": "tmle",
            "Estimator_Family": "Canonical cross-fitted TMLE",
            "Target_Identified_Functional": "ATE for a binary treatment and binary outcome.",
            "Supported_Estimand_IDs": "E01", "Treatment_Type": "Binary 0/1", "Outcome_Type": "Binary 0/1",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
            "Test_Coverage": "tests/cais/methods/test_tmle_and_bounds.py",
            "Eligibility": "Binary treatment/outcome and pre-treatment covariates.",
            "Relation_Status": "code_verified_restricted_mapping",
            "Explanation": "Continuous, survival and longitudinal TMLE are not represented by this route.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S45_MSM", "Strategy_ID": "S45", "Estimator_ID": "marginal_structural_model",
            "Estimator_Family": "Stabilized-IPTW marginal structural model",
            "Target_Identified_Functional": "Coefficient of the user-specified marginal structural model.",
            "Supported_Estimand_IDs": "E24,E25,E38", "Treatment_Type": "Binary longitudinal", "Outcome_Type": "Final numeric outcome",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; explicit longitudinal options required",
            "Test_Coverage": "Paper-based implementation; route remains non-certified", "Eligibility": "Long format, id/time columns, treatment history, confounders and final outcome.",
            "Relation_Status": "code_verified_restricted_mapping", "Explanation": "No automatic MSM specification; sequential assumptions remain advisory.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S45_g_computation", "Strategy_ID": "S45", "Estimator_ID": "longitudinal_g_computation",
            "Estimator_Family": "Two-time-point parametric g-formula",
            "Target_Identified_Functional": "E[Y^{a0,a1}] minus E[Y^{a0*,a1*}] for two static regimes.",
            "Supported_Estimand_IDs": "E24,E25", "Treatment_Type": "Binary longitudinal", "Outcome_Type": "Final numeric outcome",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; explicit longitudinal options required",
            "Test_Coverage": "Paper-based implementation; route remains non-certified", "Eligibility": "Exactly two time points, intermediate covariate, baseline covariates and final outcome.",
            "Relation_Status": "code_verified_restricted_mapping", "Explanation": "General horizons and dynamic regimes are not claimed.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S63_Manski_bounds", "Strategy_ID": "S63", "Estimator_ID": "manski_bounds",
            "Estimator_Family": "Manski worst-case bounds", "Target_Identified_Functional": "Sharp ATE identified set under declared bounded outcome support.",
            "Supported_Estimand_IDs": "E01", "Treatment_Type": "Binary 0/1", "Outcome_Type": "Numeric bounded",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; identified set only", "Test_Coverage": "tests/cais/methods/test_tmle_and_bounds.py",
            "Eligibility": "Binary treatment and explicit outcome lower/upper support bounds.", "Relation_Status": "code_verified_restricted_mapping",
            "Explanation": "Returns an interval and never a point estimate.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S64_Balke_Pearl_bounds", "Strategy_ID": "S64", "Estimator_ID": "balke_pearl_bounds",
            "Estimator_Family": "Balke–Pearl sharp IV bounds", "Target_Identified_Functional": "Sharp ATE identified set under the binary IV model.",
            "Supported_Estimand_IDs": "E01", "Treatment_Type": "Binary instrument/treatment", "Outcome_Type": "Binary 0/1",
            "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; identified set only", "Test_Coverage": "tests/cais/methods/test_tmle_and_bounds.py",
            "Eligibility": "Binary instrument, treatment and outcome; optional explicit monotonicity.", "Relation_Status": "code_verified_restricted_mapping",
            "Explanation": "Exogeneity, exclusion and consistency are not empirically certified by the LP.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S46_s_learner_RMST", "Strategy_ID": "S46",
            "Estimator_ID": "s_learner_rmst", "Estimator_Family": "S-Learner survival RMST CATE",
            "Target_Identified_Functional": "Conditional treatment effect on restricted mean survival time at a prespecified horizon.",
            "Supported_Estimand_IDs": "C25 (conditional)", "Treatment_Type": "Binary treatment",
            "Outcome_Type": "Right-censored time-to-event", "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
            "Test_Coverage": "tests/cais/methods/test_survival_meta_learners.py",
            "Eligibility": "Duration/event columns, binary treatment, prespecified RMST horizon, censoring support.",
            "Relation_Status": "code_verified_conditional_mapping", "Explanation": "IPCW or selected survival base learner; causal interpretation remains conditional on survival assumptions.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S46_t_learner_RMST", "Strategy_ID": "S46",
            "Estimator_ID": "t_learner_rmst", "Estimator_Family": "T-Learner survival RMST CATE",
            "Target_Identified_Functional": "Conditional treatment effect on restricted mean survival time at a prespecified horizon.",
            "Supported_Estimand_IDs": "C26 (conditional)", "Treatment_Type": "Binary treatment",
            "Outcome_Type": "Right-censored time-to-event", "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
            "Test_Coverage": "tests/cais/methods/test_survival_meta_learners.py",
            "Eligibility": "Duration/event columns, binary treatment, prespecified RMST horizon, censoring support.",
            "Relation_Status": "code_verified_conditional_mapping", "Explanation": "Separate treatment-arm survival learners; no automatic claim of proportional hazards.", "Safe_For_Auto_Routing": "No",
        },
        {
            "Route_ID": "R_S46_x_learner_RMST", "Strategy_ID": "S46",
            "Estimator_ID": "x_learner_rmst", "Estimator_Family": "X-Learner survival RMST CATE",
            "Target_Identified_Functional": "Conditional treatment effect on restricted mean survival time at a prespecified horizon.",
            "Supported_Estimand_IDs": "C27 (conditional)", "Treatment_Type": "Binary treatment",
            "Outcome_Type": "Right-censored time-to-event", "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
            "Test_Coverage": "tests/cais/methods/test_survival_meta_learners.py",
            "Eligibility": "Duration/event columns, binary treatment, prespecified RMST horizon, censoring support.",
            "Relation_Status": "code_verified_conditional_mapping", "Explanation": "Survival X-learner with IPCW pseudo-outcomes; calibration and inference remain restricted.", "Safe_For_Auto_Routing": "No",
        },
    ]
    # Replace legacy placeholder route metadata for the now implemented
    # point-treatment contracts.  The historical route IDs are retained so
    # workbook links remain stable.
    upsert(routes, "Route_ID", "R_S7_AIPW", {
        "Strategy_ID": "S7", "Estimator_ID": "aipw", "Estimator_Family": "Cross-fitted augmented inverse probability weighting",
        "Target_Identified_Functional": "ATE, ATT or ATC under the declared binary-treatment backdoor contract.",
        "Supported_Estimand_IDs": "E01,E02,E03", "Treatment_Type": "Binary 0/1", "Outcome_Type": "Numeric",
        "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested", "Test_Coverage": "tests/cais/methods/doubly_robust/test_estimator.py",
        "Eligibility": "Binary treatment, numeric outcome, pre-treatment covariates and explicit target.", "Relation_Status": "code_verified_restricted_mapping",
        "Explanation": "Cross-fitted AIPW; clipping is reported and does not silently trim the target population.", "Safe_For_Auto_Routing": "No",
    })
    upsert(routes, "Route_ID", "R_S7_TMLE", {
        "Strategy_ID": "S7", "Estimator_ID": "tmle", "Estimator_Family": "Canonical cross-fitted TMLE",
        "Target_Identified_Functional": "ATE for a binary treatment and binary outcome.", "Supported_Estimand_IDs": "E01",
        "Treatment_Type": "Binary 0/1", "Outcome_Type": "Binary 0/1", "Implemented_in_CAIS": "Yes — callable through METHOD_MAPPING; unit-tested",
        "Test_Coverage": "tests/cais/methods/test_tmle_and_bounds.py", "Eligibility": "Binary treatment/outcome and pre-treatment covariates.",
        "Relation_Status": "code_verified_restricted_mapping", "Explanation": "Continuous, survival and longitudinal TMLE are not represented by this route.", "Safe_For_Auto_Routing": "No",
    })
    upsert(routes, "Route_ID", "R_S45_marginal_structural_model", {
        "Implemented_in_CAIS": "No — legacy placeholder superseded by R_S45_MSM",
        "Relation_Status": "superseded_placeholder",
        "Explanation": "Use R_S45_MSM for the restricted stabilized-IPTW MSM implementation.",
        "Safe_For_Auto_Routing": "No",
    })
    existing = {r[0].value for r in routes.iter_rows(min_row=2)}
    for route in new_routes:
        if route["Route_ID"] not in existing:
            append_row(routes, route)

    contracts = wb["REVIEW_Contracts"]
    contract_rows = [
        dict(contract_id="C24", strategy_id="S30", source_route_id="R_S30_fuzzy_RD_local_Wald", estimator="fuzzy_regression_discontinuity_design", estimand_id="E07", variant="Fuzzy RDD local Wald", precise_target="Local average treatment effect for units whose treatment receipt is changed by crossing the cutoff.", intervention_contrast="Treatment receipt induced by eligibility just above versus below the cutoff", target_population="Local compliers at the cutoff", effect_scale="mean difference", identified_functional="lim_{x→c} [E(Y|X=x)-E(Y|X=c)] / [E(D|X=x)-E(D|X=c)]", design_scope="Fuzzy RDD with a binary treatment receipt and a local first stage", identification_kind="point", conditional_theoretical_review="reviewed_conditional", interpretation_note="The result is local and complier-specific; it is not a population ATE.", limitations="Requires continuity, no precise manipulation, exclusion, monotonicity and a nonzero local first stage.", reference="Imbens and Lemieux (2008)", code_status_from_codex="Yes — callable through METHOD_MAPPING", code_tests_claimed="tests/cais/methods/fuzzy_regression_discontinuity/", code_independently_verified="yes_code_and_tests", statistical_inference_verified="no", expert_approval="pending", route_enabled="no", effective_status="REVIEWED_NOT_EXECUTABLE"),
        dict(contract_id="C25", strategy_id="S46", source_route_id="R_S46_s_learner_RMST", estimator="s_learner_rmst", estimand_id="E33", variant="S-Learner RMST CATE", precise_target="E[RMST(1,τ)-RMST(0,τ)|X=x]", intervention_contrast="Treatment 1 versus treatment 0", target_population="Units represented by the covariate distribution", effect_scale="RMST difference", identified_functional="Conditional RMST contrast at prespecified horizon τ", design_scope="Binary treatment with right-censored survival outcome", identification_kind="conditional", conditional_theoretical_review="reviewed_conditional", interpretation_note="CATE on the RMST scale, not an individual-level directly observed effect.", limitations="Requires consistency, exchangeability, positivity, independent censoring, no competing-risk misspecification and adequate support.", reference="Rosenbaum and Rubin (1983); Uno et al. (2014)", code_status_from_codex="Yes — callable through METHOD_MAPPING", code_tests_claimed="tests/cais/methods/test_survival_meta_learners.py", code_independently_verified="yes_code_and_tests", statistical_inference_verified="no", expert_approval="pending", route_enabled="no", effective_status="REVIEWED_NOT_EXECUTABLE"),
        dict(contract_id="C26", strategy_id="S46", source_route_id="R_S46_t_learner_RMST", estimator="t_learner_rmst", estimand_id="E33", variant="T-Learner RMST CATE", precise_target="E[RMST(1,τ)-RMST(0,τ)|X=x]", intervention_contrast="Treatment 1 versus treatment 0", target_population="Units represented by the covariate distribution", effect_scale="RMST difference", identified_functional="Conditional RMST contrast at prespecified horizon τ", design_scope="Binary treatment with right-censored survival outcome", identification_kind="conditional", conditional_theoretical_review="reviewed_conditional", interpretation_note="Separate survival models are fitted in each treatment arm.", limitations="Requires consistency, exchangeability, positivity, independent censoring and adequate arm-specific support.", reference="Künzel et al. (2019)", code_status_from_codex="Yes — callable through METHOD_MAPPING", code_tests_claimed="tests/cais/methods/test_survival_meta_learners.py", code_independently_verified="yes_code_and_tests", statistical_inference_verified="no", expert_approval="pending", route_enabled="no", effective_status="REVIEWED_NOT_EXECUTABLE"),
        dict(contract_id="C27", strategy_id="S46", source_route_id="R_S46_x_learner_RMST", estimator="x_learner_rmst", estimand_id="E33", variant="X-Learner RMST CATE", precise_target="E[RMST(1,τ)-RMST(0,τ)|X=x]", intervention_contrast="Treatment 1 versus treatment 0", target_population="Units represented by the covariate distribution", effect_scale="RMST difference", identified_functional="Conditional RMST contrast at prespecified horizon τ", design_scope="Binary treatment with right-censored survival outcome", identification_kind="conditional", conditional_theoretical_review="reviewed_conditional", interpretation_note="Uses treatment-arm pseudo-outcomes with censoring weights.", limitations="Requires consistency, exchangeability, positivity, independent censoring, correct censoring treatment and adequate support.", reference="Künzel et al. (2019)", code_status_from_codex="Yes — callable through METHOD_MAPPING", code_tests_claimed="tests/cais/methods/test_survival_meta_learners.py", code_independently_verified="yes_code_and_tests", statistical_inference_verified="no", expert_approval="pending", route_enabled="no", effective_status="REVIEWED_NOT_EXECUTABLE"),
    ]
    contract_ids = {r[0].value for r in contracts.iter_rows(min_row=2)}
    for row in contract_rows:
        if row["contract_id"] not in contract_ids:
            append_row(contracts, row)

    coverage = wb["REVIEW_Route_Coverage"]
    for route in ["R_S30_fuzzy_RD_local_Wald", *[r["Route_ID"] for r in new_routes]]:
        contract = next((r for r in contract_rows if r["source_route_id"] == route), None)
        if contract:
            upsert(coverage, "Source_Route_ID", route, {
                "Strategy_ID": contract["strategy_id"], "Estimator_ID": contract["estimator"],
                "Source_Code_Mapping": "Yes — callable through METHOD_MAPPING; tests present",
                "Contracts": contract["contract_id"], "Independent_Code_Verified": "yes_code_and_tests",
                "Auto_Route": "no", "Reason": "Fail-closed: causal contract, inference and expert review are not all certified.",
            })

    # Make the source strategy inventory reflect executable support without
    # claiming that identification or inference has been certified.
    strategies = wb["Strategies"]
    for sid, note in {
        "S30": "Implementation present: fuzzy_regression_discontinuity_design; conditional local-Wald route.",
        "S46": "Implementations present: survival RMST S/T/X meta-learners; conditional routes only.",
    }.items():
        upsert(strategies, "Strategy_ID", sid, {
            "Implemented_in_CAIS_code": "Yes — conditional / not auto-certified",
            "Method_review_status": "Code present; contract and inference review pending",
            "Methodological_caveat": note,
        })

    # A compact, code-grounded assumption inventory for the reviewer.  The
    # original 380 source assumptions remain untouched in Assumptions.
    title = "CODE_ASSUMPTION_CONTRACTS"
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)
    headers = ["method_id", "assumption_id", "label", "classification", "evidence_type", "failure_policy", "implemented_check", "causal_claim_warning"]
    ws.append(headers)
    rows = [
        ("all", "sutva", "SUTVA", "advisory", "llm_argument", "advisory", "check_sutva", "LLM reasoning is not proof."),
        ("observational", "covariate_balance", "Observed covariate balance", "testable", "statistical_diagnostic", "warn", "check_cond_ignorability", "Balance does not prove exchangeability."),
        ("observational", "conditional_exchangeability", "Conditional exchangeability", "advisory", "llm_argument", "advisory", "check_conditional_exchangeability", "Requires design/domain justification."),
        ("propensity/GPS", "positivity", "Treatment positivity / overlap", "testable", "statistical_diagnostic", "block", "check_positivity", "Diagnostics concern observed support only."),
        ("instrumental_variable", "iv_relevance", "Instrument relevance", "testable", "statistical_diagnostic", "block", "check_iv_relevance", "A strong first stage does not prove exclusion."),
        ("instrumental_variable", "iv_exclusion", "Exclusion restriction", "advisory", "llm_argument", "advisory", "check_iv_exclusion", "Must be justified substantively."),
        ("instrumental_variable_ate", "iv_homogeneity", "Homogeneous treatment effects for ATE interpretation", "advisory", "llm_argument", "advisory", "check_iv_homogeneity + route_iv_estimand", "A negative or inconclusive advisory does not block 2SLS execution; it blocks only the ATE interpretation."),
        ("difference_in_differences", "parallel_trends", "Parallel trends", "testable", "statistical_diagnostic", "block", "check_parallel_trends", "Pre-trends are diagnostic, not proof."),
        ("regression_discontinuity", "rdd_no_manipulation", "No manipulation", "testable", "statistical_diagnostic", "warn", "check_rdd_no_manipulation", "The test is advisory for RDD routing."),
        ("regression_discontinuity", "rdd_continuity_potential_outcomes", "Potential-outcome continuity", "advisory", "llm_argument", "advisory", "check_rdd_continuity_potential_outcomes", "Not directly testable from observed outcomes."),
        ("survival_meta_learners", "independent_censoring", "Independent censoring", "advisory", "llm_argument", "advisory", "validate_survival_assumptions", "Censoring diagnostics cannot establish causal independence."),
        ("survival_meta_learners", "rmst_horizon_support", "RMST horizon support", "testable", "statistical_diagnostic", "block", "validate_survival_assumptions", "Requires adequate event/follow-up support."),
    ]
    for row in rows:
        ws.append(row)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = copy(wb["Estimator_Routes"][1][0].font)

    # Preferences are represented as conditional registry edges rather than
    # an ordered list of methods or a hard-coded decision tree.
    write_preference_graph(wb)

    # Provenance note in START_HERE without overwriting the original content.
    start = wb["START_HERE"]
    start.append(["SYNC_UPDATE", "2026-11-14", "Synchronized with current METHOD_MAPPING, survival RMST routes, fuzzy RDD and ASSUMPTION_SPECS. All new routes remain fail-closed."])

    # Refresh the human-readable review metadata, which otherwise retained the
    # original 11-route/13-contract scope after the code-alignment updates.
    readme = wb["REVIEW_README"]
    for row in readme.iter_rows(min_row=1):
        if row[0].value == "Scope":
            row[1].value = "Executable-code-aligned inventory: 91 routes, including fuzzy RDD, IPW variants, meta-learners and survival RMST meta-learners; all remain fail-closed."
        elif row[0].value == "Code status":
            row[1].value = "Repository implementation and selected tests reviewed by Codex; independent expert certification and inference review remain pending; 0 auto-routable contracts."
        elif row[0].value == "Source workbook":
            row[1].value = "CAIS_Registry_transversal_audited.xlsx, with code-alignment updates recorded in this workbook."

    try:
        wb.save(PATH)
        print(f"Updated {PATH}")
    except PermissionError:
        wb.save(OUTPUT_PATH)
        print(f"Original workbook is locked; wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

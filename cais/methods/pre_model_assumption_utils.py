"""
Pre-modeling assumption checks for causal inference methods.

These checks are run before estimation to verify whether the data and study
design satisfy the assumptions required by the chosen causal method.
Statistical checks use the raw data directly. Non-testable assumptions
are assessed via LLM reasoning based on the dataset description.

Each check returns an AssumptionResult with:
    passed   : bool | None  (None => inconclusive)
    reasoning: str
    details  : dict         (raw stats — F, p, SMDs, ...)
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from pydantic import BaseModel, Field

from cais.models import AssumptionResult, AssumptionVariables
from cais.methods.utils import (
    calculate_standardized_differences,
    check_overlap,
)
from cais.methods.instrumental_variable.diagnostics import calculate_first_stage_f_statistic
from cais.methods.difference_in_differences.diagnostics import validate_parallel_trends
from cais.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)


# _____________________________________________________________________________
# LLM-based assumption argumentation (non-statistically-testable assumptions)
# _____________________________________________________________________________

class _LLMAssumptionVerdict(BaseModel):
    passed: Optional[bool] = Field(
        None,
        description="True if the assumption is plausibly satisfied, False if there is "
                    "a clear reason to doubt it, None if there is insufficient information.",
    )
    reasoning: str = Field(
        ...,
        description="Concise (2-4 sentences) justification grounded in the dataset "
                    "description, variable semantics, or domain knowledge.",
    )
    missing_info: Optional[str] = Field(
        None,
        description="If passed is null, what additional information would be needed. "
                    "Otherwise null.",
    )


def _llm_argue_assumption(
    assumption_name: str,
    assumption_description: str,
    vars: AssumptionVariables,
    llm,
    extra_context: Optional[str] = None,
) -> AssumptionResult:
    """Ask the LLM to argue for/against a non-statistically-testable assumption."""
    if llm is None:
        return AssumptionResult(
            passed=None,
            reasoning=(
                f"'{assumption_name}' is not statistically testable and no LLM was "
                f"provided to reason about it. Must be justified by study design or "
                f"domain knowledge."
            ),
        )

    prompt = f"""
You are a causal inference expert. Assess whether the following assumption is
plausibly satisfied for the analysis described below. Use the dataset
description and variables to argue concretely. Do not assume facts that are
not present in the description.

Assumption: {assumption_name}
Definition: {assumption_description}

Dataset description:
{vars.dataset_description or "(not provided)"}

Variables involved:
{vars.variables_summary}

If you recognize this study from the literature, use your prior knowledge
about its design, mechanisms, and known methodological concerns to inform
your assessment — not just the description provided above.

{extra_context or ""}

Respond ONLY as JSON matching this schema:
{{
  "passed": true | false | null,
  "reasoning": "<2-4 sentence justification>",
  "missing_info": "<if passed is null, specify what additional information would be needed. Otherwise set to null.>"
}}

Use null for "passed" if the dataset description is insufficient to argue either way.
""".strip()

    try:
        raw = call_llm_with_json_output(llm, prompt)
        verdict = _LLMAssumptionVerdict(**(raw or {}))
        details: Dict[str, Any] = {"assumption": assumption_name}
        if verdict.missing_info:
            details["missing_info"] = verdict.missing_info

        disclaimer = (
            " Note: this assessment relies on LLM reasoning and is sensitive to the "
            "quality and completeness of the dataset description provided."
        )
        return AssumptionResult(
            passed=verdict.passed,
            reasoning=verdict.reasoning + disclaimer,
            details=details,
        )
    except Exception as exc:
        logger.warning("LLM assumption check failed for '%s': %s", assumption_name, exc)
        return AssumptionResult(
            passed=None,
            reasoning=f"LLM check failed: {exc}. Assumption must be justified manually.",
        )


# _____________________________________________________________________________
# SUTVA  (needed for every method)
# _____________________________________________________________________________

def check_sutva(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """SUTVA: no interference between units, no hidden treatment versions."""
    return _llm_argue_assumption(
        assumption_name="SUTVA (Stable Unit Treatment Value Assumption)",
        assumption_description=(
            "(1) No interference: one unit's treatment does not affect another unit's "
            "potential outcomes. (2) No hidden versions of the treatment: the treatment "
            "is administered consistently across treated units."
        ),
        vars=vars,
        llm=llm,
        extra_context=(
            "Important: SUTVA is an idealized assumption that is technically violated "
            "in most real-world settings. Do NOT fail this assumption simply because "
            "minor spillovers or small treatment variations are theoretically possible. "
            "Only return passed=false if there is a strong, concrete reason grounded in "
            "the dataset description — such as explicit network structure, shared "
            "environments where interference is the primary mechanism, or clearly "
            "documented treatment heterogeneity. If the dataset comes from a published "
            "causal study, assume the researchers judged SUTVA to be reasonable unless "
            "the description contradicts this."
        ),
    )


# _____________________________________________________________________________
# Ignorability / Conditional ignorability (RCT and observational)
# _____________________________________________________________________________

def check_cond_ignorability(
    vars: AssumptionVariables,
    smd_threshold: float = 0.1,
) -> AssumptionResult:
    """Partial test of ignorability: covariate balance on observables."""
    if not vars.covariates or vars.df is None:
        return AssumptionResult(
            passed=None,
            reasoning="No covariates or dataframe provided; balance check skipped.",
        )
    smds = calculate_standardized_differences(vars.df, vars.treatment, vars.covariates)
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    passed = len(imbalanced) == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Randomization check on {len(vars.covariates)} covariates "
            f"(|SMD| < {smd_threshold}). "
            f"{'All balanced.' if passed else f'Imbalanced: {list(imbalanced.keys())}.'}"
        ),
        details={"smds": smds, "threshold": smd_threshold, "imbalanced": imbalanced},
    )


# _____________________________________________________________________________
# Positivity / overlap  (IPW, matching, GPS)
# _____________________________________________________________________________

def check_positivity(
    vars: AssumptionVariables,
    propensity_scores: np.ndarray,
    overlap_threshold: float = 0.5,
    extreme_ps_bounds: tuple = (0.1, 0.9),
    max_extreme_pct: float = 0.05,
) -> AssumptionResult:
    """0 < P(T=1|X) < 1 across the support of X."""
    overlap = check_overlap(vars.df, vars.treatment, propensity_scores, threshold=overlap_threshold)
    lo, hi = extreme_ps_bounds
    n_extreme = int(((propensity_scores < lo) | (propensity_scores > hi)).sum())
    pct_extreme = n_extreme / len(propensity_scores) if len(propensity_scores) else 0.0
    passed = overlap["sufficient_overlap"] and pct_extreme < max_extreme_pct
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Overlap proportion: {overlap['overlap_proportion']:.3f} "
            f"(threshold {overlap_threshold}). "
            f"{n_extreme} obs ({pct_extreme:.1%}) outside [{lo}, {hi}]. "
            f"{'OK.' if passed else 'Consider trimming or restricting to common support.'}"
        ),
        details={**overlap, "n_extreme_ps": n_extreme, "pct_extreme_ps": pct_extreme},
    )


# _____________________________________________________________________________
# IV-specific checks
# _____________________________________________________________________________

def check_iv_relevance(
    vars: AssumptionVariables,
    f_threshold: float = 10.0,
) -> AssumptionResult:
    """First-stage F-test for instrument strength."""
    f, p = calculate_first_stage_f_statistic(
        vars.df, vars.treatment, vars.instruments, vars.covariates
    )
    if f is None:
        return AssumptionResult(
            passed=None,
            reasoning="First-stage F-statistic could not be computed.",
        )
    passed = f >= f_threshold
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"First-stage F = {f:.2f} (threshold {f_threshold}). "
            f"{'Strong instrument.' if passed else 'Weak instrument warning.'}"
        ),
        details={"f_statistic": f, "p_value": p, "threshold": f_threshold},
    )


def check_iv_exclusion(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Exclusion restriction: Z affects Y only through T."""
    return _llm_argue_assumption(
        "Exclusion restriction",
        "The instrument Z affects the outcome Y only through the treatment T, with no direct effect.",
        vars, llm,
    )


def check_iv_exogeneity(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Instrument exogeneity: Z is independent of unobserved confounders."""
    return _llm_argue_assumption(
        "Instrument exogeneity (independence)",
        "Z is as good as randomly assigned with respect to unobserved confounders of T and Y.",
        vars, llm,
    )


def check_iv_monotonicity(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Monotonicity: no defiers."""
    return _llm_argue_assumption(
        "Monotonicity (LATE)",
        "There are no defiers: the instrument never moves any unit in the opposite direction "
        "of its average effect on treatment uptake.",
        vars, llm,
    )


# _____________________________________________________________________________
# DiD-specific checks
# _____________________________________________________________________________

def check_parallel_trends(
    vars: AssumptionVariables,
    **kwargs,
) -> AssumptionResult:
    """Parallel trends: treatment and control had similar outcome trends pre-treatment."""
    res = validate_parallel_trends(
        vars.df, vars.time_var, vars.outcome, vars.treatment,
        vars.treatment_period_start, **kwargs
    )
    return AssumptionResult(
        passed=res.get("valid"),
        reasoning=res.get("details", ""),
        details={"p_value": res.get("p_value"), "error": res.get("error")},
    )


def check_no_anticipation(vars: AssumptionVariables) -> AssumptionResult:
    """No anticipation: treatment has no effect before implementation.

    Runs a placebo DiD restricted to pre-treatment periods, avoiding contamination
    from the actual treatment effect. Uses Q() quoting for compatibility with
    column names that contain underscores or special characters.
    """
    import statsmodels.formula.api as smf

    df = vars.df
    time_var = vars.time_var
    group_var = vars.group_var
    outcome = vars.outcome
    treatment = vars.treatment
    covariates = list(vars.covariates or [])
    t0 = vars.treatment_period_start
    t_placebo = vars.placebo_period_start

    if t_placebo is None or t0 is None:
        return AssumptionResult(
            passed=None,
            reasoning="placebo_period_start or treatment_period_start not provided.",
            details={},
        )
    if t_placebo >= t0:
        return AssumptionResult(
            passed=False,
            reasoning="placebo_period_start must be strictly before treatment_period_start.",
            details={},
        )

    # Restrict to pre-treatment periods to avoid contamination from actual treatment
    df_pre = df[df[time_var] < t0].copy()
    df_pre["__post_pl__"] = (df_pre[time_var] >= t_placebo).astype(int)
    df_pre["__did_pl__"] = df_pre[treatment] * df_pre["__post_pl__"]

    try:
        formula = f"Q('{outcome}') ~ Q('{treatment}') + __post_pl__ + __did_pl__"
        if covariates:
            cov_terms = " + ".join("Q('" + c + "')" for c in covariates)
            formula += " + " + cov_terms
        formula += f" + C(Q('{group_var}')) + C(Q('{time_var}'))"

        model = smf.ols(formula=formula, data=df_pre)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df_pre[group_var]})

        effect = float(res.params["__did_pl__"])
        p_val = float(res.pvalues["__did_pl__"])
        passed = p_val > 0.10
        return AssumptionResult(
            passed=passed,
            reasoning=(
                f"Placebo treatment effect (pre-treatment only): {effect:.4f} "
                f"(p={p_val:.4f}). Test {'passed' if passed else 'failed'}."
            ),
            details={"effect_estimate": effect, "p_value": p_val},
        )
    except Exception as exc:
        return AssumptionResult(
            passed=None,
            reasoning=f"Placebo test could not be completed: {exc}",
            details={"error": str(exc)},
        )


def check_baseline_outcome_balance(
    vars: AssumptionVariables,
    smd_threshold: float = 0.1,
) -> AssumptionResult:
    """Comparable pre-treatment outcome levels between groups."""
    pre = vars.df[vars.df[vars.time_var] < vars.treatment_period_start]
    if pre.empty:
        return AssumptionResult(passed=None, reasoning="No pre-treatment data available.")
    smd = calculate_standardized_differences(pre, vars.treatment, [vars.outcome]).get(vars.outcome, np.nan)
    if pd.isna(smd):
        return AssumptionResult(
            passed=None,
            reasoning="Could not compute SMD on baseline outcome (missing data or no variance).",
        )
    passed = abs(smd) <= smd_threshold
    return AssumptionResult(
        passed=passed,
        reasoning=f"Baseline outcome SMD = {smd:.3f} (threshold {smd_threshold}).",
        details={"smd_pre_outcome": smd, "threshold": smd_threshold},
    )


def check_stable_group_composition(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Stable group composition: no differential attrition due to treatment."""
    return _llm_argue_assumption(
        "Stable group composition",
        "Unit composition of treatment and control groups does not change as a result "
        "of treatment (no differential attrition or selective entry/exit).",
        vars, llm,
    )


# _____________________________________________________________________________
# Frontdoor-specific checks
# _____________________________________________________________________________

def check_frontdoor_full_mediation(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Full mediation: M fully captures the effect of T on Y."""
    return _llm_argue_assumption(
        "Full mediation",
        "The mediator M fully captures the effect of treatment T on outcome Y; "
        "there is no direct T→Y path outside of the T→M→Y pathway.",
        vars, llm,
    )


def check_frontdoor_no_TM_confounding(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """No unobserved confounding between treatment and mediator."""
    return _llm_argue_assumption(
        "No T-M confounding",
        "The relationship between the treatment T and the mediator M is unconfounded. "
        "There are no unobserved variables that affect both T and M.",
        vars, llm,
    )


def check_frontdoor_T_blocks_MY(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Treatment blocks all confounding paths between mediator and outcome."""
    return _llm_argue_assumption(
        "T blocks M->Y confounding",
        "Conditioning on the treatment T removes all back-door paths between "
        "the mediator M and the outcome Y.",
        vars, llm,
    )


def check_frontdoor_positivity(
    vars: AssumptionVariables,
    min_count: int = 5,
) -> AssumptionResult:
    """Frontdoor positivity: P(M=m|X=x) > 0 for all relevant (x, m) combinations."""
    combos = vars.df.groupby([vars.treatment, vars.mediator]).size().reset_index(name='count')
    total_combos = vars.df[vars.treatment].nunique() * vars.df[vars.mediator].nunique()
    observed_combos = len(combos)
    empty = total_combos - observed_combos
    sparse = int((combos['count'] < min_count).sum())
    passed = empty == 0 and sparse == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"{observed_combos}/{total_combos} (treatment, mediator) combinations observed. "
            f"{empty} empty, {sparse} sparse (< {min_count} obs). "
            f"{'Positivity satisfied.' if passed else 'Some combinations are empty or near-empty — frontdoor formula may be undefined.'}"
        ),
        details={
            "total_combos": total_combos, "observed_combos": observed_combos,
            "empty": empty, "sparse": sparse, "min_count": min_count,
        },
    )


# _____________________________________________________________________________
# RDD-specific checks
# _____________________________________________________________________________

def check_rdd_no_manipulation(
    vars: AssumptionVariables,
    n_bins: int = 50,
    bandwidth: float = None,
) -> AssumptionResult:
    """McCrary-style density test: check for bunching at the cutoff."""
    rv = vars.df[vars.running_variable].dropna()

    if bandwidth is None:
        bandwidth = (rv.max() - rv.min()) * 0.25

    near = rv[(rv >= vars.cutoff - bandwidth) & (rv <= vars.cutoff + bandwidth)]
    below = near[near < vars.cutoff]
    above = near[near >= vars.cutoff]

    if len(below) < 10 or len(above) < 10:
        return AssumptionResult(
            passed=None,
            reasoning="Too few observations near cutoff for density test.",
        )

    total = len(below) + len(above)
    try:
        p_value = scipy_stats.binomtest(len(below), total, 0.5).pvalue
    except AttributeError:
        p_value = scipy_stats.binom_test(len(below), total, 0.5)

    passed = p_value > 0.05
    status_msg = "No evidence of manipulation." if passed else "Significant density discontinuity — possible manipulation."
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Density test around cutoff ({vars.cutoff}): {len(below)} below, {len(above)} above "
            f"(p={p_value:.4f}). Note: This binomial test is a local approximation of "
            f"density continuity and may be sensitive to the underlying distribution's slope. "
            f"{status_msg}"
        ),
        details={
            "n_below": len(below), "n_above": len(above),
            "p_value": p_value, "bandwidth": bandwidth,
            "test_type": "binomial_density_approximation",
        },
    )


def check_rdd_covariate_continuity(
    vars: AssumptionVariables,
    bandwidth: float = None,
) -> AssumptionResult:
    """Check continuity of covariates at the cutoff via t-tests."""
    if not vars.covariates:
        return AssumptionResult(passed=None, reasoning="No covariates provided.")

    if bandwidth is None:
        rv_range = vars.df[vars.running_variable].max() - vars.df[vars.running_variable].min()
        bandwidth = 0.1 * rv_range

    df_bw = vars.df[
        (vars.df[vars.running_variable] >= vars.cutoff - bandwidth) &
        (vars.df[vars.running_variable] <= vars.cutoff + bandwidth)
    ]
    below = df_bw[df_bw[vars.running_variable] < vars.cutoff]
    above = df_bw[df_bw[vars.running_variable] >= vars.cutoff]

    if len(below) < 5 or len(above) < 5:
        return AssumptionResult(passed=None, reasoning="Too few observations near cutoff.")

    results = {}
    discontinuous = []
    for cov in vars.covariates:
        if cov not in df_bw.columns:
            continue
        t_stat, p_val = scipy_stats.ttest_ind(below[cov].dropna(), above[cov].dropna(), equal_var=False)
        results[cov] = {"t_stat": float(t_stat), "p_value": float(p_val)}
        if p_val < 0.05:
            discontinuous.append(cov)

    passed = len(discontinuous) == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Covariate continuity at cutoff ({vars.cutoff}) on {len(results)} covariates. "
            f"{'All continuous.' if passed else f'Discontinuous: {discontinuous}.'}"
        ),
        details={"covariate_tests": results, "discontinuous": discontinuous, "bandwidth": bandwidth},
    )


def check_rdd_continuity_potential_outcomes(vars: AssumptionVariables, llm=None) -> AssumptionResult:
    """Continuity of potential outcomes at the cutoff (local exchangeability)."""
    return _llm_argue_assumption(
        "Continuity of potential outcomes at the cutoff",
        "E[Y(1)|X=c] and E[Y(0)|X=c] are continuous at the cutoff c. "
        "In the absence of treatment, individuals just above and just below "
        "the threshold would have had, on average, the same outcome.",
        vars, llm,
    )


# _____________________________________________________________________________
# Registry: maps each method to its pre-model assumption checks
# _____________________________________________________________________________

ASSUMPTION_REGISTRY: Dict[str, List] = {
    "linear_regression": [
        check_sutva,
        check_cond_ignorability,
    ],
    "propensity_score_matching": [
        check_sutva,
        check_cond_ignorability,
        check_positivity,
    ],
    "instrumental_variable": [
        check_sutva,
        check_iv_relevance,
        check_iv_exclusion,
        check_iv_exogeneity,
        check_iv_monotonicity,
    ],
    "difference_in_differences": [
        check_sutva,
        check_parallel_trends,
        check_no_anticipation,
        check_baseline_outcome_balance,
        check_stable_group_composition,
    ],
    "frontdoor_adjustment": [
        check_frontdoor_full_mediation,
        check_frontdoor_no_TM_confounding,
        check_frontdoor_T_blocks_MY,
        check_frontdoor_positivity,
    ],
    "regression_discontinuity_design": [
        check_rdd_no_manipulation,
        check_rdd_covariate_continuity,
        check_rdd_continuity_potential_outcomes,
    ],
    "backdoor_adjustment": [
        check_sutva,
        check_cond_ignorability,
    ],
}

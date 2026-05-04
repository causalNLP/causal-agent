"""
Pre-modeling assumption checks for causal inference methods.

These checks are run before estimation to verify whether the data and study
design satisfy the assumptions required by the chosen causal method.
Statistical checks use the raw data directly. Non-testable assumptions
are assessed via LLM reasoning based on the dataset description.

Each check returns a standardized dict:
    {
        "passed": bool | None,           # None => inconclusive
        "reasoning": str,                # human-readable explanation
        "details": dict,                 # raw stats (F, p, SMDs, ...)
    }
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from pydantic import BaseModel, Field


# import some assumptions already available for each method
from cais.methods.utils import (
    calculate_standardized_differences,
    check_overlap,
)

from cais.methods.instrumental_variable.diagnostics import calculate_first_stage_f_statistic

from cais.methods.difference_in_differences.diagnostics import (
    validate_parallel_trends,
    run_placebo_test,
)
from cais.utils.llm_helpers import call_llm_with_json_output

logger = logging.getLogger(__name__)

# _____________________________________________________________________________
# Output helper
# _____________________________________________________________________________

def _result(
    passed: Optional[bool],
    reasoning: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "passed": passed,
        "reasoning": reasoning,
        "details": details or {},
    }

# _____________________________________________________________________________
# LLM-based assumption argumentation (non-statistically-testable assumptions)
# _____________________________________________________________________________

class _LLMAssumptionVerdict(BaseModel):
    passed: Optional[bool] = Field(
        None,
        description="True if the assumption is plausibly satisfied given the dataset and "
                    "domain context, False if there is a clear reason to doubt it, "
                    "None if there is insufficient information to argue either way.",
    )
    reasoning: str = Field(
        ...,
        description="Concise (2-4 sentences) justification grounded in the dataset "
                    "description, variable semantics, or domain knowledge.",
    )
    missing_info: Optional[str] = Field(
            None,
            description="If passed is null, what additional information would be needed "
                        "to make a determination. Otherwise null.",
        )

def _llm_argue_assumption(
    assumption_name: str,
    assumption_description: str,
    dataset_description: Optional[str],
    variables_summary: Dict[str, Any],
    llm,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask the LLM to argue for/against a non-statistically-testable assumption.

    Falls back to passed=None with a clear notice if no LLM is available.
    """
    if llm is None:
        return _result(
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
{dataset_description or "(not provided)"}

Variables involved:
{variables_summary}

If you recognize this study from the literature, use your prior knowledge
about its design, mechanisms, and known methodological concerns to inform
your assessment — not just the description provided above.

{extra_context or ""}

Respond ONLY as JSON matching this schema:
{{
  "passed": true | false | null,
  "reasoning": "<2-4 sentence justification>"
  "missing_info": "<if passed is null, specify what additional information would be needed to make a determination. Otherwise set to null.>"
}}
                                                                                                                                                                                                       
Use null for "passed" if the dataset description is insufficient to argue either way.
""".strip()

    try:
        raw = call_llm_with_json_output(llm, prompt)
        verdict = _LLMAssumptionVerdict(**(raw or {}))
        details = {"assumption": assumption_name}
        if verdict.missing_info:
            details["missing_info"] = verdict.missing_info

        disclaimer = (
            " Note: this assessment relies on LLM reasoning and is sensitive to the "
            "quality and completeness of the dataset description provided."
        )
        return _result(
            passed=verdict.passed,
            reasoning=verdict.reasoning + disclaimer,
            details=details,
        )
    except Exception as exc:
        logger.warning("LLM assumption check failed for '%s': %s", assumption_name, exc)
        return _result(
            passed=None,
            reasoning=f"LLM check failed: {exc}. Assumption must be justified manually.",
        )


# _____________________________________________________________________________
# SUTVA  (needed for every method)
# _____________________________________________________________________________

def check_sutva(
    dataset_description: Optional[str],
    variables_summary: Dict[str, Any],
    llm=None,
) -> Dict[str, Any]:
    """SUTVA: no interference between units, no hidden treatment versions."""
    return _llm_argue_assumption(
        assumption_name="SUTVA (Stable Unit Treatment Value Assumption)",
        assumption_description=(
            "(1) No interference: one unit's treatment does not affect another unit's "
            "potential outcomes. (2) No hidden versions of the treatment: the treatment "
            "is administered consistently across treated units."
        ),
        dataset_description=dataset_description,
        variables_summary=variables_summary,
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
    df: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    smd_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Partial test of ignorability for RCTs: covariate balance on observables."""
    if not covariates:
        return _result(
            passed=None,
            reasoning="No covariates provided; balance check skipped.",
        )
    smds = calculate_standardized_differences(df, treatment, covariates)
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    passed = len(imbalanced) == 0
    return _result(
        passed=passed,
        reasoning=(
            f"Randomization check on {len(covariates)} covariates "
            f"(|SMD| < {smd_threshold}). "
            f"{'All balanced.' if passed else f'Imbalanced: {list(imbalanced.keys())}.'}"
        ),
        details={"smds": smds, "threshold": smd_threshold, "imbalanced": imbalanced},
    )


# _____________________________________________________________________________
# Positivity / overlap  (IPW, matching, GPS)
# _____________________________________________________________________________

def check_positivity(
    df: pd.DataFrame,
    treatment: str,
    propensity_scores: np.ndarray,
    overlap_threshold: float = 0.5,
    extreme_ps_bounds: tuple = (0.1, 0.9), # values from Crump et al. 2009
    max_extreme_pct: float = 0.05,
) -> Dict[str, Any]:
    """0 < P(T=1|X) < 1 across the support of X."""
    overlap = check_overlap(df, treatment, propensity_scores, threshold=overlap_threshold)
    lo, hi = extreme_ps_bounds
    n_extreme = int(((propensity_scores < lo) | (propensity_scores > hi)).sum())
    pct_extreme = n_extreme / len(propensity_scores) if len(propensity_scores) else 0.0
    passed = overlap["sufficient_overlap"] and pct_extreme < max_extreme_pct
    return _result(
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
    df: pd.DataFrame,
    treatment: str,
    instruments: List[str],
    covariates: List[str],
    f_threshold: float = 10.0,
) -> Dict[str, Any]:
    """First-stage F-test for instrument strength."""
    f, p = calculate_first_stage_f_statistic(df, treatment, instruments, covariates)
    if f is None:
        return _result(
            passed=None,
            reasoning="First-stage F-statistic could not be computed.",
        )
    passed = f >= f_threshold
    return _result(
        passed=passed,
        reasoning=(
            f"First-stage F = {f:.2f} (threshold {f_threshold}). "
            f"{'Strong instrument.' if passed else 'Weak instrument warning.'}"
        ),
        details={"f_statistic": f, "p_value": p, "threshold": f_threshold},
    )

def check_iv_exclusion(dataset_description, variables_summary, llm=None):
    return _llm_argue_assumption(
        "Exclusion restriction",
        "The instrument Z affects the outcome Y only through the treatment T, with no direct effect.",
        dataset_description, variables_summary, llm,
    )

def check_iv_exogeneity(dataset_description, variables_summary, llm=None):
    return _llm_argue_assumption(
        "Instrument exogeneity (independence)",
        "Z is as good as randomly assigned with respect to unobserved confounders of T and Y.",
        dataset_description, variables_summary, llm,
    )

def check_iv_monotonicity(dataset_description, variables_summary, llm=None):
    return _llm_argue_assumption(
        "Monotonicity (LATE)",
        "There are no defiers: the instrument never moves any unit in the opposite direction "
        "of its average effect on treatment uptake.",
        dataset_description, variables_summary, llm,
    )


# _____________________________________________________________________________
# DiD-specific checks
# _____________________________________________________________________________

def check_parallel_trends(
    df: pd.DataFrame, time_var: str, outcome: str,
    group_indicator_col: str, treatment_period_start,
    **kwargs,
) -> Dict[str, Any]:
    """Parallel trends: treatment and control groups had similar outcome trends pre-treatment."""
    res = validate_parallel_trends(
        df, time_var, outcome, group_indicator_col, treatment_period_start, **kwargs
    )
    return _result(
        passed=res.get("valid"),
        reasoning=res.get("details", ""),
        details={"p_value": res.get("p_value"), "error": res.get("error")},
    )


def check_no_anticipation(
    df, time_var, group_var, outcome, treated_unit_indicator,
    covariates, treatment_period_start, placebo_period_start,
) -> Dict[str, Any]:
    """No anticipation: treatment has no effect before implementation."""
    res = run_placebo_test(
        df, time_var, group_var, outcome, treated_unit_indicator,
        covariates, treatment_period_start, placebo_period_start,
    )
    return _result(
        passed=res.get("passed"),
        reasoning=res.get("details", ""),
        details={k: v for k, v in res.items() if k != "details"},
    )


def check_baseline_outcome_balance(
    df: pd.DataFrame, treatment: str, outcome: str,
    time_var: str, treatment_period_start,
    smd_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Intervention unrelated to outcome at baseline: comparable pre-treatment outcome levels."""
    pre = df[df[time_var] < treatment_period_start]
    if pre.empty:
        return _result(
            passed=None,
            reasoning="No pre-treatment data available.",
        )
    smd = calculate_standardized_differences(pre, treatment, [outcome]).get(outcome, np.nan)
    if pd.isna(smd):
        return _result(
            passed=None,
            reasoning="Could not compute SMD on baseline outcome (missing data or no variance).",
        )
    passed = abs(smd) <= smd_threshold
    return _result(
        passed=passed,
        reasoning=f"Baseline outcome SMD = {smd:.3f} (threshold {smd_threshold}).",
        details={"smd_pre_outcome": smd, "threshold": smd_threshold},
    )


def check_stable_group_composition(dataset_description, variables_summary, llm=None):
    """Stable group composition: no differential attrition or selective entry/exit due to treatment."""
    return _llm_argue_assumption(
        "Stable group composition",
        "Unit composition of treatment and control groups does not change as a result "
        "of treatment (no differential attrition or selective entry/exit).",
        dataset_description, variables_summary, llm,
    )


# _____________________________________________________________________________
# Frontdoor-specific checks (LLM-reasoned)
# _____________________________________________________________________________

def check_frontdoor_full_mediation(dataset_description, variables_summary, llm=None):
    """Full mediation: M fully captures the effect of T on Y; no direct T→Y path."""
    return _llm_argue_assumption(
        "Full mediation",
        "The mediator M fully captures the effect of treatment T on outcome Y; "
        "there is no direct T→Y path outside of the T→M→Y pathway.",
        dataset_description, variables_summary, llm,
    )

def check_frontdoor_no_TM_confounding(dataset_description, variables_summary, llm=None):
    """No unobserved confounding between treatment and mediator."""
    return _llm_argue_assumption(
        "No T-M confounding",
        "The relationship between the treatment T and the mediator M is unconfounded. "
        "There are no unobserved variables that affect both T and M.",
        dataset_description, variables_summary, llm,
    )

def check_frontdoor_T_blocks_MY(dataset_description, variables_summary, llm=None):
    """Treatment blocks all confounding paths between mediator and outcome."""
    return _llm_argue_assumption(
        "T blocks M→Y confounding",
        "Conditioning on the treatment T removes all back-door paths between "
        "the mediator M and the outcome Y.",
        dataset_description, variables_summary, llm,
    )

def check_frontdoor_positivity(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    min_count: int = 5,
) -> Dict[str, Any]:
    """Frontdoor positivity: P(M=m|X=x) > 0 for all relevant (x, m) combinations."""
    combos = df.groupby([treatment, mediator]).size().reset_index(name='count')
    total_combos = df[treatment].nunique() * df[mediator].nunique()
    observed_combos = len(combos)
    empty = total_combos - observed_combos
    sparse = int((combos['count'] < min_count).sum())

    passed = empty == 0 and sparse == 0
    return _result(
        passed=passed,
        reasoning=(
            f"{observed_combos}/{total_combos} (treatment, mediator) combinations observed. "
            f"{empty} empty, {sparse} sparse (< {min_count} obs). "
            f"{'Positivity satisfied.' if passed else 'Some combinations are empty or near-empty — frontdoor formula may be undefined.'}"
        ),
        details={"total_combos": total_combos, "observed_combos": observed_combos,
                 "empty": empty, "sparse": sparse, "min_count": min_count},
    )


# _____________________________________________________________________________
# RDD-specific checks
# _____________________________________________________________________________

def check_rdd_no_manipulation(
        df: pd.DataFrame,
        running_variable: str,
        cutoff: float,
        n_bins: int = 50,
        bandwidth: float = None,
) -> Dict[str, Any]:
    """McCrary-style density test: check for bunching at the cutoff."""
    rv = df[running_variable].dropna()

    if bandwidth is None:
        bandwidth = (rv.max() - rv.min()) * 0.25

    near = rv[(rv >= cutoff - bandwidth) & (rv <= cutoff + bandwidth)]
    below = near[near < cutoff]
    above = near[near >= cutoff]

    if len(below) < 10 or len(above) < 10:
        return _result(
            passed=None,
            reasoning="Too few observations near cutoff for density test.",
        )

    # Compare density just below vs just above using bin counts
    n_below = len(below)
    n_above = len(above)

    # Binomial test: under no manipulation, ~50% should be on each side
    total = n_below + n_above
    # Using scipy.stats.binomtest (modern) or binom_test (legacy)
    try:
        p_value = scipy_stats.binomtest(n_below, total, 0.5).pvalue
    except AttributeError:
        p_value = scipy_stats.binom_test(n_below, total, 0.5)

    passed = p_value > 0.05

    # Logic adjustment: Mention that the binomial test is a local density approximation
    status_msg = "No evidence of manipulation." if passed else "Significant density discontinuity — possible manipulation."
    reasoning = (
        f"Density test around cutoff ({cutoff}): {n_below} below, {n_above} above "
        f"(p={p_value:.4f}). Note: This binomial test is a local approximation of "
        f"density continuity and may be sensitive to the underlying distribution's slope. "
        f"{status_msg}"
    )

    return _result(
        passed=passed,
        reasoning=reasoning,
        details={
            "n_below": n_below, "n_above": n_above,
            "p_value": p_value, "bandwidth": bandwidth,
            "test_type": "binomial_density_approximation"
        },
    )


def check_rdd_covariate_continuity(
    df: pd.DataFrame,
    running_variable: str,
    cutoff: float,
    covariates: List[str],
    bandwidth: float = None,
) -> Dict[str, Any]:
    """Check continuity of covariates at the cutoff via t-tests."""
    if not covariates:
        return _result(passed=None, reasoning="No covariates provided.")

    if bandwidth is None:
        rv_range = df[running_variable].max() - df[running_variable].min()
        bandwidth = 0.1 * rv_range

    df_bw = df[(df[running_variable] >= cutoff - bandwidth) & (df[running_variable] <= cutoff + bandwidth)]
    below = df_bw[df_bw[running_variable] < cutoff]
    above = df_bw[df_bw[running_variable] >= cutoff]

    if len(below) < 5 or len(above) < 5:
        return _result(passed=None, reasoning="Too few observations near cutoff.")

    results = {}
    discontinuous = []
    for cov in covariates:
        if cov not in df_bw.columns:
            continue
        t_stat, p_val = scipy_stats.ttest_ind(
            below[cov].dropna(), above[cov].dropna(), equal_var=False
        )
        results[cov] = {"t_stat": float(t_stat), "p_value": float(p_val)}
        if p_val < 0.05:
            discontinuous.append(cov)

    passed = len(discontinuous) == 0
    return _result(
        passed=passed,
        reasoning=(
            f"Covariate continuity at cutoff ({cutoff}) on {len(results)} covariates. "
            f"{'All continuous.' if passed else f'Discontinuous: {discontinuous}.'}"
        ),
        details={"covariate_tests": results, "discontinuous": discontinuous, "bandwidth": bandwidth},
    )


def check_rdd_continuity_potential_outcomes(dataset_description, variables_summary, llm=None):
    """Continuity of potential outcomes at the cutoff (local exchangeability)."""
    return _llm_argue_assumption(
        "Continuity of potential outcomes at the cutoff",
        "E[Y(1)|X=c] and E[Y(0)|X=c] are continuous at the cutoff c. "
        "In the absence of treatment, individuals just above and just below "
        "the threshold would have had, on average, the same outcome.",
        dataset_description, variables_summary, llm,
    )
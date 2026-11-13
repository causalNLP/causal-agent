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
    # A covariate with any missing value has no SMD. Never claim balance for
    # a covariate that was not actually assessed: report partial coverage.
    untestable = sorted(c for c, v in smds.items() if pd.isna(v))
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    if untestable and len(untestable) == len(smds):
        # No covariate could be assessed: a vacuous "all balanced" would be
        # a false claim, so return the honest can't-tell verdict instead.
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Balance cannot be assessed: no covariate has complete data. "
                "Untestable due to missing data: " + ", ".join(untestable) + "."
            ),
            details={
                "smds": smds,
                "threshold": smd_threshold,
                "imbalanced": {},
                "untestable_covariates": untestable,
            },
        )
    passed = len(imbalanced) == 0
    coverage = (
        f"Randomization check on {len(smds) - len(untestable)} of {len(smds)} covariates "
        if untestable
        else f"Randomization check on {len(smds)} covariates "
    )
    missing_note = (
        " Untestable due to missing data: " + ", ".join(untestable) + "."
        if untestable
        else ""
    )
    return AssumptionResult(
        passed=passed,
        reasoning=(
            coverage
            + f"(|SMD| < {smd_threshold}). "
            + ("All balanced." if passed else f"Imbalanced: {list(imbalanced.keys())}.")
            + missing_note
        ),
        details={
            "smds": smds,
            "threshold": smd_threshold,
            "imbalanced": imbalanced,
            "untestable_covariates": untestable,
        },
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


def _check_bounded_anticipation(
    vars: AssumptionVariables,
    window_start: Any,
    alpha: float,
) -> AssumptionResult:
    """Test whether outcomes already diverge before the anticipation window.

    ``window_start`` is the user-supplied beginning of the early-response
    period. Only periods before it are used, so changes inside the window do
    not count as evidence against bounded anticipation.
    """
    df = vars.df
    time_var = vars.time_var
    outcome = vars.outcome
    treatment = vars.treatment

    if df is None or not time_var or not outcome or not treatment:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Bounded-anticipation check cannot run: time, outcome, or "
                "treatment/group column is missing."
            ),
        )

    try:
        if vars.treatment_period_start is not None and window_start >= vars.treatment_period_start:
            return AssumptionResult(
                passed=None,
                reasoning=(
                    "Anticipation window must start strictly before the treatment "
                    "period start."
                ),
            )
    except TypeError:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Anticipation window and treatment start are not comparable; "
                "they must use the same time units."
            ),
        )

    try:
        pre = df[df[time_var] < window_start]
    except Exception as exc:
        return AssumptionResult(
            passed=None,
            reasoning=f"Bounded-anticipation check could not filter pre-window data: {exc}",
        )

    if pre.empty:
        return AssumptionResult(
            passed=None,
            reasoning="No observations exist before the stated anticipation window.",
        )
    if pre[time_var].nunique() < 2:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "At least two pre-window periods are needed to test for early "
                "divergence; only one is available."
            ),
        )
    if pre[treatment].nunique() < 2:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Bounded-anticipation check needs both treated and control units "
                "before the window."
            ),
        )
    if len(pre) < 20:
        return AssumptionResult(
            passed=None,
            reasoning="Too few pre-window observations to test for early divergence.",
        )

    try:
        result = validate_parallel_trends(
            df,
            time_var,
            outcome,
            treatment,
            window_start,
            alpha=alpha,
        )
    except Exception as exc:
        return AssumptionResult(
            passed=None,
            reasoning=f"Bounded-anticipation test could not run: {exc}",
        )

    passed = result.get("valid")
    p_value = result.get("p_value")
    detail = result.get("details") or ""
    p_text = f"{p_value:.4f}" if p_value is not None else "not available"
    if passed is None:
        reasoning = f"Bounded anticipation could not be tested: {detail}"
    elif passed:
        reasoning = (
            "No evidence of divergence before the stated anticipation window "
            f"(p={p_text}). This supports bounded anticipation but does not "
            "prove that no anticipation occurred."
        )
    else:
        reasoning = (
            "Outcomes already differ before the stated anticipation window "
            f"(p={p_text}), contradicting bounded anticipation."
        )

    return AssumptionResult(
        passed=passed,
        reasoning=reasoning,
        details={
            "test_type": "pre_window_trend_test",
            "window_start": window_start,
            "p_value": p_value,
            "alpha": alpha,
            "n_pre_window": int(len(pre)),
            "pre_window_periods": int(pre[time_var].nunique()),
        },
    )


def check_no_anticipation(vars: AssumptionVariables, alpha: float = 0.10) -> AssumptionResult:
    """No anticipation: treatment has no effect before implementation.

    If ``anticipation_window_start`` is supplied, tests whether the groups
    diverge before that window (bounded anticipation). Otherwise runs the older
    single placebo-period DiD restricted to pre-treatment periods.
    """
    import statsmodels.formula.api as smf

    window_start = getattr(vars, "anticipation_window_start", None)
    if window_start is not None:
        return _check_bounded_anticipation(vars, window_start, alpha)

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
            details={
                "missing_info": (
                    "Does a pre-treatment period exist during which the outcome should "
                    "not yet respond to the treatment (a placebo period)? State when it "
                    "starts and when the treatment begins, in the same units as the time "
                    "column (e.g. a year), best as 'placebo=<year>, treatment=<year>'. "
                    "Type 'skip' if unknown."
                ),
            },
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
        passed = p_val > alpha
        return AssumptionResult(
            passed=passed,
            reasoning=(
                f"Placebo treatment effect (pre-treatment only): {effect:.4f} "
                f"(p={p_val:.4f}). Test {'passed' if passed else 'failed'}."
            ),
            details={"effect_estimate": effect, "p_value": p_val, "alpha": alpha},
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


def check_stable_group_composition(
    vars: AssumptionVariables,
    llm=None,
    alpha: float = 0.05,
    min_units_per_group: int = 10,
) -> AssumptionResult:
    """Stable group composition: no differential attrition or entry.

    Deterministic panel check. Requires a unit identifier (``unit_var``, or
    ``group_var`` as a fallback), a time column, and a binary treatment/group
    indicator that is constant within units. It compares dropout among units
    observed in the first period and entry among units observed in the last
    period with Fisher's exact test.

    Returns ``passed=None`` ("can't tell") whenever the inputs are missing or too
    thin to test; the tree treats that as a warning and continues. ``llm`` is
    accepted for signature compatibility but is not used: this assumption is
    checked from the data, not argued from the description.
    """
    del llm  # group composition is tested from the data, not argued by the LLM
    df = vars.df
    if df is None or not hasattr(df, "columns"):
        return AssumptionResult(
            passed=None,
            reasoning="Group composition cannot be tested: no dataframe was provided.",
        )

    unit = vars.unit_var or vars.group_var
    unit_source = "unit_var" if vars.unit_var else "group_var (fallback)"
    time_col = vars.time_var
    group_col = vars.treatment

    missing = [
        label
        for label, name in (
            ("unit identifier (unit_var or group_var)", unit),
            ("time variable", time_col),
            ("treatment/group variable", group_col),
        )
        if not name
    ]
    if missing:
        return AssumptionResult(
            passed=None,
            reasoning="Group composition cannot be tested: missing " + ", ".join(missing) + ".",
            details={"required": ["unit_var/group_var", "time_var", "treatment"]},
        )

    absent = [name for name in (unit, time_col, group_col) if name not in df.columns]
    if absent:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Group composition cannot be tested: column(s) not in the dataset: "
                + ", ".join(repr(name) for name in absent)
                + "."
            ),
            details={"missing_columns": absent},
        )
    if len({unit, time_col, group_col}) < 3:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Group composition cannot be tested: the unit, time, and "
                "treatment/group columns must be distinct."
            ),
        )

    pairs = (
        df[[unit, time_col, group_col]]
        .dropna()
        .drop_duplicates()
    )
    n_units = int(pairs[unit].nunique())
    n_periods = int(pairs[time_col].nunique())
    if n_units < 2:
        return AssumptionResult(
            passed=None,
            reasoning="Group composition cannot be tested: fewer than two units were observed.",
        )
    if n_periods < 2:
        return AssumptionResult(
            passed=None,
            reasoning="Group composition cannot be tested: fewer than two time periods were observed.",
        )

    per_unit_groups = pairs.groupby(unit)[group_col].nunique()
    varying = int((per_unit_groups > 1).sum())
    if varying:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Group composition cannot be tested: the treatment/group indicator "
                f"varies within {varying} unit(s); a stable treatment-group "
                "indicator is required."
            ),
        )

    unit_group = pairs.groupby(unit)[group_col].first()
    group_values = list(pd.unique(unit_group))
    if len(group_values) != 2:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Group composition cannot be tested: expected a binary group "
                f"indicator, found {len(group_values)} value(s) "
                f"({', '.join(str(value) for value in group_values[:5])})."
            ),
        )
    group_values = sorted(group_values, key=lambda value: str(value))

    periods = list(pd.unique(pairs[time_col]))
    try:
        periods = sorted(periods)
    except TypeError:
        return AssumptionResult(
            passed=None,
            reasoning="Group composition cannot be tested: time values are not orderable.",
        )
    first_period, last_period = periods[0], periods[-1]

    start_units = set(pairs.loc[pairs[time_col] == first_period, unit])
    end_units = set(pairs.loc[pairs[time_col] == last_period, unit])
    if not start_units or not end_units:
        return AssumptionResult(
            passed=None,
            reasoning="Group composition cannot be tested: the first or last period has no units.",
        )

    def _group_frame(units):
        frame = pd.DataFrame({unit: list(units)})
        frame["__group__"] = frame[unit].map(unit_group)
        return frame

    start_frame = _group_frame(start_units)
    end_frame = _group_frame(end_units)

    # Dropout: baseline units absent from the last period.
    start_totals = {
        value: int(start_frame["__group__"].eq(value).sum()) for value in group_values
    }
    if min(start_totals.values()) < min_units_per_group:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "Group composition cannot be tested: too few units per group in "
                "the first period ("
                + ", ".join(f"{value}: {start_totals[value]}" for value in group_values)
                + f"); need at least {min_units_per_group} per group."
            ),
            details={"start_group_counts": {str(k): v for k, v in start_totals.items()}},
        )
    dropout_events = {
        value: int(
            (start_frame["__group__"].eq(value) & ~start_frame[unit].isin(end_units)).sum()
        )
        for value in group_values
    }

    # Entry: last-period units absent from the first period.
    end_totals = {
        value: int(end_frame["__group__"].eq(value).sum()) for value in group_values
    }
    entry_events = {
        value: int(
            (end_frame["__group__"].eq(value) & ~end_frame[unit].isin(start_units)).sum()
        )
        for value in group_values
    }

    dropout_table = [
        [dropout_events[value], start_totals[value] - dropout_events[value]]
        for value in group_values
    ]
    try:
        _, p_dropout = scipy_stats.fisher_exact(dropout_table)
        p_dropout = float(p_dropout)
    except Exception:
        p_dropout = None
    if p_dropout is None:
        return AssumptionResult(
            passed=None,
            reasoning="Group composition could not be tested: the dropout test failed to run.",
        )

    p_entry = None
    if min(end_totals.values()) >= min_units_per_group:
        entry_table = [
            [entry_events[value], end_totals[value] - entry_events[value]]
            for value in group_values
        ]
        try:
            _, p_entry = scipy_stats.fisher_exact(entry_table)
            p_entry = float(p_entry)
        except Exception:
            p_entry = None

    dropout_rates = {
        value: (dropout_events[value] / start_totals[value] if start_totals[value] else 0.0)
        for value in group_values
    }
    entry_rates = {
        value: (entry_events[value] / end_totals[value] if end_totals[value] else 0.0)
        for value in group_values
    }

    def _fmt_dropout():
        return ", ".join(
            f"{value}: {dropout_rates[value]:.1%} (n={start_totals[value]})"
            for value in group_values
        )

    def _fmt_entry():
        return ", ".join(
            f"{value}: {entry_rates[value]:.1%} (n={end_totals[value]})"
            for value in group_values
        )

    failures = []
    if p_dropout <= alpha:
        failures.append(
            "differential dropout (" + _fmt_dropout() + f"; Fisher p={p_dropout:.3f})"
        )
    if p_entry is not None and p_entry <= alpha:
        failures.append(
            "differential entry (" + _fmt_entry() + f"; Fisher p={p_entry:.3f})"
        )

    details = {
        "unit_variable": unit,
        "unit_source": unit_source,
        "time_variable": time_col,
        "group_variable": group_col,
        "n_units": n_units,
        "n_periods": n_periods,
        "first_period": str(first_period),
        "last_period": str(last_period),
        "dropout": {
            str(value): {
                "n": start_totals[value],
                "events": dropout_events[value],
                "rate": dropout_rates[value],
            }
            for value in group_values
        },
        "entry": {
            str(value): {
                "n": end_totals[value],
                "events": entry_events[value],
                "rate": entry_rates[value],
            }
            for value in group_values
        },
        "p_dropout": p_dropout,
        "p_entry": p_entry,
        "alpha": alpha,
        "min_units_per_group": min_units_per_group,
    }

    if failures:
        return AssumptionResult(
            passed=False,
            reasoning=(
                "Group composition changed differentially: "
                + "; ".join(failures)
                + ". The treated and control groups may not be comparable over time."
            ),
            details=details,
        )

    entry_sentence = (
        f"entry {_fmt_entry()} (Fisher p={p_entry:.3f})."
        if p_entry is not None
        else "entry could not be tested (too few units per group in the last period)."
    )
    return AssumptionResult(
        passed=True,
        reasoning=(
            "Group composition looks stable: "
            f"dropout {_fmt_dropout()} "
            f"(Fisher p={p_dropout:.3f}); {entry_sentence}"
        ),
        details=details,
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
    alpha: float = 0.05,
    sensitivity: bool = False,
    **kwargs,
) -> AssumptionResult:
    """Density-continuity test around the RDD cutoff.

    Delegates to the standalone CJM/McCrary test (``rddensity``). The old
    binomial 50/50 comparison is intentionally retired: it conflated a sloping
    density with a discontinuity. Returns ``passed=None`` whenever the test
    cannot run.
    """
    from cais.methods.regression_discontinuity.density_test import mccrary_density_test

    if vars is None or vars.df is None:
        return AssumptionResult(
            passed=None,
            reasoning="Density test cannot run: no dataframe was provided.",
        )
    return mccrary_density_test(
        vars.df,
        vars.running_variable,
        vars.cutoff,
        alpha=alpha,
        sensitivity=sensitivity,
    )


def check_rdd_covariate_continuity(
    vars: AssumptionVariables,
    bandwidth: float = None,
    alpha: float = 0.05,
    bandwidth_fraction: float = 0.1,
) -> AssumptionResult:
    """Check continuity of covariates at the cutoff via t-tests."""
    if not vars.covariates:
        return AssumptionResult(passed=None, reasoning="No covariates provided.")

    # The running variable cannot be tested for continuity at its own cutoff:
    # below-cutoff and above-cutoff values differ by construction, so including
    # it would turn the design rule itself into a manufactured violation.
    if vars.running_variable in vars.covariates:
        vars.covariates = [
            c for c in vars.covariates if c != vars.running_variable
        ]
        if not vars.covariates:
            return AssumptionResult(
                passed=None,
                reasoning=(
                    "After removing the running variable from the covariate "
                    "list, no covariates remain to test."
                ),
            )

    if bandwidth is None:
        rv_range = vars.df[vars.running_variable].max() - vars.df[vars.running_variable].min()
        bandwidth = bandwidth_fraction * rv_range

    df_bw = vars.df[
        (vars.df[vars.running_variable] >= vars.cutoff - bandwidth) &
        (vars.df[vars.running_variable] <= vars.cutoff + bandwidth)
    ]
    below = df_bw[df_bw[vars.running_variable] < vars.cutoff]
    above = df_bw[df_bw[vars.running_variable] >= vars.cutoff]

    if len(below) < 5 or len(above) < 5:
        return AssumptionResult(passed=None, reasoning="Too few observations near cutoff.")

    results = {}
    for cov in vars.covariates:
        if cov not in df_bw.columns:
            continue
        t_stat, p_val = scipy_stats.ttest_ind(below[cov].dropna(), above[cov].dropna(), equal_var=False)
        results[cov] = {"t_stat": float(t_stat), "p_value": float(p_val)}

    # Family-wise control: without a correction, testing k covariates at
    # alpha 0.05 rejects a clean cutoff roughly k-in-20 times by chance. The
    # gate's verdict should hold the intended 5% error rate over the whole
    # set, so each test is judged at alpha/k (Bonferroni); raw p-values stay
    # in details for inspection.
    if results:
        per_test_alpha = alpha / len(results)
    else:
        per_test_alpha = alpha
    discontinuous = [
        cov for cov, test in results.items() if test["p_value"] < per_test_alpha
    ]

    passed = len(discontinuous) == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Covariate continuity at cutoff ({vars.cutoff}) on {len(results)} covariates "
            f"(per-test alpha {per_test_alpha:.4f}, Bonferroni). "
            f"{'All continuous.' if passed else f'Discontinuous: {discontinuous}.'}"
        ),
        details={
            "covariate_tests": results,
            "discontinuous": discontinuous,
            "bandwidth": bandwidth,
            "alpha": alpha,
            "per_test_alpha": per_test_alpha,
        },
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

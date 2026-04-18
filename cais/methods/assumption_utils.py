"""
Reusable assumption checks for causal inference methods.

Each check returns a standardized dict:
    {
        "passed": bool | None,           # None => inconclusive
        "reasoning": str,                # human-readable explanation
        "details": dict,                 # raw stats (F, p, SMDs, ...)
    }

These are composed in each estimator's `validate_assumptions` method.
The agent-level `validate_method` simply dispatches to the selected estimator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# import some assumptions already available for each method

from cais.methods.utils import (
    calculate_standardized_differences,
    check_overlap,
)
from cais.methods.instrumental_variable.diagnostics import (
    calculate_first_stage_f_statistic,
)
from cais.methods.difference_in_differences.diagnostics import (
    validate_parallel_trends,
    run_placebo_test,
)
from cais.methods.generalized_propensity_score.diagnostics import (
    assess_gps_balance,
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

{extra_context or ""}

Respond ONLY as JSON matching this schema:
{{
  "passed": true | false | null,
  "reasoning": "<2-4 sentence justification>"
}}

Use null for "passed" if the dataset description is insufficient to argue either way.
""".strip()

    try:
        raw = call_llm_with_json_output(llm, prompt)
        verdict = _LLMAssumptionVerdict(**(raw or {}))
        return _result(
            passed=verdict.passed,
            reasoning=verdict.reasoning,
            details={"assumption": assumption_name},
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
            "Pay attention to: network/spillover effects (e.g., units in shared "
            "schools, households, markets), partial compliance, treatment intensity "
            "variation."
        ),
    )


# _____________________________________________________________________________
# Ignorability / Conditional ignorability (RCT and observational)
# _____________________________________________________________________________

def check_rct_balance(
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

"""
Post-modeling assumption checks for causal inference methods.

These checks require outputs from the estimation step (e.g., IPW weights,
matched samples, IV residuals, GPS model residuals) and are run after
the causal effect has been estimated.

Each check returns an AssumptionResult with:
    passed   : bool | None  (None => inconclusive)
    reasoning: str
    details  : dict         (raw stats — SMDs, p-values, ...)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from cais.models import AssumptionResult, AssumptionVariables
from cais.methods.instrumental_variable.diagnostics import run_overidentification_test
from cais.methods.utils import calculate_standardized_differences
from cais.methods.generalized_propensity_score.diagnostics import assess_gps_balance


# _____________________________________________________________________________
# Balance checks (IPW, matching)
# _____________________________________________________________________________

def check_balance_after_weighting(
    vars: AssumptionVariables,
    weights: np.ndarray,
    smd_threshold: float = 0.1,
) -> AssumptionResult:
    """Weighted SMDs after IPW: checks covariate balance in the weighted sample."""
    treated = vars.df[vars.treatment] == 1
    smds = {}
    for c in vars.covariates:
        x = vars.df[c].astype(float).values
        w = weights
        m1 = np.average(x[treated], weights=w[treated])
        m0 = np.average(x[~treated], weights=w[~treated])
        v1 = np.average((x[treated] - m1) ** 2, weights=w[treated])
        v0 = np.average((x[~treated] - m0) ** 2, weights=w[~treated])
        denom = np.sqrt((v1 + v0) / 2)
        smds[c] = float((m1 - m0) / denom) if denom > 0 else float("nan")
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    passed = len(imbalanced) == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Weighted balance on {len(vars.covariates)} covariates (|SMD| < {smd_threshold}). "
            f"{'All balanced after IPW.' if passed else f'Still imbalanced: {list(imbalanced.keys())}.'}"
        ),
        details={"weighted_smds": smds, "threshold": smd_threshold, "imbalanced": imbalanced},
    )


def check_balance_after_matching(
    vars: AssumptionVariables,
    df_matched: pd.DataFrame,
    smd_threshold: float = 0.1,
) -> AssumptionResult:
    """SMDs computed on the matched sample."""
    smds = calculate_standardized_differences(df_matched, vars.treatment, vars.covariates)
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    passed = len(imbalanced) == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Matched sample balance on {len(vars.covariates)} covariates (|SMD| < {smd_threshold}). "
            f"{'All balanced after matching.' if passed else f'Still imbalanced: {list(imbalanced.keys())}.'}"
        ),
        details={"smds": smds, "threshold": smd_threshold, "imbalanced": imbalanced},
    )


# _____________________________________________________________________________
# IV: over-identification test (requires multiple instruments)
# _____________________________________________________________________________

def check_iv_overidentification(
    vars: AssumptionVariables,
    sm_results,
) -> AssumptionResult:
    """Sargan-Hansen test: are the instruments valid (uncorrelated with residuals)?

    Only applicable when len(instruments) > 1.
    """
    stat, p, status = run_overidentification_test(
        sm_results, vars.df, vars.treatment, vars.outcome,
        vars.instruments, vars.covariates,
    )
    if stat is None:
        return AssumptionResult(
            passed=None,
            reasoning=status or "Over-identification test could not be computed.",
        )
    passed = p > 0.05
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Sargan-Hansen test: statistic={stat:.2f}, p={p:.4f}. "
            f"{'Instruments appear valid (not correlated with errors).' if passed else 'Instruments may be invalid — correlated with residuals.'}"
        ),
        details={"statistic": float(stat), "p_value": float(p), "status": status},
    )


# _____________________________________________________________________________
# GPS (Generalized Propensity Score)
# _____________________________________________________________________________

def check_gps_balance(
    vars: AssumptionVariables,
    df_with_gps: pd.DataFrame,
    gps_col_name: str,
    **kwargs,
) -> AssumptionResult:
    """Covariate balance after GPS adjustment."""
    res = assess_gps_balance(
        df_with_gps, vars.treatment, vars.covariates, gps_col_name, **kwargs
    )
    cov_balance = res.get("covariate_balance", {})
    unbalanced = [c for c, v in cov_balance.items() if not v.get("balanced", True)]
    passed = len(unbalanced) == 0
    return AssumptionResult(
        passed=passed,
        reasoning=(
            res.get("summary", "GPS balance assessed.")
            + (f" Unbalanced covariates: {unbalanced}." if unbalanced else "")
        ),
        details=res,
    )


def check_gps_specification(residuals: np.ndarray) -> AssumptionResult:
    """Residual normality of the GPS model (Shapiro-Wilk).

    The GPS is typically estimated via OLS/GLM; normally distributed residuals
    support a well-specified model.
    """
    if len(residuals) < 3:
        return AssumptionResult(
            passed=None,
            reasoning="Too few residuals for normality test.",
        )
    sample = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
    stat, p = stats.shapiro(sample)
    passed = p > 0.05
    return AssumptionResult(
        passed=passed,
        reasoning=(
            f"Shapiro-Wilk on GPS model residuals: W={stat:.3f}, p={p:.4f}. "
            f"{'Residuals consistent with normality.' if passed else 'Significant departure from normality — consider re-specifying the GPS model.'}"
        ),
        details={"shapiro_w": float(stat), "p_value": float(p)},
    )


# _____________________________________________________________________________
# Registry: maps each method to its post-model assumption checks
# _____________________________________________________________________________

POST_ASSUMPTION_REGISTRY: Dict[str, List] = {
    "propensity_score_matching": [
        check_balance_after_matching,
    ],
    "instrumental_variable": [
        check_iv_overidentification,
    ],
    "generalized_propensity_score": [
        check_gps_balance,
        check_gps_specification,
    ],
}

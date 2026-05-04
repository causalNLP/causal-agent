"""
Post-modeling assumption checks for causal inference methods.

These checks require outputs from the estimation step (e.g., IPW weights,
matched samples, IV residuals, GPS model residuals) and are run after
the causal effect has been estimated.

Each check returns a standardized dict:
    {
        "passed": bool | None,           # None => inconclusive
        "reasoning": str,                # human-readable explanation
        "details": dict,                 # raw stats (SMDs, p-values, ...)
    }
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# import some assumptions already available for each method
from cais.methods.instrumental_variable.diagnostics import (
    run_overidentification_test,
)
from cais.methods.utils import calculate_standardized_differences
from cais.methods.generalized_propensity_score.diagnostics import assess_gps_balance

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
# Balance checks (IPW, matching)
# _____________________________________________________________________________

def check_balance_after_weighting(
    df: pd.DataFrame, treatment: str, covariates: List[str],
    weights: np.ndarray, smd_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Weighted SMDs after IPW."""
    treated = df[treatment] == 1
    smds = {}
    for c in covariates:
        x = df[c].astype(float).values
        w = weights
        m1 = np.average(x[treated], weights=w[treated])
        m0 = np.average(x[~treated], weights=w[~treated])
        v1 = np.average((x[treated] - m1) ** 2, weights=w[treated])
        v0 = np.average((x[~treated] - m0) ** 2, weights=w[~treated])
        denom = np.sqrt((v1 + v0) / 2)
        smds[c] = (m1 - m0) / denom if denom > 0 else np.nan
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    passed = len(imbalanced) == 0
    return _result(
        passed=passed,
        reasoning=(
            f"Weighted balance on {len(covariates)} covariates. "
            f"{'All balanced after IPW.' if passed else f'Still imbalanced: {list(imbalanced.keys())}.'}"
        ),
        details={"weighted_smds": smds, "threshold": smd_threshold, "imbalanced": imbalanced},
    )


def check_balance_after_matching(
    df_matched: pd.DataFrame, treatment: str, covariates: List[str],
    smd_threshold: float = 0.1,
) -> Dict[str, Any]:
    """SMDs computed on the matched sample."""
    smds = calculate_standardized_differences(df_matched, treatment, covariates)
    imbalanced = {c: v for c, v in smds.items() if pd.notna(v) and abs(v) > smd_threshold}
    passed = len(imbalanced) == 0
    return _result(
        passed=passed,
        reasoning=(
            f"Matched sample balance on {len(covariates)} covariates. "
            f"{'All balanced after matching.' if passed else f'Still imbalanced: {list(imbalanced.keys())}.'}"
        ),
        details={"smds": smds, "threshold": smd_threshold, "imbalanced": imbalanced},
    )


# _____________________________________________________________________________
# IVs
# _____________________________________________________________________________

def check_iv_overidentification(
    sm_results, df, treatment, outcome, instruments, covariates,
) -> Dict[str, Any]:
    """Sargan-Hansen test: are the instruments valid (uncorrelated with errors)?"""
    stat, p, status = run_overidentification_test(
        sm_results, df, treatment, outcome, instruments, covariates,
    )
    if stat is None:
        return _result(
            passed=None,
            reasoning=status or "Over-identification test could not be computed.",
        )
    passed = p > 0.05  # non-rejet = instruments valides
    return _result(
        passed=passed,
        reasoning=(
            f"Sargan-Hansen test: statistic={stat:.2f}, p={p:.4f}. "
            f"{'Instruments appear valid.' if passed else 'Instruments may be invalid — correlated with errors.'}"
        ),
        details={"statistic": stat, "p_value": p, "status": status},
    )


# _____________________________________________________________________________
# GPS (Generalized Propensity Score)
# _____________________________________________________________________________

def check_gps_balance(
    df_with_gps: pd.DataFrame, treatment_var: str, covariate_vars: List[str],
    gps_col_name: str, **kwargs,
) -> Dict[str, Any]:
    """Covariate balance after GPS adjustment."""
    res = assess_gps_balance(df_with_gps, treatment_var, covariate_vars, gps_col_name, **kwargs)
    cov_balance = res.get("covariate_balance", {})
    unbalanced = [c for c, v in cov_balance.items() if not v.get("balanced", True)]
    passed = len(unbalanced) == 0
    return _result(
        passed=passed,
        reasoning=(
            res.get("summary", "GPS balance assessed.") +
            (f" Unbalanced: {unbalanced}." if unbalanced else "")
        ),
        details=res,
    )


def check_gps_specification(residuals: np.ndarray) -> Dict[str, Any]:
    """Residual normality of the GPS model (e.g., Shapiro-Wilk)."""
    if len(residuals) < 3:
        return _result(
            passed=None,
            reasoning="Too few residuals for normality test.",
        )
    # Shapiro caps at ~5000; subsample if needed
    sample = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
    stat, p = stats.shapiro(sample)
    passed = p > 0.05
    return _result(
        passed=passed,
        reasoning=(
            f"Shapiro-Wilk on GPS residuals: W={stat:.3f}, p={p:.4f}. "
            f"{'Residuals consistent with normality.' if passed else 'Departure from normality — reconsider GPS model.'}"
        ),
        details={"statistic": float(stat), "p_value": float(p)},
    )
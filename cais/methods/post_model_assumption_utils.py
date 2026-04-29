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

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# import some assumptions already available for each method
from cais.methods.instrumental_variable.diagnostics import (
    run_overidentification_test,
)
from cais.methods.utils import calculate_standardized_differences
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



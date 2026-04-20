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

from cais.methods.instrumental_variable.diagnostics import (
    run_overidentification_test,
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
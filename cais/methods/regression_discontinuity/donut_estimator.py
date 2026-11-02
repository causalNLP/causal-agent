"""Donut RDD estimator: RDD after dropping observations near the cutoff.

Donut RDD is the standard remedy for *local* sorting/bunching at an RDD cutoff:
drop units with ``|running_variable - cutoff| < donut_width`` and estimate RDD
on the remaining units. It does not "fix" manipulation in general; it changes
the estimand to units outside the donut and is only credible when the sorting
is confined to that band.

Standalone for now: this is registered as a method (``donut_rdd``) but the
decision tree does not route to it yet.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from cais.methods.causal_method import CausalMethod
from cais.methods.regression_discontinuity.estimator import (
    estimate_effect as rdd_estimate_effect,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_RETAINED_PER_SIDE = 10


def _validate_donut_inputs(
    df: pd.DataFrame,
    running_variable: str,
    cutoff_value: Any,
    donut_width: Any,
) -> tuple[float, float, pd.Series]:
    """Validate shared inputs and return (cutoff, donut width, numeric running var)."""
    if df is None or not hasattr(df, "columns"):
        raise ValueError("Donut RDD requires a dataframe.")
    if not running_variable or running_variable not in df.columns:
        raise ValueError(
            f"Donut RDD running variable {running_variable!r} is not a column."
        )
    if cutoff_value is None:
        raise ValueError("Donut RDD requires a cutoff value.")
    try:
        cutoff = float(cutoff_value)
    except (TypeError, ValueError):
        raise ValueError(f"Cutoff value {cutoff_value!r} is not numeric.")
    if not np.isfinite(cutoff):
        raise ValueError("Cutoff value is not finite.")

    if donut_width is None:
        raise ValueError(
            "Donut RDD requires an explicit donut_width (in running-variable units)."
        )
    try:
        width = float(donut_width)
    except (TypeError, ValueError):
        raise ValueError(f"donut_width {donut_width!r} is not numeric.")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("donut_width must be a positive finite number.")

    numeric = pd.to_numeric(df[running_variable], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if numeric.dropna().empty:
        raise ValueError(
            f"Donut RDD running variable {running_variable!r} has no numeric observations."
        )
    return cutoff, width, numeric


def _finalize_donut_result(
    result: Dict[str, Any],
    donut_width: float,
    n_total: int,
    n_dropped: int,
) -> Dict[str, Any]:
    """Normalize a donut result and record the changed estimand."""
    if not isinstance(result, dict):
        result = {"effect_estimate": None, "error": "Base RDD returned no dict."}

    result["method_used"] = "donut_rdd"
    result["donut_width"] = donut_width
    result["n_dropped"] = n_dropped
    result["share_dropped"] = (n_dropped / n_total) if n_total else 0.0
    result["estimand_population"] = (
        f"units with |running variable - {result.get('cutoff_value', 'cutoff')}| "
        f">= {donut_width:g}"
    )

    base_details = result.get("method_details")
    prefix = (
        f"Donut RDD (donut width {donut_width:g}; {n_dropped}/{n_total} observations "
        "dropped near the cutoff); the estimate refers to units outside the donut"
    )
    result["method_details"] = (
        f"{prefix}. {base_details}" if base_details else prefix
    )

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics.update(
        {
            "donut_width": donut_width,
            "n_dropped": n_dropped,
            "share_dropped": result["share_dropped"],
            "estimand": result["estimand_population"],
        }
    )
    result["diagnostics"] = diagnostics
    return result


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    running_variable: str,
    cutoff_value: Any,
    covariates: Optional[Sequence[str]] = None,
    donut_width: Optional[float] = None,
    bandwidth: Optional[float] = None,
    treat_above_cutoff: Optional[bool] = None,
    query: Optional[str] = None,
    llm: Any = None,
    min_retained_per_side: int = DEFAULT_MIN_RETAINED_PER_SIDE,
    **kwargs,
) -> Dict[str, Any]:
    """Estimate RDD after dropping observations within ``donut_width`` of the cutoff.

    Args:
        df: Dataframe with running variable, outcome, treatment and covariates.
        treatment: Treatment indicator column (for API consistency; RDD is driven
            by the cutoff).
        outcome: Outcome column.
        running_variable: Numeric running variable.
        cutoff_value: Treatment threshold.
        covariates: Optional covariates (currently ignored by the base RDD
            implementations, kept for API consistency).
        donut_width: Half-width of the dropped band, in running-variable units.
            Required; there is no silent default.
        bandwidth: Optional RDD bandwidth for the retained sample.
        treat_above_cutoff: Whether treatment is assigned above the cutoff.
        query: Optional original query for interpretation.
        llm: Optional LLM for interpretation; ``None`` uses the default text.
        min_retained_per_side: Minimum retained observations on each side of the
            cutoff after the donut; below this a ValueError is raised.

    Returns:
        The base RDD result dictionary extended with donut width, dropped counts,
        changed estimand, and method metadata.
    """
    covariates_list = list(covariates or [])
    cutoff, width, numeric = _validate_donut_inputs(
        df, running_variable, cutoff_value, donut_width
    )

    if not (numeric.min() < cutoff < numeric.max()):
        raise ValueError(
            "Donut RDD cutoff is outside the observed running-variable support."
        )

    keep = (numeric - cutoff).abs() >= width
    filtered = df.loc[keep].copy()
    n_total = len(df)
    n_dropped = int(n_total - len(filtered))
    if filtered.empty:
        raise ValueError(
            f"Donut RDD would drop every observation (donut_width={width:g})."
        )
    if n_dropped == 0:
        logger.warning(
            "Donut RDD dropped no observations with donut_width=%s; this is "
            "equivalent to standard RDD on the supplied data.",
            width,
        )

    retained_left = int((pd.to_numeric(filtered[running_variable], errors="coerce") < cutoff).sum())
    retained_right = int((pd.to_numeric(filtered[running_variable], errors="coerce") >= cutoff).sum())
    if retained_left < min_retained_per_side or retained_right < min_retained_per_side:
        raise ValueError(
            "Too few observations remain after the donut "
            f"(left={retained_left}, right={retained_right}; need at least "
            f"{min_retained_per_side} per side)."
        )

    try:
        result = rdd_estimate_effect(
            df=filtered,
            treatment=treatment,
            outcome=outcome,
            running_variable=running_variable,
            cutoff_value=cutoff,
            covariates=covariates_list,
            bandwidth=bandwidth,
            treat_above_cutoff=treat_above_cutoff,
            query=query,
            llm=llm,
            **kwargs,
        )
    except Exception as exc:
        raise ValueError(f"Donut RDD base estimation failed: {exc}") from exc

    result.setdefault("cutoff_value", cutoff)
    return _finalize_donut_result(result, width, n_total, n_dropped)


class DonutRDD(CausalMethod):
    """RDD estimated after dropping observations close to the cutoff.

    The reported effect refers to units outside the donut, not to the immediate
    cutoff limit. It is appropriate only when sorting/manipulation is believed
    to be local to the dropped band.
    """

    name = "Donut Regression Discontinuity Design"
    description = (
        "Regression discontinuity design estimated after dropping observations "
        "within a donut around the cutoff. This is a robustness/identification "
        "adjustment for local sorting or bunching; the estimand changes to units "
        "outside the donut."
    )
    assumptions = [
        "Manipulation/sorting is local to the dropped band around the cutoff",
        "Units outside the donut are comparable across the cutoff",
        "Density/covariate continuity holds on the retained sample",
    ]

    def __init__(self):
        super().__init__()

    def validate_assumptions(self, df, variables) -> Dict[str, Any]:
        return {
            "assumptions_valid": True,
            "failed_assumptions": [],
            "warnings": [
                "Donut RDD drops units near the cutoff; the estimate refers to "
                "the population outside the donut and is only credible if sorting "
                "is local to that band."
            ],
            "suggestions": [],
        }

    def estimate_effect(
        self,
        df: pd.DataFrame,
        variables,
        query: Optional[str] = None,
        donut_width: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        running_variable = getattr(variables, "running_variable", None)
        cutoff_value = getattr(variables, "cutoff_value", None)
        treatment = getattr(variables, "treatment_variable", None)
        outcome = getattr(variables, "outcome_variable", None)
        covariates = list(
            getattr(variables, "confounders", None)
            or getattr(variables, "covariates", None)
            or []
        )
        treat_above = getattr(variables, "treat_above_cutoff", None)

        try:
            return estimate_effect(
                df=df,
                treatment=treatment,
                outcome=outcome,
                running_variable=running_variable,
                cutoff_value=cutoff_value,
                covariates=covariates,
                donut_width=donut_width,
                treat_above_cutoff=treat_above,
                query=query,
                **kwargs,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "error": str(exc),
                "effect_estimate": None,
                "method_used": "donut_rdd",
                "diagnostics": {"error": str(exc), "donut_width": donut_width},
            }

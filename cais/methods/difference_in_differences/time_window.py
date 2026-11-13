"""Time-window DiD: drop the contaminated anticipation window before estimating.

The user supplies the start of the early-response window and the official
treatment start. Rows with ``window_start <= time < treatment_start`` are
removed; the remaining pre-window periods form the baseline, and the usual
difference-in-differences regression is estimated on the retained data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from cais.methods.causal_method import CausalMethod
from cais.methods.difference_in_differences.estimator import estimate_twfe_did

logger = logging.getLogger(__name__)

DID_TIME_WINDOW_METHOD = "difference_in_differences_time_window"


def _is_binary(series: pd.Series) -> bool:
    values = set(pd.Series(series).dropna().unique().tolist())
    return values.issubset({0, 1, False, True}) and len(values) <= 2


def estimate_effect_time_window(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[Sequence[str]] = None,
    dataset_description: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Estimate DiD after dropping the contaminated pre-treatment window.

    Required keyword arguments:
        time_variable: time column
        group_variable: unit/group identifier used for fixed effects
        treatment_period_start: official treatment start
        anticipation_window_start: start of the early-response window

    Rows in ``[anticipation_window_start, treatment_period_start)`` are removed.
    The result reports how many rows were removed and that the estimate refers
    to the retained time window.
    """
    time_variable = kwargs.get("time_variable")
    group_variable = kwargs.get("group_variable")
    treatment_start = kwargs.get("treatment_period_start")
    window_start = kwargs.get("anticipation_window_start")

    if df is None or not hasattr(df, "columns"):
        raise ValueError("Time-window DiD requires a dataframe.")
    for label, name in (
        ("time variable", time_variable),
        ("group variable", group_variable),
        ("treatment variable", treatment),
        ("outcome variable", outcome),
    ):
        if not name or name not in df.columns:
            raise ValueError(f"Time-window DiD: {label} {name!r} is not a column.")

    if treatment_start is None:
        raise ValueError("Time-window DiD requires treatment_period_start.")
    if window_start is None:
        raise ValueError("Time-window DiD requires anticipation_window_start.")
    try:
        if window_start >= treatment_start:
            raise ValueError(
                "anticipation_window_start must be strictly before "
                "treatment_period_start."
            )
    except TypeError as exc:
        raise ValueError(
            "anticipation_window_start and treatment_period_start must use "
            "comparable time units."
        ) from exc

    time_values = df[time_variable].dropna()
    if time_values.nunique() <= 2:
        raise ValueError(
            "Time-window DiD needs at least three distinct time periods; the "
            "time column looks like a single before/after indicator."
        )
    if not _is_binary(df[treatment]):
        raise ValueError(
            "Time-window DiD requires a binary treatment/group indicator."
        )

    in_window = (df[time_variable] >= window_start) & (df[time_variable] < treatment_start)
    filtered = df.loc[~in_window].copy()
    n_removed = int(in_window.sum())
    n_total = len(df)
    if filtered.empty:
        raise ValueError("Time-window DiD would drop every observation.")
    if n_removed == 0:
        logger.warning(
            "Time-window DiD found no observations inside the stated window; "
            "the result is equivalent to standard DiD on the supplied data."
        )

    pre = filtered[filtered[time_variable] < treatment_start]
    post = filtered[filtered[time_variable] >= treatment_start]
    if pre.empty or post.empty:
        raise ValueError(
            "Time-window DiD needs both pre-window and post-treatment "
            "observations after dropping the contaminated period."
        )
    if pre[treatment].nunique() < 2 or post[treatment].nunique() < 2:
        raise ValueError(
            "Time-window DiD needs both treated and control units before and "
            "after the treatment start."
        )

    filtered["__did_time_window__"] = (
        filtered[treatment].astype(int)
        * (filtered[time_variable] >= treatment_start).astype(int)
    )

    result = estimate_twfe_did(
        filtered,
        outcome=outcome,
        time_var=time_variable,
        group_var=group_variable,
        did_term="__did_time_window__",
        covariates=list(covariates or []),
    )

    result["method_used"] = DID_TIME_WINDOW_METHOD
    result["anticipation_window_start"] = window_start
    result["treatment_period_start"] = treatment_start
    result["n_rows_removed"] = n_removed
    result["share_rows_removed"] = (n_removed / n_total) if n_total else 0.0
    result["time_window_adjustment"] = True
    result["estimand_population"] = (
        "units in the retained periods: pre-window baseline and post-treatment "
        "periods; the contaminated anticipation window is excluded"
    )

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics.update(
        {
            "time_window_adjustment": True,
            "anticipation_window_start": window_start,
            "treatment_period_start": treatment_start,
            "n_rows_removed": n_removed,
            "share_rows_removed": result["share_rows_removed"],
            "estimand": result["estimand_population"],
        }
    )
    result["diagnostics"] = diagnostics

    details = result.get("method_details")
    prefix = (
        f"Time-window DiD (dropped {n_removed}/{n_total} rows from the "
        f"anticipation window [{window_start}, {treatment_start}))"
    )
    result["method_details"] = f"{prefix}. {details}" if details else prefix
    return result


class DiDTimeWindow(CausalMethod):
    """Difference-in-differences with the contaminated anticipation window removed."""

    name = "Time-Window Difference-in-Differences"
    description = (
        "Difference-in-differences estimated after dropping observations from "
        "the user-supplied early-response window before treatment starts. This "
        "is the estimator counterpart of the bounded-anticipation check."
    )
    assumptions = [
        "Bounded anticipation: the only contaminated period is the supplied window",
        "Parallel trends holds on the retained pre-window and post-treatment periods",
        "The treatment/group indicator is binary and stable within units",
    ]

    def __init__(self):
        super().__init__()

    def validate_assumptions(self, df, variables) -> Dict[str, Any]:
        return {
            "assumptions_valid": True,
            "failed_assumptions": [],
            "warnings": [
                "Time-window DiD drops the supplied anticipation window; the "
                "estimate refers to the retained periods."
            ],
            "suggestions": [],
        }

    def estimate_effect(
        self,
        df: pd.DataFrame,
        variables,
        query: Optional[str] = None,
        anticipation_window_start: Optional[Any] = None,
        treatment_period_start: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        treatment_start = treatment_period_start
        if treatment_start is None:
            treatment_start = getattr(variables, "treatment_time", None)

        try:
            return estimate_effect_time_window(
                df=df,
                treatment=getattr(variables, "treatment_variable", None),
                outcome=getattr(variables, "outcome_variable", None),
                covariates=list(
                    getattr(variables, "confounders", None)
                    or getattr(variables, "covariates", None)
                    or []
                ),
                time_variable=getattr(variables, "time_variable", None),
                group_variable=getattr(variables, "group_variable", None),
                treatment_period_start=treatment_start,
                anticipation_window_start=anticipation_window_start,
                **kwargs,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "error": str(exc),
                "effect_estimate": None,
                "method_used": DID_TIME_WINDOW_METHOD,
                "diagnostics": {
                    "error": str(exc),
                    "anticipation_window_start": anticipation_window_start,
                    "treatment_period_start": treatment_start,
                },
            }

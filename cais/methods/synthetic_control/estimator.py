"""Synthetic control following Abadie, Diamond, and Hainmueller (2010).

The donor weights are constrained to the simplex, so the counterfactual is a
convex combination of untreated units.  When a small number of units is
treated, the paper's recommendation is followed and their outcomes and
predictors are aggregated before fitting one synthetic control.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cais.methods.causal_method import CausalMethod


def _simplex_weights(x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Minimize ||X1 - X0 W||² subject to W being on the simplex."""
    n_donors = x0.shape[1]
    initial = np.repeat(1.0 / n_donors, n_donors)
    result = minimize(
        lambda w: float(np.sum((x1 - x0 @ w) ** 2)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-12, "maxiter": 10_000},
    )
    if not result.success:
        raise RuntimeError(f"Synthetic-control optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def _scaled_predictors(
    outcomes: pd.DataFrame,
    covariates: Dict[str, pd.DataFrame],
    target_units: Iterable[Any],
    donor_units: List[Any],
    pre_periods: List[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build X1 and X0 from pre outcomes and pre-treatment covariate means."""
    target_units = list(target_units)
    target = [outcomes.loc[target_units, pre_periods].mean(axis=0).to_numpy(float)]
    donors = [outcomes.loc[donor_units, pre_periods].to_numpy(float).T]

    for values in covariates.values():
        target.append(np.array([values.loc[target_units, pre_periods].to_numpy(float).mean()]))
        donors.append(values.loc[donor_units, pre_periods].mean(axis=1).to_numpy(float)[None, :])

    x1 = np.concatenate(target)
    x0 = np.vstack(donors)
    # A diagonal positive-semidefinite V: scale predictors to comparable units.
    scale = np.std(np.column_stack([x1, x0]), axis=1, ddof=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return x1 / scale, x0 / scale[:, None]


def _fit_one(
    outcomes: pd.DataFrame,
    covariates: Dict[str, pd.DataFrame],
    target_units: List[Any],
    donor_units: List[Any],
    pre_periods: List[Any],
) -> Dict[str, Any]:
    x1, x0 = _scaled_predictors(outcomes, covariates, target_units, donor_units, pre_periods)
    weights = _simplex_weights(x1, x0)
    actual = outcomes.loc[target_units].mean(axis=0).to_numpy(float)
    synthetic = weights @ outcomes.loc[donor_units].to_numpy(float)
    gap = actual - synthetic
    pre_index = [outcomes.columns.get_loc(period) for period in pre_periods]
    pre_mspe = float(np.mean(np.square(gap[pre_index])))
    return {
        "weights": weights,
        "actual": actual,
        "synthetic": synthetic,
        "gap": gap,
        "pre_mspe": pre_mspe,
    }


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[List[str]] = None,
    *,
    time_variable: str,
    group_variable: str,
    treatment_time: Optional[Any] = None,
    treatment_state: Optional[Any] = None,
    run_placebos: bool = True,
    placebo_mspe_multiplier: float = 20.0,
    **_: Any,
) -> Dict[str, Any]:
    """Estimate the average post-intervention effect with synthetic control.

    ``treatment`` may be a time-varying exposure indicator or a time-invariant
    treated-group indicator.  The latter requires ``treatment_time``.
    """
    covariates = list(covariates or [])
    required = [group_variable, time_variable, treatment, outcome, *covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing synthetic-control columns: {missing}")
    if df.duplicated([group_variable, time_variable]).any():
        raise ValueError("Synthetic control requires one observation per unit and time period.")

    work = df[required].dropna().copy()
    treatment_values = set(work[treatment].unique())
    if not treatment_values.issubset({0, 1, False, True}):
        raise ValueError("Synthetic control requires a binary treatment indicator.")

    exposure = work.pivot(index=group_variable, columns=time_variable, values=treatment).sort_index(axis=1)
    times = list(exposure.columns)
    if exposure.isna().any().any():
        raise ValueError("Synthetic control requires a balanced treatment panel.")

    varies_over_time = bool((exposure.nunique(axis=1) > 1).any())
    if varies_over_time:
        treated_units = list(exposure.index[exposure.max(axis=1) == 1])
        starts = [next(t for t in times if exposure.loc[unit, t] == 1) for unit in treated_units]
        if len(set(starts)) != 1:
            raise ValueError("This implementation requires a common intervention time (no staggered adoption).")
        inferred_time = starts[0]
        if treatment_time is not None and treatment_time != inferred_time:
            raise ValueError("treatment_time conflicts with the observed treatment onset.")
        treatment_time = inferred_time
    else:
        treated_units = list(exposure.index[exposure.iloc[:, 0] == 1])
        if treatment_state is not None:
            treated_units = [treatment_state]
        if treatment_time is None:
            raise ValueError("treatment_time is required for a time-invariant treatment indicator.")

    donor_units = [unit for unit in exposure.index if unit not in treated_units and exposure.loc[unit].max() == 0]
    if not treated_units:
        raise ValueError("No treated unit was identified.")
    if len(donor_units) < 2:
        raise ValueError("Synthetic control requires at least two never-treated donor units.")

    pre_periods = [period for period in times if period < treatment_time]
    post_periods = [period for period in times if period >= treatment_time]
    if len(pre_periods) < 2 or not post_periods:
        raise ValueError("Synthetic control requires at least two pre-treatment and one post-treatment periods.")

    outcomes = work.pivot(index=group_variable, columns=time_variable, values=outcome).reindex(index=exposure.index, columns=times)
    covariate_panels = {
        column: work.pivot(index=group_variable, columns=time_variable, values=column).reindex(index=exposure.index, columns=times)
        for column in covariates
    }
    if outcomes.isna().any().any() or any(panel.isna().any().any() for panel in covariate_panels.values()):
        raise ValueError("Synthetic control requires a balanced panel without missing analysis values.")

    fitted = _fit_one(outcomes, covariate_panels, treated_units, donor_units, pre_periods)
    post_idx = [times.index(period) for period in post_periods]
    post_mspe = float(np.mean(np.square(fitted["gap"][post_idx])))
    ratio = np.inf if fitted["pre_mspe"] <= 1e-15 else post_mspe / fitted["pre_mspe"]

    placebo_ratios: Dict[str, float] = {}
    eligible_ratios: List[float] = []
    if run_placebos:
        for placebo in donor_units:
            placebo_donors = [unit for unit in donor_units if unit != placebo]
            if len(placebo_donors) < 2:
                continue
            placebo_fit = _fit_one(outcomes, covariate_panels, [placebo], placebo_donors, pre_periods)
            if placebo_fit["pre_mspe"] > placebo_mspe_multiplier * max(fitted["pre_mspe"], 1e-15):
                continue
            placebo_post_mspe = float(np.mean(np.square(placebo_fit["gap"][post_idx])))
            placebo_ratio = np.inf if placebo_fit["pre_mspe"] <= 1e-15 else placebo_post_mspe / placebo_fit["pre_mspe"]
            placebo_ratios[str(placebo)] = placebo_ratio
            eligible_ratios.append(placebo_ratio)

    p_value = None
    if run_placebos and eligible_ratios:
        p_value = float((1 + sum(value >= ratio for value in eligible_ratios)) / (1 + len(eligible_ratios)))

    average_effect = float(np.mean(fitted["gap"][post_idx]))
    return {
        "effect_estimate": average_effect,
        "standard_error": None,
        "confidence_interval": [None, None],
        "p_value": p_value,
        "method_used": "Synthetic Control",
        "diagnostics": {
            "treated_units": [str(unit) for unit in treated_units],
            "donor_weights": {str(unit): float(weight) for unit, weight in zip(donor_units, fitted["weights"])},
            "treatment_time": treatment_time,
            "pre_periods": pre_periods,
            "post_periods": post_periods,
            "pre_mspe": fitted["pre_mspe"],
            "post_mspe": post_mspe,
            "post_pre_mspe_ratio": ratio,
            "placebo_ratios": placebo_ratios,
            "n_placebos": len(eligible_ratios),
            "period_effects": {str(period): float(fitted["gap"][times.index(period)]) for period in post_periods},
            "actual_path": {str(period): float(value) for period, value in zip(times, fitted["actual"])},
            "synthetic_path": {str(period): float(value) for period, value in zip(times, fitted["synthetic"])},
        },
        "raw_results": {
            "estimand": "average post-intervention treated-minus-synthetic gap",
            "inference": "in-space placebo permutation (Abadie, Diamond, and Hainmueller, 2010)",
        },
    }


class SyntheticControl(CausalMethod):
    """Pipeline adapter for the Abadie-Diamond-Hainmueller estimator."""

    name = "Synthetic Control"
    description = "Constructs a convex combination of never-treated units to reproduce the treated unit's pre-intervention predictors and outcomes."
    assumptions = [
        "No interference between treated and donor units.",
        "No anticipation before the intervention period.",
        "The treated predictors lie sufficiently close to the donor-pool convex hull.",
        "Donor units are not exposed to the intervention.",
    ]

    def validate_assumptions(self, df: pd.DataFrame, variables: Any) -> Dict[str, Any]:
        required = [variables.treatment_variable, variables.outcome_variable, variables.time_variable, variables.group_variable]
        missing = [name for name in required if not name or name not in df.columns]
        return {"assumptions_valid": not missing, "failed_assumptions": missing, "warnings": [], "suggestions": []}

    def estimate_effect(self, df: pd.DataFrame, variables: Any, query: Optional[str] = None) -> Dict[str, Any]:
        return estimate_effect(
            df,
            treatment=variables.treatment_variable,
            outcome=variables.outcome_variable,
            covariates=variables.covariates,
            time_variable=variables.time_variable,
            group_variable=variables.group_variable,
            treatment_time=variables.treatment_time,
            treatment_state=variables.treatment_state,
        )

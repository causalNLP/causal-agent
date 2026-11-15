"""Local-linear fuzzy regression-discontinuity estimator.

The estimator targets the local Wald estimand at a documented cutoff: the
outcome discontinuity divided by the treatment-receipt discontinuity.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from cais.methods.causal_method import CausalMethod


def _default_bandwidth(running: pd.Series) -> float:
    running_range = float(running.max() - running.min())
    if not np.isfinite(running_range) or running_range <= 0:
        raise ValueError("The running variable must have a positive finite range.")
    return 0.1 * running_range


def _cross_covariance_hc1(
    design: np.ndarray,
    weights: np.ndarray,
    residuals_a: np.ndarray,
    residuals_b: np.ndarray,
    coefficient_index: int,
) -> float:
    """HC1 cross-equation covariance for matching weighted regressions."""
    bread = np.linalg.pinv(design.T @ (weights[:, None] * design))
    projection = design @ bread[coefficient_index]
    influence_a = projection * weights * residuals_a
    influence_b = projection * weights * residuals_b
    n_obs, n_parameters = design.shape
    correction = n_obs / (n_obs - n_parameters)
    return float(correction * np.sum(influence_a * influence_b))


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[List[str]] = None,
    running_variable: Optional[str] = None,
    cutoff_value: Optional[float] = None,
    treat_above_cutoff: bool = True,
    bandwidth: Optional[float] = None,
    min_first_stage_f: float = 10.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate a fuzzy-RD LATE using a triangular-kernel local Wald ratio.

    ``treatment`` is actual treatment receipt. Eligibility is constructed from
    the running variable and cutoff and acts as the local instrument.
    """
    covariates = covariates or []
    if not running_variable:
        raise ValueError("Fuzzy RDD requires a running_variable.")
    if cutoff_value is None or not np.isfinite(cutoff_value):
        raise ValueError("Fuzzy RDD requires a finite cutoff_value.")

    required = [treatment, outcome, running_variable, *covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Fuzzy RDD is missing required columns: {missing}")

    analysis = df[required].dropna().copy()
    if analysis.empty:
        raise ValueError("No complete observations are available for fuzzy RDD.")
    if not np.isin(analysis[treatment].unique(), [0, 1]).all():
        raise ValueError("Fuzzy RDD currently requires treatment receipt coded 0/1.")

    final_bandwidth = bandwidth if bandwidth is not None else _default_bandwidth(analysis[running_variable])
    if not np.isfinite(final_bandwidth) or final_bandwidth <= 0:
        raise ValueError("bandwidth must be a positive finite number.")

    analysis["running_centered"] = analysis[running_variable].astype(float) - float(cutoff_value)
    analysis = analysis[analysis["running_centered"].abs() <= final_bandwidth].copy()
    if analysis.empty:
        raise ValueError("No observations lie within the fuzzy-RD bandwidth.")

    if treat_above_cutoff:
        analysis["eligibility"] = (analysis["running_centered"] >= 0).astype(float)
    else:
        analysis["eligibility"] = (analysis["running_centered"] < 0).astype(float)

    side_counts = analysis["eligibility"].value_counts()
    if side_counts.get(0.0, 0) < 2 or side_counts.get(1.0, 0) < 2:
        raise ValueError("Fuzzy RDD requires at least two observations on each side of the cutoff.")

    analysis["eligibility_x_running"] = analysis["eligibility"] * analysis["running_centered"]
    design_columns = ["eligibility", "running_centered", "eligibility_x_running", *covariates]
    design_df = sm.add_constant(analysis[design_columns], has_constant="add")
    design = design_df.to_numpy(dtype=float)
    if len(analysis) <= design.shape[1]:
        raise ValueError("Insufficient observations for the requested fuzzy-RD specification.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The fuzzy-RD local-linear design matrix is rank deficient.")

    kernel_weights = 1.0 - analysis["running_centered"].abs().to_numpy() / final_bandwidth
    kernel_weights = np.maximum(kernel_weights, np.finfo(float).eps)
    outcome_fit = sm.WLS(analysis[outcome].astype(float), design_df, weights=kernel_weights).fit(cov_type="HC1")
    treatment_fit = sm.WLS(analysis[treatment].astype(float), design_df, weights=kernel_weights).fit(cov_type="HC1")

    reduced_form = float(outcome_fit.params["eligibility"])
    first_stage = float(treatment_fit.params["eligibility"])
    first_stage_se = float(treatment_fit.bse["eligibility"])
    if not np.isfinite(first_stage_se) or first_stage_se <= 0:
        raise ValueError("The fuzzy-RD first-stage uncertainty could not be estimated.")
    first_stage_f = float((first_stage / first_stage_se) ** 2)
    if first_stage <= 0:
        raise ValueError(
            "The treatment-receipt jump is not positive in the documented eligibility direction; "
            "check treatment and cutoff coding."
        )
    if first_stage_f < min_first_stage_f:
        raise ValueError(
            f"Weak fuzzy-RD first stage: F={first_stage_f:.3f} < {min_first_stage_f:.3f}."
        )

    estimate = reduced_form / first_stage
    eligibility_index = list(design_df.columns).index("eligibility")
    cross_covariance = _cross_covariance_hc1(
        design,
        kernel_weights,
        np.asarray(outcome_fit.resid, dtype=float),
        np.asarray(treatment_fit.resid, dtype=float),
        eligibility_index,
    )
    reduced_form_variance = float(outcome_fit.cov_params().loc["eligibility", "eligibility"])
    first_stage_variance = float(treatment_fit.cov_params().loc["eligibility", "eligibility"])
    ratio_variance = (
        reduced_form_variance / first_stage**2
        + reduced_form**2 * first_stage_variance / first_stage**4
        - 2.0 * reduced_form * cross_covariance / first_stage**3
    )
    standard_error = float(np.sqrt(max(ratio_variance, 0.0)))
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("The fuzzy-RD delta-method standard error is not positive and finite.")

    z_value = estimate / standard_error
    p_value = float(2.0 * norm.sf(abs(z_value)))
    confidence_interval = [
        float(estimate - 1.96 * standard_error),
        float(estimate + 1.96 * standard_error),
    ]
    return {
        "effect_estimate": float(estimate),
        "standard_error": standard_error,
        "confidence_interval": confidence_interval,
        "p_value": p_value,
        "method_used": "Fuzzy RDD local Wald",
        "diagnostics": {
            "reduced_form_jump": reduced_form,
            "first_stage_jump": first_stage,
            "first_stage_standard_error": first_stage_se,
            "first_stage_f_statistic": first_stage_f,
            "bandwidth_used": float(final_bandwidth),
            "kernel": "triangular",
            "n_observations": int(len(analysis)),
            "n_eligible_side": int(side_counts.get(1.0, 0)),
            "n_ineligible_side": int(side_counts.get(0.0, 0)),
            "treat_above_cutoff": bool(treat_above_cutoff),
        },
        "raw_results": {
            "estimand": "local Wald LATE at the cutoff among fuzzy-RD compliers",
            "formula": "jump in E[Y|X] / jump in E[D|X] at the cutoff",
            "inference": "HC1 delta method with cross-equation covariance",
        },
    }


class FuzzyRDD(CausalMethod):
    """Pipeline adapter for the local-linear fuzzy-RD estimator."""

    name = "Fuzzy Regression Discontinuity Design"
    description = "Estimates the complier LATE at a cutoff as a local Wald ratio."
    assumptions = [
        "Treatment eligibility changes discontinuously at the cutoff.",
        "Potential outcomes and potential treatment receipt are continuous at the cutoff.",
        "Eligibility has no direct effect on the outcome except through treatment receipt.",
        "Treatment receipt is monotone in eligibility near the cutoff.",
        "There is no precise manipulation of the running variable around the cutoff.",
    ]

    def validate_assumptions(self, df: pd.DataFrame, variables: Any) -> Dict[str, Any]:
        required = [
            variables.treatment_variable,
            variables.outcome_variable,
            variables.running_variable,
        ]
        missing = [name for name in required if not name or name not in df.columns]
        return {
            "assumptions_valid": not missing,
            "failed_assumptions": missing,
            "warnings": ["Continuity, exclusion, and monotonicity require substantive review."],
            "suggestions": [],
        }

    def estimate_effect(self, df: pd.DataFrame, variables: Any, query: Optional[str] = None) -> Dict[str, Any]:
        return estimate_effect(
            df=df,
            treatment=variables.treatment_variable,
            outcome=variables.outcome_variable,
            covariates=variables.confounders or [],
            running_variable=variables.running_variable,
            cutoff_value=variables.cutoff_value,
            treat_above_cutoff=(
                True if variables.treat_above_cutoff is None else variables.treat_above_cutoff
            ),
        )

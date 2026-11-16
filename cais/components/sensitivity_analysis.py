"""Post-estimation sensitivity diagnostics.

These diagnostics are deliberately advisory.  They describe sampling
stability of an already selected estimator; they do not test exchangeability,
exclusion restrictions, SUTVA, or any other identifying assumption.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd


def run_sensitivity_analysis(
    *,
    df: Optional[pd.DataFrame],
    estimator: Optional[Callable[..., Mapping[str, Any]]],
    variables: Any,
    query: Optional[str],
    method_options: Optional[dict[str, Any]] = None,
    base_results: Optional[Mapping[str, Any]] = None,
    n_bootstrap: int = 50,
    seed: int = 2027,
) -> dict[str, Any]:
    """Run an optional bootstrap stability analysis for the selected route.

    The selected estimator is re-fitted on ordinary iid bootstrap samples.  A
    route is marked ``not_run`` when it cannot be safely re-fitted through the
    common ``CausalMethod`` interface (for example, a function-based fallback
    route or an identified set without a scalar effect).  Failures in
    individual resamples are recorded rather than stopping the main pipeline.
    """
    output: dict[str, Any] = {
        "status": "not_run",
        "advisory": True,
        "estimand_or_assumption_tested": "finite-sample estimator stability only",
        "warnings": [],
    }
    if df is None or estimator is None:
        output["warnings"].append(
            "Bootstrap sensitivity was unavailable for the executed fallback route."
        )
        return output

    base = (base_results or {}).get("effect_estimate")
    if not isinstance(base, (int, float, np.integer, np.floating)) or not np.isfinite(base):
        output["warnings"].append(
            "Bootstrap sensitivity requires a finite scalar effect estimate."
        )
        return output
    if len(df) < 5:
        output["warnings"].append("Too few rows for a bootstrap stability diagnostic.")
        return output

    estimates: list[float] = []
    failures: list[str] = []
    options = dict(method_options or {})
    n_bootstrap = max(10, min(int(n_bootstrap), 250))
    for draw in range(n_bootstrap):
        sample = df.sample(n=len(df), replace=True, random_state=seed + draw)
        try:
            result = estimator(df=sample, variables=variables, query=query, **options)
            value = result.get("effect_estimate") if isinstance(result, Mapping) else None
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                estimates.append(float(value))
            else:
                failures.append("non_scalar_effect")
        except Exception as exc:  # advisory diagnostics must never block execution
            failures.append(type(exc).__name__)

    output.update({
        "status": "supported" if estimates else "failed",
        "base_effect_estimate": float(base),
        "requested_resamples": n_bootstrap,
        "successful_resamples": len(estimates),
        "failed_resamples": len(failures),
        "failure_types": {str(key): int(value) for key, value in pd.Series(failures).value_counts().items()}
        if failures else {},
    })
    if estimates:
        values = np.asarray(estimates, dtype=float)
        low, high = np.quantile(values, [0.025, 0.975])
        output.update({
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_interval_95": [float(low), float(high)],
            "base_inside_bootstrap_interval": bool(low <= float(base) <= high),
            "relative_shift_from_base": float((np.mean(values) - float(base)) / max(abs(float(base)), 1e-12)),
        })
    if len(estimates) < max(10, n_bootstrap // 2):
        output["warnings"].append("Many bootstrap refits failed; treat the stability summary cautiously.")

    # Model-specification sensitivity: omit one declared adjustment variable at
    # a time.  This is not a test that the omitted variable is unnecessary; it
    # only reports how dependent the fitted estimate is on the chosen set.
    covariates = list(getattr(variables, "covariates", None) or [])
    confounders = list(getattr(variables, "confounders", None) or [])
    adjustment_set = list(dict.fromkeys(covariates + confounders))[:20]
    omission_results: list[dict[str, Any]] = []
    for omitted in adjustment_set:
        try:
            if hasattr(variables, "model_copy"):
                reduced_variables = variables.model_copy(deep=True)
                reduced_variables.covariates = [x for x in covariates if x != omitted]
                reduced_variables.confounders = [x for x in confounders if x != omitted]
            else:
                reduced_variables = deepcopy(variables)
                if isinstance(reduced_variables, dict):
                    reduced_variables["covariates"] = [x for x in covariates if x != omitted]
                    reduced_variables["confounders"] = [x for x in confounders if x != omitted]
            result = estimator(df=df, variables=reduced_variables, query=query, **options)
            value = result.get("effect_estimate") if isinstance(result, Mapping) else None
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                omission_results.append({
                    "omitted_covariate": omitted,
                    "effect_estimate": float(value),
                    "absolute_change": float(value - float(base)),
                })
        except Exception as exc:  # one unsupported omission must not block others
            omission_results.append({"omitted_covariate": omitted, "status": "failed", "error": type(exc).__name__})
    if omission_results:
        output["covariate_omission_sensitivity"] = omission_results
    output["warnings"].append(
        "This diagnostic does not establish causal identification or validate untestable assumptions."
    )
    return output


__all__ = ["run_sensitivity_analysis"]

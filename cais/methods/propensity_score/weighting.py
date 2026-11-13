# Propensity Score Weighting (IPW) Implementation 

import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import Dict, List, Optional, Any, Sequence

from cais.methods.causal_method import CausalMethod

from .base import estimate_propensity_scores, format_ps_results, select_propensity_model
from .diagnostics import assess_weight_distribution, plot_overlap, plot_balance # Import diagnostic functions
from .llm_assist import determine_optimal_weight_type, determine_optimal_trim_threshold, get_llm_parameters # Import LLM helpers

logger = logging.getLogger(__name__)

# Minimum observations required in each treatment group after trimming.
# Below this the weighted contrast is too fragile to report.
MIN_TRIMMED_GROUP_SIZE = 5


def _validate_ps_bounds(bounds: Sequence[float]) -> tuple:
    '''Validate a (low, high) propensity-score overlap band.

    Args:
        bounds: Two numbers, 0 <= low < high <= 1.

    Returns:
        Tuple (low, high) as floats.

    Raises:
        ValueError: If the pair is malformed or out of range.
    '''
    try:
        lo, hi = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"ps_bounds must be a pair (low, high) of numbers; got {bounds!r}."
        ) from exc
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"ps_bounds must satisfy 0 <= low < high <= 1; got [{lo}, {hi}].")
    return lo, hi


def default_trim_bounds() -> tuple:
    '''Propensity-score bounds used by the trimmed-IPW leaf.

    Reads ``positivity.extreme_ps_bounds`` from the user-editable thresholds
    config (cais.config.load_thresholds), falling back to (0.1, 0.9) when the
    key is missing or malformed. Explicitly supplied ``ps_bounds`` are still
    validated strictly by :func:`estimate_effect`.
    '''
    from cais.config import load_thresholds

    thresholds = load_thresholds() or {}
    positivity = thresholds.get("positivity") or {}
    bounds = positivity.get("extreme_ps_bounds")
    if bounds is None:
        return 0.1, 0.9
    try:
        return _validate_ps_bounds(bounds)
    except ValueError:
        return 0.1, 0.9


def _post_checks_after_weighting(df: pd.DataFrame, treatment: str, outcome: str,
                                 covariates: List[str], weights: np.ndarray) -> Dict[str, Any]:
    '''Advisory post-modeling check for IPW: covariate balance after weighting.

    Runs after the estimate is produced, so it never changes the result. The
    verdict is carried in ``diagnostics["post_checks"]`` and the agent turns it
    into an affirming/warning sentence in CAIS's output.
    '''
    try:
        from cais.models import AssumptionVariables
        from cais.methods.post_model_assumption_utils import check_balance_after_weighting

        variables = AssumptionVariables(
            df=df, treatment=treatment, outcome=outcome, covariates=list(covariates)
        )
        verdict = check_balance_after_weighting(variables, np.asarray(weights, dtype=float))
        return {"balance_after_weighting": verdict.model_dump()}
    except Exception as exc:  # advisory checks must never break execution
        logger.warning("Post-check balance_after_weighting could not be evaluated: %s", exc)
        return {
            "balance_after_weighting": {
                "passed": None,
                "reasoning": f"Post-check could not be evaluated: {exc}",
                "details": {},
            }
        }


def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, 
                      covariates: List[str],
                      ps_bounds: Optional[Sequence[float]] = None,
                      **kwargs) -> Dict[str, Any]:
    '''Generic propensity score weighting (IPW) implementation.
    
    Args:
        df: Dataset containing causal variables
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        covariates: List of covariate names
        ps_bounds: Optional (low, high) propensity-score band. When given, units
            whose propensity score falls outside the band are dropped before
            weighting (the trimmed-IPW remedy for a positivity failure). Takes
            precedence over ``trim_threshold``.
        **kwargs: Method-specific parameters (e.g., weight_type, trim_threshold, query)
        
    Returns:
        Dictionary with effect estimate and diagnostics
    '''
    query = kwargs.get('query')

    # --- LLM-Assisted Parameter Optimization / Defaults --- 
    llm_params = get_llm_parameters(df, query, "PS.Weighting")
    llm_suggested_params = llm_params.get("parameters", {})
    
    # Explicitly check LLM suggestion before falling back to default helper
    llm_weight_type = llm_suggested_params.get('weight_type')
    default_weight_type = determine_optimal_weight_type(df, treatment, query) if llm_weight_type is None else llm_weight_type
    weight_type = kwargs.get('weight_type', default_weight_type)
    
    # Similar explicit check for trim_threshold
    llm_trim_thresh = llm_suggested_params.get('trim_threshold')
    default_trim_thresh = determine_optimal_trim_threshold(df, treatment, query=query) if llm_trim_thresh is None else llm_trim_thresh
    trim_threshold = kwargs.get('trim_threshold', default_trim_thresh)
        
    propensity_model_type = kwargs.get('propensity_model_type', 
                                   llm_suggested_params.get('propensity_model_type', 
                                                          select_propensity_model(df, treatment, covariates, query)))
    robust_se = kwargs.get('robust_se', True)

    # --- Step 1: Estimate propensity scores --- 
    propensity_scores = estimate_propensity_scores(df, treatment, covariates, 
                                                   model_type=propensity_model_type, 
                                                   **kwargs) # Pass other kwargs like C, penalty etc.
    df_ps = df.copy()
    df_ps['propensity_score'] = propensity_scores

    # --- Step 2: Calculate weights --- 
    if weight_type.upper() == 'ATE':
        weights = np.where(df_ps[treatment] == 1, 
                           1 / df_ps['propensity_score'], 
                           1 / (1 - df_ps['propensity_score']))
    elif weight_type.upper() == 'ATT':
        weights = np.where(df_ps[treatment] == 1, 
                           1, 
                           df_ps['propensity_score'] / (1 - df_ps['propensity_score']))
    # TODO: Add other weight types like ATC if needed
    else:
        raise ValueError(f"Unsupported weight type: {weight_type}")
        
    df_ps['ipw'] = weights
    n_before = int(len(df_ps))

    # --- Step 3: Apply trimming if needed --- 
    # Two mutually exclusive trim rules:
    #   * ps_bounds: drop units whose propensity score falls outside the
    #     overlap band. This is the decision tree's remedy when the positivity
    #     gate fails; the bounds come from the user-editable positivity
    #     thresholds (extreme_ps_bounds).
    #   * trim_threshold: the legacy percentile trim.
    trim_info: Dict[str, Any] = {
        "trim_rule": "none",
        "n_before": n_before,
        "n_after": n_before,
        "n_trimmed": 0,
        "dropped_fraction": 0.0,
    }
    if ps_bounds is not None:
        lo, hi = _validate_ps_bounds(ps_bounds)
        keep_indices = (df_ps['propensity_score'] >= lo) & (df_ps['propensity_score'] <= hi)
        df_trimmed = df_ps[keep_indices].copy()
        n_after = int(len(df_trimmed))
        if n_after == 0:
            raise ValueError(
                f"All units removed after trimming to propensity-score bounds [{lo}, {hi}]. "
                "Widen the bounds or use a different method."
            )
        if df_trimmed[treatment].nunique() < 2:
            raise ValueError(
                f"Trimming to propensity-score bounds [{lo}, {hi}] removed all units "
                "of one treatment group."
            )
        smallest_group = int(df_trimmed[treatment].value_counts().min())
        if smallest_group < MIN_TRIMMED_GROUP_SIZE:
            raise ValueError(
                f"Trimming to propensity-score bounds [{lo}, {hi}] left only "
                f"{smallest_group} unit(s) in the smaller treatment group "
                f"(minimum required: {MIN_TRIMMED_GROUP_SIZE})."
            )
        print(f"Trimming {n_before - n_after} units (propensity scores outside [{lo}, {hi}])")
        df_analysis = df_trimmed
        trim_threshold = 0  # percentile trim is not used when ps_bounds is set
        trim_info = {
            "trim_rule": "ps_bounds",
            "ps_bounds": [lo, hi],
            "n_before": n_before,
            "n_after": n_after,
            "n_trimmed": n_before - n_after,
            "dropped_fraction": (n_before - n_after) / n_before if n_before else 0.0,
        }
    elif trim_threshold is not None and trim_threshold > 0:
        # Trim based on propensity score percentile
        min_ps_thresh = np.percentile(propensity_scores, trim_threshold * 100)
        max_ps_thresh = np.percentile(propensity_scores, (1 - trim_threshold) * 100)
        
        keep_indices = (df_ps['propensity_score'] >= min_ps_thresh) & (df_ps['propensity_score'] <= max_ps_thresh)
        df_trimmed = df_ps[keep_indices].copy()
        print(f"Trimming {len(df_ps) - len(df_trimmed)} units ({trim_threshold*100:.1f}% percentile trim)")
        if df_trimmed.empty:
            raise ValueError("All units removed after trimming. Try a smaller trim_threshold.")
        df_analysis = df_trimmed
        trim_info = {
            "trim_rule": "percentile",
            "n_before": n_before,
            "n_after": int(len(df_analysis)),
            "n_trimmed": n_before - int(len(df_analysis)),
            "dropped_fraction": (n_before - len(df_analysis)) / n_before if n_before else 0.0,
        }
    else:
        # Trim based on weight percentile (alternative approach)
        # q_low, q_high = np.percentile(weights, [trim_threshold*100, (1-trim_threshold)*100])
        # df_ps['ipw'] = np.clip(df_ps['ipw'], q_low, q_high)
        df_analysis = df_ps.copy()
        trim_threshold = 0 # Explicitly set for parameters output

    # --- Step 4: Normalize weights (optional but common) --- 
    # Normalize weights to sum to sample size within treated/control groups if ATT
    if weight_type.upper() == 'ATT':
        sum_weights_treated = df_analysis.loc[df_analysis[treatment] == 1, 'ipw'].sum()
        sum_weights_control = df_analysis.loc[df_analysis[treatment] == 0, 'ipw'].sum()
        n_treated = (df_analysis[treatment] == 1).sum()
        n_control = (df_analysis[treatment] == 0).sum()
        
        if sum_weights_treated > 0:
             df_analysis.loc[df_analysis[treatment] == 1, 'ipw'] *= n_treated / sum_weights_treated
        if sum_weights_control > 0:
             df_analysis.loc[df_analysis[treatment] == 0, 'ipw'] *= n_control / sum_weights_control 
    else: # ATE normalization
        df_analysis['ipw'] *= len(df_analysis) / df_analysis['ipw'].sum()

    # --- Step 5: Estimate weighted treatment effect --- 
    X_treat = sm.add_constant(df_analysis[[treatment]]) # Use only treatment variable for direct effect
    wls_model = sm.WLS(df_analysis[outcome], X_treat, weights=df_analysis['ipw'])
    results = wls_model.fit(cov_type='HC1' if robust_se else 'nonrobust')
    
    effect = results.params[treatment]
    effect_se = results.bse[treatment]

    # --- Step 6: Validate weight quality / Diagnostics --- 
    diagnostics = assess_weight_distribution(df_analysis['ipw'], df_analysis[treatment])
    # Could also add balance assessment on the weighted sample
    # weighted_diagnostics = assess_balance(df, df_analysis, treatment, covariates, method="PSW", weights=df_analysis['ipw'])
    # diagnostics.update(weighted_diagnostics)
    diagnostics["propensity_score_model"] = propensity_model_type
    diagnostics["trimming"] = dict(trim_info)
    # Advisory post-modeling check (surfaced in CAIS's output, never blocking):
    # did the weights actually balance the observed covariates?
    diagnostics["post_checks"] = _post_checks_after_weighting(
        df_analysis, treatment, outcome, covariates, df_analysis['ipw'].to_numpy()
    )

    # --- Step 7: Format and return results --- 
    return format_ps_results(effect, effect_se, diagnostics,
                           method_details="PS.Weighting",
                           parameters={"weight_type": weight_type, 
                                         "trim_threshold": trim_threshold,
                                         "trim_rule": trim_info["trim_rule"],
                                         "ps_bounds": trim_info.get("ps_bounds"),
                                         "n_before": trim_info["n_before"],
                                         "n_after": trim_info["n_after"],
                                         "n_trimmed": trim_info["n_trimmed"],
                                         "propensity_model": propensity_model_type,
                                         "robust_se": robust_se})


# _____________________________________________________________________________
# Trimmed IPW: the decision tree's remedy for a positivity/overlap failure
# _____________________________________________________________________________

def _finalize_trimmed_result(result: Dict[str, Any],
                             bounds: Sequence[float]) -> Dict[str, Any]:
    '''Normalize a trimmed-IPW result to the pipeline's standard contract.

    Adds the keys the agent/explainer layer expects (standard_error, p_value,
    method_used) and records the estimand that trimming implies.
    '''
    lo, hi = _validate_ps_bounds(bounds)
    effect = result.get("effect_estimate")
    effect_se = result.get("effect_se")
    result["standard_error"] = effect_se
    if effect is not None and effect_se is not None and float(effect_se) > 0:
        result["p_value"] = float(2.0 * stats.norm.sf(abs(float(effect) / float(effect_se))))
    else:
        result["p_value"] = None
    result["method_used"] = "trimmed_ipw"
    diagnostics = result.setdefault("diagnostics", {})
    diagnostics["estimand"] = (
        f"ATE among units with propensity score within [{lo:g}, {hi:g}] "
        "(overlap-satisfying subpopulation)"
    )
    result["method_details"] = "Trimmed IPW (propensity-score bounds trim)"
    return result


def estimate_effect_trimmed_ipw(df: pd.DataFrame, treatment: str, outcome: str,
                                covariates: List[str],
                                dataset_description: Optional[str] = None,
                                **kwargs) -> Dict[str, Any]:
    '''METHOD_MAPPING entry: IPW restricted to the positivity overlap band.

    Used by the method executor fallback path; ``ps_bounds`` may be supplied
    explicitly, otherwise the configured positivity bounds are used.
    '''
    bounds = kwargs.pop("ps_bounds", None)
    if bounds is None:
        bounds = default_trim_bounds()
    bounds = _validate_ps_bounds(bounds)
    result = estimate_effect(
        df=df,
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        ps_bounds=bounds,
        **kwargs
    )
    return _finalize_trimmed_result(result, bounds)


class TrimmedIPW(CausalMethod):
    '''Inverse probability weighting restricted to the positivity overlap band.

    Used by the decision tree when the positivity/overlap gate fails: units
    whose propensity score falls outside the configured bounds are dropped
    before weighting. The reported estimand is the ATE for the retained
    (overlap-satisfying) subpopulation, not the full-sample ATE.
    '''

    name = "Trimmed Inverse Probability Weighting"
    description = (
        "Inverse probability weighting restricted to the units whose propensity "
        "scores lie inside the overlap band. Units with extreme scores (near 0 or 1) "
        "are trimmed before weighting, which stabilizes the estimate when the "
        "positivity assumption fails."
    )
    assumptions = [
        "Positivity/overlap holds inside the retained propensity-score range",
        "No unmeasured confounders (conditional ignorability) given the covariates",
        "Correct specification of the propensity-score model",
    ]

    def __init__(self, ps_bounds: Optional[Sequence[float]] = None):
        super().__init__()
        self.ps_bounds = list(_validate_ps_bounds(ps_bounds)) if ps_bounds is not None else None

    def validate_assumptions(self, df: pd.DataFrame, variables) -> Dict[str, Any]:
        return {
            "assumptions_valid": True,
            "failed_assumptions": [],
            "warnings": [
                "Trimmed IPW drops units outside the overlap band; the estimate "
                "targets the overlap-satisfying subpopulation."
            ],
            "suggestions": [],
        }

    def estimate_effect(self, df: pd.DataFrame, variables, query: Optional[str] = None,
                        ps_bounds: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        bounds = ps_bounds if ps_bounds is not None else self.ps_bounds
        if bounds is None:
            bounds = default_trim_bounds()
        bounds = list(_validate_ps_bounds(bounds))

        covariates = list(getattr(variables, "confounders", None)
                          or getattr(variables, "covariates", None) or [])
        if not covariates:
            return {
                "error": "Trimmed IPW requires at least one covariate for the propensity model.",
                "effect_estimate": None,
                "method_used": "trimmed_ipw",
                "diagnostics": {"ps_bounds": bounds},
            }

        try:
            result = estimate_effect(
                df=df,
                treatment=variables.treatment_variable,
                outcome=variables.outcome_variable,
                covariates=covariates,
                ps_bounds=bounds,
                weight_type="ATE",
                query=query,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "error": str(exc),
                "effect_estimate": None,
                "method_used": "trimmed_ipw",
                "diagnostics": {"error": str(exc), "ps_bounds": bounds},
            }
        return _finalize_trimmed_result(result, bounds)
"""Frontdoor adjustment for causal effect estimation through a mediator.

Implements the frontdoor formula (Pearl) for a discrete treatment T and a
discrete mediator M:

    E[Y | do(T=t)] = sum_m P(M=m | T=t) * [ sum_t' E[Y | M=m, T=t'] * P(T=t') ]

The inner sum estimates E[Y | do(M=m)] using the treatment distribution to
remove unmeasured confounding of M and Y; the outer sum re-expresses the
outcome under each treatment level through the mediator distribution.

Identification requires (all argued, not tested, by the tree's advisories):
  (i)   M intercepts every directed path from T to Y (full mediation),
  (ii)  no unmeasured confounding of T and M,
  (iii) T blocks every backdoor path between M and Y.
The tree additionally gates on (treatment, mediator) cell support.

Only discrete treatment levels and a discrete, low-cardinality mediator are
supported. Continuous treatments/mediators need a different estimator, so they
are refused here; the tree's data-suitability check falls back to the backdoor
branch for such data.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from cais.methods.causal_method import CausalMethod

logger = logging.getLogger(__name__)

# Frontdoor needs finite level sets for both variables; beyond this the plug-in
# sums are not meaningful (and are better handled by a different estimator).
MAX_DISCRETE_LEVELS = 10

DEFAULT_N_BOOTSTRAPS = 500
DEFAULT_SEED = 0  # deterministic bootstrap by default, matching the repo's reproducibility goal
DEFAULT_MIN_CELL_COUNT = 5


class FrontdoorNotSupported(ValueError):
    """Raised when the data cannot support the frontdoor plug-in estimator."""


def _sorted_levels(series: pd.Series) -> List[Any]:
    """Deterministic ordering of a discrete column's levels."""
    uniques = [value for value in pd.unique(series.dropna())]
    try:
        return sorted(uniques)
    except TypeError:
        return sorted(uniques, key=lambda value: str(value))


def _prepare_data(df: pd.DataFrame, treatment: str, outcome: str, mediator: str,
                  min_cell_count: int) -> pd.DataFrame:
    """Validate columns/levels/support and return the complete-case frame."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise FrontdoorNotSupported("Frontdoor adjustment requires a pandas DataFrame.")
    missing = [column for column in (treatment, outcome, mediator) if column not in df.columns]
    if missing:
        raise FrontdoorNotSupported(
            f"Frontdoor adjustment is missing required column(s): {missing}."
        )

    data = df[[treatment, mediator, outcome]].dropna()
    if len(data) < 20:
        raise FrontdoorNotSupported(
            f"Frontdoor adjustment needs at least 20 complete observations; got {len(data)}."
        )
    if not pd.api.types.is_numeric_dtype(data[outcome]):
        raise FrontdoorNotSupported("Frontdoor adjustment requires a numeric outcome.")

    treatment_levels = _sorted_levels(data[treatment])
    mediator_levels = _sorted_levels(data[mediator])
    if len(treatment_levels) < 2:
        raise FrontdoorNotSupported("Frontdoor adjustment requires at least two treatment levels.")
    if len(treatment_levels) > MAX_DISCRETE_LEVELS:
        raise FrontdoorNotSupported(
            f"Frontdoor adjustment needs a discrete treatment; '{treatment}' has "
            f"{len(treatment_levels)} levels (max {MAX_DISCRETE_LEVELS})."
        )
    if len(mediator_levels) > MAX_DISCRETE_LEVELS:
        raise FrontdoorNotSupported(
            f"Frontdoor adjustment needs a discrete mediator; '{mediator}' has "
            f"{len(mediator_levels)} levels (max {MAX_DISCRETE_LEVELS})."
        )

    # (treatment, mediator) support: the frontdoor sums require every cell the
    # treatment distribution puts mass on.
    counts = data.groupby([treatment, mediator], observed=True).size()
    observed = set(counts.index)
    sparse = []
    for treatment_level in treatment_levels:
        for mediator_level in mediator_levels:
            n_cell = int(counts.get((treatment_level, mediator_level), 0))
            if n_cell < min_cell_count:
                sparse.append((treatment_level, mediator_level, n_cell))
    if sparse:
        raise FrontdoorNotSupported(
            "Frontdoor adjustment needs non-empty (treatment, mediator) cells with at "
            f"least {min_cell_count} observations; sparse cells: {sparse[:5]}"
            + (" ..." if len(sparse) > 5 else "")
            + "."
        )
    return data


def _adrf(data: pd.DataFrame, treatment: str, outcome: str, mediator: str,
          treatment_levels: Sequence[Any], mediator_levels: Sequence[Any]) -> Dict[Any, float]:
    """Plug-in estimate of E[Y | do(T=t)] for every treatment level.

    Cell means give E[Y | M=m, T=t']; the observed treatment distribution
    P(T=t') turns those into E[Y | do(M=m)]; the conditional mediator
    distribution P(M=m | T=t) then yields E[Y | do(T=t)].
    """
    p_treatment = data[treatment].value_counts(normalize=True)
    n_by_treatment = data[treatment].value_counts()

    cell_counts: Dict[tuple, int] = {}
    cell_means: Dict[tuple, float] = {}
    grouped = data.groupby([treatment, mediator], observed=True)[outcome]
    for (treatment_level, mediator_level), values in grouped:
        cell_counts[(treatment_level, mediator_level)] = int(len(values))
        cell_means[(treatment_level, mediator_level)] = float(values.mean())

    # E[Y | do(M=m)]: average the cell outcome over the treatment distribution.
    do_mediator: Dict[Any, float] = {}
    for mediator_level in mediator_levels:
        do_mediator[mediator_level] = sum(
            cell_means[(treatment_level, mediator_level)] * float(p_treatment.get(treatment_level, 0.0))
            for treatment_level in treatment_levels
        )

    # E[Y | do(T=t)]: push P(M=m | T=t) through E[Y | do(M=m)].
    adrf: Dict[Any, float] = {}
    for treatment_level in treatment_levels:
        n_t = int(n_by_treatment.get(treatment_level, 0))
        adrf[treatment_level] = sum(
            (cell_counts.get((treatment_level, mediator_level), 0) / n_t) * do_mediator[mediator_level]
            for mediator_level in mediator_levels
        )
    return adrf


def _summarize_effect(adrf: Dict[Any, float], treatment_levels: Sequence[Any],
                      treatment_reference_level: Optional[Any],
                      p_treatment: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Reduce the ADRF to the scalar effect the pipeline reports.

    * Two treatment levels: contrast of the non-reference level vs reference.
    * More than two levels: average deviation from the reference level across
      the observed treatment distribution (an ADRF summary; the full vector is
      reported in diagnostics).
    """
    levels = list(treatment_levels)
    reference = treatment_reference_level if treatment_reference_level in levels else levels[0]
    if treatment_reference_level is not None and treatment_reference_level not in levels:
        logger.warning(
            "Treatment reference level %r is not an observed level; using %r.",
            treatment_reference_level, reference,
        )
    contrasts = {
        str(level): float(adrf[level] - adrf[reference])
        for level in levels if level != reference
    }
    if len(levels) == 2:
        effect = next(iter(contrasts.values()))
        estimand = (
            f"ATE: E[Y | do(T={levels[1] if levels[0] == reference else levels[0]})] "
            f"- E[Y | do(T={reference})]"
        )
    else:
        if p_treatment is None:
            weights = {str(level): 1.0 for level in levels if level != reference}
        else:
            weights = {
                str(level): float(p_treatment.get(level, 0.0))
                for level in levels if level != reference
            }
        total_weight = sum(weights.values())
        weights = ({key: value / total_weight for key, value in weights.items()}
                   if total_weight > 0 else {key: 1.0 / len(weights) for key in weights})
        effect = float(sum(weights[key] * contrasts[key] for key in contrasts))
        estimand = (
            "Average deviation of E[Y | do(T=t)] from the reference level "
            f"T={reference}, weighted by the observed treatment distribution "
            "(multi-level ADRF summary)"
        )
    return {"effect": effect, "contrasts": contrasts, "reference_level": reference,
            "estimand": estimand}


def estimate_frontdoor_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    mediator: str,
    *,
    treatment_reference_level: Optional[Any] = None,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
) -> Dict[str, Any]:
    """Estimate the frontdoor-adjusted effect of ``treatment`` on ``outcome``.

    Args:
        df: Dataset with the treatment, mediator, and outcome columns.
        treatment: Discrete treatment column.
        outcome: Continuous outcome column.
        mediator: Discrete mediator column.
        treatment_reference_level: Level used as the reference for the reported
            contrast. Defaults to the lowest observed level.
        n_bootstraps: Bootstrap replications for the standard error/CI.
        seed: Seed for the (deterministic by default) row bootstrap.
        min_cell_count: Minimum observations per (treatment, mediator) cell.

    Returns:
        Standard method result dict (effect_estimate, standard_error,
        confidence_interval, p_value, diagnostics, ...).

    Raises:
        FrontdoorNotSupported: If the data cannot support the estimator.
    """
    data = _prepare_data(df, treatment, outcome, mediator, min_cell_count)
    treatment_levels = _sorted_levels(data[treatment])
    mediator_levels = _sorted_levels(data[mediator])
    p_treatment = data[treatment].value_counts(normalize=True)

    adrf = _adrf(data, treatment, outcome, mediator, treatment_levels, mediator_levels)
    summary = _summarize_effect(adrf, treatment_levels, treatment_reference_level, p_treatment)
    effect = summary["effect"]

    # Bootstrap rows; skip resamples whose support collapses (a sparse cell).
    rng = np.random.default_rng(seed)
    bootstrap_effects: List[float] = []
    n_bootstraps = int(n_bootstraps)
    for _ in range(n_bootstraps):
        sample = data.iloc[rng.integers(0, len(data), len(data))]
        try:
            sample_data = _prepare_data(sample, treatment, outcome, mediator, 1)
            sample_levels = _sorted_levels(sample_data[treatment])
            sample_mediators = _sorted_levels(sample_data[mediator])
            sample_adrf = _adrf(sample_data, treatment, outcome, mediator,
                                sample_levels, sample_mediators)
            sample_summary = _summarize_effect(
                sample_adrf, sample_levels, summary["reference_level"],
                sample_data[treatment].value_counts(normalize=True),
            )
            bootstrap_effects.append(sample_summary["effect"])
        except (FrontdoorNotSupported, KeyError, ZeroDivisionError):
            continue

    if len(bootstrap_effects) < max(20, n_bootstraps // 2):
        raise FrontdoorNotSupported(
            f"Frontdoor bootstrap failed for too many resamples "
            f"({len(bootstrap_effects)}/{n_bootstraps} succeeded); the mediator support is too sparse."
        )

    bootstrap = np.asarray(bootstrap_effects, dtype=float)
    standard_error = float(bootstrap.std(ddof=1)) if len(bootstrap) > 1 else None
    if standard_error is not None and np.isfinite(standard_error):
        ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
        share_below = float((bootstrap <= 0.0).mean())
        share_above = float((bootstrap >= 0.0).mean())
        p_value = float(min(1.0, 2.0 * min(share_below, share_above)))
        # Keep the p-value strictly positive when every draw agrees on the sign.
        p_value = max(p_value, 1.0 / (len(bootstrap) + 1.0))
        confidence_interval = [float(ci_low), float(ci_high)]
    else:
        standard_error = None
        confidence_interval = None
        p_value = None

    diagnostics = {
        "adrf": {str(level): float(value) for level, value in adrf.items()},
        "contrasts_vs_reference": summary["contrasts"],
        "reference_level": summary["reference_level"],
        "estimand": summary["estimand"],
        "treatment_levels": [str(level) for level in treatment_levels],
        "mediator_levels": [str(level) for level in mediator_levels],
        "n_observations": int(len(data)),
        "cell_counts": {
            f"{treatment_level}|{mediator_level}": int(
                data[(data[treatment] == treatment_level) & (data[mediator] == mediator_level)].shape[0]
            )
            for treatment_level in treatment_levels
            for mediator_level in mediator_levels
        },
        "n_bootstraps": n_bootstraps,
        "n_bootstrap_success": len(bootstrap_effects),
        "seed": int(seed),
        "frontdoor_assumptions": [
            "M intercepts every directed path from T to Y (full mediation; advisory)",
            "No unmeasured confounding of T and M (advisory)",
            "T blocks every backdoor path between M and Y (advisory)",
            "(treatment, mediator) support: positivity gate",
        ],
    }

    return {
        "effect_estimate": float(effect),
        "standard_error": standard_error,
        "confidence_interval": confidence_interval,
        "p_value": p_value,
        "method_used": "frontdoor_adjustment",
        "method_details": (
            "Frontdoor adjustment: discrete (treatment, mediator) plug-in formula "
            "with row bootstrap inference."
        ),
        "diagnostics": diagnostics,
        "parameters": {
            "treatment_variable": treatment,
            "outcome_variable": outcome,
            "mediator_variable": mediator,
            "treatment_levels": [str(level) for level in treatment_levels],
            "mediator_levels": [str(level) for level in mediator_levels],
            "reference_level": summary["reference_level"],
            "n_bootstraps": n_bootstraps,
            "seed": int(seed),
            "min_cell_count": int(min_cell_count),
        },
    }


def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str,
                    covariates: List[str], dataset_description: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
    """METHOD_MAPPING entry point used by the method executor."""
    mediator = kwargs.get("mediator_variable") or kwargs.get("mediator")
    if not mediator:
        raise FrontdoorNotSupported(
            "Frontdoor adjustment requires a mediator column; none was identified."
        )
    return estimate_frontdoor_effect(
        df,
        treatment=treatment,
        outcome=outcome,
        mediator=mediator,
        treatment_reference_level=kwargs.get("treatment_reference_level"),
        n_bootstraps=kwargs.get("n_bootstraps", DEFAULT_N_BOOTSTRAPS),
        seed=kwargs.get("seed", DEFAULT_SEED),
        min_cell_count=kwargs.get("min_cell_count", DEFAULT_MIN_CELL_COUNT),
    )


class FrontdoorAdjustment(CausalMethod):
    """CausalMethod wrapper: frontdoor adjustment through an identified mediator."""

    name = "Frontdoor Adjustment"
    description = (
        "Frontdoor adjustment identifies a causal effect by routing it through a "
        "mediator: M intercepts every path from T to Y, T does not confound M, and T "
        "blocks confounding between M and Y. Used when unmeasured confounding of T and "
        "Y is suspected but a clean mediator is available."
    )
    assumptions = [
        "Full mediation: the mediator intercepts all directed paths from treatment to outcome",
        "No unmeasured confounding between treatment and mediator",
        "Treatment blocks all backdoor paths between mediator and outcome",
        "Positivity/overlap of the (treatment, mediator) cells",
    ]

    def validate_assumptions(self, df: pd.DataFrame, variables) -> Dict[str, Any]:
        mediator = getattr(variables, "mediator_variable", None)
        return {
            "assumptions_valid": bool(mediator),
            "failed_assumptions": [] if mediator else ["No mediator identified"],
            "warnings": [] if mediator else [
                "Frontdoor adjustment requires a mediator column."
            ],
            "suggestions": [],
        }

    def estimate_effect(self, df: pd.DataFrame, variables, query: Optional[str] = None,
                        mediator_variable: Optional[str] = None,
                        treatment_reference_level: Optional[Any] = None,
                        n_bootstraps: Optional[int] = None,
                        seed: Optional[int] = None) -> Dict[str, Any]:
        mediator = mediator_variable or getattr(variables, "mediator_variable", None)
        reference_level = (treatment_reference_level if treatment_reference_level is not None
                           else getattr(variables, "treatment_reference_level", None))
        if not mediator:
            return {
                "error": "Frontdoor adjustment requires a mediator column; none was identified.",
                "effect_estimate": None,
                "method_used": "frontdoor_adjustment",
                "diagnostics": {"error": "missing mediator"},
            }
        try:
            result = estimate_frontdoor_effect(
                df,
                treatment=variables.treatment_variable,
                outcome=variables.outcome_variable,
                mediator=mediator,
                treatment_reference_level=reference_level,
                n_bootstraps=n_bootstraps if n_bootstraps is not None else DEFAULT_N_BOOTSTRAPS,
                seed=seed if seed is not None else DEFAULT_SEED,
            )
        except (FrontdoorNotSupported, KeyError, TypeError) as exc:
            logger.warning("Frontdoor adjustment failed: %s", exc)
            return {
                "error": str(exc),
                "effect_estimate": None,
                "method_used": "frontdoor_adjustment",
                "diagnostics": {"error": str(exc)},
            }
        return result

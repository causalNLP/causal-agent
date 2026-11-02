"""McCrary / Cattaneo-Jansson-Ma density discontinuity test for RDD.

Standalone diagnostic: it does not select a method and is not wired into the
decision tree yet. It is intentionally separate from ``rdd_no_manipulation`` so
the existing gate keeps its current behaviour until the tree is reworked.

The test asks whether the density of the running variable is continuous at the
cutoff. A rejection is evidence of a density discontinuity (e.g. bunching from
sorting), not proof of manipulation; a non-rejection is not proof of absence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cais.models import AssumptionResult

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_OBS_PER_SIDE = 20
DEFAULT_ORDER = 2
SENSITIVITY_BWSELECT = ("comb", "each", "diff")


def _finite_float(value: Any) -> Optional[float]:
    """Best-effort float conversion that returns None for NaN/inf/failures."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _series_value(container: Any, key: str) -> Optional[float]:
    """Read a value from a pandas Series, dict, or array-like result field."""
    if container is None:
        return None
    try:
        value = container[key]
    except (KeyError, IndexError, TypeError):
        try:
            value = getattr(container, key)
        except AttributeError:
            return None
    return _finite_float(value)


def _run_once(
    x: np.ndarray,
    cutoff: float,
    bwselect: str,
    p_order: int,
    mass_points: bool,
) -> Optional[Dict[str, Any]]:
    """Run one rddensity test; return its diagnostics or None if untestable."""
    try:
        import rddensity
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        logger.warning("rddensity is not installed: %s", exc)
        return None

    try:
        result = rddensity.rddensity(
            x,
            c=cutoff,
            p=p_order,
            fitselect="unrestricted",
            kernel="triangular",
            vce="jackknife",
            massPoints=mass_points,
            bwselect=bwselect,
        )
    except Exception as exc:
        logger.warning("rddensity failed for bwselect=%s: %s", bwselect, exc)
        return None

    p_jackknife = _series_value(getattr(result, "test", None), "p_jk")
    p_asymptotic = _series_value(getattr(result, "test", None), "p_asy")
    p_value = p_jackknife if p_jackknife is not None else p_asymptotic
    jump = _series_value(getattr(result, "hat", None), "diff")
    bandwidth_left = _series_value(getattr(result, "h", None), "left")
    bandwidth_right = _series_value(getattr(result, "h", None), "right")
    n_left = _series_value(getattr(result, "n", None), "left")
    n_right = _series_value(getattr(result, "n", None), "right")

    if p_value is None:
        return None

    return {
        "p_value": p_value,
        "p_value_type": "jackknife" if p_jackknife is not None else "asymptotic",
        "jump": jump,
        "bandwidth_left": bandwidth_left,
        "bandwidth_right": bandwidth_right,
        "n_left": int(n_left) if n_left is not None else None,
        "n_right": int(n_right) if n_right is not None else None,
        "bwselect": bwselect,
        "mass_points": bool(mass_points),
        "p_order": int(p_order),
    }


def mccrary_density_test(
    df: pd.DataFrame,
    running_variable: str,
    cutoff_value: float,
    alpha: float = DEFAULT_ALPHA,
    bwselect: str = "comb",
    sensitivity: bool = False,
    p_order: int = DEFAULT_ORDER,
    mass_points: bool = True,
    min_obs_per_side: int = DEFAULT_MIN_OBS_PER_SIDE,
) -> AssumptionResult:
    """Test whether the running-variable density is continuous at the cutoff.

    Args:
        df: Dataframe containing the running variable.
        running_variable: Numeric column used for RDD assignment.
        cutoff_value: Threshold in the running variable.
        alpha: Significance level for the density-continuity verdict.
        bwselect: ``rddensity`` bandwidth selector (``comb``, ``each``, ``diff``,
            ``sum``).
        sensitivity: Also run ``each`` and ``diff`` bandwidth selectors and
            report them in ``details`` (the verdict still uses ``bwselect``).
        p_order: Local polynomial order for the density estimator.
        mass_points: Adjust for mass points / discrete running variables.
        min_obs_per_side: Minimum observations required on each side. Below this
            the test returns "can't tell" instead of failing.

    Returns:
        ``AssumptionResult`` with ``passed=True`` if no density discontinuity is
        detected, ``False`` if a discontinuity is detected, and ``None`` when
        the test cannot be run. ``details`` carries the jump, p-value, selected
        bandwidths, sample sizes, and optional sensitivity results.
    """
    if df is None or not hasattr(df, "columns"):
        return AssumptionResult(
            passed=None,
            reasoning="McCrary density test cannot be run: no dataframe was provided.",
        )

    if not running_variable or running_variable not in df.columns:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "McCrary density test cannot be run: running variable "
                f"{running_variable!r} is not a column in the dataset."
            ),
        )

    if cutoff_value is None:
        return AssumptionResult(
            passed=None,
            reasoning="McCrary density test cannot be run: cutoff value is missing.",
        )

    try:
        cutoff = float(cutoff_value)
    except (TypeError, ValueError):
        return AssumptionResult(
            passed=None,
            reasoning=(
                "McCrary density test cannot be run: cutoff value "
                f"{cutoff_value!r} is not numeric."
            ),
        )
    if not np.isfinite(cutoff):
        return AssumptionResult(
            passed=None,
            reasoning="McCrary density test cannot be run: cutoff value is not finite.",
        )

    raw = df[running_variable]
    numeric = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = numeric.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "McCrary density test cannot be run: the running variable has no "
                "numeric observations."
            ),
        )

    n_left = int((x < cutoff).sum())
    n_right = int((x >= cutoff).sum())

    if not (x.min() < cutoff < x.max()):
        return AssumptionResult(
            passed=None,
            reasoning=(
                "McCrary density test cannot be run: the cutoff is outside the "
                "observed running-variable support."
            ),
            details={
                "cutoff": cutoff,
                "running_min": float(x.min()),
                "running_max": float(x.max()),
            },
        )

    if n_left < min_obs_per_side or n_right < min_obs_per_side:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "McCrary density test cannot be run: too few observations near the "
                f"cutoff (left={n_left}, right={n_right}; need at least "
                f"{min_obs_per_side} per side)."
            ),
            details={
                "n_left": n_left,
                "n_right": n_right,
                "cutoff": cutoff,
                "min_obs_per_side": min_obs_per_side,
            },
        )

    primary = _run_once(x, cutoff, bwselect, p_order, mass_points)
    if primary is None:
        return AssumptionResult(
            passed=None,
            reasoning=(
                "McCrary density test could not produce a finite p-value "
                "(possibly too little local variation or a discrete running variable)."
            ),
            details={
                "cutoff": cutoff,
                "n_left": n_left,
                "n_right": n_right,
                "bwselect": bwselect,
            },
        )

    p_value = float(primary["p_value"])
    jump = primary["jump"]
    passed = bool(p_value >= alpha)

    if passed:
        reasoning = (
            "No evidence of a density discontinuity at the cutoff "
            f"(density jump={jump:.4f}, p={p_value:.4f} >= {alpha:g}). "
            "This does not prove that manipulation did not occur."
            if jump is not None
            else
            "No evidence of a density discontinuity at the cutoff "
            f"(p={p_value:.4f} >= {alpha:g}). This does not prove that "
            "manipulation did not occur."
        )
    else:
        reasoning = (
            "Density discontinuity at the cutoff is statistically significant "
            f"(density jump={jump:.4f}, p={p_value:.4f} < {alpha:g}), consistent "
            "with sorting/bunching. This is evidence, not direct proof, of "
            "manipulation."
            if jump is not None
            else
            "Density discontinuity at the cutoff is statistically significant "
            f"(p={p_value:.4f} < {alpha:g}), consistent with sorting/bunching."
        )

    details: Dict[str, Any] = {
        "test_type": "rddensity_cjm",
        "running_variable": running_variable,
        "cutoff": cutoff,
        "alpha": alpha,
        "jump": jump,
        "p_value": p_value,
        "p_value_type": primary["p_value_type"],
        "bandwidth_left": primary["bandwidth_left"],
        "bandwidth_right": primary["bandwidth_right"],
        "n_left": primary["n_left"],
        "n_right": primary["n_right"],
        "bwselect": primary["bwselect"],
        "mass_points": primary["mass_points"],
        "p_order": primary["p_order"],
    }

    if sensitivity:
        variants: Dict[str, Any] = {}
        for variant in SENSITIVITY_BWSELECT:
            if variant == bwselect:
                variants[variant] = {
                    "p_value": p_value,
                    "jump": jump,
                    "bwselect": variant,
                }
                continue
            run = _run_once(x, cutoff, variant, p_order, mass_points)
            if run is None:
                variants[variant] = {"status": "could_not_test", "bwselect": variant}
            else:
                variants[variant] = {
                    "p_value": run["p_value"],
                    "jump": run["jump"],
                    "bwselect": variant,
                    "bandwidth_left": run["bandwidth_left"],
                    "bandwidth_right": run["bandwidth_right"],
                }
        details["sensitivity"] = variants

    return AssumptionResult(passed=passed, reasoning=reasoning, details=details)


class McCraryDensityTest:
    """Small diagnostic wrapper around :func:`mccrary_density_test`.

    This is a density-continuity diagnostic, not a causal estimator, so it does
    not subclass ``CausalMethod`` and is not registered with the estimator
    library or the decision tree.
    """

    name = "McCrary Density Test"
    description = (
        "Tests whether the density of the running variable is continuous at the "
        "RDD cutoff. A discontinuity is evidence of sorting/bunching, not proof "
        "of manipulation, and continuity is not proof of its absence."
    )
    assumptions = [
        "Running variable is numeric and has enough observations on both sides of the cutoff",
        "The density is estimated locally around the cutoff; bandwidth and mass points matter",
    ]

    def test(self, df: pd.DataFrame, running_variable: str, cutoff_value: float, **kwargs) -> AssumptionResult:
        return mccrary_density_test(df, running_variable, cutoff_value, **kwargs)

    def __call__(self, df: pd.DataFrame, running_variable: str, cutoff_value: float, **kwargs) -> AssumptionResult:
        return self.test(df, running_variable, cutoff_value, **kwargs)

"""Observable statistical reliability diagnostics for conditional routing."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


def effective_sample_size(weights: Any) -> Optional[float]:
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0 or np.sum(values ** 2) == 0:
        return None
    return float(np.sum(values) ** 2 / np.sum(values ** 2))


def assess_binary_treatment_reliability(
    df: pd.DataFrame,
    treatment: str,
    propensity: Optional[Any] = None,
    required_columns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return diagnostics without interpreting them as causal validity.

    The score is intentionally descriptive. It can break ties in the routing
    lexicographic rule but cannot prove exchangeability, exclusion or any
    other substantive identification assumption.
    """
    result: dict[str, Any] = {"status": "insufficient_evidence", "warnings": []}
    if treatment not in df.columns:
        result["warnings"].append(f"Treatment column '{treatment}' is missing.")
        return result
    observed = df[treatment].dropna()
    levels = set(observed.unique())
    if not levels.issubset({0, 1, False, True}) or len(levels) < 2:
        result["warnings"].append("Binary treatment support is unavailable.")
        return result
    counts = observed.astype(int).value_counts()
    result["n"] = int(len(observed))
    result["treated_n"] = int(counts.get(1, 0))
    result["control_n"] = int(counts.get(0, 0))
    result["treatment_prevalence"] = float(observed.astype(int).mean())
    result["missing_required_share"] = float(df[required_columns].isna().any(axis=1).mean()) if required_columns else 0.0
    if propensity is not None:
        ps = np.asarray(propensity, dtype=float)
        ps = ps[np.isfinite(ps)]
        if ps.size:
            result["propensity_min"] = float(np.min(ps))
            result["propensity_max"] = float(np.max(ps))
            result["propensity_overlap_share"] = float(np.mean((ps >= 0.05) & (ps <= 0.95)))
            weights = 1.0 / np.where(observed.astype(int).to_numpy() == 1, ps[:len(observed)], 1 - ps[:len(observed)])
            result["effective_sample_size"] = effective_sample_size(weights)
    result["status"] = "supported"
    # A descriptive tie-breaking score, not a probability of causal validity.
    overlap = result.get("propensity_overlap_share", 1.0)
    missing = result.get("missing_required_share", 0.0)
    result["reliability_score"] = float(max(0.0, min(1.0, overlap * (1.0 - missing))))
    return result


__all__ = ["effective_sample_size", "assess_binary_treatment_reliability"]

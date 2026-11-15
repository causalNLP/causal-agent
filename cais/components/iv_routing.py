"""Fail-closed routing for IV LATE versus homogeneity-based ATE claims."""

from typing import Any, Dict, Optional

from cais.models import AssumptionResult


def route_iv_estimand(
    target_estimand: str,
    homogeneity_assessment: Optional[AssumptionResult] = None,
) -> Dict[str, Any]:
    """Choose an IV interpretation without changing the user's target.

    The 2SLS implementation is executable in both cases.  The routing decision
    controls only which causal interpretation may be attached to its output.
    An inconclusive or negative advisory never blocks computation, but it does
    prevent silently reporting a population ATE.
    """
    target = str(target_estimand or "").upper()
    if target not in {"E01", "ATE", "LATE", "E07"}:
        return {
            "route_id": "R_S16_2SLS",
            "estimator_id": "instrumental_variable",
            "identified_estimand": target_estimand,
            "execution_allowed": True,
            "causal_interpretation_allowed": False,
            "status": "unsupported_target",
            "warning": "The requested estimand is not an IV route target; no estimand was relabelled.",
        }

    if target in {"LATE", "E07"}:
        return {
            "route_id": "R_S16_2SLS",
            "estimator_id": "instrumental_variable",
            "identified_estimand": "E07",
            "execution_allowed": True,
            "causal_interpretation_allowed": True,
            "status": "late_route",
            "warning": "Interpret as a local IV/LATE effect for compliers, conditional on IV assumptions.",
        }

    assessment = homogeneity_assessment
    if assessment is not None and assessment.passed is True:
        return {
            "route_id": "R_S18_2SLS_with_homogeneity",
            "estimator_id": "instrumental_variable",
            "identified_estimand": "E01",
            "execution_allowed": True,
            "causal_interpretation_allowed": True,
            "status": "ate_route_conditionally_supported",
            "warning": "ATE interpretation is conditional on the advisory homogeneity argument and all standard IV assumptions.",
        }

    status = "homogeneity_evidence_against" if assessment is not None and assessment.passed is False else "homogeneity_insufficient_evidence"
    return {
        "route_id": "R_S18_2SLS_with_homogeneity",
        "fallback_route_id": "R_S16_2SLS",
        "estimator_id": "instrumental_variable",
        "identified_estimand": "E01",
        "execution_allowed": True,
        "causal_interpretation_allowed": False,
        "status": status,
        "warning": "2SLS may be computed, but the result must not be reported as a population ATE without homogeneity justification.",
    }

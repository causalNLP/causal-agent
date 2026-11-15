from cais.components.iv_routing import route_iv_estimand
from cais.models import AssumptionResult


def test_homogeneity_is_advisory_and_does_not_block_execution():
    result = route_iv_estimand("E01", AssumptionResult(passed=None, reasoning="insufficient information"))

    assert result["execution_allowed"] is True
    assert result["causal_interpretation_allowed"] is False
    assert result["fallback_route_id"] == "R_S16_2SLS"


def test_positive_homogeneity_advisory_allows_conditional_ate_route():
    result = route_iv_estimand("ATE", AssumptionResult(passed=True, reasoning="homogeneity argued by design"))

    assert result["route_id"] == "R_S18_2SLS_with_homogeneity"
    assert result["causal_interpretation_allowed"] is True


def test_late_route_does_not_require_homogeneity():
    result = route_iv_estimand("E07", AssumptionResult(passed=False, reasoning="not supported"))

    assert result["route_id"] == "R_S16_2SLS"
    assert result["identified_estimand"] == "E07"
    assert result["causal_interpretation_allowed"] is True

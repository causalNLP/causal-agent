from cais.components.conditional_preference_graph import ConditionalPreferenceGraph, PreferenceEdge
from cais.components.conditional_strategy_router import (
    ConditionalStrategyRouter,
    EstimatorRoute,
)


def test_router_filters_only_blocking_candidates_and_keeps_advisory_unknown():
    graph = ConditionalPreferenceGraph([
        PreferenceEdge(
            edge_id="iv-late",
            preferred_strategy_id="S16",
            fallback_strategy_id="S7",
            estimand_ids=frozenset({"E07"}),
            conditions=({"field": "instrument_relevance", "equals": "supported"},),
        )
    ])
    router = ConditionalStrategyRouter(graph, [
        EstimatorRoute("S16", "instrumental_variable", frozenset({"E07"}), True, False),
        EstimatorRoute("S7", "backdoor_adjustment", frozenset({"E07"}), True, False),
    ])
    result = router.route(
        [
            {"strategy_id": "S16", "assumption_statuses": {"iv_exclusion": "insufficient_evidence"}},
            {"strategy_id": "S7", "structural_status": "eligible"},
            {"strategy_id": "S29", "structural_status": "confirmed_violation"},
        ],
        estimand_id="E07",
        evidence={"instrument_relevance": "supported"},
    )
    assert result.excluded_candidates == {"S29": "confirmed_violation"}
    assert result.primary_strategy_id == "S16"
    assert result.primary_estimator_id == "instrumental_variable"
    assert result.interpretation_status == "causal_claim_requires_unverified_assumptions"
    assert result.execution_status == "available_but_not_certified"


def test_router_retains_multiple_maximal_candidates_when_preference_is_unknown():
    graph = ConditionalPreferenceGraph([
        PreferenceEdge(
            edge_id="iv-late",
            preferred_strategy_id="S16",
            fallback_strategy_id="S7",
            estimand_ids=frozenset({"E07"}),
            conditions=({"field": "iv_exclusion", "equals": "supported"},),
        )
    ])
    router = ConditionalStrategyRouter(graph, [
        EstimatorRoute("S16", "instrumental_variable", frozenset({"E07"}), True),
        EstimatorRoute("S7", "backdoor_adjustment", frozenset({"E07"}), True),
    ])
    result = router.route(
        [{"strategy_id": "S16"}, {"strategy_id": "S7"}],
        estimand_id="E07",
        evidence={"iv_exclusion": "insufficient_evidence"},
    )
    assert result.primary_strategy_id is None
    assert {item["strategy_id"] for item in result.alternatives} == {"S16", "S7"}
    assert result.graph_decision.status == "indeterminate"

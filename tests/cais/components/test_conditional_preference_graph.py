import json

from cais.components.conditional_preference_graph import (
    ConditionalPreferenceGraph,
    PreferenceEdge,
)


def edge(edge_id, preferred, fallback, estimand, conditions, unknown="inactive", edge_type="preference"):
    return PreferenceEdge.from_row({
        "Edge_ID": edge_id,
        "Preferred_Strategy_ID": preferred,
        "Fallback_Strategy_ID": fallback,
        "Estimand_IDs": estimand,
        "Conditions_JSON": json.dumps({"all": conditions}),
        "Unknown_Policy": unknown,
        "Edge_Type": edge_type,
        "Rationale": "test",
    })


def test_graph_returns_unique_maximal_candidate_when_edge_is_supported():
    graph = ConditionalPreferenceGraph([
        edge("p1", "S16", "S7", "E07", [
            {"field": "instrument_relevance", "equals": "supported"},
            {"field": "iv_exclusion", "equals": "supported"},
        ])
    ])
    result = graph.decide(
        [{"strategy_id": "S16"}, {"strategy_id": "S7"}],
        estimand_id="E07",
        context={"instrument_relevance": "supported", "iv_exclusion": "supported"},
    )
    assert result.primary_strategy_id == "S16"
    assert result.maximal_strategy_ids == ["S16"]
    assert result.status == "preferred"


def test_unknown_advisory_does_not_activate_preference():
    graph = ConditionalPreferenceGraph([
        edge("p1", "S16", "S7", "E07", [{"field": "iv_exclusion", "equals": "supported"}])
    ])
    result = graph.decide(
        [{"strategy_id": "S16"}, {"strategy_id": "S7"}],
        estimand_id="E07",
        context={"iv_exclusion": "insufficient_evidence"},
    )
    assert result.primary_strategy_id is None
    assert result.maximal_strategy_ids == ["S16", "S7"]
    assert result.status == "indeterminate"
    assert result.inactive_edges["p1"] == "condition_unknown:inactive"


def test_first_stage_alone_does_not_activate_iv_edge():
    graph = ConditionalPreferenceGraph([
        edge("iv", "S16", "S7", "E07", [
            {"field": "instrument_relevance", "equals": "supported"},
            {"field": "iv_exclusion", "equals": "supported"},
        ])
    ])
    result = graph.decide(
        [{"strategy_id": "S16"}, {"strategy_id": "S7"}],
        estimand_id="E07",
        context={"instrument_relevance": "supported"},
    )
    assert result.status == "indeterminate"
    assert result.primary_strategy_id is None


def test_non_applicable_edge_is_not_used_for_other_estimand():
    graph = ConditionalPreferenceGraph([
        edge("late", "S16", "S7", "E07", [{"field": "iv_exclusion", "equals": "supported"}])
    ])
    result = graph.decide(
        [{"strategy_id": "S16"}, {"strategy_id": "S7"}],
        estimand_id="E01",
        context={"iv_exclusion": "supported"},
    )
    assert result.status == "indeterminate"
    assert result.inactive_edges["late"] == "estimand_not_applicable"


def test_override_edges_are_reported_separately():
    graph = ConditionalPreferenceGraph([
        edge("override", "S16", "S7", "E01", [{"field": "iv_homogeneity", "equals": "supported"}], edge_type="override")
    ])
    result = graph.decide(
        [{"strategy_id": "S16"}, {"strategy_id": "S7"}],
        estimand_id="E01",
        context={"iv_homogeneity": "supported"},
    )
    assert result.primary_strategy_id == "S16"
    assert result.active_override_edges == ["override"]


def test_lexicographic_burden_prefers_lower_unresolved_assumption_burden():
    graph = ConditionalPreferenceGraph()
    result = graph.decide(
        [
            {"strategy_id": "S16", "u_k": 2.0, "statistical_reliability": 0.9},
            {"strategy_id": "S7", "u_k": 1.0, "statistical_reliability": 0.8},
        ],
        estimand_id="E01",
    )
    assert result.primary_strategy_id == "S7"
    assert result.maximal_strategy_ids == ["S7"]

"""Registry-driven conditional preference graph for identification strategies.

The graph deliberately does not assign a total ranking to methods.  A directed
edge is activated only when its registry conditions are satisfied for the
requested estimand and the available evidence.  The output is therefore a set
of maximal candidates, which may contain more than one strategy when the
available evidence does not justify a preference.

Registry edge rows use the following columns::

    Edge_ID, Preferred_Strategy_ID, Fallback_Strategy_ID, Estimand_IDs,
    Conditions_JSON, Unknown_Policy, Edge_Type, Rationale

``Conditions_JSON`` is an ``all`` list of predicates.  A predicate can contain
``field`` and one of ``equals``, ``in``, ``gte``, ``lte`` or ``truthy``.  The
candidate fields are merged with the run context before evaluation.  Missing
fields are ``unknown`` and never activate an edge unless ``Unknown_Policy`` is
``activate`` (which is rejected by default for safety).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


UNKNOWN = object()


@dataclass(frozen=True)
class PreferenceEdge:
    edge_id: str
    preferred_strategy_id: str
    fallback_strategy_id: str
    estimand_ids: frozenset[str] = frozenset()
    conditions: tuple[Mapping[str, Any], ...] = ()
    unknown_policy: str = "inactive"
    edge_type: str = "preference"
    rationale: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PreferenceEdge":
        raw_estimands = str(row.get("Estimand_IDs") or "").strip()
        estimands = frozenset(x.strip() for x in raw_estimands.replace(";", ",").split(",") if x.strip())
        raw_conditions = row.get("Conditions_JSON") or "{}"
        parsed = json.loads(raw_conditions) if isinstance(raw_conditions, str) else raw_conditions
        if isinstance(parsed, Mapping):
            parsed = parsed.get("all", [])
        if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
            raise ValueError(f"{row.get('Edge_ID')}: Conditions_JSON must contain an all-list")
        unknown_policy = str(row.get("Unknown_Policy") or "inactive").strip().lower()
        if unknown_policy not in {"inactive", "retain", "activate"}:
            raise ValueError(f"{row.get('Edge_ID')}: unknown policy {unknown_policy!r}")
        return cls(
            edge_id=str(row.get("Edge_ID") or "").strip(),
            preferred_strategy_id=str(row.get("Preferred_Strategy_ID") or "").strip(),
            fallback_strategy_id=str(row.get("Fallback_Strategy_ID") or "").strip(),
            estimand_ids=estimands,
            conditions=tuple(dict(p) for p in parsed),
            unknown_policy=unknown_policy,
            edge_type=str(row.get("Edge_Type") or "preference").strip().lower(),
            rationale=str(row.get("Rationale") or "").strip(),
        )

    def applies_to_estimand(self, estimand_id: Optional[str]) -> bool:
        return not self.estimand_ids or str(estimand_id or "") in self.estimand_ids


@dataclass
class GraphDecision:
    primary_strategy_id: Optional[str]
    maximal_strategy_ids: list[str]
    dominated_strategy_ids: list[str]
    active_edges: list[str] = field(default_factory=list)
    active_override_edges: list[str] = field(default_factory=list)
    inactive_edges: dict[str, str] = field(default_factory=dict)
    status: str = "indeterminate"
    warnings: list[str] = field(default_factory=list)


def _lookup(mapping: Mapping[str, Any], field_name: str) -> Any:
    value: Any = mapping
    for part in field_name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return UNKNOWN
        value = value[part]
    return value


def _predicate_holds(predicate: Mapping[str, Any], values: Mapping[str, Any]) -> Optional[bool]:
    field_name = predicate.get("field")
    if not field_name:
        raise ValueError("Preference predicate requires a field")
    value = _lookup(values, str(field_name))
    if value is UNKNOWN or value is None or str(value).lower() in {
        "unknown", "insufficient_evidence", "unverified", "not_assessed",
    }:
        return None
    if "equals" in predicate:
        expected = predicate["equals"]
        return value == expected or str(value).lower() == str(expected).lower()
    if "in" in predicate:
        return value in predicate["in"]
    if "gte" in predicate:
        try:
            return float(value) >= float(predicate["gte"])
        except (TypeError, ValueError):
            return False
    if "lte" in predicate:
        try:
            return float(value) <= float(predicate["lte"])
        except (TypeError, ValueError):
            return False
    if "truthy" in predicate:
        return bool(value) is bool(predicate["truthy"])
    raise ValueError(f"Unsupported preference predicate: {predicate}")


class ConditionalPreferenceGraph:
    """Evaluate registry edges and return the maximal eligible strategies."""

    def __init__(self, edges: Iterable[PreferenceEdge] = ()):
        self.edges = tuple(edges)

    @classmethod
    def from_registry(cls, workbook_path: str | Path, sheet_name: str = "Preference_Graph"):
        """Load graph edges from an Excel registry.

        A missing sheet is treated as an empty graph rather than silently
        activating hard-coded preferences.  This keeps routing fail-closed
        until the registry has been explicitly populated and reviewed.
        """
        import openpyxl

        wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
        if sheet_name not in wb.sheetnames:
            return cls()
        ws = wb[sheet_name]
        headers = [cell.value for cell in next(ws.iter_rows(max_row=1))]
        return cls(PreferenceEdge.from_row(dict(zip(headers, row))) for row in ws.iter_rows(min_row=2, values_only=True))

    def decide(
        self,
        candidates: Iterable[Mapping[str, Any]],
        estimand_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> GraphDecision:
        candidate_map = {str(c["strategy_id"]): dict(c) for c in candidates if c.get("strategy_id")}
        if not candidate_map:
            return GraphDecision(None, [], [], status="no_candidates")
        values = dict(context or {})
        active: list[PreferenceEdge] = []
        inactive: dict[str, str] = {}
        for edge in self.edges:
            if not edge.applies_to_estimand(estimand_id):
                inactive[edge.edge_id] = "estimand_not_applicable"
                continue
            if edge.preferred_strategy_id not in candidate_map or edge.fallback_strategy_id not in candidate_map:
                inactive[edge.edge_id] = "candidate_missing"
                continue
            merged = dict(values)
            merged["preferred"] = candidate_map[edge.preferred_strategy_id]
            merged["fallback"] = candidate_map[edge.fallback_strategy_id]
            outcomes = [_predicate_holds(predicate, merged) for predicate in edge.conditions]
            if any(result is False for result in outcomes):
                inactive[edge.edge_id] = "condition_false"
            elif any(result is None for result in outcomes):
                inactive[edge.edge_id] = f"condition_unknown:{edge.unknown_policy}"
                if edge.unknown_policy == "activate":
                    active.append(edge)
            else:
                active.append(edge)

        dominated: set[str] = set()
        active_ids: list[str] = []
        active_override_ids: list[str] = []
        for edge in active:
            dominated.add(edge.fallback_strategy_id)
            active_ids.append(edge.edge_id)
            if edge.edge_type == "override":
                active_override_ids.append(edge.edge_id)
        maximal = sorted(set(candidate_map) - dominated)
        if len(maximal) == 1:
            status = "preferred" if active_ids else "single_candidate"
            primary = maximal[0]
        elif maximal:
            status = "indeterminate"
            primary = None
        else:
            status = "cycle_or_conflicting_preferences"
            primary = None
            maximal = sorted(candidate_map)
        warnings = []
        if len(maximal) > 1:
            warnings.append("No registry edge establishes a unique preference among the maximal candidates.")
        return GraphDecision(primary, maximal, sorted(dominated), active_ids, active_override_ids, inactive, status, warnings)


__all__ = ["PreferenceEdge", "GraphDecision", "ConditionalPreferenceGraph"]

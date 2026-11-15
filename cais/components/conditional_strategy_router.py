"""Conditional strategy-to-estimator routing.

This component is the registry-facing boundary between identification and
execution.  It removes only structural incompatibilities or explicitly
blocking violations.  Non-testable assumptions marked as
``insufficient_evidence`` remain attached to the candidate and do not remove
it from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from cais.components.conditional_preference_graph import (
    ConditionalPreferenceGraph,
    GraphDecision,
)


BLOCKING_STATUSES = frozenset({
    "incompatible",
    "structural_incompatibility",
    "confirmed_violation",
    "blocking_violation",
})


@dataclass(frozen=True)
class EstimatorRoute:
    strategy_id: str
    estimator_id: str
    estimand_ids: frozenset[str] = frozenset()
    implemented: bool = False
    safe_for_auto_routing: bool = False


@dataclass
class ConditionalRoutingResult:
    estimand_id: Optional[str]
    graph_decision: GraphDecision
    primary_strategy_id: Optional[str]
    primary_estimator_id: Optional[str]
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    identification_status: str = "conditional"
    execution_status: str = "not_routed"
    interpretation_status: str = "causal_claim_requires_unverified_assumptions"
    excluded_candidates: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ConditionalStrategyRouter:
    """Route eligible strategy candidates using registry graph edges."""

    def __init__(
        self,
        graph: ConditionalPreferenceGraph,
        routes: Iterable[EstimatorRoute] = (),
    ):
        self.graph = graph
        self.routes = tuple(routes)

    @classmethod
    def from_registry(cls, workbook_path: str | Path):
        import openpyxl

        graph = ConditionalPreferenceGraph.from_registry(workbook_path)
        wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
        routes: list[EstimatorRoute] = []
        ws = wb["Estimator_Routes"]
        headers = [cell.value for cell in next(ws.iter_rows(max_row=1))]
        for values in ws.iter_rows(min_row=2, values_only=True):
            row = dict(zip(headers, values))
            estimator = str(row.get("Estimator_ID") or "").strip()
            strategy = str(row.get("Strategy_ID") or "").strip()
            if not estimator or not strategy:
                continue
            supported = frozenset(
                part.strip().replace(" (conditional)", "")
                for part in str(row.get("Supported_Estimand_IDs") or "").replace(";", ",").split(",")
                if part.strip() and not part.strip().lower().startswith("not")
            )
            implemented = str(row.get("Implemented_in_CAIS") or "").lower().startswith("yes")
            safe = str(row.get("Safe_For_Auto_Routing") or "").lower() in {"yes", "true"}
            routes.append(EstimatorRoute(strategy, estimator, supported, implemented, safe))
        return cls(graph, routes)

    def _route_for(self, strategy_id: Optional[str], estimand_id: Optional[str]) -> Optional[EstimatorRoute]:
        if not strategy_id:
            return None
        matches = [r for r in self.routes if r.strategy_id == strategy_id]
        matches = [r for r in matches if not r.estimand_ids or estimand_id in r.estimand_ids]
        return next((r for r in matches if r.implemented), None) or (matches[0] if matches else None)

    def route(
        self,
        candidates: Iterable[Mapping[str, Any]],
        estimand_id: Optional[str],
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> ConditionalRoutingResult:
        eligible: list[Mapping[str, Any]] = []
        excluded: dict[str, str] = {}
        for candidate in candidates:
            candidate = dict(candidate)
            strategy_id = str(candidate.get("strategy_id") or "")
            status = str(candidate.get("structural_status") or "eligible").lower()
            if status in BLOCKING_STATUSES or candidate.get("blocking_violation") is True:
                excluded[strategy_id] = status or "blocking_violation"
                continue
            candidate.setdefault("u_k", self.unresolved_assumption_burden(candidate))
            eligible.append(candidate)

        decision = self.graph.decide(eligible, estimand_id=estimand_id, context=evidence)
        primary_route = self._route_for(decision.primary_strategy_id, estimand_id)
        alternatives = []
        for strategy_id in decision.maximal_strategy_ids:
            route = self._route_for(strategy_id, estimand_id)
            alternatives.append({
                "strategy_id": strategy_id,
                "estimator_id": route.estimator_id if route else None,
                "implemented": route.implemented if route else False,
                "safe_for_auto_routing": route.safe_for_auto_routing if route else False,
            })

        warnings = list(decision.warnings)
        if decision.primary_strategy_id and primary_route is None:
            warnings.append("The selected strategy has no compatible estimator route in the registry.")
        if primary_route and not primary_route.implemented:
            warnings.append("The compatible route is declared but not implemented in CAIS.")
        if primary_route and not primary_route.safe_for_auto_routing:
            warnings.append("The route is executable or declared, but is not certified for automatic causal routing.")

        def _status(value: Any) -> Any:
            return value.get("status") if isinstance(value, Mapping) else value

        unresolved = any(
            _status(value) in {"insufficient_evidence", "unknown", "unverified", "not_assessed"}
            for candidate in eligible
            for value in candidate.get("assumption_statuses", {}).values()
        )
        contradicted = any(
            _status(value) in {"evidence_against", "confirmed_violation"}
            for candidate in eligible
            for value in candidate.get("assumption_statuses", {}).values()
        )
        interpretation = (
            "causal_claim_contradicted_by_available_evidence" if contradicted
            else "causal_claim_requires_unverified_assumptions" if unresolved
            else "conditional_on_identification_contract"
        )
        execution = "available_but_not_certified"
        if primary_route is None:
            execution = "not_routed"
        elif not primary_route.implemented:
            execution = "route_not_implemented"
        identification = "preferred" if decision.status == "preferred" else "conditional"
        return ConditionalRoutingResult(
            estimand_id=estimand_id,
            graph_decision=decision,
            primary_strategy_id=decision.primary_strategy_id,
            primary_estimator_id=primary_route.estimator_id if primary_route else None,
            alternatives=alternatives,
            identification_status=identification,
            execution_status=execution,
            interpretation_status=interpretation,
            excluded_candidates=excluded,
            warnings=warnings,
        )

    @staticmethod
    def unresolved_assumption_burden(candidate: Mapping[str, Any]) -> float:
        """Compute registry-weighted unresolved burden for one candidate.

        Only assumptions marked unresolved contribute.  Testable failures that
        are blocking should already have been removed; advisory failures are
        retained and remain visible in the interpretation status.
        """
        explicit = candidate.get("u_k", candidate.get("unresolved_assumption_burden"))
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        statuses = candidate.get("assumption_statuses", {}) or {}
        weights = candidate.get("assumption_weights", {}) or {}
        burden = 0.0
        unresolved = {"insufficient_evidence", "unknown", "unverified", "not_assessed"}
        if isinstance(statuses, Mapping):
            for name, assessment in statuses.items():
                status = assessment.get("status") if isinstance(assessment, Mapping) else assessment
                if str(status).lower() in unresolved:
                    weight = assessment.get("weight") if isinstance(assessment, Mapping) else weights.get(name, 1.0)
                    try:
                        burden += float(weight)
                    except (TypeError, ValueError):
                        burden += 1.0
        return burden


__all__ = ["EstimatorRoute", "ConditionalRoutingResult", "ConditionalStrategyRouter"]

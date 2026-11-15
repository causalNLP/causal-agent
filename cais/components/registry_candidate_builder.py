"""Build identification candidates from the CAIS registry.

The builder is deliberately conservative: unknown feature evidence retains a
candidate, while explicit structural contradiction removes it.  It does not
turn an unreviewed registry row into a causal certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class RegistryCandidateBuilder:
    links: tuple[Mapping[str, Any], ...]
    rules: tuple[Mapping[str, Any], ...]
    routes: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_registry(cls, workbook_path: str | Path) -> "RegistryCandidateBuilder":
        import openpyxl

        wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)

        def rows(sheet: str) -> tuple[Mapping[str, Any], ...]:
            ws = wb[sheet]
            headers = [cell.value for cell in next(ws.iter_rows(max_row=1))]
            return tuple(dict(zip(headers, row)) for row in ws.iter_rows(min_row=2, values_only=True))

        return cls(rows("Estimand_Strategy_Links"), rows("Strategy_Feature_Rules"), rows("Estimator_Routes"))

    @staticmethod
    def _matches_required(value: Any, required: Any) -> bool:
        if isinstance(required, str) and required.lower() in {"yes", "true", "1"}:
            return value is True or str(value).lower() in {"yes", "true", "1", "supported", "eligible"}
        return value == required or str(value).lower() == str(required).lower()

    def _implemented_strategy_ids(self, estimand_id: Optional[str]) -> set[str]:
        result = set()
        for route in self.routes:
            if not str(route.get("Implemented_in_CAIS") or "").lower().startswith("yes"):
                continue
            supported = str(route.get("Supported_Estimand_IDs") or "")
            if estimand_id and estimand_id not in supported:
                continue
            strategy = str(route.get("Strategy_ID") or "").strip()
            if strategy:
                result.add(strategy)
        return result

    def build(
        self,
        estimand_id: Optional[str],
        feature_evidence: Optional[Mapping[str, Any]] = None,
        assumption_statuses: Optional[Mapping[str, Mapping[str, Any]]] = None,
        implemented_only: bool = True,
    ) -> list[dict[str, Any]]:
        evidence = dict(feature_evidence or {})
        statuses = dict(assumption_statuses or {})
        strategy_ids = self._implemented_strategy_ids(estimand_id) if implemented_only else None
        linked: list[str] = []
        for link in self.links:
            if str(link.get("Estimand_ID") or "").strip() != str(estimand_id or ""):
                continue
            strategy = str(link.get("Strategy_ID") or "").strip()
            review = str(link.get("Review_status") or "").lower()
            if strategy and (strategy_ids is None or strategy in strategy_ids) and "incompatible" not in review:
                linked.append(strategy)
        # Keep one candidate per strategy even when several registry links are
        # duplicated by source or review status.
        candidates = []
        for strategy in sorted(set(linked)):
            strategy_rules = [r for r in self.rules if str(r.get("Strategy_ID") or "") == strategy]
            blocking = []
            unknown = []
            for rule in strategy_rules:
                feature = str(rule.get("Feature_ID") or "")
                required = rule.get("Required_value")
                value = evidence.get(feature)
                if value is None or str(value).lower() in {"unknown", "insufficient_evidence", "unverified"}:
                    unknown.append(feature)
                elif not self._matches_required(value, required):
                    absent_policy = str(rule.get("When_confirmed_absent") or "").lower()
                    if "exclude" in absent_policy or str(rule.get("Role") or "").lower() in {"required", "blocking"}:
                        blocking.append(feature)
            candidate = {
                "strategy_id": strategy,
                "structural_status": "confirmed_violation" if blocking else "eligible",
                "structural_failures": blocking,
                "unknown_structural_features": unknown,
                "assumption_statuses": dict(statuses.get(strategy, {})),
            }
            candidates.append(candidate)
        return candidates


__all__ = ["RegistryCandidateBuilder"]

from __future__ import annotations

import re
from typing import Iterable, Optional


def extract_available_columns(context: str) -> list[str]:
    """Extracts available dataset column names from a context string.

    Expected pattern in context (as used in tests):
        "Available columns: col_a, col_b, col_c."
    """

    if not context:
        return []

    # Capture everything after "Available columns:" up to newline or end.
    match = re.search(r"Available columns\s*:\s*(.+)", context, flags=re.IGNORECASE)
    if not match:
        return []

    raw = match.group(1)
    # Stop at newline; and strip trailing sentence punctuation.
    raw = raw.splitlines()[0].strip().rstrip(". ")

    parts = [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
    # Strip quotes/backticks/markdown, preserve original spelling.
    cols: list[str] = []
    for part in parts:
        cleaned = _strip_formatting(part)
        if cleaned:
            cols.append(cleaned)
    return cols


def _strip_formatting(value: str) -> str:
    value = value.strip()
    # remove markdown bold/italics/backticks and surrounding quotes
    value = value.strip("`" )
    value = value.strip().strip("\"'")
    value = value.strip("*")
    # remove trailing type/category suffixes like "col_name (binary)"
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    return value.strip()


def normalize_name(value: str) -> str:
    """Normalization for matching LLM outputs to dataset columns."""

    value = _strip_formatting(value)
    value = value.lower()
    # allow space/underscore interchange and remove other punctuation
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def map_to_available(value: str, available: Iterable[str]) -> Optional[str]:
    """Map a candidate name to an exact available column name (or None)."""

    target = normalize_name(value)
    if not target:
        return None

    available_list = list(available)
    normalized_map = {normalize_name(c): c for c in available_list}
    return normalized_map.get(target)


def filter_to_available(values: Iterable[str], available: Iterable[str]) -> list[str]:
    available_list = list(available)
    seen: set[str] = set()
    kept: list[str] = []
    for v in values:
        mapped = map_to_available(v, available_list)
        if mapped and mapped not in seen:
            seen.add(mapped)
            kept.append(mapped)
    return kept


def fallback_candidates(available: Iterable[str], *, exclude: Iterable[str] = ()) -> list[str]:
    available_list = list(available)
    excluded_norm = {normalize_name(x) for x in exclude}
    return [c for c in available_list if normalize_name(c) not in excluded_norm]

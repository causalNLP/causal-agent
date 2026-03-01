from __future__ import annotations
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────────────

COLUMN_TYPES = [
    "constant",
    "id",
    "temporal",
    "continuous",
    "binary_numeric",
    "binary_string",
    "nominal",
    "low_cardinality_string",
    "likely_categorical_numeric",
    "ordinal",
    "log_derived",
    "polynomial_derived",
    "cumulative_derived",
    "duplicate",
    "count",
    "rate_or_proportion",
]


@dataclass
class ClassificationResult:
    # col_name -> {col_type, is_string, n_unique, n_missing, note}
    metadata: dict[str, dict] = field(default_factory=dict)
    cols_to_drop: list[str] = field(default_factory=list)
    cols_for_discovery: list[str] = field(default_factory=list)
    discrete_flags: list[str] = field(default_factory=list)
    temporal_var: str | None = None
    warnings: list[str] = field(default_factory=list)

_ID_PATTERNS = re.compile(
    r"^(id|_id|row_?id|index|obs|observation|unit_?id|"
    r"subject_?id|participant_?id|record_?id|serial|"
    r"respondent_?id|sample_?id|case_?id)$",
    re.IGNORECASE,
)
_ID_SUFFIX = re.compile(r"_id$", re.IGNORECASE)

_TEMPORAL_PATTERNS = re.compile(
    r"^(year|yr|month|mon|date|day|time|period|quarter|qtr|"
    r"week|t|trend|wave|round|session|epoch)$",
    re.IGNORECASE,
)

_LOG_PREFIX = re.compile(r"^(ln|log|lg|l_)", re.IGNORECASE)

_RATE_PATTERNS = re.compile(
    r"(rate|ratio|proportion|pct|percent|share|frac|frequency|prevalence)",
    re.IGNORECASE,
)

_COUNT_PATTERNS = re.compile(
    r"(count|total|tot|num_|number|n_|pop|cases|deaths|births|incidents|events)",
    re.IGNORECASE,
)

_CUMULATIVE_PATTERNS = re.compile(
    r"(cumul|cum_|acc|running_total|cumsum)", re.IGNORECASE
)

_ORDINAL_PATTERNS = re.compile(
    r"(level|grade|stage|rank|score|scale|rating|class|tier|severity|priority|order)",
    re.IGNORECASE,
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 when either input is constant."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def _is_integer_valued(s: pd.Series) -> bool:
    """True if a numeric series contains only integer values (ignoring NaN)."""
    s = s.dropna()
    if len(s) == 0:
        return False
    return np.allclose(s, s.round(), equal_nan=True)


def _looks_sequential(s: pd.Series) -> bool:
    """True if sorted unique values are roughly 1,2,3,...,n (an ID or time index)."""
    u = np.sort(s.dropna().unique())
    if len(u) < 5:
        return False
    ideal = np.arange(1, len(u) + 1, dtype=float)
    # Allow for small scaling/offset
    if np.allclose(u, ideal, atol=1) or np.allclose(u - u[0], np.arange(len(u)), atol=1):
        return True
    return False


def _corr_with_log(df: pd.DataFrame, col: str) -> tuple[str, float] | None:
    """
    If `col` looks like a log-transform of another column, return (parent, corr).
    Checks: is there a column X such that log(X) ≈ col?
    """
    if not _LOG_PREFIX.match(col):
        return None
    s = df[col].dropna()
    if len(s) < 10:
        return None
    # Strip prefix to guess parent name
    stem = _LOG_PREFIX.sub("", col)
    # Try exact stem, stem with various cases
    candidates = [c for c in df.columns if c != col and c.lower().startswith(stem.lower())]
    # Also try all other numeric columns
    if not candidates:
        candidates = [c for c in df.select_dtypes(include="number").columns if c != col]
    for cand in candidates:
        other = df[cand].dropna()
        if (other <= 0).any():
            continue
        common = s.index.intersection(other.index)
        if len(common) < 10:
            continue
        log_other = np.log(other.loc[common])
        corr = _safe_corr(s.loc[common].values, log_other.values)
        if abs(corr) > 0.999:
            return (cand, corr)
    return None


def _find_polynomial_parent(df: pd.DataFrame, col: str) -> tuple[str, int, float] | None:
    """
    Check if col ≈ parent^k for k in {2, 3}.
    Returns (parent_name, degree, corr) or None.
    """
    s = df[col].dropna()
    if len(s) < 10:
        return None
    for cand in df.select_dtypes(include="number").columns:
        if cand == col:
            continue
        other = df[cand].dropna()
        common = s.index.intersection(other.index)
        if len(common) < 10:
            continue
        for k in [2, 3]:
            powered = other.loc[common] ** k
            corr = _safe_corr(s.loc[common].values, powered.values)
            if abs(corr) > 0.999:
                return (cand, k, corr)
    return None


def _find_cumulative_parent(df: pd.DataFrame, col: str, group_col: str | None = None) -> str | None:
    """
    Check if col ≈ cumsum of another column (optionally within groups).
    """
    if not _CUMULATIVE_PATTERNS.search(col):
        return None
    s = df[col].dropna()
    if len(s) < 10:
        return None
    for cand in df.select_dtypes(include="number").columns:
        if cand == col:
            continue
        if group_col and group_col in df.columns:
            cumulated = df.groupby(group_col)[cand].cumsum()
        else:
            cumulated = df[cand].cumsum()
        common = s.index.intersection(cumulated.dropna().index)
        if len(common) < 10:
            continue
        corr = _safe_corr(s.loc[common].values, cumulated.loc[common].values)
        if abs(corr) > 0.999:
            return cand
    return None


def _find_duplicate(df: pd.DataFrame, col: str, already_seen: list[str]) -> str | None:
    """Check if col is essentially a duplicate of a previously-seen column."""
    s = df[col].dropna()
    if len(s) < 5:
        return None
    for prev in already_seen:
        other = df[prev].dropna()
        common = s.index.intersection(other.index)
        if len(common) < 5:
            continue
        # Exact or near-exact match
        if np.allclose(s.loc[common], other.loc[common], rtol=1e-4, atol=1e-8, equal_nan=True):
            return prev
        # Or perfect correlation with same scale
        if s.loc[common].std() > 1e-12 and other.loc[common].std() > 1e-12:
            corr = _safe_corr(s.loc[common].values, other.loc[common].values)
            if abs(corr) > 0.9999:
                return prev
    return None


# ──────────────────────────────────────────────────────────────────────
#  Main classifier
# ──────────────────────────────────────────────────────────────────────

def classify_columns(
    df: pd.DataFrame,
    *,
    max_onehot_levels: int = 5,
    id_uniqueness_threshold: float = 0.95,
    categorical_nunique_threshold: int = 15,
) -> ClassificationResult:
    """
    Classify each column in `df` and return a ClassificationResult.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    max_onehot_levels : int
        Max unique values for one-hot encoding nominal columns.
    id_uniqueness_threshold : float
        If nunique/nrows > this AND column looks like an ID, classify as ID.
    categorical_nunique_threshold : int
        Numeric columns with nunique <= this that don't match other patterns
        are flagged as likely_categorical_numeric.
    """
    result = ClassificationResult()
    n_rows = len(df)
    seen_numeric: list[str] = []  # for duplicate detection
    # Track which cols were classified as "id" for cumulative parent detection
    id_cols: list[str] = []

    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        is_string = pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
        n_unique = s.nunique()
        n_missing = int(s.isna().sum())

        col_type = None
        note = ""
        classified = False

        # ── 1. CONSTANT ──────────────────────────────────────────────
        if n_unique <= 1:
            col_type = "constant"
            note = f"Only {n_unique} unique value(s)"
            classified = True

        # ── 2. ID detection ──────────────────────────────────────────
        if not classified:
            is_id = False
            if _ID_PATTERNS.match(col) or _ID_SUFFIX.search(col):
                is_id = True
            if is_string and n_unique / n_rows > id_uniqueness_threshold:
                is_id = True
            if is_numeric and n_unique / n_rows > id_uniqueness_threshold:
                if _is_integer_valued(s) and _looks_sequential(s):
                    is_id = True
            if is_numeric and _is_integer_valued(s):
                low_name = col.lower()
                if any(tag in low_name for tag in ["fip", "fips", "state_id", "county", "country_code", "entity", "unit"]):
                    is_id = True

            if is_id:
                col_type = "id"
                note = "Detected as identifier"
                classified = True

        # ── 3. TEMPORAL detection ────────────────────────────────────
        if not classified and n_unique > 2:
            is_temporal = False
            low = col.lower().strip()
            if _TEMPORAL_PATTERNS.match(low):
                is_temporal = True
            if is_numeric and ("year" in low or "yr" in low):
                vals = s.dropna()
                if len(vals) > 0 and vals.min() >= 1900 and vals.max() <= 2100:
                    is_temporal = True
            if is_numeric and "date" in low:
                is_temporal = True

            if is_temporal:
                col_type = "temporal"
                note = "Time / period variable"
                classified = True

        # ── 4. DERIVED: log-transform ────────────────────────────────
        if not classified and is_numeric:
            log_result = _corr_with_log(df, col)
            if log_result is not None:
                parent, corr = log_result
                col_type = "log_derived"
                note = f"log({parent}), corr={corr:.6f}"
                classified = True

        # ── 5. DERIVED: polynomial (tsq = t^2) ──────────────────────
        if not classified and is_numeric:
            poly_result = _find_polynomial_parent(df, col)
            if poly_result is not None:
                parent, degree, corr = poly_result
                col_type = "polynomial_derived"
                note = f"{parent}^{degree}, corr={corr:.6f}"
                classified = True

        # ── 6. DERIVED: cumulative ───────────────────────────────────
        if not classified and is_numeric:
            group_col = id_cols[0] if id_cols else None
            cum_parent = _find_cumulative_parent(df, col, group_col)
            if cum_parent is not None:
                col_type = "cumulative_derived"
                note = f"Cumulative sum of {cum_parent}"
                classified = True

        # ── 7. DUPLICATE of already-seen column ──────────────────────
        if not classified and is_numeric:
            dup = _find_duplicate(df, col, seen_numeric)
            if dup is not None:
                col_type = "duplicate"
                note = f"Near-duplicate of {dup}"
                classified = True

        # ── 8. BINARY NUMERIC (exactly 2 unique numeric values) ──────
        if not classified and is_numeric and n_unique == 2:
            col_type = "binary_numeric"
            note = f"Binary: values {sorted(s.dropna().unique().tolist())}"
            classified = True

        # ── 9. BINARY STRING ─────────────────────────────────────────
        if not classified and is_string and n_unique == 2:
            col_type = "binary_string"
            note = f"Binary string: values {sorted(s.dropna().unique().tolist())}"
            classified = True

        # ── 10. STRING with low cardinality → nominal ────────────────
        if not classified and is_string:
            if n_unique <= max_onehot_levels:
                col_type = "nominal"
                note = f"Nominal string, {n_unique} levels"
            elif n_unique <= categorical_nunique_threshold:
                col_type = "low_cardinality_string"
                note = f"Low-cardinality string, {n_unique} levels"
            else:
                col_type = "id"
                note = f"High-cardinality string ({n_unique} unique), treating as ID"
            classified = True

        # ── 11. NUMERIC: rate / proportion ───────────────────────────
        if not classified and is_numeric:
            vals = s.dropna()
            is_rate = False
            if _RATE_PATTERNS.search(col):
                is_rate = True
            if len(vals) > 0 and vals.min() >= 0 and vals.max() <= 1.0 and not _is_integer_valued(s):
                is_rate = True
            if is_rate:
                col_type = "rate_or_proportion"
                note = "Rate/proportion — treat as continuous"
                classified = True

        # ── 12. NUMERIC: count data ──────────────────────────────────
        if not classified and is_numeric:
            vals = s.dropna()
            if _COUNT_PATTERNS.search(col) and _is_integer_valued(s) and vals.min() >= 0:
                col_type = "count"
                note = "Count data — treat as continuous"
                classified = True

        # ── 13. NUMERIC: likely categorical ──────────────────────────
        if not classified and is_numeric:
            if n_unique <= categorical_nunique_threshold and _is_integer_valued(s):
                unique_ratio = n_unique / n_rows
                if unique_ratio < 0.05 or n_unique <= 10:
                    col_type = "likely_categorical_numeric"
                    note = f"{n_unique} unique integer values, ratio={unique_ratio:.3f}"
                    classified = True

        # ── 14. ORDINAL (name-based heuristic) ───────────────────────
        if col_type == "likely_categorical_numeric" and _ORDINAL_PATTERNS.search(col):
            col_type = "ordinal"
            note += " — name suggests ordinal"

        # ── 15. FALLBACK: continuous ─────────────────────────────────
        if not classified:
            if is_numeric:
                col_type = "continuous"
                note = "Numeric, high cardinality"
            else:
                col_type = "nominal"
                note = "Fallback: unclassified string column"

        # Store metadata
        result.metadata[col] = {
            "col_type": col_type,
            "is_string": is_string,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "note": note,
        }

        if col_type == "id":
            id_cols.append(col)
        if is_numeric:
            seen_numeric.append(col)

    # ── Build action lists ───────────────────────────────────────────
    for col, meta in result.metadata.items():
        ct = meta["col_type"]
        match ct:
            case "constant" | "id" | "temporal":
                result.cols_to_drop.append(col)
                if ct == "temporal":
                    result.temporal_var = col

            case "log_derived" | "polynomial_derived" | "cumulative_derived" | "duplicate":
                result.cols_to_drop.append(col)

            case "continuous" | "rate_or_proportion" | "count":
                result.cols_for_discovery.append(col)

            case "binary_numeric" | "binary_string":
                result.cols_for_discovery.append(col)
                result.discrete_flags.append(col)

            case "nominal" | "low_cardinality_string":
                if meta["n_unique"] <= max_onehot_levels:
                    result.cols_for_discovery.append(col)
                    result.discrete_flags.append(col)
                else:
                    result.cols_to_drop.append(col)
                    result.warnings.append(
                        f"Dropping {col}: {meta['n_unique']} levels, too many for one-hot"
                    )

            case "likely_categorical_numeric" | "ordinal":
                result.cols_for_discovery.append(col)
                result.discrete_flags.append(col)

    return result


# ──────────────────────────────────────────────────────────────────────
#  Clean data for causal discovery
# ──────────────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    result: ClassificationResult,
) -> pd.DataFrame:
    """
    Given raw data and its ClassificationResult, return a cleaned numeric
    DataFrame ready for causal-learn.

    Steps
    -----
    1. Keep only cols_for_discovery (drop IDs, constants, derived, etc.).
    2. Handle missing values (drop rows with any NaN).
    3. Encode string columns:
       - binary_string  → 0/1
       - nominal / low_cardinality_string → one-hot (drop_first)
    4. Standardise binary numerics whose values aren't {0, 1} → remap to 0/1.
    5. Return a float64 ndarray-backed DataFrame (causal-learn expects this).
    """
    kept = [c for c in result.cols_for_discovery if c in df.columns]
    out = df[kept].copy()

    # ── Drop rows with missing values ────────────────────────────────
    out = out.dropna()

    # ── Per-column transforms based on col_type ──────────────────────
    cols_to_onehot: list[str] = []

    for col in list(out.columns):
        meta = result.metadata[col]
        ct = meta["col_type"]

        if ct == "binary_string":
            # Map the two unique string values to 0 / 1 alphabetically
            vals = sorted(out[col].unique())
            out[col] = out[col].map({vals[0]: 0, vals[1]: 1}).astype(np.float64)

        elif ct in ("nominal", "low_cardinality_string"):
            cols_to_onehot.append(col)

        elif ct == "binary_numeric":
            # Ensure values are exactly {0, 1}
            vals = sorted(out[col].dropna().unique())
            if vals != [0, 1]:
                out[col] = out[col].map({vals[0]: 0, vals[1]: 1}).astype(np.float64)
            else:
                out[col] = out[col].astype(np.float64)

        elif ct == "likely_categorical_numeric" or ct == "ordinal":
            out[col] = out[col].astype(np.float64)

        else:
            # continuous, rate_or_proportion, count — already numeric
            out[col] = out[col].astype(np.float64)

    # ── One-hot encode nominal / low-cardinality string columns ──────
    if cols_to_onehot:
        out = pd.get_dummies(out, columns=cols_to_onehot, drop_first=True, dtype=np.float64)

    out = out.reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Pretty printer
# ──────────────────────────────────────────────────────────────────────

def print_classification(res: ClassificationResult) -> None:
    print(f"{'Column':<25s} {'Type':<28s} {'Action':<8s} {'Note'}")
    print("─" * 100)
    for col, meta in res.metadata.items():
        action = "DROP" if col in res.cols_to_drop else "KEEP"
        disc = " [D]" if col in res.discrete_flags else ""
        print(f"{col:<25s} {meta['col_type']:<28s} {action:<8s} {meta['note']}{disc}")

    print(f"\n── Summary ──")
    print(f"  Keep for discovery: {len(res.cols_for_discovery)} columns")
    print(f"  Drop: {len(res.cols_to_drop)} columns")
    print(f"  Discrete flags: {len(res.discrete_flags)}")
    if res.temporal_var:
        print(f"  Temporal variable: {res.temporal_var}")
    for w in res.warnings:
        print(f"  ⚠ {w}")


# ──────────────────────────────────────────────────────────────────────
#  Test on the four datasets
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "all_data"

    # The four original datasets
    named_files = [
        DATA_DIR / "abortion_bf15.csv",
        DATA_DIR / "drinking.csv",
        DATA_DIR / "fulton.csv",
        DATA_DIR / "ihdp_0.csv",
    ]

    # Pass --all to run on every CSV in the data folder instead
    if "--all" in sys.argv:
        files = sorted(DATA_DIR.glob("*.csv"))
    else:
        files = named_files

    for path in files:
        if not path.exists():
            print(f"  ⚠ Skipping {path.name}: file not found")
            continue
        df = pd.read_csv(path)
        print(f"\n{'='*100}")
        print(f"  {path.name}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        print(f"{'='*100}")
        res = classify_columns(df)
        print(res)
        print_classification(res)

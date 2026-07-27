"""
Profiles every CSV in data/all_data and data/synthetic_data.
Per file: total variable count + named lists for each semantic category.
Goal: see what's directly usable by causal-learn (needs continuous numeric)
and what needs cleaning (and what kind).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import json
import warnings
warnings.filterwarnings("ignore")


# ── keyword / pattern banks for heuristic classification ─────────────────────

# Patterns matched case-insensitively against column names.
# The distinction between ordinal-int and nominal-int is inherently ambiguous
# from data alone — the heuristic here leans on naming conventions seen in
# this specific corpus (severity levels, scales, grade_level, etc.).

ID_PATTERNS = re.compile(
    r"^(unnamed|row|index|Unnamed)"
    r"|_?id$|^id$|^id_"
    r"|^fip$|^fips$|^sid$"
    r"|^unit$|^cluster$"
    r"|^hh_id$|^store_id$|^factory_id$"
    r"|^household_id$|^student_id$|^school_id$"
    r"|^region_id$|^club_id$|^village$|^villnum$"
    r"|^statedisdec$",
    re.IGNORECASE,
)

YEAR_TIME_PATTERNS = re.compile(
    r"^year$|^yr$|^yob$|^yod$"
    r"|year_of_birth|quarter_of_birth"
    r"|^quarter$|^quarter_num$"
    r"|^date\d*$|^date2$"
    r"|^academic_year$|^measurement_year$"
    r"|_year$|^t$|^time$",
    re.IGNORECASE,
)

LOG_PREFIXES = re.compile(r"^l_|^ln|^log_|^lh", re.IGNORECASE)

COUNT_PATTERNS = re.compile(
    r"^tot|total|^num_|^n_|_count$|^count"
    r"|^ncalls|^nregs|^popwt$|^population$|^police$",
    re.IGNORECASE,
)

RATE_PATTERNS = re.compile(
    r"rate$|^pct|^perc|percent|proportion|_mean$|_pct$|share$|^p\d{4}",
    re.IGNORECASE,
)

# Names that suggest an ordinal scale (severity, grade_level, satisfaction, etc.)
ORDINAL_HINTS = re.compile(
    r"severity|grade_level|satisfaction|education|experience|income_level"
    r"|quality|score|level|scale|rank|priority",
    re.IGNORECASE,
)


def profile_column(col_name: str, series: pd.Series, all_col_names: list[str]) -> str:
    s = series.dropna()
    n = len(series)

    if s.nunique() <= 1:
        return "constant_or_near_constant"

    # ── ID / index ───────────────────────────────────────────────────────
    if ID_PATTERNS.search(col_name):
        return "id_index"
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and s.nunique() / max(n, 1) > 0.90:
        if numeric.min() in (0, 1) and (numeric.diff().dropna() == 1).mean() > 0.8:
            return "id_index"

    # ── string-typed columns ─────────────────────────────────────────────
    if series.dtype == object or s.dtype == object:
        s_str = s.astype(str).str.strip()
        nuniq = s_str.nunique()
        avg_len = s_str.str.len().mean()

        lower_vals = set(s_str.str.lower().unique())
        if lower_vals <= {"true", "false", "yes", "no", "t", "f", "y", "n"}:
            return "binary"

        if avg_len > 40 or nuniq > 100:
            return "natural_language"

        return "nominal_categorical_str"

    # ── numeric from here ────────────────────────────────────────────────
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.isna().any():
        return "nominal_categorical_str"

    vals = set(numeric.dropna().unique())
    nuniq = len(vals)

    if vals <= {0, 1, 0.0, 1.0}:
        return "binary"

    if YEAR_TIME_PATTERNS.search(col_name):
        return "year_time"
    if nuniq > 2 and numeric.min() >= 1900 and numeric.max() <= 2030 and (numeric == numeric.astype(int)).all():
        return "year_time"

    # log-transformed: check if a plausible raw counterpart exists in the same file
    if LOG_PREFIXES.search(col_name):
        raw_candidates = [
            re.sub(r"^l_", "", col_name, flags=re.I),
            re.sub(r"^ln", "", col_name, flags=re.I),
            re.sub(r"^log_", "", col_name, flags=re.I),
            re.sub(r"^lh", "", col_name, flags=re.I),
        ]
        has_raw = any(rc in all_col_names for rc in raw_candidates if rc and rc != col_name)
        return "log_transformed_has_raw" if has_raw else "log_transformed"

    if COUNT_PATTERNS.search(col_name) and numeric.min() >= 0 and (numeric == numeric.round(0)).all():
        return "count"

    if RATE_PATTERNS.search(col_name):
        return "bounded_rate_proportion"
    if 0 <= numeric.min() and numeric.max() <= 1 and nuniq > 2:
        return "bounded_rate_proportion"

    # ordinal vs nominal for low-cardinality integers
    is_int = (numeric == numeric.astype(int)).all()
    if is_int and 2 < nuniq <= 10:
        if ORDINAL_HINTS.search(col_name):
            return "ordinal_int"
        return "nominal_int"
    if is_int and 10 < nuniq <= 20:
        return "nominal_int"

    return "continuous"


def profile_file(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path, low_memory=False, nrows=50_000)
    except Exception as e:
        return {"error": str(e)}

    all_col_names = list(df.columns)
    buckets = defaultdict(list)
    missing_cols = {}

    for col in df.columns:
        tag = profile_column(col, df[col], all_col_names)
        buckets[tag].append(col)
        miss = df[col].isna().mean() * 100
        if miss > 0:
            missing_cols[col] = round(miss, 1)

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "buckets": dict(buckets),
        "missing": missing_cols,
    }


DISPLAY_ORDER = [
    ("continuous",              "Continuous"),
    ("binary",                  "Binary indicators"),
    ("ordinal_int",             "Ordinal categoricals (int-encoded)"),
    ("nominal_int",             "Nominal categoricals (int-encoded)"),
    ("nominal_categorical_str", "Nominal categoricals (string)"),
    ("year_time",               "Year / time columns"),
    ("log_transformed_has_raw", "Log-transformed (raw version present)"),
    ("log_transformed",         "Log-transformed (standalone)"),
    ("count",                   "Count data"),
    ("bounded_rate_proportion", "Bounded rates / proportions"),
    ("id_index",                "IDs / indices"),
    ("constant_or_near_constant", "Constant or near-constant"),
    ("natural_language",        "Natural language descriptions"),
]

READY_TAGS = {"continuous", "binary"}


def main():
    base = Path(__file__).parent
    folders = [base / "all_data", base / "synthetic_data"]

    all_profiles = {}
    for folder in folders:
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            key = f"{folder.name}/{csv_path.name}"
            all_profiles[key] = profile_file(csv_path)

    agg_type_counts = defaultdict(int)
    agg_file_ready = []
    agg_file_needs_work = []

    for fname, prof in sorted(all_profiles.items()):
        print(f"\n{'━' * 80}")
        print(f"  {fname}   ({prof['rows']} rows × {prof['cols']} cols)")
        print(f"{'━' * 80}")

        if "error" in prof:
            print(f"  ERROR: {prof['error']}")
            continue

        buckets = prof["buckets"]
        file_clean = True

        for tag, label in DISPLAY_ORDER:
            cols = buckets.get(tag, [])
            if not cols:
                continue
            agg_type_counts[tag] += len(cols)
            if tag not in READY_TAGS:
                file_clean = False
            marker = "✅" if tag in READY_TAGS else "🔧"
            print(f"  {marker} {label} ({len(cols)}): {', '.join(cols)}")

        if prof["missing"]:
            items = [f"{c} ({v}%)" for c, v in prof["missing"].items()]
            print(f"  ⚠️  Missing values: {', '.join(items)}")

        if file_clean:
            agg_file_ready.append(fname)
        else:
            agg_file_needs_work.append(fname)

    total_cols = sum(agg_type_counts.values())
    ready_cols = agg_type_counts.get("continuous", 0) + agg_type_counts.get("binary", 0)

    print(f"\n{'═' * 80}")
    print("  AGGREGATE SUMMARY")
    print(f"{'═' * 80}")
    print(f"  Files scanned:        {len(all_profiles)}")
    print(f"  Ready for causal-learn (all continuous/binary): {len(agg_file_ready)}")
    print(f"  Need some cleaning:   {len(agg_file_needs_work)}")
    print(f"  Total columns:        {total_cols}   (ready: {ready_cols},  need work: {total_cols - ready_cols})")

    print(f"\n  Column type breakdown:")
    for tag, label in DISPLAY_ORDER:
        n = agg_type_counts.get(tag, 0)
        if n == 0:
            continue
        marker = "✅" if tag in READY_TAGS else "🔧"
        print(f"    {marker} {label:48s} {n:4d}")

    print(f"\n  Files ready as-is:")
    for f in sorted(agg_file_ready):
        print(f"    ✅ {f}")

    out_path = base / "data_profile.json"
    with open(out_path, "w") as fh:
        json.dump(all_profiles, fh, indent=2, default=str)
    print(f"\n  💾 Full profile → {out_path}")


if __name__ == "__main__":
    main()

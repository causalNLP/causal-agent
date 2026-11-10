"""Oracle-facts test of decision_tree_v2.

Feeds each CauSciBench row's ground-truth design metadata (treatment, outcome,
controls, instrument, running/temporal variable, is_rct) straight into the
deterministic tree, with the real data checks but NO LLM. This isolates the
tree's rules from the LLM interpreter: if the tree cannot pick the intended
method with oracle facts, the rules are the problem; if it can, the interpreter
is.

Usage (from the repo root, after the CauSciBench metadata and CSVs are downloaded):
    python experiments/tree_oracle_test.py [--limit N]
"""
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from cais.components.decision_tree_v2 import select_method_v2

BASE = Path("experiments/data/causcibench")
SPLITS = [
    ("qr", BASE / "metadata_csv" / "qr_input.csv", BASE / "qrdata"),
    ("synthetic", BASE / "metadata_csv" / "synthetic_input.csv", BASE / "synthetic_data"),
]

# Benchmark method -> tree leaves that count as a match. Second entry in each
# pair is the "family" match (same identification strategy, different estimator).
ALLOWED = {
    "ols": ({"linear_regression"}, {"linear_regression", "diff_in_means"}),
    "ipw": ({"propensity_score_weighting"}, {"propensity_score_weighting", "trimmed_ipw", "propensity_score_matching"}),
    "matching": ({"propensity_score_matching"}, {"propensity_score_matching", "propensity_score_weighting", "trimmed_ipw"}),
    "did": ({"difference_in_differences"}, {"difference_in_differences", "difference_in_differences_time_window"}),
    "rdd": ({"regression_discontinuity_design"}, {"regression_discontinuity_design", "donut_rdd"}),
    "iv": ({"instrumental_variable"}, {"instrumental_variable"}),
    "frontdoor": ({"frontdoor_adjustment"}, {"frontdoor_adjustment"}),
}


def resolve_file(directory: Path, name: str):
    name = str(name)
    for cand in (name, f"cleaned_{name}", name.replace("cleaned_", "")):
        p = directory / cand
        if p.exists():
            return p
    return None


def split_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def blank(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "nan" else text


def build_props(row, df):
    treatment = blank(row.get("treatment"))
    outcome = blank(row.get("outcome"))
    covariates = split_list(row.get("controls"))
    instrument = blank(row.get("instrument_var"))
    running = blank(row.get("running_var"))
    temporal = blank(row.get("temporal_var"))
    state = blank(row.get("state_var"))
    mediator = blank(row.get("mediator"))
    is_rct = bool(int(row["is_rct"])) if not pd.isna(row.get("is_rct")) else False

    if treatment and treatment in df.columns:
        n_levels = int(df[treatment].dropna().nunique())
    else:
        n_levels = 2
    ttype = "binary" if n_levels <= 2 else "continuous"

    return {
        "treatment_variable": treatment,
        "outcome_variable": outcome,
        "treatment_variable_type": ttype,
        "covariates": covariates,
        "instrument_variable": instrument,
        "mediator_variable": mediator,
        "running_variable": running,
        "cutoff_value": None,
        "time_variable": temporal,
        "group_variable": state,
        "treatment_period_start": None,
        "placebo_period_start": None,
        "is_rct": is_rct,
        "has_temporal_structure": bool(temporal),
        "is_encouragement_design": None,
        "has_pre_treatment_variables": bool(covariates),
        "has_valid_instrument": bool(instrument),
        "has_candidate_mediator": None,
        "has_valid_backdoor_set": bool(covariates),
        "is_structure_supported": True,
        "anticipation_known_bounded": False,
        "local_manipulation": False,
        "treatment_reference_level": None,
    }, n_levels


def summarize_path(result, n=4):
    steps = [f"{s.node}[{s.outcome}]" for s in result.steps]
    return " -> ".join(steps[-n:]) if steps else ""


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    rows_out = []
    errors = []
    for split, meta_path, data_dir in SPLITS:
        meta = pd.read_csv(meta_path)
        if limit:
            meta = meta.head(limit)
        for _, row in meta.iterrows():
            name = row["dataset_name"]
            path = resolve_file(data_dir, name)
            expected = blank(row.get("method"))
            entry = {
                "split": split,
                "id": int(row["id"]),
                "dataset": name,
                "expected": expected,
            }
            if path is None:
                entry.update(selected=None, status="file_missing")
                rows_out.append(entry)
                continue
            try:
                df = pd.read_csv(path, low_memory=False)
                props, n_levels = build_props(row, df)
                result = select_method_v2(
                    props, df=df, description=None, llm=None, prompt_callback=None
                )
                strict, family = ALLOWED.get(expected, (set(), set()))
                selected = result.method
                entry.update(
                    selected=selected,
                    ended=bool(result.ended),
                    end_reason=result.end_reason,
                    match=selected in strict,
                    family_match=selected in family,
                    constant_treatment=n_levels < 2,
                    path_tail=summarize_path(result),
                    warnings=result.warnings[-3:],
                    status="ok",
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                errors.append((split, int(row["id"]), name, repr(exc)))
                entry.update(selected=None, status="error", error=repr(exc))
                if len(errors) <= 5:
                    traceback.print_exc()
            rows_out.append(entry)

    out = Path("experiments/data/causcibench/runs")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "tree_oracle_test.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in rows_out:
            fh.write(json.dumps(entry) + "\n")

    ok = [r for r in rows_out if r["status"] == "ok"]
    print(f"\nrows={len(rows_out)} ok={len(ok)} errors={len(errors)} file_missing={sum(1 for r in rows_out if r['status']=='file_missing')}")
    print(f"strict match {sum(1 for r in ok if r['match'])}/{len(ok)}")
    print(f"family match {sum(1 for r in ok if r['family_match'])}/{len(ok)}")
    ended = [r for r in ok if r["ended"]]
    print(f"END (no estimate) {len(ended)}/{len(ok)}")

    print("\nby expected method (strict / family / n / ended):")
    by_method = defaultdict(lambda: [0, 0, 0, 0])
    for r in ok:
        bucket = by_method[r["expected"]]
        bucket[0] += int(bool(r["match"]))
        bucket[1] += int(bool(r["family_match"]))
        bucket[2] += 1
        bucket[3] += int(bool(r["ended"]))
    for method, (strict_n, family_n, total, end_n) in sorted(by_method.items()):
        print(f"  {method:10s} strict {strict_n:3d}/{total:<3d} family {family_n:3d}/{total:<3d} ended {end_n}")

    print("\nmismatches (expected -> selected), excluding ended:")
    mismatch = Counter(
        (r["expected"], r["selected"]) for r in ok if not r["match"] and not r["ended"]
    )
    for (expected, selected), count in mismatch.most_common(15):
        print(f"  {expected:10s} -> {selected:30s} {count}")

    print("\nEND reasons:")
    for reason, count in Counter(r["end_reason"] for r in ended).most_common():
        print(f"  {count:3d}x {reason}")

    constant = [r for r in ok if r["constant_treatment"]]
    print(f"\nconstant-treatment rows (benchmark quirk): {len(constant)}")


if __name__ == "__main__":
    main()

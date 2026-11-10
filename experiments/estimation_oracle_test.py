"""Oracle-estimation test: oracle-selected method, executed offline, no LLM.

Extends tree_oracle_test from *selection* to *numbers*: for every CauSciBench
row, walk the deterministic tree (oracle facts, real checks, no LLM), then run
the selected estimator directly (llm=None, defaults) and compare the effect
against the bench reference answer.

DiD/RDD parameters are inferred from the data itself (no LLM parsing):
  - RDD cutoff = the boundary of treated/untreated mass on the running variable
    (only when assignment is sharp; violations are recorded, not fatal).
  - DiD: time/group columns come from the metadata (temporal_var/state_var),
    falling back to post*/unit* column names; treatment_time is not needed by
    the estimators when the post flag itself is the time/binary column.

Usage:  PYTHONPATH=. python experiments/estimation_oracle_test.py
Output: experiments/data/causcibench/runs/estimation_oracle.jsonl
"""
import json
import logging
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from cais.methods import METHOD_MAPPING
from cais.components.decision_tree_v2 import select_method_v2

BASE = Path("experiments/data/causcibench")
SPLITS = [
    ("qr", BASE / "metadata_csv" / "qr_input.csv", BASE / "qrdata"),
    ("synthetic", BASE / "metadata_csv" / "synthetic_input.csv", BASE / "synthetic_data"),
]
OUT = BASE / "runs" / "estimation_oracle.jsonl"

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
    for cand in (str(name), f"cleaned_{name}", str(name).replace("cleaned_", "")):
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
    return [p.strip() for p in text.split(",") if p.strip()]


def blank(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "nan" else text


def build_props(row, df):
    """Same oracle facts as tree_oracle_test.build_props (kept in sync)."""
    treatment = blank(row.get("treatment"))
    outcome = blank(row.get("outcome"))
    covariates = [c for c in split_list(row.get("controls")) if c in df.columns]
    instrument = blank(row.get("instrument_var"))
    running = blank(row.get("running_var"))
    temporal = blank(row.get("temporal_var"))
    state = blank(row.get("state_var"))
    mediator = blank(row.get("mediator"))
    is_rct = bool(row["is_rct"]) if not pd.isna(row.get("is_rct")) else False
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


def infer_rdd(df, running, treatment):
    """Sharp cutoff inference: boundary of the treated mass on the running var."""
    unit_dosage = df.groupby(running)[treatment].agg(lambda s: s.mean())
    treated_vals = unit_dosage[unit_dosage >= 0.5].index
    control_vals = unit_dosage[unit_dosage < 0.5].index
    if len(treated_vals) == 0 or len(control_vals) == 0:
        return None
    upper_edge = min(treated_vals)   # smallest running value that is treated
    below_bad = int((df[running] < upper_edge).mul(df[treatment] == 1).sum())
    top_bad = int((df[running] >= upper_edge).mul(df[treatment] == 0).sum())
    return {
        "cutoff_value": float(upper_edge),
        "treat_above_cutoff": True,
        "violations": below_bad + top_bad,
    }


def _binary_cols(df):
    return [c for c in df.columns
            if set(df[c].dropna().unique()).issubset({0, 1, True, False}) and df[c].nunique() == 2]


def infer_did(df, meta):
    """Time/group columns from metadata, else name-based fallbacks."""
    time_var = None
    group_var = None
    did_term = None
    notes = []
    temporal = blank(meta.get("temporal_var"))
    state = blank(meta.get("state_var"))
    if temporal and temporal in df.columns:
        is_bin = df[temporal].dropna().nunique() == 2
        if is_bin:
            time_var, did_term = temporal, temporal
            notes.append("temporal_var is a binary post flag (canonical)")
        else:
            time_var = temporal
            notes.append("temporal_var is a real time axis (TWFE)")
    else:
        post_like = [c for c in _binary_cols(df) if c.lower().startswith(("post", "after"))]
        if post_like:
            time_var = post_like[0]
            did_term = time_var
            notes.append(f"post flag found by name: {time_var} (canonical)")
        else:
            return None
    if state and state in df.columns:
        if group_var is None and df[state].nunique() > (2 if did_term else 2):
            pass
        group_var = state
    else:
        unit_like = [c for c in df.columns if c.lower().startswith(("unit", "state"))]
        if unit_like:
            group_var = unit_like[0]
            notes.append(f"group column found by name: {group_var}")
    if did_term is None:
        treat = blank(meta.get("treatment"))
        if treat and treat in df.columns and df[treat].dropna().nunique() == 2:
            did_term = treat
    return {"time_variable": time_var, "group_variable": group_var,
            "did_term": did_term, "notes": "; ".join(notes)}


def execute(method, df, treatment, outcome, covariates, row_extra):
    """Run one estimator offline (llm=None). Returns (effect, error, options)."""
    func = METHOD_MAPPING[method]
    kwargs = dict(
        df=df, treatment=treatment, outcome=outcome, covariates=covariates,
        dataset_description="", query_str="", llm=None,
    )
    opts = {}
    per_method = {
        "instrumental_variable": lambda: kwargs.update(
            instrument_variable=[c for c in row_extra.get("instruments", []) if c in df.columns]
        ),
        "regression_discontinuity_design": lambda: kwargs.update(
            running_variable=row_extra["running"], **{k: row_extra["rdd"][k] for k in ("cutoff_value", "treat_above_cutoff")}
        ),
        "difference_in_differences": lambda: kwargs.update(**row_extra["did"]),
        "diff_in_means": lambda: kwargs.pop("covariates", None),
    }
    if method in per_method:
        per_method[method]()
    if method.startswith("propensity_score"):
        opts["covariates"] = list(covariates)
    res = func(**kwargs)
    if method.startswith("propensity_score"):
        opts["covariates"] = list(covariates)
    effect = res.get("effect_estimate")
    return effect, None, res


def main():
    rows_out = []
    for split, meta_path, data_dir in SPLITS:
        meta = pd.read_csv(meta_path)
        for _, mrow in meta.iterrows():
            name = mrow["dataset_name"]
            path = resolve_file(data_dir, name)
            expected = blank(mrow.get("method"))
            entry = {"split": split, "id": int(mrow["id"]), "dataset": name,
                     "expected_method": expected, "expected_answer": float(mrow["answer"])}
            if path is None:
                entry.update(status="file_missing")
                rows_out.append(entry)
                continue
            df = pd.read_csv(path, low_memory=False)
            props, _ = build_props(mrow, df)
            treatment, outcome = props["treatment_variable"], props["outcome_variable"]
            covariates = list(props["covariates"])
            try:
                tree = select_method_v2(props, df=df, description=None, llm=None, prompt_callback=None)
            except Exception as exc:
                entry.update(status="error", phase="selection", error=repr(exc))
                rows_out.append(entry)
                continue
            entry.update(selected=tree.method, ended=bool(tree.ended),
                         end_reason=tree.end_reason, estimand=tree.estimand)
            strict, family = ALLOWED.get(expected, (set(), set()))
            entry.update(match=tree.method in strict, family_match=tree.method in family)

            method = tree.method
            row_extra = {
                "instruments": split_list(mrow.get("instrument_var")),
                "running": blank(mrow.get("running_var")),
                "rdd": None, "did": None,
            }
            if method == "regression_discontinuity_design":
                if not row_extra["running"] or row_extra["running"] not in df.columns or treatment not in df.columns:
                    entry.update(status="error", phase="did_or_rdd_params", error="running variable missing")
                    rows_out.append(entry)
                    continue
                row_extra["rdd"] = infer_rdd(df, row_extra["running"], treatment)
                entry["rdd_inference"] = row_extra["rdd"]
            elif method == "difference_in_differences":
                row_extra["did"] = infer_did(df, mrow)
                entry["did_inference"] = None if not row_extra["did"] else dict(row_extra["did"])
                if not row_extra["did"]:
                    entry.update(status="error", phase="did_or_rdd_params", error="no time/group structure found")
                    rows_out.append(entry)
                    continue

            if method is None or method not in METHOD_MAPPING:
                entry.update(status="no_method", phase="estimation")
                rows_out.append(entry)
                continue
            try:
                effect, err, _res = execute(method, df, treatment, outcome, covariates, row_extra)
            except Exception as exc:
                entry.update(status="error", phase="estimation",
                             error=f"{type(exc).__name__}: {str(exc)[:180]}")
                rows_out.append(entry)
                continue
            ref = float(mrow["answer"])
            rel = abs(effect - ref) / abs(ref) if isinstance(effect, (int, float)) and ref != 0 else None
            entry.update(status="ok", effect=effect, rel_error=rel,
                         standard_error=_res.get("standard_error"),
                         p_value=_res.get("p_value"), method_used=_res.get("method_used"))
            rows_out.append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for e in rows_out:
            fh.write(json.dumps(e, default=str) + "\n")

    ok = [e for e in rows_out if e.get("status") == "ok"]
    errors = [e for e in rows_out if e.get("status") == "error"]
    n = len(rows_out)
    print(f"\nrows={n} ok={len(ok)} errors={len(errors)} "
          f"no_method={sum(1 for e in rows_out if e.get('status')=='no_method')} "
          f"file_missing={sum(1 for e in rows_out if e.get('status')=='file_missing')}")

    dom = [e for e in ok if e.get("effect") is not None]
    print(f"estimates produced: {len(dom)}/{len(ok)}")
    hits = [e for e in dom if e.get("rel_error") is not None and e["rel_error"] <= 0.05]
    print(f"within 5% of reference: {len(hits)}/{len(dom)}")

    print("\nby expected method (n / estimated / <=5% / <=20% / median rel_err):")
    by = defaultdict(lambda: [])
    for e in dom:
        bucket = by[e.get("expected_method")]
        bucket.append(e.get("rel_error") if e.get("rel_error") is not None else np.inf)
    for method, errs in sorted(by.items()):
        med = float(np.median(errs))
        print(f"  {method:12s} n={len(errs):3d}  <=5%: {sum(r <= 0.05 for r in errs):3d}  "
              f"<=20%: {sum(r <= 0.20 for r in errs):3d}  median rel_err: {med:.3f}")

    worst = sorted([e for e in dom if e.get("rel_error")], key=lambda e: -e["rel_error"])[:12]
    print("\nworst 12 rows (split#id dataset expected->selected ref vs effect):")
    for e in worst:
        print(f"  {e['split']}#{e['id']:3d} {e['dataset'][:28]:28s} {e['expected_method']}->{e['selected']} "
              f"ref {e['expected_answer']:.3f} vs {e['effect']:.4f} (rel {e['rel_error']:.2})")

    errs = [e for e in errors]
    if errs:
        print("\nerror phases:", Counter((e.get('phase'), str(e.get('error', ''))[:60]) for e in errs).most_common(6))
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()

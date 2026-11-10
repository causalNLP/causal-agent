"""Interpreter accuracy harness: measure CauSciBench identification against metadata.

For each sampled bench row, runs the real interpreter path (deterministic
dataset profile -> LLM variable identification -> design-fact prompts ->
post-selection controls refiner) and scores every named field against the
metadata columns that CauSciBench records as ground truth (treatment, outcome,
controls, instrument, mediator, running variable, temporal/unit columns, RCT
flag). No estimation and no decision tree: this isolates the NLP stage.

Note: `interpret_query` builds its own LLM client from .env (openrouter/gpt-4o),
so this measures the configured production interpreter, not a test double.

Usage:  PYTHONPATH=. python experiments/interpreter_accuracy_test.py [--rows-per-method N]
Output: experiments/data/causcibench/runs/interpreter_accuracy.jsonl
"""
import argparse
import contextlib
import io
import json
import sys
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cais.config import get_llm_client
from cais.components.dataset_analyzer import analyze_dataset
from cais.components.query_interpreter import interpret_query
from cais.components.controls_selector import identify_controls

BASE = Path("experiments/data/causcibench")
SPLITS = [
    ("qr", BASE / "metadata_csv" / "qr_input.csv", BASE / "qrdata"),
    ("synthetic", BASE / "metadata_csv" / "synthetic_input.csv", BASE / "synthetic_data"),
]
MODEL_TAG = (os.getenv("LLM_MODEL") or "default").split("/")[-1].replace(":", "-").replace(".", "")
OUT = BASE / "runs" / f"interpreter_accuracy_{MODEL_TAG}.jsonl"

METHOD_NAMES = {
    "ols": "linear_regression",
    "ipw": "propensity_score_weighting",
    "matching": "propensity_score_matching",
    "did": "difference_in_differences",
    "rdd": "regression_discontinuity_design",
    "iv": "instrumental_variable",
    "frontdoor": "frontdoor_adjustment",
}
print_lock = threading.Lock()


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


def sample_rows(per_method: int):
    picked = []
    for split, meta_path, _ in SPLITS:
        meta = pd.read_csv(meta_path)
        for method in meta["method"].dropna().astype(str).str.strip().unique():
            sub = meta[meta["method"].astype(str).str.strip() == method]
            if len(sub) == 0:
                continue
            picks = sub.iloc[np.linspace(0, len(sub) - 1, min(per_method, len(sub))).round().astype(int)]
            for _, row in picks.iterrows():
                picked.append((split, method, row))
    return picked


def judge_list(picked_list, truth_list):
    """Precision/recall of one list against the metadata list."""
    picked_list = [p for p in picked_list if p]
    truth_list = [t for t in truth_list if t]
    if not picked_list and not truth_list:
        return {"prec": None, "rec": None, "empty_both": True}
    hit = [c for c in picked_list if c in truth_list]
    prec = len(hit) / len(picked_list) if picked_list else None
    rec = len(hit) / len(truth_list) if truth_list else None
    return {"prec": prec, "rec": rec, "empty_both": False}


def run_row(args):
    split, expected_method, row, path = args
    record = {
        "split": split, "id": int(row["id"]), "dataset": row["dataset_name"],
        "expected_method": expected_method,
        "meta": {
            "treatment": blank(row.get("treatment")),
            "outcome": blank(row.get("outcome")),
            "controls": split_list(row.get("controls")),
            "instrument": split_list(row.get("instrument_var")),
            "mediator": blank(row.get("mediator")),
            "running": blank(row.get("running_var")),
            "temporal": blank(row.get("temporal_var")),
            "state": blank(row.get("state_var")),
            "is_rct": bool(row["is_rct"]) if not pd.isna(row.get("is_rct")) else None,
        },
        "picked": {}, "scores": {}, "notes": [], "status": "running",
    }
    meta = record["meta"]
    llm = get_llm_client()
    query = str(row["query"])
    description = str(row["data_description"])
    started = time.time()
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analysis = analyze_dataset(
                dataset_path=str(path), llm_client=llm,
                dataset_description=description, original_query=query,
                use_iv_pipeline=False,
            )
            query_info = {
                "query_text": query,
                "potential_treatments": analysis.get("potential_treatments", []),
                "potential_outcomes": analysis.get("potential_outcomes", []),
                "covariate_hints": None,
                "instrument_hints": None,
            }
            variables = interpret_query(query_info, analysis, dataset_description=description)

            refined = identify_controls(
                treatment_variable=variables.get("treatment_variable"),
                outcome_variable=variables.get("outcome_variable"),
                method_name=METHOD_NAMES.get(expected_method, expected_method),
                columns=analysis["columns"],
                column_categories=analysis.get("column_categories") or {},
                instrument=variables.get("instrument_variable"),
                running_variable=variables.get("running_variable"),
                time_variable=variables.get("time_variable"),
                state_variable=variables.get("group_variable"),
                query_text=query, dataset_description=description, llm=llm,
            )
        vars_picked = {
            k: variables.get(k) for k in (
                "treatment_variable", "treatment_variable_type", "outcome_variable",
                "covariates", "confounders", "instrument_variable", "mediator_variable",
                "running_variable", "cutoff_value", "time_variable", "group_variable",
                "is_rct", "is_encouragement_design",
            )
        }
        vars_picked["controls_refined"] = refined
        record["picked"] = vars_picked
        record["stdout_tail"] = buf.getvalue()[-400:]

        # ---- scoring -------------------------------------------------------
        s = {}
        s["treatment_ok"] = vars_picked["treatment_variable"] == meta["treatment"]
        s["outcome_ok"] = vars_picked["outcome_variable"] == meta["outcome"]
        instr_pick = vars_picked["instrument_variable"]
        s["instrument"] = (
            {"has_truth": False, "picked": instr_pick} if not meta["instrument"]
            else {"has_truth": True, "ok": instr_pick in meta["instrument"]}
        )
        med_pick = vars_picked["mediator_variable"]
        s["mediator_ok"] = (med_pick or None) == (meta["mediator"] or None)
        if meta["is_rct"] is not None:
            s["is_rct_ok"] = vars_picked["is_rct"] == meta["is_rct"]
        s["covariates"] = judge_list(vars_picked["covariates"] or [], meta["controls"])
        s["controls_refined"] = judge_list(refined or [], meta["controls"])
        for field, pick_key, truth_key in (
            ("running", "running_variable", "running"),
            ("time", "time_variable", "temporal"),
            ("group", "group_variable", "state"),
        ):
            truth = meta[truth_key]
            pick = vars_picked[pick_key]
            s[field] = (
                {"has_truth": True, "ok": pick == truth} if truth
                else {"has_truth": False, "picked": pick}
            )
        record["scores"] = s
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        record["trace_tail"] = traceback.format_exc()[-600:]
    record["seconds"] = round(time.time() - started, 1)
    return record


def summarize(records):
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\nrows run: {len(records)} (ok {len(ok)}, errors {len(records)-len(ok)})")

    def pct(fraction, denom):
        return f"{fraction}/{denom} ({fraction/denom:.0%})" if denom else "n/a"

    print("\nfield accuracy (vs metadata truth):")
    t_ok = sum(r["scores"]["treatment_ok"] for r in ok)
    o_ok = sum(r["scores"]["outcome_ok"] for r in ok)
    m_ok = sum(r["scores"]["mediator_ok"] for r in ok)
    print(f"  treatment   {pct(t_ok, len(ok))}")
    print(f"  outcome     {pct(o_ok, len(ok))}")
    print(f"  mediator    {pct(m_ok, len(ok))}")
    with_rct = [r for r in ok if "is_rct_ok" in r["scores"]]
    print(f"  is_rct      {pct(sum(r['scores']['is_rct_ok'] for r in with_rct), len(with_rct))}")

    instr_truth = [r for r in ok if r["scores"]["instrument"]["has_truth"]]
    instr_ok = sum(r["scores"]["instrument"]["ok"] for r in instr_truth)
    instr_nofp = [r for r in ok if not r["scores"]["instrument"]["has_truth"] and not r["scores"]["instrument"]["picked"]]
    print(f"  instrument  {pct(instr_ok, len(instr_truth))}  (no-false-positive {pct(len(instr_nofp), len(ok)-len(instr_truth))})")

    for field in ("running", "time", "group"):
        truth_rows = [r for r in ok if r["scores"][field]["has_truth"]]
        hits = sum(r["scores"][field]["ok"] for r in truth_rows)
        print(f"  {field:10s} {pct(hits, len(truth_rows))}  (rows with truth: {len(truth_rows)})")

    print("\ncovariate lists vs metadata controls (mean precision / recall):")
    for label in ("covariates", "controls_refined"):
        prec = [r["scores"][label]["prec"] for r in ok if r["scores"][label]["prec"] is not None]
        rec = [r["scores"][label]["rec"] for r in ok if r["scores"][label]["rec"] is not None]
        print(f"  {label:16s} precision {np.mean(prec):.2f} (n={len(prec)})   recall {np.mean(rec):.2f} (n={len(rec)})")

    print("\nworst treatment misses:")
    for r in ok:
        if r["scores"]["treatment_ok"]:
            continue
        print(f"  {r['split']}#{r['id']:3d} {r['dataset'][:26]:26s} truth={r['meta']['treatment']} picked={r['picked']['treatment_variable']}")
    print("\ninstrument misses:")
    for r in ok:
        s = r["scores"]["instrument"]
        if s["has_truth"] and not s["ok"]:
            print(f"  {r['split']}#{r['id']:3d} {r['dataset'][:26]:26s} truth={r['meta']['instrument']} picked={s.get('picked') or '(none)'}")
    print("\nlowest covariate recall:")
    scored = [r for r in ok if r["scores"]["controls_refined"]["rec"] is not None]
    for r in sorted(scored, key=lambda r: r["scores"]["controls_refined"]["rec"])[:8]:
        print(f"  {r['split']}#{r['id']:3d} {r['dataset'][:24]:24s} rec={r['scores']['controls_refined']['rec']:.2f} "
              f"truth={r['meta']['controls']} refined={r['picked']['controls_refined']}")
    if ok:
        secs = [r["seconds"] for r in ok]
        print(f"\ntime per row (s): median {np.median(secs):.0f}, max {max(secs):.0f} -> est full 182-row run "
              f"~{np.median(secs)*182/5/60:.0f}-{max(secs)*182/5/60:.0f} min wall (5 threads)")


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--rows-per-method", type=int, default=2)
    arg.add_argument("--limit", type=int, default=None)
    args = arg.parse_args()

    sampling = []
    for split, method, row in sample_rows(args.rows_per_method):
        path = resolve_file(SPLITS[0][2] if split == "qr" else SPLITS[1][2], row["dataset_name"])
        if path is None:
            continue
        sampling.append((split, method, row, path))
    if args.limit:
        sampling = sampling[: args.limit]
    print(f"running {len(sampling)} rows ({args.rows_per_method}/method/split), 5 workers")

    records = [None] * len(sampling)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(run_row, item): i for i, item in enumerate(sampling)}
        for fut in futures.items():
            pass
        done = 0
        for fut, idx in futures.items():
            res = fut.result()
            records[idx] = res
            done += 1
            with print_lock:
                msg = (f"[{done}/{len(sampling)}] {res['split']}#{res['id']} ({res['seconds']}s)"
                       if res["status"] != "error" else
                       f"[{done}/{len(sampling)}] {res['split']}#{res['id']} ERROR {res['error'][:80]}")
                print(msg, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    print("\nsaved:", OUT)
    summarize(records)


if __name__ == "__main__":
    main()

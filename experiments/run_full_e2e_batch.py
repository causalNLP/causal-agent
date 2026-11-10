"""Full-pipeline batch benchmark runner for CauSciBench (qr+synthetic splits).

Generalizes run_causcibench_e2e_pilot.py to all 182 resolvable rows, run in
contiguous batches with resume support: results append to one jsonl and
already-run rows (matched by split+id) are skipped on every invocation, so a
crash or a deliberate stop costs nothing on restart.

Batch mode exactly as production: no prompt_callback, interactive=False, so
information asks are skipped and gate overrides fail closed.

Usage:
  PYTHONPATH=. python experiments/run_full_e2e_batch.py --count 6          # first 6 rows
  PYTHONPATH=. python experiments/run_full_e2e_batch.py --start 6 --count 18
  PYTHONPATH=. python experiments/run_full_e2e_batch.py --workers 5
"""
import argparse
import json
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from cais.agent import CausalAgent

BASE = Path("experiments/data/causcibench")
SPLITS = [
    ("qr", BASE / "metadata_csv" / "qr_input.csv", BASE / "qrdata"),
    ("synthetic", BASE / "metadata_csv" / "synthetic_input.csv", BASE / "synthetic_data"),
]

STRICT = {
    "ols": {"linear_regression"},
    "ipw": {"propensity_score_weighting"},
    "matching": {"propensity_score_matching"},
    "did": {"difference_in_differences"},
    "rdd": {"regression_discontinuity_design"},
    "iv": {"instrumental_variable"},
    "frontdoor": {"frontdoor_adjustment"},
}
FAMILY = {
    "ols": {"linear_regression", "diff_in_means"},
    "ipw": {"propensity_score_weighting", "trimmed_ipw", "propensity_score_matching"},
    "matching": {"propensity_score_matching", "propensity_score_weighting", "trimmed_ipw"},
    "did": {"difference_in_differences", "difference_in_differences_time_window"},
    "rdd": {"regression_discontinuity_design", "donut_rdd"},
    "iv": {"instrumental_variable"},
    "frontdoor": {"frontdoor_adjustment"},
}

write_lock = threading.Lock()


def resolve_file(directory: Path, name: str):
    name = str(name)
    for candidate in (name, f"cleaned_{name}", name.replace("cleaned_", "")):
        path = directory / candidate
        if path.exists():
            return path
    return None


def build_rowlist():
    rows = []
    for split, meta_path, data_dir in SPLITS:
        meta = pd.read_csv(meta_path)
        for _, row in meta.iterrows():
            path = resolve_file(data_dir, row["dataset_name"])
            rows.append((split, row, path))
    return rows  # deterministic: qr rows in id order, then synthetic


def payload_from_output(output):
    if not isinstance(output, dict):
        return {}
    payload = output.get("results", {})
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("results")
    return nested if isinstance(nested, dict) else payload


def variables_snapshot(agent):
    variables = getattr(agent, "variables", None)
    if variables is None:
        return None
    dump = variables.model_dump() if hasattr(variables, "model_dump") else dict(variables)
    keep = [
        "treatment_variable", "treatment_variable_type", "outcome_variable", "covariates",
        "instrument_variable", "mediator_variable", "running_variable", "cutoff_value",
        "time_variable", "group_variable", "is_rct", "treatment_period_start",
    ]
    return {k: dump.get(k) for k in keep if k in dump}


def tree_summary(agent):
    tree = getattr(agent, "tree_result", None)
    if tree is None:
        return {}
    return {
        "ended": bool(tree.ended),
        "end_reason": tree.end_reason,
        "method": tree.method,
        "variant": tree.variant,
        "estimand": getattr(tree, "estimand", None),
        "path_tail": [f"{s.node}[{s.outcome}]" for s in tree.steps][-6:],
        "warnings": list(tree.warnings)[:8],
        "assumptions": {
            name: {"passed": v.passed, "reasoning": (v.reasoning or "")[:160]}
            for name, v in tree.assumptions.items()
        },
    }


def score(run, record):
    inner = payload_from_output(run.get("output"))
    effect = inner.get("effect_estimate", inner.get("causal_effect"))
    try:
        effect_value = float(effect) if effect is not None else None
    except (TypeError, ValueError):
        effect_value = None
    try:
        reference = float(record["expected_answer"])
    except (TypeError, ValueError):
        reference = None
    rel_error = None
    if effect_value is not None and reference not in (None, 0):
        rel_error = abs(effect_value - reference) / abs(reference)

    selected = getattr(run["agent"], "selected_method", None) or inner.get("method_used")
    expected = str(record["expected_method"]).lower().strip()
    return {
        "selected_method": selected,
        "effect": effect_value,
        "rel_error": rel_error,
        "method_ok": bool(selected in STRICT.get(expected, set())),
        "family_ok": bool(selected in FAMILY.get(expected, set())),
        "results": inner.get("explanation", "")[:240],
    }


def run_row(args):
    row_index, split, row, path, model, provider = args
    record = {
        "split": split, "id": int(row["id"]), "dataset": str(row["dataset_name"]),
        "expected_method": str(row["method"]),
        "expected_answer": float(row["answer"]) if not pd.isna(row["answer"]) else None,
        "model": model, "status": "running",
    }
    if path is None:
        record["status"] = "file_missing"
        record["seconds"] = 0.0
        return record
    started = time.time()
    attempt = 0
    while attempt < 2:
        attempt += 1
        try:
            agent = CausalAgent(
                dataset_path=str(path),
                dataset_description=str(row["data_description"]),
                model_name=model,
                provider=provider,
            )
            output = agent.run_analysis(query=str(row["query"]), prompt_callback=None, interactive=False)
            record.update(score({"output": output, "agent": agent}, record))
            record["tree"] = tree_summary(agent)
            record["variables"] = variables_snapshot(agent)
            record["status"] = "ok"
            break
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:220]}"
            record["trace"] = traceback.format_exc()[-700:]
            if attempt == 1:
                time.sleep(5)
    record["seconds"] = round(time.time() - started, 1)
    return record


def summarize(records, cumulative):
    run = [r for r in records if r["status"] == "ok"]
    print(f"\nthis batch: {len(records)} rows, ok {len(run)}, errors {len(records) - len(run)}")
    _print_score(run, "batch")
    if cumulative:
        with cumulative.open() as fh:
            all_rows = [json.loads(line) for line in fh if line.strip()]
        done = [r for r in all_rows if r["status"] == "ok"]
        print(f"\ncumulative: {len(all_rows)} rows run so far (ok {len(done)})")
        _print_score(done, "all")


def _print_score(rows, label):
    if not rows:
        return
    strict = sum(r.get("method_ok") for r in rows)
    family = sum(r.get("family_ok") for r in rows)
    effects = sum(r.get("effect") is not None for r in rows)
    print(f"[{label}] strict {strict}/{len(rows)}  family {family}/{len(rows)}  "
          f"estimates {effects}/{len(rows)}  ended-without-answer "
          f"{sum(r.get('tree', {}).get('ended') for r in rows)}/{len(rows)}")
    misses = [r for r in rows if not r.get("method_ok")]
    if misses:
        print(f"[{label}] misses:")
        for r in misses[:12]:
            print(f"  {r['split']}#{r['id']:3d} {r['dataset'][:24]:24s} expect={r['expected_method']:9s} "
                  f"got={str(r.get('selected_method')):28s} rel_err={r.get('rel_error')}")
    errs = [r for r in rows if r["status"] == "error"]
    if errs:
        for r in errs[:5]:
            print(f"  ERROR {r['split']}#{r['id']} {r['dataset'][:24]} {r.get('error', '')[:110]}")


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--start", type=int, default=0)
    arg.add_argument("--count", type=int, default=10)
    arg.add_argument("--workers", type=int, default=5)
    arg.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    arg.add_argument("--provider", default="openrouter")
    arg.add_argument("--out-tag", default="deepseek_v41flash")
    args = arg.parse_args()

    out_path = BASE / "runs" / f"e2e_full_{args.out_tag}.jsonl"
    rowlist = build_rowlist()
    done_keys = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["split"], r["id"]))
    pending = [
        (i, split, row, path) for i, (split, row, path) in enumerate(rowlist)
        if (split, int(row["id"])) not in done_keys and i >= args.start
    ]
    batch = pending[: args.count]
    print(f"model {args.model} | pending {len(pending)} | running rows {args.start}.. "
          f"batch of {len(batch)} with {args.workers} workers | out: {out_path}")
    if not batch:
        return

    from concurrent.futures import ThreadPoolExecutor
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_row, (i, split, row, path, args.model, args.provider))
                   for i, split, row, path in batch]
        for fut in futures:
            record = fut.result()
            records.append(record)
            with write_lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            mark = (f"{record['split']}#{record['id']} ({record['seconds']}s)"
                    if record["status"] != "error"
                    else f"{record['split']}#{record['id']} ERROR {record.get('error', '')[:70]}")
            print(mark, flush=True)

    summarize(records, out_path)


if __name__ == "__main__":
    main()

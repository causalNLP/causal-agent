"""Batch end-to-end pilot on CauSciBench rows with the current tree.

Runs the same 9 rows as the earlier pilot so results are comparable:
- synthetic: 1, 21, 41, 61, 76, 113
- qr: 19, 20, 23

No user callback: this is batch mode, so questions are skipped and gates fail
closed exactly as in a real batch run. Uses the provider/model from .env.
"""
import json
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from cais.agent import CausalAgent

BASE = Path("experiments/data/causcibench")
OUT = BASE / "runs" / "e2e_pilot_current.jsonl"
ROWS = [
    ("synthetic", 1),
    ("synthetic", 21),
    ("synthetic", 41),
    ("synthetic", 61),
    ("synthetic", 76),
    ("synthetic", 113),
    ("qr", 19),
    ("qr", 20),
    ("qr", 23),
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


def resolve_file(directory: Path, name: str):
    name = str(name)
    for candidate in (name, f"cleaned_{name}", name.replace("cleaned_", "")):
        path = directory / candidate
        if path.exists():
            return path
    return None


def payload_from_output(output):
    if not isinstance(output, dict):
        return {}
    payload = output.get("results", {})
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("results")
    return nested if isinstance(nested, dict) else payload


def tree_summary(agent):
    tree = getattr(agent, "tree_result", None)
    if tree is None:
        return {}
    assumptions = {}
    for name, verdict in tree.assumptions.items():
        assumptions[name] = {
            "passed": verdict.passed,
            "reasoning": verdict.reasoning,
        }
    return {
        "ended": bool(tree.ended),
        "end_reason": tree.end_reason,
        "method": tree.method,
        "path": [f"{step.node}[{step.outcome}]" for step in tree.steps],
        "assumptions": assumptions,
        "warnings": list(tree.warnings),
    }


def summarize(output, agent, expected_method, expected_answer):
    inner = payload_from_output(output)
    effect = inner.get("effect_estimate", inner.get("causal_effect"))
    try:
        effect_value = float(effect) if effect is not None else None
    except (TypeError, ValueError):
        effect_value = None
    try:
        reference = float(expected_answer)
    except (TypeError, ValueError):
        reference = None
    rel_error = None
    if effect_value is not None and reference is not None and reference != 0:
        rel_error = abs(effect_value - reference) / abs(reference)

    selected = getattr(agent, "selected_method", None) or inner.get("method_used")
    expected_method = str(expected_method).lower().strip()
    return {
        "selected_method": selected,
        "effect": effect_value,
        "standard_error": inner.get("standard_error"),
        "p_value": inner.get("p_value"),
        "method_used": inner.get("method_used"),
        "rel_error": rel_error,
        "method_ok": selected in STRICT.get(expected_method, set()),
        "family_ok": selected in FAMILY.get(expected_method, set()),
        "tree": tree_summary(agent),
    }


def main():
    metas = {
        "synthetic": pd.read_csv(BASE / "metadata_csv" / "synthetic_input.csv"),
        "qr": pd.read_csv(BASE / "metadata_csv" / "qr_input.csv"),
    }
    dirs = {
        "synthetic": BASE / "synthetic_data",
        "qr": BASE / "qrdata",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for split, row_id in ROWS:
        meta = metas[split]
        row = meta[meta["id"] == row_id].iloc[0]
        path = resolve_file(dirs[split], row["dataset_name"])
        record = {
            "split": split,
            "id": int(row_id),
            "dataset": str(row["dataset_name"]),
            "expected_method": str(row["method"]),
            "expected_answer": float(row["answer"]),
            "status": "running",
        }
        if path is None:
            record["status"] = "file_missing"
            records.append(record)
            print("missing file:", split, row_id, row["dataset_name"])
            continue
        started = time.time()
        try:
            agent = CausalAgent(
                dataset_path=str(path),
                dataset_description=str(row["data_description"]),
                model_name="openai/gpt-4o",
                provider="openrouter",
            )
            output = agent.run_analysis(
                query=str(row["query"]),
                prompt_callback=None,
                interactive=False,
            )
            record.update(summarize(output, agent, row["method"], row["answer"]))
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)
            record["trace"] = traceback.format_exc()
        record["seconds"] = round(time.time() - started, 1)
        records.append(record)
        print(
            f"{split}#{row_id} {row['dataset_name']}: "
            f"expected={record.get('expected_method')} "
            f"selected={record.get('selected_method')} "
            f"ended={record.get('tree', {}).get('ended')} "
            f"effect={record.get('effect')} "
            f"rel_err={record.get('rel_error')} "
            f"({record['seconds']}s)"
        )

    with OUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")

    ok = [r for r in records if r.get("status") == "ok"]
    print("\nSUMMARY")
    print("rows:", len(records), "ok:", len(ok), "errors:", len(records) - len(ok))
    print("method matches:", sum(bool(r.get("method_ok")) for r in ok), "/", len(ok))
    print("family matches:", sum(bool(r.get("family_ok")) for r in ok), "/", len(ok))
    print("estimates:", sum(r.get("effect") is not None for r in ok), "/", len(ok))
    print("ended:", sum(bool(r.get("tree", {}).get("ended")) for r in ok), "/", len(ok))
    print("saved:", OUT)


if __name__ == "__main__":
    main()

"""A/B: deterministic vs LLM-argued group-composition gate, plus one-shot baseline.

Runs CauSciBench QR row 23 (smoking2.csv, reference DiD, answer -27.349):
  1. current pipeline (deterministic group-composition check)
  2. same pipeline with the old LLM-argued check monkeypatched in
  3. the LLM-argued check's verdict directly
  4. a one-shot GPT-4o baseline (description + query -> method + effect)

Uses .env (openrouter / openai/gpt-4o).
"""
import json
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from cais.agent import CausalAgent
from cais.config import get_llm_client
from cais.models import AssumptionVariables
from cais.utils.llm_helpers import call_llm_with_json_output
import cais.methods.pre_model_assumption_utils as pm

BASE = Path("experiments/data/causcibench")
META = pd.read_csv(BASE / "metadata_csv" / "qr_input.csv")
ROW = META[META["dataset_name"] == "smoking2.csv"].iloc[0]
DATA = str(BASE / "qrdata" / "smoking2.csv")
QUERY = str(ROW["query"])
DESC = str(ROW["data_description"])
REF_METHOD = str(ROW["method"])
REF_EFFECT = float(ROW["answer"])
REF_SE = float(ROW["std_error"])
MODEL = "openai/gpt-4o"
PROVIDER = "openrouter"
OUT = BASE / "runs" / "composition_ab.json"

COMPOSITION_ASSUMPTION = (
    "Unit composition of treatment and control groups does not change as a result "
    "of treatment (no differential attrition or selective entry/exit)."
)


def summarize(agent, out):
    tr = agent.tree_result
    payload = out.get("results", {}) if isinstance(out, dict) else {}
    inner = payload.get("results", payload) if isinstance(payload, dict) else {}
    verdict = tr.assumptions.get("stable_group_composition") if tr is not None else None
    return {
        "tree_ended": bool(tr.ended) if tr is not None else None,
        "end_reason": tr.end_reason if tr is not None else None,
        "selected_method": agent.selected_method,
        "composition_passed": verdict.passed if verdict is not None else None,
        "composition_reasoning": verdict.reasoning if verdict is not None else None,
        "effect_estimate": inner.get("effect_estimate", inner.get("causal_effect")),
        "standard_error": inner.get("standard_error"),
        "confidence_interval": inner.get("confidence_interval"),
        "p_value": inner.get("p_value"),
        "method_used": inner.get("method_used"),
        "path": [f"{step.node}[{step.outcome}]" for step in tr.steps] if tr is not None else [],
        "warnings": list(tr.warnings) if tr is not None else [],
    }


def run_agent():
    agent = CausalAgent(
        dataset_path=DATA,
        dataset_description=DESC,
        model_name=MODEL,
        provider=PROVIDER,
    )
    out = agent.run_analysis(query=QUERY, prompt_callback=None, interactive=False)
    return summarize(agent, out)


def llm_composition_check(df, llm):
    variables = AssumptionVariables(
        df=df,
        treatment=str(ROW["treatment"]),
        outcome=str(ROW["outcome"]),
        time_var="year",
        group_var="state",
        dataset_description=DESC,
        variables_summary={
            "treatment": ROW["treatment"],
            "outcome": ROW["outcome"],
            "covariates": [],
            "instrument": None,
            "mediator": None,
            "running_variable": None,
        },
    )
    return pm._llm_argue_assumption(
        "Stable group composition", COMPOSITION_ASSUMPTION, variables, llm
    )


def one_shot_baseline(df, llm):
    prompt = f"""You are an expert in causal inference. Choose the most appropriate causal
inference method for the query below and estimate the causal effect.

Available methods (choose one):
- ols (linear regression / covariate adjustment)
- did (difference-in-differences)
- rdd (regression discontinuity)
- iv (instrumental variables)
- matching (propensity score matching)
- ipw (inverse probability weighting)
- frontdoor (frontdoor adjustment)

Dataset: smoking2.csv
Columns: {', '.join(df.columns)}
Dataset description:
{DESC}

Query: {QUERY}

Respond ONLY with JSON:
{{
  "method": "ols|did|rdd|iv|matching|ipw|frontdoor",
  "effect_estimate": <number, estimated causal effect on the outcome>,
  "reasoning": "<brief justification>"
}}"""
    return call_llm_with_json_output(llm, prompt)


def main():
    results = {"reference": {"method": REF_METHOD, "effect": REF_EFFECT, "std_error": REF_SE}}
    try:
        results["deterministic"] = run_agent()
    except Exception as exc:
        results["deterministic"] = {"error": repr(exc), "trace": traceback.format_exc()}

    llm = get_llm_client(provider=PROVIDER, model_name=MODEL)
    df = pd.read_csv(DATA)

    try:
        direct = llm_composition_check(df, llm)
        results["llm_composition_direct"] = {
            "passed": direct.passed,
            "reasoning": direct.reasoning,
        }
    except Exception as exc:
        results["llm_composition_direct"] = {"error": repr(exc), "trace": traceback.format_exc()}

    # Old-gate A/B: monkeypatch the old LLM-argued check back in.
    original = pm.check_stable_group_composition

    def old_llm_gate(vars, llm=None):
        return pm._llm_argue_assumption(
            "Stable group composition", COMPOSITION_ASSUMPTION, vars, llm
        )

    pm.check_stable_group_composition = old_llm_gate
    try:
        results["llm_gate_agent"] = run_agent()
    except Exception as exc:
        results["llm_gate_agent"] = {"error": repr(exc), "trace": traceback.format_exc()}
    finally:
        pm.check_stable_group_composition = original

    try:
        results["one_shot_llm"] = one_shot_baseline(df, llm)
    except Exception as exc:
        results["one_shot_llm"] = {"error": repr(exc), "trace": traceback.format_exc()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("REFERENCE:", results["reference"])
    for name in ("deterministic", "llm_gate_agent"):
        block = results[name]
        print(f"\n=== {name} ===")
        print("ended:", block.get("tree_ended"), "| reason:", block.get("end_reason"))
        print("method:", block.get("selected_method"), "| composition:", block.get("composition_passed"))
        print("effect:", block.get("effect_estimate"), "| SE:", block.get("standard_error"))
        print("composition reasoning:", str(block.get("composition_reasoning"))[:300])
    print("\n=== LLM composition (direct) ===")
    print(json.dumps(results.get("llm_composition_direct"), indent=2, default=str)[:1200])
    print("\n=== One-shot LLM ===")
    print(json.dumps(results.get("one_shot_llm"), indent=2, default=str)[:1200])
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()

"""Reproduce the interpreter's instrument-identification abstention on CauSciBench IV rows.

Background (2026-11 investigation): in the 182-row batch e2e sweep
(``experiments/data/causcibench/runs/e2e_full_deepseek_v41flash.jsonl``), 13 of
39 IV rows ended with no method because the interpreter named no instrument.
This probe shows the abstention is deterministic and instruction-induced:

* The production prompt (``IV_IDENTIFICATION_PROMPT_TEMPLATE``) requires the
  model to certify all four IV conditions. Exclusion and independence are
  untestable from column names plus prose, so an instruction-following model
  returns ``null`` every time (measured: gpt-4o 3/3 nulls, DeepSeek 3/3 nulls).
* Softening the wording to nomination semantics ("relevance is enough; the
  program verifies strength with an F-test") makes the model name the correct
  instrument (matching metadata `instrument_var`) on most rows.

Usage (requires `experiments/data/download_data.py` to have run first):
    PYTHONPATH=. python experiments/iv_abstention_probe.py                 # 13 known rows
    PYTHONPATH=. python experiments/iv_abstention_probe.py --ids 76 84
    PYTHONPATH=. python experiments/iv_abstention_probe.py --model deepseek/deepseek-v4.1-flash

Output: experiments/results/iv_abstention_probe.jsonl + a summary table.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("experiments/data/causcibench")
OUT = Path("experiments/results/iv_abstention_probe.jsonl")

PRODUCTION_HEAD = (
    "You need to be strict with the assessment.\n"
    "Do not suggest IV based on weak reasoning. "
)
SOFTENED_HEAD = (
    "Note: relevance is enough for candidacy - the downstream program will "
    "verify instrument strength statistically with an F-test and discard weak ones.\n"
    "If a column plausibly determines who received the treatment (its name or the "
    "description says so), include it even if other conditions are untestable from the data.\n"
)
PRODUCTION_TAIL = "- Select the best candidate only if it satisfies all conditions"
SOFTENED_TAIL = "- Return the best candidate for the role (not a verdict on validity)"

DEFAULT_IDS = [76, 77, 78, 80, 83, 84, 85, 86, 87, 88, 90, 91, 92]


def build_prompt(intro, description, columns, categories, query, treatment, outcome):
    from cais.components.query_interpreter import _create_identify_prompt

    prompt = _create_identify_prompt(
        "instrumental variable", query=query, description=description,
        columns=columns, categories=categories,
        treatment=treatment, outcome=outcome,
    )
    return prompt.replace(PRODUCTION_HEAD, intro).replace(PRODUCTION_TAIL, outro_of(intro))


def outro_of(intro):
    return SOFTENED_TAIL if intro == SOFTENED_HEAD else PRODUCTION_TAIL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash",
                        help="falls back to .env value if omitted via --model-follow-env")
    parser.add_argument("--model-follow-env", action="store_true")
    args = parser.parse_args()

    if not args.model_follow_env:
        os.environ["LLM_MODEL"] = args.model
        os.environ.setdefault("LLM_PROVIDER", "openrouter")

    from dotenv import load_dotenv
    load_dotenv(BASE / ".." / ".." / ".env")
    from cais.config import get_llm_client
    from cais.components.dataset_analyzer import analyze_dataset
    from cais.models import LLMIVars

    llm = get_llm_client()
    meta = pd.read_csv(BASE / "metadata_csv" / "synthetic_input.csv")
    records = []
    for rid in args.ids:
        row = meta[meta["id"] == rid].iloc[0]
        ds = str(row["dataset_name"])
        path = BASE / "synthetic_data" / ds
        analysis = analyze_dataset(dataset_path=str(path), llm_client=None)
        prompt_kwargs = dict(
            query=str(row["query"]), description=str(row["data_description"]),
            columns=analysis["columns"], categories=analysis["column_categories"],
            treatment=str(row["treatment"]), outcome=str(row["outcome"]),
        )
        truth = str(row.get("instrument_var"))
        print(f"#{rid} {ds[:22]:22s} roster={list(analysis['columns'])}")
        print(f"      truth instrument (metadata): {truth}")
        record = {"id": int(rid), "dataset": ds, "truth": truth, "trials": {}}
        for label, head in (("production", PRODUCTION_HEAD), ("softened", SOFTENED_HEAD)):
            prompt = build_prompt(head, **prompt_kwargs)
            etc = []
            for _ in range(args.trials):
                out = llm.with_structured_output(LLMIVars).invoke(prompt)
                etc.append(out.instrument_variable)
            record["trials"][label] = etc
            print(f"      {label:10s} x{args.trials}: {etc}")
        records.append(record)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nsummary (softened-prompt agreement with metadata truth):")
    recovered = sum(
        len({p for p in r["trials"]["softened"]}) == 1 and list({p for p in r["trials"]["softened"]})[0] in r["truth"]
        for r in records
    )
    print(f"  rows where softened naming matches truth: {recovered}/{len(records)}")
    still_silent = [r["dataset"] for r in records if len({p for p in r['trials']['softened']}) == 1 and list({p for p in r['trials']['softened']})[0] is None]
    print(f"  rows still silent even softened: {still_silent}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()

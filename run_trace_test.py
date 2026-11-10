"""Selection-boundary e2e trace: real data + real GPT-4o, up to and including tree traversal.

Replicates test_e2e_did.py's query/description, then runs analyse_dataset() +
select_method_v2() and dumps the full decision-tree trace (path, steps, check
verdicts, warnings). Stops before controls/cleaning/execution.
"""
import os, json
from dotenv import load_dotenv
load_dotenv()

from cais.agent import CausalAgent

QUERY = "What was the impact of cigarette taxation rules on cigarette sales in California?"
DESCRIPTION = (
    "To estimate the effect of cigarette taxation on its consumption, data from "
    "cigarette sales were collected and analyzed across 39 states in the United States "
    "from the years 1970 to 2000. Proposition 99, a Tobacco Tax and Health Protection Act "
    "passed in California in 1988, imposed a 25-cent per pack state excise tax on tobacco "
    "cigarettes and implemented additional restrictions. We aim to determine if the "
    "imposition of this tax and the subsequent regulations led to a reduction in cigarette "
    "sales. The data is in the CSV file smoking2.csv."
)

agent = CausalAgent(
    dataset_path="tests/test_data/smoking2.csv",
    dataset_description=DESCRIPTION,
    model_name="openai/gpt-4o",
    provider="openrouter",
)

print("=== Stage 1: analyse_dataset ===")
agent.analyse_dataset(query=QUERY)
print("Treatment:", agent.variables.treatment_variable)
print("Outcome:", agent.variables.outcome_variable)
print("Time var:", agent.variables.time_variable, "| Group var:", agent.variables.group_variable)
print("is_rct:", agent.variables.is_rct)
print("Covariates:", agent.variables.covariates)

print("\n=== Stage 2: select_method_v2 (tree traversal) ===")
agent.select_method_v2(query=QUERY)
tr = agent.tree_result
print("Selected method:", tr.method)
print("Implemented:", tr.implemented, "| Ended:", tr.ended, "| End reason:", tr.end_reason)
print("\nPath:")
print(" ", tr.path)
print("\nSteps:")
for s in tr.steps:
    print(f"  [{s.kind:9s}] {s.node}  ->  {s.outcome}")
print("\nCheck verdicts:")
for name, r in tr.assumptions.items():
    print(f"  {name}: passed={r.passed}")
    if r.reasoning:
        print(f"    reasoning: {r.reasoning[:220]}")
print("\nWarnings:", tr.warnings or "(none)")

print("\n=== logged to trace_output.json ===")
with open("trace_output.json", "w", encoding="utf-8") as f:
    json.dump(tr.to_dict(), f, indent=2, default=str)

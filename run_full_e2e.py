"""Full-pipeline e2e on smoking2.csv with GPT-4o via OpenRouter.

Uses run_analysis() defaults (tree_v2 selection). prompt_callback simulates an
interactive user: answers 'y' (proceed) to gate-override consultations and
'skip' to information asks, logging everything it was asked.
"""
import json, os
from dotenv import load_dotenv
load_dotenv()

from cais.agent import CausalAgent

QUERY = "What was the impact of cigarette taxation rules on cigarette sales in California?"
DESCRIPTION = (
    "To estimate the effect of cigarette taxation on its consumption, data from "
    "cigarette sales were collected and analyzed across 39 states in the United States "
    "from the years 1970 to 2000. Proposition 99, a Tobacco Tax and Health Protection Act "
    "passed in California in 1988, imposed a 25-cent per pack state excise tax on tobacco "
    "cigarettes and implemented additional restrictions, including the ban on cigarette "
    "vending machines. We aim to determine if the imposition of this tax and the "
    "subsequent regulations led to a reduction in cigarette sales. The data is in the CSV "
    "file smoking2.csv."
)

ASKED = []

def simulated_user(question: str):
    ASKED.append(question)
    q = (question or "").lower()
    if "override" in q or "proceed" in q or "continue" in q:
        return "y"
    return "skip"

agent = CausalAgent(
    dataset_path="tests/test_data/smoking2.csv",
    dataset_description=DESCRIPTION,
    model_name="openai/gpt-4o",
    provider="openrouter",
)

out = agent.run_analysis(query=QUERY, prompt_callback=simulated_user, interactive=True)

tr = agent.tree_result
print("############ TREE ############")
if tr is not None:
    print("Path:", tr.path)
    print("Warnings:", tr.warnings)
    print("Verdicts:")
    for name, r in tr.assumptions.items():
        print(f"  {name}: passed={r.passed}")
elif agent.method_info is not None:
    print("Legacy method_info:", agent.method_info)

print("############ SELECTED ############")
print("selected_method:", agent.selected_method)

print("############ EFFECT ############")
res = out.get("results", {}) if isinstance(out, dict) else {}
inner = res.get("results", res) if isinstance(res, dict) else {}
for k in ("effect_estimate", "causal_effect", "standard_error", "confidence_interval", "p_value", "method_used"):
    v = inner.get(k) if isinstance(inner, dict) else None
    print(f"  {k}: {v}")

print("############ QUESTIONS ASKED OF USER ############")
for q in ASKED:
    print(" -", q[:200])

print("############ EXPLANATION (first 1200 chars) ############")
expl = out.get("explanation")
print((expl if isinstance(expl, str) else json.dumps(expl, indent=2))[:1200])

with open("full_e2e_output.json", "w", encoding="utf-8") as f:
    json.dump({"tree": tr.to_dict() if tr else None,
               "selected": agent.selected_method,
               "results": {k: str(v) for k, v in inner.items()} if isinstance(inner, dict) else {},
               "asked": ASKED}, f, indent=2, default=str)

"""End-to-end run of the agent on the IHDP data, with a scripted LLM.

Every LLM call in the pipeline is answered from a fixed script (see
``fake_llm.py``), so the test needs no API key and no network. The statistics
(decision tree gates, propensity model, weighting) run for real on the data.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from cais.agent import CausalAgent

from .fake_llm import ScriptedChatModel, install_llm

DATA_PATH = Path(__file__).resolve().parents[1] / "test_data" / "ihdp_1.csv"

QUERY = "What is the effect of home visits from specialist doctors on the cognitive scores of premature infants?"
DESCRIPTION = (
    "The CSV file ihdp_1.csv contains data obtained from the Infant Health and Development Program (IHDP). "
    "The study is designed to evaluate the effect of home visit from specialist doctors on the cognitive test "
    "scores of premature infants. The confounders x (x1-x25) correspond to collected measurements of the "
    "children and their mothers, including measurements on the child (birth weight, head circumference, weeks "
    "born preterm, birth order, first born, neonatal health index, sex, twin status), as well as behaviors "
    "engaged in during the pregnancy (smoked cigarettes, drank alcohol, took drugs) and measurements on the "
    "mother at the time she gave birth (age, marital status, educational attainment, whether she worked during "
    "pregnancy, whether she received prenatal care) and the site (8 total) in which the family resided at the "
    "start of the intervention. There are 6 continuous covariates and 19 binary covariates."
)

EXPECTED_EFFECT = 4.05
TOLERANCE = 0.5

COVARIATES = [f"x{i}" for i in range(1, 26)]
INTERPRETATION = "Scripted interpretation: home visits raise the cognitive score."

CLEANING_SCRIPT = (
    "import pandas as pd\n"
    "df = pd.read_csv(__DATASET_PATH__)\n"
    "df.to_csv(__CLEANED_PATH__, index=False)\n"
    "print(len(df))\n"
)


def _selected_variable(prompt: str) -> dict:
    return {"variable_name": "treatment" if "identify the treatment variable" in prompt else "y"}


def scripted_ihdp_llm() -> ScriptedChatModel:
    """The fixed answers for each pipeline stage on the IHDP query."""
    return ScriptedChatModel(
        structured={
            "LLMSelectedVariable": _selected_variable,
            "LLMSelectedCovariates": {"covariates": COVARIATES, "reasoning": "x1-x25 are pre-treatment covariates"},
            "LLMRCTCheck": {"is_rct": False, "reasoning": "Observational comparison."},
            "LLMIVars": {"instrument_variable": None, "candidates": []},
            "LLMRDDVars": {"running_variable": None, "cutoff_value": None, "treat_above_cutoff": None},
            "LLMEstimand": {"estimand": "ATE"},
            "LLMDesignContext": {"reasoning": "Nothing to extract."},
            "LLMTreatmentReferenceLevel": {"reference_level": None, "reasoning": "Binary treatment."},
            "LLMInteractionSuggestion": {"interaction_needed": False, "interaction_variable": None, "reasoning": "None."},
        },
        text_rules=[
            ("Identify potential treatment and outcome variables",
             json.dumps({"potential_treatments": ["treatment"], "potential_outcomes": ["y"]})),
            ("Identify potential instrumental variables", "[]"),
            ("identify variables for Difference-in-Differences",
             json.dumps({"time_variable": None, "unit_variable": None, "did_canonical": None,
                         "did_term": None, "treatment_time": None, "treatment_state": None})),
            ("Assess whether the following assumption is",
             json.dumps({"passed": True, "reasoning": "Plausible for this study.", "missing_info": None})),
            ("CausalPrep-Planner",
             json.dumps({"method": "trimmed_ipw", "roles": {"treatment": "treatment", "outcome": "y"}})),
            ("CausalPrep-Codegen", CLEANING_SCRIPT),
            ("Interpret causal inference results", INTERPRETATION),
        ],
    )


@pytest.fixture
def scripted_llm(monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "TOGETHER_API_KEY",
                "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return install_llm(monkeypatch, scripted_ihdp_llm())


def test_ihdp_e2e(scripted_llm, tmp_path):
    dataset = tmp_path / "ihdp_1.csv"
    shutil.copy(DATA_PATH, dataset)

    agent = CausalAgent(dataset_path=str(dataset), dataset_description=DESCRIPTION)
    output = agent.run_analysis(QUERY, interactive=False)

    assert scripted_llm.unscripted == []

    assert agent.variables.treatment_variable == "treatment"
    assert agent.variables.outcome_variable == "y"

    assert agent.selected_method in {"propensity_score_matching", "trimmed_ipw", "propensity_score_weighting"}

    results = output["results"]
    assert results["effect_estimate"] == pytest.approx(EXPECTED_EFFECT, abs=TOLERANCE)
    low, high = results["confidence_interval"]
    assert low < results["effect_estimate"] < high

    explanation = output["explanation"]
    assert re.search(r"Method Used:\**\s*\S+", explanation["final_explanation_text"])
    assert explanation["interpretation_text"] == INTERPRETATION

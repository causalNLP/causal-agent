# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CAIS (Causal AI Scientist)** is an LLM-powered tool that automates causal inference end-to-end: it takes a natural language causal query + dataset, selects the appropriate causal inference method via a decision tree, executes it, validates assumptions, and interprets results in plain language.

## Environment Setup

```bash
cp .env.example .env          # Add your LLM API keys
conda create -n cais python=3.10
conda activate cais
pip install -r requirements.txt
pip install -e .               # Install cais as editable package
```

Required API keys (in `.env`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`

Optional: `OPENROUTER_API_KEY` for OpenRouter. LiteLLM provider manages its own credentials — no additional key needed in `.env`.

## Commands

```bash
# Run tests
pytest tests/                                          # All tests
pytest tests/cais/components/                         # Component tests only
pytest tests/cais/methods/difference_in_differences/  # Single method tests
pytest tests/cais/test_e2e_did.py                     # Single test file

# Run CAIS batch evaluation
python run_cais_new.py \
    --metadata_path data/qr_info.csv \
    --data_dir data/all_data \
    --output_dir output \
    --output_name results.json \
    --llm_name gpt-4o-mini \
    --llm_provider openai

# CLI (after pip install -e .)
cais run --query "Does X cause Y?" --data path/to/data.csv
cais batch --metadata path/to/metadata.csv
```

## Architecture

### Pipeline Stages

The `CausalAgent` (`cais/agent.py`) orchestrates a sequential tool-based pipeline:

1. **Input parsing** → Parse query, dataset path/DataFrame, description
2. **Dataset analysis** → Detect columns, sample size, temporal structure, instruments, discontinuities
3. **Query interpretation** → Extract treatment, outcome, confounders from natural language
4. **Method selection** → Traverse decision tree (`components/decision_tree.py`) or LLM-based (`decision_tree_llm.py`) to select method
5. **Controls selection** → Identify covariates to control for
6. **Method validation** → Check method assumptions (`components/assumption_checks.py`)
7. **Data cleaning** → Preprocess for estimation
8. **Method execution** → Run the selected estimator
9. **Explanation generation** → LLM interprets numerical results
10. **Output formatting** → Return JSON with effect estimate, CI, p-value, explanation

Each stage is implemented as both a `component` and a LangChain `tool` wrapper in `cais/tools/`.

### Key Modules

| Module | Role |
|--------|------|
| `cais/agent.py` | `CausalAgent` class — main entry point, orchestrates all tools |
| `cais/estimator_lib.py` | `Estimators` registry — dict-like container for all causal methods |
| `cais/methods/causal_method.py` | `CausalMethod` ABC — interface all estimators must implement |
| `cais/models.py` | Pydantic models for typed data flow (`Variables`, `MethodInfo`, `DatasetAnalysis`, etc.) |
| `cais/config.py` | `get_llm_client()` — initializes LLM from provider/model_name or env vars |
| `cais/components/` | Core logic for each pipeline stage |
| `cais/tools/` | LangChain `Tool` wrappers around components |
| `cais/prompts/` | LLM prompt strings for each stage |
| `cais/iv_llm/` | LLM-assisted IV discovery module (hypothesizer, critics, confounder miner) |

### IV-LLM Module

`cais/iv_llm/` is a sub-package for LLM-assisted instrumental variable discovery. It is invoked via `IVDiscovery` in `cais/components/iv_discovery.py` and exposes:

- `src/agents/hypothesizer.py` — proposes candidate IVs for a given treatment/outcome pair
- `src/agents/confounder_miner.py` — identifies confounders from context
- `src/critics/exclusion_critic.py` — checks the exclusion restriction for proposed IVs
- `src/critics/independence_critic.py` — checks instrument independence
- `src/llm/client.py` — `LLMClient` thin adapter wrapping `get_llm_client()` for use within the module
- `src/experiments/iv_co_scientist.py` — orchestration experiment combining agents and critics
- `src/prompts/` — prompt templates loaded via `PromptLoader`

### Causal Methods

All methods inherit from `CausalMethod` ABC and implement `validate_assumptions()` and `estimate_effect()`. Each method lives in its own subdirectory under `cais/methods/`:

- `linear_regression/` — OLS for RCTs and observational data with controls
- `difference_in_differences/` — DiD for panel data with treatment/control groups over time
- `regression_discontinuity/` — RDD for sharp cutoff-based treatment assignment
- `instrumental_variable/` — IV/2SLS for endogeneity with valid instruments
- `propensity_score/` — Matching and weighting for observational studies
- `backdoor_adjustment/` — Graph-based adjustment using DoWhy

Each method directory typically contains: `estimator.py`, `diagnostics.py`, `llm_assist.py`.

### Adding a Custom Estimator

```python
from cais.methods.causal_method import CausalMethod

class MyMethod(CausalMethod):
    name = "my_method"
    description = "..."
    assumptions = [...]

    def validate_assumptions(self, df, variables): ...
    def estimate_effect(self, df, variables, query=None): ...

agent = CausalAgent(dataset_path="data.csv", ...)
agent.estimators.add_estimator(MyMethod.name, MyMethod())
```

### LLM Provider Configuration

`cais/config.py` reads from env vars if provider/model_name not specified. Supported providers: `openai`, `anthropic`, `together`, `gemini`, `deepseek`, `openrouter`, `litellm`.

The `litellm` provider uses `langchain_litellm.ChatLiteLLM` and requires no API key configuration in `config.py` — LiteLLM handles routing and credentials internally. This allows using any model supported by LiteLLM (local models, proxies, etc.) without adding provider-specific code.

### Data Conventions

- Metadata CSVs (`data/qr_info.csv`, `real_info.csv`, `synthetic_info.csv`) define: filename, description, causal query, reference effect, intended method
- `CausalAgent` accepts either a file path or a `pd.DataFrame` directly as `dataset_path`
- Output is JSON with keys: `effect_estimate`, `confidence_interval`, `p_value`, `method`, `explanation`

## Test Structure

Tests mirror the source structure under `tests/cais/`:
- `tests/cais/components/` — unit tests per component
- `tests/cais/methods/<method>/` — unit tests per estimator
- `tests/cais/test_e2e_*.py` — end-to-end tests on real datasets (DiD, IV, RDD, IHDP)
- `tests/cais/test_agent_workflow.py` — full agent execution tests

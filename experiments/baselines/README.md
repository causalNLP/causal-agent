## CAIS Baselines 

This module provides a runnable baseline (CAIS: Causal AI Scientist) that uses an LLM to plan and execute causal analysis code inside a sandboxed Python environment (Docker). It supports multiple query formats (standard, Veridical, Program-of-Thoughts, ReAct, and a sequential thinking workflow) and multiple LLM providers (OpenAI, Azure OpenAI, Vertex, Together).

### Directory

- `run_baselines.py`: CLI entrypoint to run the baseline
- `cais_baseline.py`: Orchestrates LLM ↔ code execution workflow
- `query_formats.py`: Prompt templates and formats
- `chatbot.py`: LLM client adapters (OpenAI, Azure, Vertex, Together, Test, RPC)
- `coderunner.py`: Safe code execution via Docker; persistent session support via HTTP server
- `kernel_http.py`: HTTP execution server used for persistent sessions
- `Dockerfile.http`: Persistent execution server image
- `docker_dependencies.txt`: Python libs installed into the Docker images for code execution
- `method_explanations.txt`: Optional method explanations included in prompts when enabled

## Prerequisites

- Python 3.10+ (local host)
- Docker (required for safe code execution)
- Provider credentials (optional, depending on `--api`)

Install Python dependencies (host):
```bash
pip install -r requirements.txt
```

Build the Docker images used by the baseline code runner:
```bash


docker build -t python-causalscientist-http \
  -f baselines/Dockerfile.http \
  baselines
```

These images install the libraries listed in `docker_dependencies.txt` (e.g., pandas, numpy, statsmodels, dowhy). If your LLM-generated code needs extra libraries, add them to that file and rebuild.

## Data layout

Place CSVs under one of:
- `data/all_data` (qrdata)
- `data/real_data`
- `data/synthetic_data`

In your queries JSON, set `dataset_path` to just the filename (the runner will prepend the base path according to `--data-type`).

Example queries file:
```json
[
  {
        "query": "What is the effect of home visits from specialist doctors on the cognitive scores of premature infants on the control?",
        "method": "propensity_score_weighting",
        "answer": 4.02,
        "dataset_description": "The CSV file ihdp_0.csv contains data obtained from the Infant Health and Development Program (IHDP). The study is designed to evaluate the effect of home visit from specialist doctors on the cognitive test scores of premature infants. The confounders x (x1-x25) correspond to collected measurements of the children and their mothers, including measurements on the child (birth weight, head circumference, weeks born preterm, birth order, first born, neonatal health index, sex, twin status), as well as behaviors engaged in during the pregnancy (smoked cigarettes, drank alcohol, took drugs) and measurements on the mother at the time she gave birth (age, marital status, educational attainment, whether she worked during pregnancy, whether she received prenatal care) and the site (8 total) in which the family resided at the start of the intervention. There are 6 continuous covariates and 19 binary covariates.",
        "dataset_path": "ihdp_0.csv",
        "keywords": "Causality, Average treatment effect, Observational data"
    }
]
```

## Environment variables (providers)

- OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`
- Vertex (Google): `PROJECT_ID`, `LOCATION`
- Together: `TOGETHER_API_KEY`

## Run



OpenAI example (persistent session enabled):
```bash
python baselines/run_baselines.py \
  --queries baselines/sample_queries.json \
  --output runs/output.json \
  --api openai \
  --model gpt-4o-mini \
  --persistent
```


## CLI options 

- `--queries`: Path to queries file (`.json` or `.csv`). For CSV, columns will be renamed to `{query, dataset_description, dataset_path}`
- `--output`: Path to save results JSON
- `--data-type`: One of `qrdata`, `real`, `synthetic` (controls dataset base path)
- `--api`: `openai`, `azure`, `vertex`, `together`, `test`, `local` (local not implemented)
- `--model`: Provider-specific model id (e.g., `gpt-4o-mini`, `google/gemini-1.5-flash-002`)
- `--persistent`: Use a persistent Python environment (HTTP server inside Docker)
- Prompting modes (pick any):
  - `--veridical` (Veridical Data Science)
  - `--sequential` (sequential thinking workflow)
  - `--potm` (Program of Thoughts)
  - `--react` (ReAct)
- `--session-timeout`: Timeout for persistent sessions (seconds)

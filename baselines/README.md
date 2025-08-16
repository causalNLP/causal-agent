## CAIS Baselines (Causal AI Scientist)

This module provides a runnable baseline (CAIS: Causal AI Scientist) that uses an LLM to plan and execute causal analysis code inside a sandboxed Python environment (Docker). It supports multiple query formats (standard, Veridical, Program-of-Thoughts, ReAct, and a sequential thinking workflow) and multiple LLM providers (OpenAI, Azure OpenAI, Vertex, Together).

### Directory

- `run_baselines.py`: CLI entrypoint to run the baseline
- `cais_baseline.py`: Orchestrates LLM ↔ code execution workflow
- `query_formats.py`: Prompt templates and formats
- `chatbot.py`: LLM client adapters (OpenAI, Azure, Vertex, Together, Test, RPC)
- `coderunner.py`: Safe code execution via Docker; persistent session support via HTTP server
- `kernel_http.py`: HTTP execution server used for persistent sessions
- `Dockerfile.runner`: One-off execution image for running Python code
- `Dockerfile.http`: Persistent execution server image
- `docker_dependencies.txt`: Python libs installed into the Docker images for code execution
- `method_explanations.txt`: Optional method explanations included in prompts when enabled

## Prerequisites

- Python 3.10+ (local host)
- Docker (required for safe code execution)
- Provider credentials (optional, depending on `--api`)

Install Python dependencies (host):
```bash
pip install -r /Users/vishal/Projects/causal-agent/requirements.txt
```

Build the Docker images used by the baseline code runner:
```bash
docker build -t python-causalscientistrunner \
  -f /Users/vishal/Projects/causal-agent/baselines/Dockerfile.runner \
  /Users/vishal/Projects/causal-agent/baselines

docker build -t python-causalscientist-http \
  -f /Users/vishal/Projects/causal-agent/baselines/Dockerfile.http \
  /Users/vishal/Projects/causal-agent/baselines
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
    "query": "Estimate the ATE of treatment T on outcome Y controlling for X1,X2.",
    "dataset_description": "Short description.",
    "dataset_path": "ihdp_0.csv"
  }
]
```

## Environment variables (providers)

- OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`
- Vertex (Google): `PROJECT_ID`, `LOCATION`
- Together: `TOGETHER_API_KEY`

## Run

Basic smoke test (uses a test chatbot; still needs Docker images for code execution):
```bash
python /Users/vishal/Projects/causal-agent/baselines/run_baselines.py \
  --queries /Users/vishal/Projects/causal-agent/baselines/sample_queries.json \
  --output /Users/vishal/Projects/causal-agent/runs/output.json \
  --api test
```

OpenAI example (persistent session enabled):
```bash
python /Users/vishal/Projects/causal-agent/baselines/run_baselines.py \
  --queries /Users/vishal/Projects/causal-agent/baselines/sample_queries.json \
  --output /Users/vishal/Projects/causal-agent/runs/output.json \
  --api openai \
  --model gpt-4o-mini \
  --persistent
```

Vertex example:
```bash
python /Users/vishal/Projects/causal-agent/baselines/run_baselines.py \
  --queries /Users/vishal/Projects/causal-agent/baselines/sample_queries.json \
  --output /Users/vishal/Projects/causal-agent/runs/output.json \
  --api vertex \
  --model google/gemini-1.5-flash-002
```

## CLI options (high level)

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
  - `--method-explanation` (include `method_explanations.txt` in prompt)
- `--session-timeout`: Timeout for persistent sessions (seconds)

## How it works (brief)

1) LLM receives a prompt (from selected query format) that includes dataset info
2) LLM returns Python code inside a single fenced block
3) Code is executed in a container (one-off or persistent HTTP server)
4) Output is fed back to the LLM for analysis/corrections
5) Final result is summarized as JSON and saved to `--output`

## Troubleshooting

- Invalid model ID: Ensure `--model` matches the chosen `--api` provider
- Import/module errors during code execution:
  - Add the library to `baselines/docker_dependencies.txt` and rebuild the images
- Dataset not found: Ensure your queries JSON uses only the filename and that the file exists under `data/<type>/`
- Persistent server doesn’t start: Check Docker daemon and build the `python-causalscientist-http` image; port 8888 must be free
- Slow runs: Large CSVs lead to long `df.describe()`/`head()` prints. Consider sampling your data.

## Extending

- Add/modify prompt formats in `query_formats.py`
- Add provider adapters in `chatbot.py`
- Add runtime libs in `docker_dependencies.txt` and rebuild images



<h1 align="center">
<img src="blob/main/asset/cais.png" width="400" alt="CAIS" />
<br>
Causal AI Scientist: Facilitating Causal Data Science with Large Language Models
</h1>

<p align="center">
  <a href="https://github.com/causalNLP/causal-agent"><b>[Code]</b></a> •
  <a href="https://drive.google.com/file/d/1KvxAVIE3Ooe5V392Gw-1ZBAlQe39VVQI/view?usp=drive_link"><b>[Paper]</b></a>
</p>

**Causal AI Scientist (CAIS)** is an LLM-powered tool for generating data-driven answers to natural language causal queries. Given a natural language query (e.g., *"Does participating in a job training program lead to higher income?"*), an accompanying dataset, and its description, CAIS frames a suitable causal estimation problem, selects an appropriate inference method, executes it, runs diagnostic checks, and interprets the results in plain language.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Pipeline](#2-pipeline)
3. [Getting Started](#3-getting-started)
4. [Running CAIS](#4-running-cais)
5. [Reproducing Paper Results](#5-reproducing-paper-results)
6. [Citation](#6-citation)
7. [License](#7-license)

---

## 1. Introduction

Causal effect estimation is central to evidence-based decision-making across domains such as the social sciences, healthcare, and economics, but it requires substantial methodological expertise to apply correctly.

**CAIS** automates this process end-to-end using Large Language Models (LLMs) to:
- Parse a natural language causal query and analyze dataset characteristics.
- Select an appropriate causal inference method via a decision tree and structured prompting.
- Execute the method using predefined code templates and validate the results.
- Interpret the numerical output in the context of the original query.

**Supported Methods:**
- **Econometric:** Difference-in-Differences (DiD), Instrumental Variables (IV), Ordinary Least Squares (OLS), Regression Discontinuity Design (RDD).
- **Causal Graph-based:** Backdoor adjustment, Frontdoor adjustment.

---

## 2. Pipeline

CAIS consists of four successive stages, powered by a decision-tree-driven reasoning pipeline:

### Stage 1 — Data Preprocessing & Query Decomposition
- Profiles the dataset (column types, missing values, statistical distributions) and uses an LLM to identify treatment, outcome, and covariate variables.
- Scans for method-specific variables such as instruments and running variables based on the dataset description and causal query.

### Stage 2 — Method Selection
- Traverses a rule-based decision tree that evaluates dataset properties (e.g., randomization, presence of temporal structure, availability of instruments) to select a valid causal inference method.
- Breaking selection into explicit, verifiable steps ensures interpretability and avoids the opacity of direct LLM-based method selection.

### Stage 2b — IV-LLM Pipeline *(activated when IV is selected)*

If the **Instrumental Variable (IV)** method is selected and the `--iv_llm` pipeline is enabled (based on [IV Co-Scientist](https://arxiv.org/abs/2602.07943)):
1. **Hypothesis Generation**: The LLM hypothesizes potential instruments based on dataset context and variable names.
2. **Confounder Mining**: Identifies potential confounders that might violate the independence or exclusion restrictions.
3. **Critic Validation**: Uses specialized LLM "critics" (Exclusion, Independence) to reason about the validity of each candidate instrument.
4. **Final Selection**: Selects the most robust instrument for the estimation stage.

### Stage 3 — Validation
- Runs standard statistical assumption checks for the selected method (e.g., the F-statistic for IV, covariate balance for OLS).
- If any check fails, initiates a **feedback loop back to Stage 2**, incorporating information from the failure to skip the invalid method and identify the next plausible candidate.

### Stage 4 — Method Execution & Interpretation
- Executes the chosen method using predefined Python code templates with placeholders substituted from Stage 1, maximizing reliability over LLM-generated code.
- Prompts an LLM to interpret the estimated causal effect, standard error, and confidence interval in the context of the original query, alongside validation caveats and a clear statement of assumptions and limitations.

---

## 3. Getting Started

**Prerequisites:**
- Python 3.10
- Conda (recommended)

**Step 1: Clone the repository and copy the example configuration**
```bash
git clone https://github.com/causalNLP/causal-agent.git
cd causal-agent
cp .env.example .env
```

Open `.env` and set the API keys for the provider(s) you intend to use:

```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
TOGETHER_API_KEY=your_together_key_here
```

Only the key corresponding to your chosen `--llm_provider` is required.

> ⚠️ Keep your `.env` file secure and never commit it to version control.

**Step 2: Create and activate a Python 3.10 environment**
```bash
conda create -n cais python=3.10
conda activate cais
pip install -r requirements.txt
```

**Step 3: Install the CAIS library**
```bash
pip install -e .
```

---


## 4. Reproducing Paper Results

### Step 1 — Download the dataset

The CauSciBench benchmark dataset is hosted on [HuggingFace](https://huggingface.co/datasets/causal-nlp/causcibench). To download it:

```bash
python experiments/download_data.py
```

This will populate `experiments/data/causcibench/` with the following structure:

| Path | Description |
|---|---|
| `data/real_data/` | Real-world study datasets |
| `data/synthetic_data/` | Synthetic datasets |
| `data/qrdata/` | Textbook examples |
| `data/real_info.csv` | Metadata for real-world datasets |
| `data/synthetic_info.csv` | Metadata for synthetic datasets |
| `data/qr_info.csv` | Metadata for textbook examples |

> ⚠️ The dataset is 583MB. Ensure you have sufficient disk space before downloading.

### Step 2 — Run the evaluation

```bash
python experiments/scripts/run_cais.py \
    --metadata_path <path_to_metadata_csv> \
    --data_dir <path_to_data_folder> \
    --output_dir <output_folder> \
    --output_name <output_filename> \
    --llm_name <llm_name> \
    --llm_provider <llm_provider> \
    [--iv_llm]
```
**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `--metadata_path` | `str` | Path to the CSV file containing queries, dataset descriptions, and filenames |
| `--data_dir` | `str` | Path to the folder containing the data in CSV format |
| `--output_dir` | `str` | Path to the folder where output JSON results will be saved |
| `--output_name` | `str` | Name of the output JSON file |
| `--llm_name` | `str` | Name of the LLM to use (e.g., `gpt-4o`, `claude-3-5-sonnet`) |
| `--llm_provider` | `str` | LLM service provider (e.g., `openai`, `anthropic`, `together`) |
| `--iv_llm` | `bool` | *(Optional)* If present, enables the advanced experimental [IV-LLM pipeline](https://arxiv.org/abs/2602.07943) for instrument discovery. |

**Example:**
```bash
python experiments/scripts/run_cais.py \
    --metadata_path "experiments/data/causcibench/data/qr_info.csv" \
    --data_dir "experiments/data/causcibench/data/qrdata" \
    --output_dir "experiments/outputs" \
    --output_name "results_qr_4o" \
    --llm_name "gpt-4o-mini" \
    --llm_provider "openai" \
    --iv_llm
```

---

## 6. Citation

If you use CAIS or build on this work, we would appreciate it if you could cite:

```bibtex
@inproceedings{
verma2025causal,
title={Causal {AI} Scientist: Facilitating Causal Data Science with Large Language Models},
author={Vishal Verma and Sawal Acharya and Devansh Bhardwaj and Samuel Simko and Yongjin Yang and Anahita Haghighat and Dominik Janzing and Mrinmaya Sachan and Bernhard Sch{\"o}lkopf and Zhijing Jin},
booktitle={NeurIPS 2025 Workshop on CauScien: Uncovering Causality in Science},
year={2025},
url={https://openreview.net/forum?id=EDWTHMVOCj}
}
```

The IV-LLM pipeline builds on the methodology introduced in [IV Co-Scientist](https://arxiv.org/abs/2602.07943). If you use that component, please also cite:

```bibtex
@misc{sheth2026ivcoscientistmultiagentllm,
      title={IV Co-Scientist: Multi-Agent LLM Framework for Causal Instrumental Variable Discovery}, 
      author={Ivaxi Sheth and Zhijing Jin and Bryan Wilder and Dominik Janzing and Mario Fritz},
      year={2026},
      eprint={2602.07943},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.07943}
}
```

---

## 7. License

Distributed under the MIT License. See `LICENSE` for more information.

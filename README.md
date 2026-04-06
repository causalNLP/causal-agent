<h1 align="center">
<img src="blob/main/asset/cais.png" width="400" alt="CAIS" />
<br>
Causal AI Scientist: Facilitating Causal Data Science with Large Language Models
</h1>

<p align="center">
  <a href="https://github.com/your-repo/cais"><b>[Code]</b></a> •
  <a href=""><b>[Paper (coming soon)]</b></a>
</p>

**Causal AI Scientist (CAIS)** is an LLM-powered tool for generating data-driven answers to natural language causal queries. Given a natural language query (e.g., *"Does participating in a job training program lead to higher income?"*), an accompanying dataset, and its description, CAIS frames a suitable causal estimation problem, selects an appropriate inference method, executes it, runs diagnostic checks, and interprets the results in plain language.

> **Note:** This repository is a work in progress and will be updated with additional instructions and files.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Pipeline](#2-pipeline)
3. [Getting Started](#3-getting-started)
4. [Dataset Information](#4-dataset-information)
5. [Running CAIS](#5-running-cais)
6. [Reproducing Paper Results](#6-reproducing-paper-results)
7. [License](#7-license)

---

## 1. Introduction

Causal effect estimation is central to evidence-based decision-making across domains like social sciences, healthcare, and economics, but it requires substantial methodological expertise to apply correctly.

**CAIS** automates this process end-to-end using Large Language Models (LLMs) to:
- Parse a natural language causal query and analyze dataset characteristics.
- Select an appropriate causal inference method via a decision tree and structured prompting.
- Execute the method using predefined code templates and validate the results.
- Interpret the numerical output in the context of the original query.

<div align="center">
  <img src="blob/main/asset/CAIS-arch.png" width="990" alt="CAIS Architecture" />
</div>

**Supported Methods:**
- **Econometric:** Difference-in-Differences (DiD), Instrumental Variables (IV), Ordinary Least Squares (OLS), Regression Discontinuity Design (RDD)
- **Causal Graph-based:** Backdoor adjustment, Frontdoor adjustment

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
git clone https://github.com/your-repo/cais.git
cd cais
cp .env.example .env
```

**Step 2: Load necessary compute modules**
```bash
module load rust
module load gcc
module load openblas
```

**Step 3: Create and activate a Python 3.10 environment**
```bash
conda create -n cais python=3.10
conda activate cais
pip install -r requirements.txt
```

**Step 4: Install the CAIS library**
```bash
pip install -e .
```

> ⚠️ Keep your `.env` file secure and never commit it to version control.

---

## 4. Dataset Information

All datasets used to evaluate CAIS and the baseline models are available in the `data/` directory:

| Path | Description |
|---|---|
| `data/all_data/` | CSV files from the QRData and real-world study collections |
| `data/synthetic_data/` | CSV files for synthetic datasets |
| `data/qr_info.csv` | Metadata for QRData: filename, description, query, reference effect, intended method, remarks |
| `data/real_info.csv` | Metadata for real-world datasets |
| `data/synthetic_info.csv` | Metadata for synthetic datasets |

---

## 5. Running CAIS

```bash
python main/run_cais.py \
    --metadata_path <path_to_metadata_csv> \
    --data_dir <path_to_data_folder> \
    --output_dir <output_folder> \
    --output_name <output_filename> \
    --llm_name <llm_name> \
    --llm_provider <llm_provider>
```

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `--metadata_path` | `str` | Path to the CSV file containing queries, dataset descriptions, and filenames |
| `--data_dir` | `str` | Path to the folder containing the data in CSV format |
| `--output_dir` | `str` | Path to the folder where output JSON results will be saved |
| `--output_name` | `str` | Name of the output JSON file |
| `--llm_name` | `str` | Name of the LLM to use (e.g., `gpt-4o`, `claude-3`) |
| `--llm_provider` | `str` | LLM service provider (e.g., `openai`, `anthropic`, `together`) |

**Example:**
```bash
python main/run_cais.py \
    --metadata_path "data/qr_info.csv" \
    --data_dir "data/all_data" \
    --output_dir "output" \
    --output_name "results_qr_4o" \
    --llm_name "gpt-4o-mini" \
    --llm_provider "openai"
```

---

## 6. Reproducing Paper Results

Instructions for reproducing results from the paper will be updated soon.

---

## 7. Citation

If you use CAIS in your research, please cite our paper:

If you use CAIS or build on this work, please cite:

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

## 8. License

Distributed under the MIT License. See `LICENSE` for more information.

<h1 align="center">
<img src="blob/main/asset/cais.png" width="400" alt="CAIS" />
<br>
Causal AI Scientist: Facilitating Causal Data Science with
Large Language Models
</h1>
<!-- <p align="center">
  <a href="https://causalcopilot.com/"><b>[Demo]</b></a> •
  <a href="https://github.com/Lancelot39/Causal-Copilot"><b>[Code]</b></a> •
  <a href="">"Coming Soon"<b>[Arxiv(coming soon)]</b></a>
</p> -->

**Causal AI Scientist (CAIS)** is an LLM-powered tool for generating data-driven answers to natural language causal queries. It takes a natural language query (for example, "Does participating in a job training program lead to higher income?"), an accompanying dataset, and the corresponding description as inputs. CAIS then frames a suitable causal estimation problem by selecting appropriate treatment and outcome variables. It finds the suitable method for causal effect estimation, implements it, runs diagnostic tests, and finally interprets the numerical results in the context of the original query.

This repo includes instructions on both using the tool to perform causal analysis on a dataset of interest and reproducing results from our paper.

**Note** : This repository is a work in progress and will be updated with additional instructions and files.

<!-- ## 1. Introduction

Causal effect estimation is central to evidence-based decision-making across domains like social sciences, healthcare, and economics. However, it requires specialized expertise to select the right inference method, identify valid variables, and validate results.  

**CAIS (Causal AI Scientist)** automates this process using Large Language Models (LLMs) to:
- Parse a natural language causal query.
- Analyze the dataset characteristics.
- Select the appropriate causal inference method via a decision tree and prompting strategies.
- Execute the method using pre-defined code templates.
- Validate and interpret the results.

<div style="text-align: center;">
    <img src="blob/main/asset/CAIS-arch.png" width="990" alt="CAIS" />
</div>
</h1>

**Key Features:**
- End-to-end causal estimation with minimal user input.
- Supports a wide range of methods:  
  - **Econometric:** Difference-in-Differences (DiD), Instrumental Variables (IV), Ordinary Least Squares (OLS), Regression Discontinuity Design (RDD).
  - **Causal Graph-based:** Backdoor adjustment, Frontdoor adjustment.
- Combines structured reasoning (decision tree) with LLM-powered interpretation.
- Works on clean textbook datasets, messy real-world datasets, and synthetic scenarios.


CAIS consists of three main stages, powered by a **decision-tree-driven reasoning pipeline**:

### **Stage 1: Variable and Method Selection**
1. **Dataset & Query Analysis**
   - The LLM inspects the dataset description, variable names, and statistical summaries.
   - Identifies treatment, outcome, and covariates.
2. **Property Detection**
   - Uses targeted prompts to detect dataset properties:
     - Randomized vs observational
     - Presence of temporal/running variables
     - Availability of valid instruments
3. **Decision Tree Traversal**
   - Traverses a predefined causal inference decision tree (Fig. B in paper).
   - Maps detected properties to the most appropriate estimation method.

### **Stage 1.5: IV Discovery (Optional)**
If the **Instrumental Variable (IV)** method is selected and the `--iv_llm` pipeline is enabled:
1. **Hypothesis Generation**: The LLM hypothesizes potential instruments based on dataset context and variable names.
2. **Confounder Mining**: Identifies potential confounders that might violate the independence or exclusion restrictions.
3. **Critic Validation**: Uses specialized LLM "critics" (Exclusion, Independence) to reason about the validity of each candidate instrument.
4. **Final Selection**: Selects the most robust instrument for the estimation stage.

---

### **Stage 2: Causal Inference Execution**
1. **Template-based Code Generation**
   - Predefined Python templates for each method (e.g., DiD, IV, OLS).
   - Variables from Stage 1 are substituted into templates.
2. **Diagnostics & Validation**
   - Runs statistical tests and checks assumptions where applicable.
   - Handles basic data preprocessing (e.g., type conversion for DoWhy).

---

### **Stage 3: Result Interpretation**
- LLM interprets numerical results and diagnostics in the context of the user’s causal query.
- Outputs:
  - Estimated causal effect (ATE, ATT, or LATE).
  - Standard errors, confidence intervals.
  - Plain-language explanation.

---
## 3. Evaluation

We evaluate **CAIS** across three diverse dataset collections:  
1. **QRData (Textbook Examples)** – curated, clean datasets with known causal effects.  
2. **Real-World Studies** – empirical datasets from research papers (economics, health, political science).  
3. **Synthetic Data** – generated with controlled causal structures to ensure balanced method coverage.

### **Metrics**
We assess CAIS on:
- **Method Selection Accuracy (MSA)** – % of cases where CAIS selects the correct inference method as per the reference.
- **Mean Relative Error (MRE)** – Average relative error between CAIS’s estimated causal effect and the reference value.


<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="blob/main/asset/CAIS-MRE.png" width="450" alt="CAIS MRE"/>
      </td>
      <td align="center">
        <img src="blob/main/asset/CAIS-msa.png" width="450" alt="CAIS MSA"/>
      </td>
    </tr>
  </table>
</p>
--> 

## Getting Started

#### 🔧 Environment Installation


**Prerequisites:**
- **Python 3.10** (create a new conda environment first)
- Required Python libraries (specified in `requirements.txt`)


**Step 1: Copy the example configuration**
```bash
cp .env.example .env
```

**Step 2: Load necessary modules**
```bash
# Load necessary compute modules to ensure dependencies are installed correctly
module load rust
module load gcc 
module load openblas 
```

**Step 3: Create Python 3.10 environment**
```bash
# Create a new conda environment with Python 3.10
conda create -n cais python=3.10
conda activate cais
pip install -r requirements.txt
```

**Step 4: Setup CAIS library**
```bash
pip install -e .
```

## Dataset Information 

All datasets used to evaluate CAIs and the baseline models are available in the data/ directory. Specifically:

* `all_data`: Folder containing all CSV files from the QRData and real-world study collections.
* `synthetic_data`: Folder containing all CSV files corresponding to synthetic datasets.
* `qr_info.csv`: Metadata for QRData files. For each file, this includes the filename, description, causal query, reference causal effect, intended inference method, and additional remarks.
* `real_info.csv`: Metadata for the real-world datasets.
* `synthetic_info.csv`: Metadata for the synthetic datasets.

## Run 
To execute CAIS, run
```python
python run_cais.py \
    --metadata_path {path_to_metadata} \
    --data_dir {path_to_data_folder} \
    --output_dir {output_folder} \
    --output_name {output_filename} \
    --llm_name {llm_name} \
    --llm_provider {llm_provider} \
    [--iv_llm]
```
Args:

* metadata_path (str): Path to the CSV file containing the queries, dataset descriptions, and data file names
* data_dir (str): Path to the folder containing the data in CSV format
* output_dir (str): Path to the folder where the output JSON results will be saved
* output_name (str): Name of the JSON file where the outputs will be saved
* llm_name (str): Name of the LLM to be used (e.g., 'gpt-4o', 'claude-3-5-sonnet', etc.)
* llm_provider (str): Name of the LLM service provider (e.g., 'openai', 'anthropic', 'together', etc.)
* iv_llm (bool, optional): If flag is present, enables the advanced experimental [IV LLM pipeline](https://arxiv.org/pdf/2602.07943) for instrument discovery and validation.
  
A specific example, 
```python
python run_cais.py \
    --metadata_path "data/qr_info.csv" \
    --data_dir "data/all_data" \
    --output_dir "output" \
    --output_name "results_qr_4o" \
    --llm_name "gpt-4o-mini" \
    --llm_provider "openai"
```


## Reproducing paper results
**Will be updated soon**

**⚠️ Important Notes:**
- Keep your `.env` file secure and never commit it to version control

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!--## Contributors



**Core Contributors**: Vishal Verma, Sawal Acharya, Devansh Bhardwaj

**Other Contributors**:  Zhijing Jin, Ana Hagihat, Samuel Simko

---

## Contact

For additional information, questions, or feedback, please contact ours **[Vishal Verma](vishalv@andrew.cmu.edu)**, **[Sawal Acharya](sawal386@stanford.edu)**, **[Devansh Bhardwaj](bhardwajdevansh398@gmail.com)**. We welcome contributions! Come and join us now!
-->

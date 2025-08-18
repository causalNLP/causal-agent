# CAIS - Causal AI Scientist

[![PyPI version](https://badge.fury.io/py/causal-agent.svg)](https://badge.fury.io/py/causal-agent)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Causal AI Scientist (CAIS)** is an LLM-powered tool for generating data-driven answers to natural language causal queries. It takes a natural language query (for example, "Does participating in a job training program lead to higher income?"), an accompanying dataset, and the corresponding description as inputs. CAIS then frames a suitable causal estimation problem by selecting appropriate treatment and outcome variables. It finds the suitable method for causal effect estimation, implements it, runs diagnostic tests, and finally interprets the numerical results in the context of the original query

## 🚀 Quick Start

### Installation

```bash
pip install causal-agent
```

### Basic Usage

```python
from cais import run_causal_analysis

# Run causal analysis with a simple question
result = run_causal_analysis(
    query="What is the effect of education on income?",
    dataset_path="your_data.csv",
    dataset_description="Dataset containing education and income data"
)

print(f"Causal effect: {result['results']['results']['effect_estimate']}")
print(f"Method used: {result['results']['results']['method_used']}")
```

### Command Line Interface

```bash
# Single analysis
cais run dataset.csv "What is the effect of treatment on outcome?"

# Batch analysis
cais batch metadata.csv data_folder/ results.json
```

## 🔧 Setup

### 1. Configure LLM Provider

Set your API key for your preferred LLM provider:

```python
import os

# OpenAI (default)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Or use Anthropic
os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Or use Google Gemini
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

### 2. Prepare Your Data

- CSV format with clear column names
- Include relevant variables for causal analysis
- Ensure sufficient sample size (typically >100 observations)

## 📊 What CAIS Does

1. **Parses** your natural language causal question
2. **Analyzes** your dataset structure and variables
3. **Selects** the most appropriate causal inference method:
   - Randomized Controlled Trials (RCT)
   - Difference-in-Differences (DiD)
   - Instrumental Variables (IV)
   - Regression Discontinuity Design (RDD)
   - Propensity Score Matching/Weighting
   - Linear Regression with controls
   - And more...
4. **Executes** the analysis with proper diagnostics
5. **Interprets** results in the context of your original question

## 🎯 Example Use Cases

### Education Research
```python
result = run_causal_analysis(
    query="Does smaller class size improve student test scores?",
    dataset_path="education_data.csv",
    dataset_description="Student data with class sizes and test scores"
)
```

### Healthcare
```python
result = run_causal_analysis(
    query="What is the effect of the new treatment on patient recovery time?",
    dataset_path="clinical_trial_data.csv",
    dataset_description="Randomized trial data comparing treatments"
)
```

### Economics
```python
result = run_causal_analysis(
    query="How does minimum wage increase affect employment?",
    dataset_path="employment_data.csv",
    dataset_description="Employment data before and after policy change"
)
```

## 📈 Advanced Features

### Batch Processing
Process multiple datasets at once:

```python
import pandas as pd

# Create metadata file
metadata = pd.DataFrame({
    'natural_language_query': [
        'Effect of education on income',
        'Impact of training on employment'
    ],
    'data_files': ['education.csv', 'training.csv'],
    'data_description': ['Education dataset', 'Training program data']
})

# Save metadata to CSV file first
metadata.to_csv('metadata.csv', index=False)

# Run batch analysis using CLI
# cais batch metadata.csv ./data/ results.json
```

### Custom LLM Configuration
```python
# Use different models
os.environ["LLM_MODEL"] = "gpt-4o-mini"  # Faster, cheaper
# os.environ["LLM_MODEL"] = "gpt-4"      # More accurate
# os.environ["LLM_MODEL"] = "claude-3-haiku-20240307"  # Anthropic
```

## 🔍 Understanding Results

CAIS returns structured results including:

- **Effect Estimate**: The causal effect size
- **Standard Error**: Uncertainty in the estimate
- **Confidence Interval**: Range of plausible values
- **Method Used**: Which causal inference technique was applied
- **Variables Identified**: Treatment, outcome, and control variables
- **Explanation**: Plain-language interpretation of results

```python
result = run_causal_analysis(query, dataset_path, description)

# Access key results
effect = result['results']['results']['effect_estimate']
method = result['results']['results']['method_used']
variables = result['results']['variables']
explanation = result['explanation']

print(f"Using {method}, we found that {variables['treatment_variable']} "
      f"has an effect of {effect} on {variables['outcome_variable']}")
```

## 🛠️ Supported Methods

CAIS automatically selects from:

- **Experimental Methods**: RCT analysis
- **Quasi-Experimental**: DiD, RDD, IV
- **Observational**: Propensity scoring, backdoor adjustment
- **Machine Learning**: Causal forests, double ML (coming soon)

## 📚 Best Practices

### Writing Good Causal Questions
- ✅ **Good**: "What is the causal effect of education on income?"
- ✅ **Good**: "Does job training increase employment rates?"
- ❌ **Avoid**: "Are education and income related?" (correlation, not causation)

### Dataset Requirements
- Clear variable names
- Sufficient sample size
- Relevant control variables
- Clean data (handle missing values)

### Providing Context
Include dataset descriptions with:
- Variable definitions
- Data collection method
- Time period covered
- Known confounders

## 🤝 Support

- **Documentation**: [Full documentation](https://causal-agent.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/causalNLP/causal-agent/issues)
- **Examples**: Check the [demo notebook](https://github.com/causalNLP/causal-agent/blob/main/cais_demo.ipynb)

## 📄 License

MIT License - see [LICENSE](https://github.com/causalNLP/causal-agent/blob/main/LICENSE) for details.

## Citation

If you use CAIS in your research, please cite:

```bibtex
@software{cais2025,
  title={CAIS: Causal AI Scientist for Automated Causal Inference},
  author={Verma, Vishal and Acharya, Sawal and Simko, Samuel and Bhardwaj, Devansh and Haghighat, Anahita and Jin, Zhijing},
  year={2025},
  url={https://github.com/causalNLP/causal-agent}
}
```

---

**Get started with causal inference in minutes, not hours!** 🎉
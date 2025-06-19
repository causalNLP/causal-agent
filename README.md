# Causal Agent

Causal Agent is an AI-powered agent for automated causal inference. It leverages Large Language Models (LLMs) to understand natural language queries, analyze datasets, select and execute appropriate causal inference methods, and provide interpretable results.

## Features

- **Automated Causal Inference**: End-to-end workflow from a natural language question to a causal conclusion.
- **Multiple Causal Methods**: Supports a variety of causal inference techniques:
  - Difference in Means
  - Linear Regression / Backdoor Adjustment
  - Propensity Score Matching (PSM) and Weighting (PSW)
  - Difference-in-Differences (DiD)
  - Instrumental Variable (IV)
  - Regression Discontinuity Design (RDD)
  - Generalized Propensity Score (GPS) for continuous treatments
- **LLM-Powered**: Uses LLMs for complex tasks like variable identification, method selection, and results interpretation.
- **Modular and Extensible**: Built with a clear separation of components for parsing, analysis, method selection, execution, and explanation.
- **Structured I/O**: Utilizes Pydantic for robust and predictable data structures throughout the pipeline.
- **LangChain Integration**: Built upon the LangChain framework for creating powerful AI agents and tools.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/causalNLP/causal-agent.git
    cd causal-agent
    ```

2.  Install the required dependencies from `requirement.txt`:
    ```bash
    pip install -r requirement.txt
    ```

3.  Set up your environment variables. Create a `.env` file in the root of the project and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    # Add other keys like ANTHROPIC_API_KEY or TOGETHER_API_KEY if you plan to use other models
    ```

## Usage

The primary entry point for the agent is the `run_causal_analysis` function. Here is a simple example of how to use it:

```python
from auto_causal.agent import run_causal_analysis
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# 1. Define your causal question and dataset information
query = "What is the effect of a new teaching method on student test scores?"
dataset_path = "path/to/your/student_data.csv"
dataset_description = "This dataset contains student test scores, whether they received the new teaching method, and various demographic and prior academic performance covariates."

# 2. Run the causal analysis
results = run_causal_analysis(
    query=query,
    dataset_path=dataset_path,
    dataset_description=dataset_description
)

# 3. Print the results
print(results)
```

## Generating Synthetic Data + Context
### Step 1: Simulate Data

The first step is to generate synthetic datasets using a specified causal inference method. Run the following command:

```python
python main/generate_synthetic.py --method {METHOD_NAME} --size {DSET_SIZE} --metadata_path {METADATA_DESTINATION} --data_path {DATA_DESTINATION} --observations {N_OBSERVATIONS}
```
#### Arguments
- METHOD_NAME: Causal inference method (e.g., rct, iv, rdd)
- DSET_SIZE: Number of datasets to generate
- METADATA_DESTINATION: Path to the folder for saving metadata
- DATA_DESTINATION: Path to the folder for saving data files
- N_OBSERVATIONS: Number of observations in each dataset

#### Output:
- CSV data files in the folder {DATA_DESTINATION}/{METHOD_NAME}/data
- Metadata file in the folder {METADATA_DESTINATION}/{METHOD_NAME}/metadata

#### Specific Example:
```python
python main/generate_synthetic.py --method rct --size 10 --metadata_path synthetic_data --data_path synthetic_data --observations 1000
```
This will generate 10 datasets (rct_data_0.csv, ..., rct_data_9.csv) in synthetic_data/rct/data/ and a metadata file rct.json in synthetic_data/rct/metadata/

### Step 2: Generating context + query using GPT
Once the synthetic data is generated, the next step is to create a hypothetical context and causal query associated with each dataset. This is done using GPT.

```python
python main/generate_context.py --method {METHOD_NAME} --metadata_path {METADATA_PATH} --dataset_folder {DATASET_PATH} --output_folder {OUTPUT_PATH} -- domain {DOMAIN_NAME}
```
#### Arguments
- METHOD_NAME: Causal inference method (e.g., rct, iv, rdd)
- METADATA_PATH: Path to folder containing the metadata 
- DATASET_PATH: Path to folder containing the CSV files 
- OUTPUT_PATH: Path to the folder where the output is saved as a JSON file
- DOMAIN_NAME: Name of the domain

The script provides GPT via API calls the numerical summary of the synthetic data along with the metadata. GPT then generates the column names, their descriptions, and a hypothetical story describing how and why the data was collected. Finally, it generates a causal query relevant to the dataset. 

### Outout 
- A JSON file that contains variable labels, dataset description, query, and a one-to-two-sentence summary of the dataset. The file is saved in the folder OUTPUT_PATH. 

### Specific Example 
```python
   python main/generate_context.py --method rct --dataset_folder synthetic_data/rct/data/ --metadata_path synthetic_data/rct/metadata/rct.json --output_folder synthetic_data/rct/context --domain "economics"
```
This will generate a JSON file synthetic_data/rct/context/rct.json that contains the relevant context and query for the rct datasets. 
## Project Structure

The project is organized into several key directories:

-   `auto_causal/`: The main source code for the library.
    -   `agent.py`: Defines the main LangChain agent and the `run_causal_analysis` entry point.
    -   `components/`: Core logic for different steps of the causal workflow (e.g., `dataset_analyzer`, `query_interpreter`, `method_validator`).
    -   `methods/`: Implementations of various causal inference methods (e.g., `difference_in_differences`, `propensity_score`).
    -   `prompts/`: Contains prompt templates for interacting with LLMs for specific tasks.
    -   `tools/`: LangChain tools that wrap the logic from `components/` to be used by the agent.
    -   `models.py`: Pydantic models defining the data structures used throughout the agent.
    -   `config.py`: Configuration for LLM clients.
-   `tests/`: Unit and end-to-end tests for the library.
-   `requirement.txt`: A list of project dependencies.

## Testing

To run the test suite, you can use `pytest` from the root directory of the project:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or suggestions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

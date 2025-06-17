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

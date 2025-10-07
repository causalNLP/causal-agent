"""
Causal Agent - Automated Causal Inference with Large Language Models.

The causal_agent module provides an LLM-powered tool for generating data-driven 
answers to natural language causal queries. It automatically:

- Parses natural language causal questions
- Analyzes dataset characteristics and variables  
- Selects appropriate causal inference methods
- Executes causal analysis with proper diagnostics
- Interprets results in plain language

Example:
    >>> from causal_agent import run_causal_analysis
    >>> result = run_causal_analysis(
    ...     query="What is the effect of education on income?",
    ...     dataset_path="data.csv", 
    ...     dataset_description="Education and income dataset"
    ... )
    >>> print(f"Effect: {result['results']['results']['effect_estimate']}")

The module supports various causal inference methods including:
- Randomized Controlled Trials (RCT)
- Difference-in-Differences (DiD) 
- Instrumental Variables (IV)
- Regression Discontinuity Design (RDD)
- Propensity Score Matching/Weighting
- Backdoor Adjustment
- Linear Regression with controls
"""

__version__ = "0.1.2"

# Import components
from causal_agent.components import (
    parse_input,
    analyze_dataset,
    interpret_query,
    validate_method,
    generate_explanation,
    format_output,
    create_workflow_state_update
)

# Import tools
from causal_agent.tools import (
    input_parser_tool,
    dataset_analyzer_tool,
    query_interpreter_tool,
    method_selector_tool,
    method_validator_tool,
    method_executor_tool,
    explanation_generator_tool,
    output_formatter_tool
)

from .agent import run_causal_analysis


__all__ = [
    'run_causal_analysis'
]

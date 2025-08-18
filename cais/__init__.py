"""
Auto Causal module for causal inference.

This module provides automated causal inference capabilities
through a pipeline that selects and applies appropriate causal methods.
"""

__version__ = "0.1.1"

# Import components
from cais.components import (
    parse_input,
    analyze_dataset,
    interpret_query,
    validate_method,
    generate_explanation,
    format_output,
    create_workflow_state_update
)

# Import tools
from cais.tools import (
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

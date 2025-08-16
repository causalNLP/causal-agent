"""
Auto Causal module for causal inference.

This module provides automated causal inference capabilities
through a pipeline that selects and applies appropriate causal methods.
"""

__version__ = "0.1.0"

# Import components
from auto_causal.components import (
    parse_input,
    analyze_dataset,
    interpret_query,
    validate_method,
    generate_explanation,
    format_output,
    create_workflow_state_update
)

# Import tools
from auto_causal.tools import (
    input_parser_tool,
    dataset_analyzer_tool,
    query_interpreter_tool,
    method_selector_tool,
    method_validator_tool,
    method_executor_tool,
    explanation_generator_tool,
    output_formatter_tool
)

# Import the main agent function
from .agent import run_causal_analysis

# Remove backward compatibility for old pipeline
# try:
#     from .pipeline import CausalInferencePipeline
# except ImportError:
#     # Define a placeholder class if the old pipeline doesn't exist
#     class CausalInferencePipeline:
#         """Placeholder for CausalInferencePipeline."""
#         
#         def __init__(self, *args, **kwargs):
#             pass

# Update __all__ to export the main function
__all__ = [
    'run_causal_analysis'
]

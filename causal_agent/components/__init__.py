"""
Causal Agent components package.

This package contains the core components for the causal_agent module,
each handling a specific part of the causal inference workflow:

- input_parser: Parses and validates user inputs
- dataset_analyzer: Analyzes dataset characteristics and variables
- query_interpreter: Interprets natural language causal queries
- decision_tree: Selects appropriate causal inference methods
- method_validator: Validates method selection and assumptions
- explanation_generator: Generates human-readable explanations
- output_formatter: Formats results for output
- state_manager: Manages workflow state updates
"""

from causal_agent.components.input_parser import parse_input
from causal_agent.components.dataset_analyzer import analyze_dataset
from causal_agent.components.query_interpreter import interpret_query
from causal_agent.components.decision_tree import select_method
from causal_agent.components.method_validator import validate_method
from causal_agent.components.explanation_generator import generate_explanation
from causal_agent.components.output_formatter import format_output
from causal_agent.components.state_manager import create_workflow_state_update

__all__ = [
    "parse_input",
    "analyze_dataset",
    "interpret_query",
    "select_method",
    "validate_method",
    "generate_explanation",
    "format_output",
    "create_workflow_state_update"
]

# This file makes Python treat the directory as a package.

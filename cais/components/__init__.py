"""
Auto Causal components package.

This package contains the core components for the auto_causal module,
each handling a specific part of the causal inference workflow.
"""

from auto_causal.components.input_parser import parse_input
from auto_causal.components.dataset_analyzer import analyze_dataset
from auto_causal.components.query_interpreter import interpret_query
from auto_causal.components.decision_tree import select_method
from auto_causal.components.method_validator import validate_method
from auto_causal.components.explanation_generator import generate_explanation
from auto_causal.components.output_formatter import format_output
from auto_causal.components.state_manager import create_workflow_state_update

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

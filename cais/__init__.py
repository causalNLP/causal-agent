"""
Auto Causal module for causal inference.

This module provides automated causal inference capabilities
through a pipeline that selects and applies appropriate causal methods.
"""

from __future__ import annotations

__version__ = "0.1.1"

# NOTE: Keep imports lazy.
#
# `cais` is used as a package namespace for submodules (e.g. `cais.iv_llm`).
# Importing many components/tools at module import time can trigger optional
# dependencies and/or import-time errors in parts of the pipeline that aren't
# needed for basic package usage.

import importlib
from typing import Dict, Tuple


__all__ = [
    "run_causal_analysis",
    "parse_input",
    "analyze_dataset",
    "interpret_query",
    "validate_method",
    "generate_explanation",
    "format_output",
    "create_workflow_state_update",
    "input_parser_tool",
    "dataset_analyzer_tool",
    "query_interpreter_tool",
    "method_selector_tool",
    "method_validator_tool",
    "method_executor_tool",
    "explanation_generator_tool",
    "output_formatter_tool",
]


_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # High-level API
    "run_causal_analysis": ("cais.agent", "run_causal_analysis"),

    # Components
    "parse_input": ("cais.components", "parse_input"),
    "analyze_dataset": ("cais.components", "analyze_dataset"),
    "interpret_query": ("cais.components", "interpret_query"),
    "validate_method": ("cais.components", "validate_method"),
    "generate_explanation": ("cais.components", "generate_explanation"),
    "format_output": ("cais.components", "format_output"),
    "create_workflow_state_update": ("cais.components", "create_workflow_state_update"),

    # Tools
    "input_parser_tool": ("cais.tools", "input_parser_tool"),
    "dataset_analyzer_tool": ("cais.tools", "dataset_analyzer_tool"),
    "query_interpreter_tool": ("cais.tools", "query_interpreter_tool"),
    "method_selector_tool": ("cais.tools", "method_selector_tool"),
    "method_validator_tool": ("cais.tools", "method_validator_tool"),
    "method_executor_tool": ("cais.tools", "method_executor_tool"),
    "explanation_generator_tool": ("cais.tools", "explanation_generator_tool"),
    "output_formatter_tool": ("cais.tools", "output_formatter_tool"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'cais' has no attribute {name!r}")

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

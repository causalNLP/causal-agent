"""
Tool for analyzing datasets for causal inference.

This module provides a LangChain tool for analyzing datasets to detect
characteristics relevant for causal inference, such as temporal structure,
potential instrumental variables, and variable relationships.
"""

from typing import Dict, Any, Optional
from langchain.tools import tool
import logging

from causal_agent.components.dataset_analyzer import analyze_dataset
from causal_agent.components.state_manager import create_workflow_state_update
from langchain_core.language_models import BaseChatModel

from causal_agent.config import get_llm_client

from causal_agent.models import DatasetAnalysis, DatasetAnalyzerOutput
from causal_agent import models

logger = logging.getLogger(__name__)


@tool
def dataset_analyzer_tool(dataset_path: str,
                            dataset_description: Optional[str] = None,
                            original_query: Optional[str] = None) -> DatasetAnalyzerOutput:
    """
    Analyze dataset to identify important characteristics for causal inference.
    
    This tool loads the dataset, calculates summary statistics, checks for temporal
    structure, identifies potential treatments/outcomes/instruments, and assesses
    variable relationships relevant for selecting a causal method.
    
    Args:
        dataset_path: Path to the dataset file.
        dataset_description: Optional description string from input.
        llm: Optional LLM client for enhanced analysis.
        
    Returns:
        A Pydantic model containing the structured dataset analysis results and workflow state.
    """
    logger.info(f"Running dataset_analyzer_tool on path: {dataset_path}")
    # Call the component function with the LLM if available
    llm = get_llm_client()

    try:
        # Call the component function 
        analysis_dict = analyze_dataset(dataset_path, llm_client=llm, dataset_description=dataset_description, original_query=original_query)

        # Check for errors returned explicitly by the component
        if isinstance(analysis_dict, dict) and "error" in analysis_dict:
            logger.error(f"Dataset analysis component failed: {analysis_dict['error']}")
            raise ValueError(analysis_dict['error'])
            
        # Validate and structure the analysis using Pydantic
        # This assumes analyze_dataset returns a dict compatible with DatasetAnalysis
        # Handle potential missing keys or type mismatches gracefully
        analysis_results_model = DatasetAnalysis(**analysis_dict)

    except Exception as e:
        logger.error(f"Error during dataset analysis or Pydantic model creation: {e}", exc_info=True)
        error_state = create_workflow_state_update(
            current_step="data_analysis",
            step_completed_flag=False,
            next_tool="dataset_analyzer_tool", # Retry or error handler?
            next_step_reason=f"Dataset analysis failed: {e}"
        )

        minimal_info = models.DatasetInfo(num_rows=0, num_columns=0, file_path=dataset_path, file_name="unknown")
        empty_temporal = models.TemporalStructure(has_temporal_structure=False, temporal_columns=[], is_panel_data=False)
        error_analysis = models.DatasetAnalysis(
            dataset_info=minimal_info, 
            columns=[], 
            potential_treatments=[], 
            potential_outcomes=[],
            temporal_structure_detected=False,
            panel_data_detected=False,
            potential_instruments_detected=False,
            discontinuities_detected=False,
            temporal_structure=empty_temporal,
            sample_size=0,
            num_covariates_estimate=0
        )
        return DatasetAnalyzerOutput(
            analysis_results=error_analysis, 
            dataset_description=dataset_description,
            workflow_state=error_state.get('workflow_state', {})
        )

    # Create workflow state update for success
    workflow_update = create_workflow_state_update(
        current_step="data_analysis",
        step_completed_flag="dataset_analyzed",
        next_tool="query_interpreter_tool",
        next_step_reason="Now we need to map query concepts to actual dataset variables"
    )

    # Construct the final Pydantic output object
    output = DatasetAnalyzerOutput(
        analysis_results=analysis_results_model,
        dataset_description=dataset_description,
        dataset_path=dataset_path,
        workflow_state=workflow_update.get('workflow_state', {})
    )

    # print(output)

    logger.info("dataset_analyzer_tool finished successfully.")
    return output
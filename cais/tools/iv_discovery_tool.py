"""
Tool for discovering instrumental variables using IV-LLM.

This module provides a LangChain tool for discovering valid instrumental variables
for given treatment and outcome variables using the IV-LLM pipeline.
"""

from typing import Dict, List, Any, Optional
from langchain.tools import tool
import logging

from langchain_core.language_models import BaseChatModel

from cais.components.iv_discovery import discover_instruments
from cais.components.state_manager import create_workflow_state_update
from cais.config import get_llm_client

from cais.models import Variables, DatasetAnalysis, IVDiscoveryInput, IVDiscoveryOutput

logger = logging.getLogger(__name__)


@tool(args_schema=IVDiscoveryInput)
def iv_discovery_tool(
    variables: Variables,
    dataset_analysis: DatasetAnalysis,
    dataset_description: Optional[str] = None,
    original_query: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
) -> IVDiscoveryOutput:
    """
    Discover valid instrumental variables for the identified treatment and outcome.

    Uses the IV-LLM pipeline to hypothesize potential instruments and validate them
    using exclusion and independence criteria. If valid instruments are found,
    updates the variables with the instrument_variable.

    Args:
        variables: Pydantic model containing identified variables (treatment, outcome, etc.)
        dataset_analysis: Pydantic model containing dataset analysis results
        dataset_description: Optional textual description of the dataset
        original_query: Optional original user query string

    Returns:
        Updated variables with instrument_variable if found, plus discovery results and workflow state
    """
    logger.info("Running iv_discovery_tool")

    # Extract treatment and outcome
    treatment = variables.treatment_variable
    outcome = variables.outcome_variable

    if not treatment or not outcome:
        logger.warning("No treatment or outcome variable identified, skipping IV discovery")
        workflow_update = create_workflow_state_update(
            current_step="iv_discovery",
            step_completed_flag=True,  # Completed but no IVs found
            next_tool="method_selector_tool",
            next_step_reason="No treatment/outcome variables available for IV discovery",
        )
        return IVDiscoveryOutput(
            variables=variables,
            dataset_analysis=dataset_analysis,
            dataset_description=dataset_description,
            original_query=original_query,
            iv_discovery_results={
                "proposed_ivs": [],
                "valid_ivs": [],
                "validation_results": [],
            },
            workflow_state=workflow_update.get("workflow_state", {}),
        )

    # Prepare context from dataset description and analysis
    context_parts = []
    if dataset_description:
        context_parts.append(dataset_description)

    # Add column information
    columns = dataset_analysis.columns or []
    column_categories = dataset_analysis.column_categories or {}
    if columns:
        column_info = []
        for col in columns:
            category = column_categories.get(col, "unknown")
            column_info.append(f"{col} ({category})")
        context_parts.append("Available columns: " + ", ".join(column_info))

    context = ". ".join(context_parts)

    # Get confounders from variables if available
    confounders = variables.covariates or []

    # Use provided LLM or get default from config
    if llm is None:
        llm = get_llm_client()

    try:
        # Run IV discovery
        discovery_results = discover_instruments(
            treatment=treatment,
            outcome=outcome,
            context=context,
            confounders=confounders,
            llm=llm,
        )

        # Update variables if valid IVs found
        updated_variables = variables.model_copy()
        valid_ivs = discovery_results.get("valid_ivs", [])
        if valid_ivs:
            # Select the first valid IV (could be enhanced to select best one)
            updated_variables.instrument_variable = valid_ivs[0]
            logger.info("Found valid instrument: %s", valid_ivs[0])

        # Create workflow state
        workflow_update = create_workflow_state_update(
            current_step="iv_discovery",
            step_completed_flag=True,
            next_tool="method_selector_tool",
            next_step_reason="IV discovery completed, proceeding to method selection",
        )

        return IVDiscoveryOutput(
            variables=updated_variables,
            dataset_analysis=dataset_analysis,
            dataset_description=dataset_description,
            original_query=original_query,
            iv_discovery_results=discovery_results,
            workflow_state=workflow_update.get("workflow_state", {}),
        )

    except Exception as e:
        logger.error("Error during IV discovery: %s", e, exc_info=True)
        workflow_update = create_workflow_state_update(
            current_step="iv_discovery",
            step_completed_flag=False,
            next_tool="method_selector_tool",
            next_step_reason=f"IV discovery failed: {e}",
        )
        return IVDiscoveryOutput(
            variables=variables,
            dataset_analysis=dataset_analysis,
            dataset_description=dataset_description,
            original_query=original_query,
            iv_discovery_results={
                "proposed_ivs": [],
                "valid_ivs": [],
                "validation_results": [],
                "error": str(e),
            },
            workflow_state=workflow_update.get("workflow_state", {}),
        )
"""
Tool for parsing causal inference queries.

This module provides a LangChain tool for parsing causal inference queries,
extracting key elements, and guiding the workflow to the next step.
"""

import logging
import re
from typing import Dict, Any, Optional
from langchain_core.tools import tool

from causal_agent.components.input_parser import parse_input
from causal_agent.config import get_llm_client
from causal_agent.components.state_manager import create_workflow_state_update
import json
logger = logging.getLogger(__name__)

@tool
def input_parser_tool(input_text: str) -> Dict[str, Any]:
    """
    Parse the user's initial input text to extract query, dataset path, and description.
    
    This tool uses regex to find structured information within the input text 
    and then leverages an LLM for more complex NLP tasks on the extracted query.
    
    Args:
        input_text: The combined initial input string from the user/system.
        
    Returns:
        Dict containing parsed query information, path, description, and workflow state.
    """
    logger.info(f"Running input_parser_tool on input: '{input_text[:100]}...'")
    
    # --- Extract structured info using Regex --- 
    query = None
    dataset_path = None
    dataset_description = None
    
    query_match = re.search(r"My question is: (.*?)\n", input_text, re.IGNORECASE)
    if query_match:
        query = query_match.group(1).strip()
        
    path_match = re.search(r"The dataset is located at: (.*?)\n", input_text, re.IGNORECASE)
    if path_match:
        dataset_path = path_match.group(1).strip()
        
    # Use re.search to find the description potentially anywhere after its label
    desc_match = re.search(r"Dataset Description: (.*)", input_text, re.DOTALL | re.IGNORECASE)
    if desc_match:
        # Strip leading/trailing whitespace/newlines from the captured group
        dataset_description = desc_match.group(1).strip()
        
    if not query:
        logger.warning("Could not extract query from input_text using regex. Attempting full text as query.")
        # Fallback: This is risky if input_text contains boilerplate
        query = input_text 
        
    logger.info(f"Extracted - Query: '{query[:50]}...', Path: '{dataset_path}', Desc: '{str(dataset_description)[:50]}...'")

    # --- Get LLM and Parse Query --- 
    try:
        llm_instance = get_llm_client()
    except Exception as e:
        logger.error(f"Failed to initialize LLM for input_parser_tool: {e}")
        return {"error": f"LLM Initialization failed: {e}", "workflow_state": {}} 

    # Call the component function to parse the extracted query
    try:
        parsed_info = parse_input(
            query=query, 
            dataset_path_arg=dataset_path, # Use extracted path
            dataset_info=None, # This arg seems unused by parse_input now
            llm=llm_instance
        )
    except Exception as e:
        logger.error(f"Error during parse_input execution: {e}", exc_info=True)
        return {"error": f"Input parsing failed: {e}", "workflow_state": {}} 
    
    # Create workflow state update
    workflow_update = create_workflow_state_update(
        current_step="input_processing",
        step_completed_flag="query_parsed",
        next_tool="dataset_analyzer_tool",
        next_step_reason="Now that we understand the query, we need to analyze the dataset structure"
    )
    
    # Combine results with workflow state
    result = {
        "original_query": parsed_info.get("original_query", query), # Fallback to regex query
        "dataset_path": parsed_info.get("dataset_path") or dataset_path, # Use extracted if component missed it
        "query_type": parsed_info.get("query_type"),
        "extracted_variables": parsed_info.get("extracted_variables", {}),
        "constraints": parsed_info.get("constraints", []),
        # Pass dataset_description along
        "dataset_description": dataset_description 
    }
    print('before workflow: ', result)
    # Add workflow state to the result
    result.update(workflow_update)
    print('after workflow: ', result)
    logger.info("input_parser_tool finished successfully.")
    return result
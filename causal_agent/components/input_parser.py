"""
Input parser component for extracting information from causal queries.

This module provides functionality to parse user queries and extract key
elements such as the causal question, relevant variables, and constraints.
"""

import re
import os
import json
import logging 
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from functools import partial 


from dotenv import load_dotenv


from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException 
from langchain_core.language_models import BaseChatModel
# --- Load .env file --- 
load_dotenv() # Load environment variables from .env file

# --- Configure Logging --- 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Instantiate LLM Client --- 
# Ensure OPENAI_API_KEY environment variable is set
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
except ImportError:
    logger.error("langchain_openai not installed. Please install it to use OpenAI models.")
    llm = None
except Exception as e:
    logger.error(f"Error initializing LLM: {e}. Input parsing will rely on fallbacks.")
    llm = None

# --- Pydantic Models for Structured Output --- 
class ParsedVariables(BaseModel):
    treatment: List[str] = Field(default_factory=list, description="Variable(s) representing the treatment/intervention.")
    outcome: List[str] = Field(default_factory=list, description="Variable(s) representing the outcome/result.")
    covariates_mentioned: Optional[List[str]] = Field(default_factory=list, description="Covariate/control variable(s) explicitly mentioned in the query.")
    grouping_vars: Optional[List[str]] = Field(default_factory=list, description="Variable(s) identifying groups or units for analysis.")
    instruments_mentioned: Optional[List[str]] = Field(default_factory=list, description="Potential instrumental variable(s) mentioned.")

class ParsedQueryInfo(BaseModel):
    query_type: str = Field(..., description="Type of query (e.g., EFFECT_ESTIMATION, COUNTERFACTUAL, CORRELATION, DESCRIPTIVE, OTHER). Required.")
    variables: ParsedVariables = Field(..., description="Variables identified in the query.")
    constraints: Optional[List[str]] = Field(default_factory=list, description="Constraints or conditions mentioned (e.g., 'X > 10', 'country = USA').")
    dataset_path_mentioned: Optional[str] = Field(None, description="Dataset path explicitly mentioned in the query, if any.")

# Add Pydantic model for path extraction
class ExtractedPath(BaseModel):
    dataset_path: Optional[str] = Field(None, description="File path or URL for the dataset mentioned in the query.")

# --- End Pydantic Models --- 

def _build_llm_prompt(query: str, dataset_info: Optional[Dict] = None) -> str:
    """Builds the prompt for the LLM to extract query information."""
    dataset_context = "No dataset context provided."
    if dataset_info:
        columns = dataset_info.get('columns', [])
        column_details = "\n".join([f"- {col} (Type: {dataset_info.get('column_types', {}).get(col, 'Unknown')})" for col in columns])
        sample_rows = dataset_info.get('sample_rows', 'Not available')
        # Ensure sample rows are formatted reasonably
        if isinstance(sample_rows, list):
             sample_rows_str = json.dumps(sample_rows[:3], indent=2) # Show first 3 sample rows
        elif isinstance(sample_rows, str):
             sample_rows_str = sample_rows
        else:
             sample_rows_str = 'Not available'
             
        dataset_context = f"""
Dataset Context:
Columns:
{column_details}
Sample Rows (first few):
{sample_rows_str}
"""

    prompt = f"""
Analyze the following causal query **strictly in the context of the provided dataset information (if available)**. Identify the query type, key variables (mapping query terms to actual column names when possible), constraints, and any explicitly mentioned dataset path.

User Query: "{query}"

{dataset_context}

# Add specific guidance for query types
Guidance for Identifying Query Type:
- EFFECT_ESTIMATION: Look for keywords like 'effect', 'impact', 'influence', 'cause', 'affect', 'consequence'. Also consider questions asking "how does X affect Y?" or comparing outcomes between groups based on an intervention.
- COUNTERFACTUAL: Look for hypothetical scenarios, often using phrases like 'what if', 'if X had been', 'would Y have changed', 'imagine if', 'counterfactual'.
- CORRELATION: Look for keywords like 'correlation', 'association', 'relationship', 'linked to', 'related to'. These queries ask about statistical relationships without necessarily implying causality.
- DESCRIPTIVE: These queries ask for summaries, descriptions, trends, or statistics about the data without investigating causal links or relationships (e.g., "Show sales over time", "What is the average age?").
- OTHER: Use this if the query does not fit any of the above categories.

Choose the most appropriate type from: EFFECT_ESTIMATION, COUNTERFACTUAL, CORRELATION, DESCRIPTIVE, OTHER.

Variable Roles to Identify:
- treatment: The intervention or variable whose effect is being studied.
- outcome: The result or variable being measured.
- covariates_mentioned: Variables explicitly mentioned to control for or adjust for.
- grouping_vars: Variables identifying specific subgroups for analysis (e.g., 'for men', 'in the sales department').
- instruments_mentioned: Variables explicitly mentioned as potential instruments.

Constraints: Conditions applied to the analysis (e.g., filters on columns, specific time periods).

Dataset Path Mentioned: Extract the file path or URL if explicitly stated in the query.

**Output ONLY a valid JSON object** matching this exact schema (no explanations, notes, or surrounding text):
```json
{{
  "query_type": "<Identified Query Type>",
  "variables": {{
    "treatment": ["<Treatment Variable(s) Mentioned>"],
    "outcome": ["<Outcome Variable(s) Mentioned>"],
    "covariates_mentioned": ["<Covariate(s) Mentioned>"],
    "grouping_vars": ["<Grouping Variable(s) Mentioned>"],
    "instruments_mentioned": ["<Instrument(s) Mentioned>"]
  }},
  "constraints": ["<Constraint 1>", "<Constraint 2>"],
  "dataset_path_mentioned": "<Path Mentioned or null>"
}}
```
If Dataset Context is provided, ensure variable names in the output JSON correspond to actual column names where possible. If no context is provided, or if a mentioned variable doesn't map directly, use the phrasing from the query.
Respond with only the JSON object.
"""
    return prompt

def _validate_llm_output(parsed_info: ParsedQueryInfo, dataset_info: Optional[Dict] = None) -> bool:
    """Perform basic assertions on the parsed LLM output."""
    # 1. Check required fields exist (Pydantic handles this on parsing)
    # 2. Check query type is one of the allowed types (can add enum to Pydantic later)
    allowed_types = {"EFFECT_ESTIMATION", "COUNTERFACTUAL", "CORRELATION", "DESCRIPTIVE", "OTHER"}
    print(parsed_info)
    assert parsed_info.query_type in allowed_types, f"Invalid query_type: {parsed_info.query_type}"
    
    # 3. Check that if it's an effect query, treatment and outcome are likely present
    if parsed_info.query_type == "EFFECT_ESTIMATION":
        # Check that the lists are not empty
        assert parsed_info.variables.treatment, "Treatment variable list is empty for effect query."
        assert parsed_info.variables.outcome, "Outcome variable list is empty for effect query."
        
    # 4. If dataset_info provided, check if extracted variables exist in columns
    if dataset_info and (columns := dataset_info.get('columns')):
        all_extracted_vars = set()
        for var_list in parsed_info.variables.model_dump().values(): # Iterate through variable lists
            if var_list: # Ensure var_list is not None or empty
                all_extracted_vars.update(var_list)
                
        unknown_vars = all_extracted_vars - set(columns)
        # Allow for non-column variables if context is missing? Maybe relax this.
        # For now, strict check if columns are provided.
        if unknown_vars:
             logger.warning(f"LLM mentioned variables potentially not in dataset columns: {unknown_vars}")
             # Decide if this should be a hard failure (AssertionError) or just a warning.
             # Let's make it a hard failure for now to enforce mapping.
             raise AssertionError(f"LLM hallucinated variables not in dataset columns: {unknown_vars}")
        
    logger.info("LLM output validation passed.")
    return True

def _extract_query_information_with_llm(query: str, dataset_info: Optional[Dict] = None, llm: Optional[BaseChatModel] = None, max_retries: int = 3) -> Optional[ParsedQueryInfo]:
    """Extracts query type, variables, and constraints using LLM with retries and validation."""
    if not llm:
        logger.error("LLM client not provided. Cannot perform LLM extraction.")
        return None
        
    last_error = None
    # Bind the Pydantic model to the LLM for structured output
    structured_llm = llm.with_structured_output(ParsedQueryInfo)
    
    # Initial prompt construction
    system_prompt_content = _build_llm_prompt(query, dataset_info)
    messages = [HumanMessage(content=system_prompt_content)] # Start with just the detailed prompt as Human message

    for attempt in range(max_retries):
        logger.info(f"LLM Extraction Attempt {attempt + 1}/{max_retries}...")
        try:
            # --- Invoke LangChain LLM with structured output (using passed llm) ---
            parsed_info = structured_llm.invoke(messages)
            # ---------------------------------------------------
            print(messages)
            print('---------------------------------------------------')
            print(parsed_info)
            # Perform custom assertions/validation
            if _validate_llm_output(parsed_info, dataset_info):
                return parsed_info # Success!
                
        # Catch errors specific to structured output parsing or Pydantic validation
        except (OutputParserException, ValidationError, AssertionError) as e:
            logger.warning(f"Validation/Parsing Error (Attempt {attempt + 1}): {e}")
            last_error = e
            # Add feedback message for retry
            messages.append(SystemMessage(content=f"Your previous response failed validation: {str(e)}. Please revise your response to be valid JSON conforming strictly to the schema and ensure variable names exist in the dataset context."))
            continue # Go to next retry
        except Exception as e: # Catch other potential LLM API errors
            logger.error(f"Unexpected LLM Error (Attempt {attempt + 1}): {e}", exc_info=True)
            last_error = e
            break # Stop retrying on unexpected API errors
            
    logger.error(f"LLM extraction failed after {max_retries} attempts.")
    if last_error:
         logger.error(f"Last error: {last_error}")
    return None # Indicate failure

# Add helper function to call LLM for path - needs llm argument
def _call_llm_for_path(query: str, llm: Optional[BaseChatModel] = None, max_retries: int = 2) -> Optional[str]:
    """Uses LLM as a fallback to extract just the dataset path."""
    if not llm:
        logger.warning("LLM client not provided. Cannot perform LLM path fallback.")
        return None

    logger.info("Attempting LLM fallback for dataset path extraction...")
    path_extractor_llm = llm.with_structured_output(ExtractedPath)
    prompt = f"Extract the dataset file path (e.g., /path/to/file.csv or https://...) mentioned in the following query. Respond ONLY with the JSON object.\nQuery: \"{query}\""
    messages = [HumanMessage(content=prompt)]
    last_error = None

    for attempt in range(max_retries):
        try:
            parsed_info = path_extractor_llm.invoke(messages)
            if parsed_info.dataset_path:
                logger.info(f"LLM fallback extracted path: {parsed_info.dataset_path}")
                return parsed_info.dataset_path
            else:
                logger.info("LLM fallback did not find a path.")
                return None # LLM explicitly found no path
        except (OutputParserException, ValidationError) as e:
            logger.warning(f"LLM path extraction parsing/validation error (Attempt {attempt+1}): {e}")
            last_error = e
            messages.append(SystemMessage(content=f"Parsing Error: {e}. Please ensure you provide valid JSON with only the 'dataset_path' key."))
            continue
        except Exception as e:
            logger.error(f"Unexpected LLM Error during path fallback (Attempt {attempt+1}): {e}", exc_info=True)
            last_error = e
            break # Don't retry on unexpected errors

    logger.error(f"LLM path fallback failed after {max_retries} attempts. Last error: {last_error}")
    return None

# Renamed and modified function for regex path extraction + LLM fallback - needs llm argument
def extract_dataset_path(query: str, llm: Optional[BaseChatModel] = None) -> Optional[str]:
    """
    Extract dataset path from the query using regex patterns, with LLM fallback.
    
    Args:
        query: The user's causal question text
        llm: The shared LLM client instance for fallback.
        
    Returns:
        String with dataset path or None if not found
    """
    # --- Regex Part (existing logic) ---
    # Check for common patterns indicating dataset paths
    path_patterns = [
        # More specific patterns first
        r"(?:dataset|data|file) (?:at|in|from|located at) [\"\']?([^\"\'.,\s]+\.csv(?:[\\/][^\"\'.,\s]+)*)[\"\']?", # Handles subdirs in path
        r"(?:use|using|analyze|analyse) (?:the |)(?:dataset|data|file) [\"\']?([^\"\'.,\s]+\.csv(?:[\\/][^\"\'.,\s]+)*)[\"\']?",
        # Simpler patterns
        r"[\"']([^\"']+\.csv(?:[\\/][^\"\'.,\s]+)*)[\"']", # Path in quotes
        r"([a-zA-Z0-9_/.:-]+[\\/][a-zA-Z0-9_.:-]+\.csv)", # More generic path-like structure ending in .csv
        r"([^\"\'.,\s]+\.csv)" # Just a .csv file name (least specific)
    ]
    
    for pattern in path_patterns:
        matches = re.search(pattern, query, re.IGNORECASE)
        if matches:
            path = matches.group(1).strip()
            
            # Basic check if it looks like a path
            if '/' in path or '\\' in path or os.path.exists(path):
                 # Check if this is a valid file path immediately
                if os.path.exists(path):
                    logger.info(f"Regex found existing path: {path}")
                    return path
                
                # Check if it's in common data directories
                data_dir_paths = ["data/", "datasets/", "causalscientist/data/"]
                for data_dir in data_dir_paths:
                    potential_path = os.path.join(data_dir, os.path.basename(path))
                    if os.path.exists(potential_path):
                        logger.info(f"Regex found path in {data_dir}: {potential_path}")
                        return potential_path
                
                # If not found but looks like a path, return it anyway - let downstream handle non-existence
                logger.info(f"Regex found potential path (existence not verified): {path}")
                return path 
            # Else: it might just be a word ending in .csv, ignore unless it exists
            elif os.path.exists(path):
                 logger.info(f"Regex found existing path (simple pattern): {path}")
                 return path
                 
    # --- LLM Fallback ---
    logger.info("Regex did not find dataset path. Trying LLM fallback...")
    llm_fallback_path = _call_llm_for_path(query, llm=llm)
    if llm_fallback_path:
         # Optional: Add existence check here too? Or let downstream handle it.
         # For now, return what LLM found.
         return llm_fallback_path

    logger.info("No dataset path found via regex or LLM fallback.")
    return None

def parse_input(query: str, dataset_path_arg: Optional[str] = None, dataset_info: Optional[Dict] = None, llm: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """
    Parse the user's causal query using LLM and regex.
    
    Args:
        query: The user's causal question text.
        dataset_path_arg: Path to dataset if provided directly as an argument.
        dataset_info: Dictionary with dataset context (columns, types, etc.).
        llm: The shared LLM client instance.
        
    Returns:
        Dict containing parsed query information.
    """
    result = {
        "original_query": query,
        "dataset_path": dataset_path_arg, # Start with argument path
        "query_type": "OTHER", # Default values
        "extracted_variables": {},
        "constraints": []
    }

    # --- 1. Use LLM for core NLP tasks --- 
    parsed_llm_info = _extract_query_information_with_llm(query, dataset_info, llm=llm)
    
    if parsed_llm_info:
        result["query_type"] = parsed_llm_info.query_type
        result["extracted_variables"] = {k: v if v is not None else [] for k, v in parsed_llm_info.variables.model_dump().items()}
        result["constraints"] = parsed_llm_info.constraints if parsed_llm_info.constraints is not None else []
        llm_mentioned_path = parsed_llm_info.dataset_path_mentioned
    else:
        logger.warning("LLM-based query information extraction failed.")
        llm_mentioned_path = None
        # Consider falling back to old regex methods here if critical
        # logger.info("Falling back to regex-based parsing (if implemented).")

    # --- 2. Determine Dataset Path (Hybrid Approach) --- 
    final_dataset_path = dataset_path_arg # Priority 1: Explicit argument
    
    # Pass llm instance to the path extractor for its fallback mechanism
    path_extractor = partial(extract_dataset_path, llm=llm)
    
    if not final_dataset_path:
        # Priority 2: Path mentioned in query (extracted by main LLM call)
        if llm_mentioned_path and os.path.exists(llm_mentioned_path):
             logger.info(f"Using dataset path mentioned by LLM: {llm_mentioned_path}")
             final_dataset_path = llm_mentioned_path
        elif llm_mentioned_path: # Check data dirs if path not absolute
            data_dir_paths = ["data/", "datasets/", "causalscientist/data/"]
            base_name = os.path.basename(llm_mentioned_path)
            for data_dir in data_dir_paths:
                potential_path = os.path.join(data_dir, base_name)
                if os.path.exists(potential_path):
                    logger.info(f"Using dataset path mentioned by LLM (found in {data_dir}): {potential_path}")
                    final_dataset_path = potential_path
                    break
            if not final_dataset_path:
                 logger.warning(f"LLM mentioned path '{llm_mentioned_path}' but it was not found.")
                 
    if not final_dataset_path:
        # Priority 3: Path extracted by dedicated Regex + LLM fallback function
        logger.info("Attempting dedicated dataset path extraction (Regex + LLM Fallback)...")
        extracted_path = path_extractor(query) # Call the partial function with llm bound
        if extracted_path:
            final_dataset_path = extracted_path
            
    result["dataset_path"] = final_dataset_path
    
    # Check if a path was found ultimately
    if not result["dataset_path"]:
        logger.warning("Could not determine dataset path from query or arguments.")
    else:
        logger.info(f"Final dataset path determined: {result['dataset_path']}")

    return result

# --- Old Regex-based functions (Commented out or removed) --- 
# def determine_query_type(query: str) -> str:
#     ... (implementation removed)

# def extract_variables(query: str) -> Dict[str, Any]:
#     ... (implementation removed)

# def detect_constraints(query: str) -> List[str]:
#     ... (implementation removed)
# --- End Old Functions --- 

# Renamed function for regex path extraction
def extract_dataset_path_regex(query: str) -> Optional[str]:
    """
    Extract dataset path from the query using regex patterns.
    
    Args:
        query: The user's causal question text
        
    Returns:
        String with dataset path or None if not found
    """
    # Check for common patterns indicating dataset paths
    path_patterns = [
        # More specific patterns first
        r"(?:dataset|data|file) (?:at|in|from|located at) [\"\']?([^\"\'.,\s]+\.csv(?:[\\/][^\"\'.,\s]+)*)[\"\']?", # Handles subdirs in path
        r"(?:use|using|analyze|analyse) (?:the |)(?:dataset|data|file) [\"\']?([^\"\'.,\s]+\.csv(?:[\\/][^\"\'.,\s]+)*)[\"\']?",
        # Simpler patterns
        r"[\"']([^\"']+\.csv(?:[\\/][^\"\'.,\s]+)*)[\"']", # Path in quotes
        r"([a-zA-Z0-9_/.:-]+[\\/][a-zA-Z0-9_.:-]+\.csv)", # More generic path-like structure ending in .csv
        r"([^\"\'.,\s]+\.csv)" # Just a .csv file name (least specific)
    ]
    
    for pattern in path_patterns:
        matches = re.search(pattern, query, re.IGNORECASE)
        if matches:
            path = matches.group(1).strip()
            
            # Basic check if it looks like a path
            if '/' in path or '\\' in path or os.path.exists(path):
                 # Check if this is a valid file path immediately
                if os.path.exists(path):
                    logger.info(f"Regex found existing path: {path}")
                    return path
                
                # Check if it's in common data directories
                data_dir_paths = ["data/", "datasets/", "causalscientist/data/"]
                # Also check relative to current dir (often useful)
                # base_name = os.path.basename(path)
                for data_dir in data_dir_paths:
                    potential_path = os.path.join(data_dir, os.path.basename(path))
                    if os.path.exists(potential_path):
                        logger.info(f"Regex found path in {data_dir}: {potential_path}")
                        return potential_path
                
                # If not found but looks like a path, return it anyway - let downstream handle non-existence
                logger.info(f"Regex found potential path (existence not verified): {path}")
                return path 
            # Else: it might just be a word ending in .csv, ignore unless it exists
            elif os.path.exists(path):
                 logger.info(f"Regex found existing path (simple pattern): {path}")
                 return path
                 
    # TODO: Optional: Add LLM fallback call here if regex fails
    # if no path found:
    #     llm_fallback_path = call_llm_for_path(query)
    #     return llm_fallback_path
    
    return None 
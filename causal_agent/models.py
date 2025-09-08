from typing import List, Optional, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator
import json


class LLMSelectedVariable(BaseModel):
    """Pydantic model for selecting a single variable."""
    variable_name: Optional[str] = Field(None, description="The single best column name selected.")

class LLMSelectedCovariates(BaseModel):
    """Pydantic model for selecting a list of covariates."""
    covariates: List[str] = Field(default_factory=list, description="The list of selected covariate column names.")

class LLMIVars(BaseModel):
    """Pydantic model for identifying IVs."""
    instrument_variable: Optional[str] = Field(None, description="The identified instrumental variable column name.")
    
class LLMEstimand(BaseModel):
    """Pydantic model for identifying estimand"""
    estimand: Optional[str] = Field(None, description="The identified estimand")

class LLMRDDVars(BaseModel):
    """Pydantic model for identifying RDD variables."""
    running_variable: Optional[str] = Field(None, description="The identified running variable column name.")
    cutoff_value: Optional[Union[float, int]] = Field(None, description="The identified cutoff value.")

class LLMRCTCheck(BaseModel):
    """Pydantic model for checking if data is RCT."""
    is_rct: Optional[bool] = Field(None, description="True if the data is from a randomized controlled trial, False otherwise, None if unsure.")
    reasoning: Optional[str] = Field(None, description="Brief reasoning for the RCT conclusion.")

class LLMTreatmentReferenceLevel(BaseModel):
    reference_level: Optional[str] = Field(None, description="The identified reference/control level for the treatment variable, if specified in the query. Should be one of the actual values in the treatment column.")
    reasoning: Optional[str] = Field(None, description="Brief reasoning for identifying this reference level.")


class LLMInteractionSuggestion(BaseModel):
    """Pydantic model for LLM suggestion on interaction terms."""
    interaction_needed: Optional[bool] = Field(None, description="True if an interaction term is strongly suggested by the query or context. LLM should provide true, false, or omit for None.")
    interaction_variable: Optional[str] = Field(None, description="The name of the covariate that should interact with the treatment. Null if not applicable or if the interaction is complex/multiple.")
    reasoning: Optional[str] = Field(None, description="Brief reasoning for the suggestion for or against an interaction term.")

# --- Pydantic models for Tool Inputs/Outputs and Data Structures ---

class TemporalStructure(BaseModel):
    """Represents detected temporal structure in the data."""
    has_temporal_structure: bool
    temporal_columns: List[str]
    is_panel_data: bool
    id_column: Optional[str] = None
    time_column: Optional[str] = None
    time_periods: Optional[int] = None
    units: Optional[int] = None

class DatasetInfo(BaseModel):
    """Basic information about the dataset file."""
    num_rows: int
    num_columns: int
    file_path: str
    file_name: str

class DatasetAnalysis(BaseModel):
    """Results from the dataset analysis component."""
    dataset_info: DatasetInfo
    columns: List[str]
    potential_treatments: List[str]
    potential_outcomes: List[str]
    temporal_structure_detected: bool
    panel_data_detected: bool
    potential_instruments_detected: bool
    discontinuities_detected: bool
    temporal_structure: TemporalStructure
    column_categories: Optional[Dict[str, str]] = None
    column_nunique_counts: Optional[Dict[str, int]] = None
    sample_size: int
    num_covariates_estimate: int
    per_group_summary_stats: Optional[Dict[str, Dict[str, Any]]] = None
    potential_instruments: Optional[List[str]] = None
    overlap_assessment: Optional[Dict[str, Any]] = None

# --- Model for Dataset Analyzer Tool Output ---

class DatasetAnalyzerOutput(BaseModel):
    """Structured output for the dataset analyzer tool."""
    analysis_results: DatasetAnalysis
    dataset_description: Optional[str] = None
    workflow_state: Dict[str, Any]

#TODO make query info consistent with the Data analysis out put
class QueryInfo(BaseModel):
    """Information extracted from the user's initial query."""
    query_text: str
    potential_treatments: Optional[List[str]] = None
    potential_outcomes: Optional[List[str]] = None
    covariates_hints: Optional[List[str]] = None
    instrument_hints: Optional[List[str]] = None
    running_variable_hints: Optional[List[str]] = None
    cutoff_value_hint: Optional[Union[float, int]] = None

class QueryInterpreterInput(BaseModel):
    """Input structure for the query interpreter tool."""
    query_info: QueryInfo
    dataset_analysis: DatasetAnalysis
    dataset_description: str
    # Add original_query if it should be part of the standard input
    original_query: Optional[str] = None

class Variables(BaseModel):
    """Structured variables identified by the query interpreter component."""
    treatment_variable: Optional[str] = None
    treatment_variable_type: Optional[str] = Field(None, description="Type of the treatment variable (e.g., 'binary', 'continuous', 'categorical_multi_value')")
    outcome_variable: Optional[str] = None
    instrument_variable: Optional[str] = None
    covariates: Optional[List[str]] = Field(default_factory=list)
    time_variable: Optional[str] = None
    group_variable: Optional[str] = None # Often the unit ID
    running_variable: Optional[str] = None
    cutoff_value: Optional[Union[float, int]] = None
    is_rct: Optional[bool] = Field(False, description="Flag indicating if the dataset is from an RCT.")
    treatment_reference_level: Optional[str] = Field(None, description="The specified reference/control level for a multi-valued treatment variable.")
    interaction_term_suggested: Optional[bool] = Field(False, description="Whether the query or context suggests an interaction term with the treatment might be relevant.")
    interaction_variable_candidate: Optional[str] = Field(None, description="The covariate identified as a candidate for interaction with the treatment.")
    
class QueryInterpreterOutput(BaseModel):
    """Structured output for the query interpreter tool."""
    variables: Variables 
    dataset_analysis: DatasetAnalysis 
    dataset_description: Optional[str] 
    workflow_state: Dict[str, Any]
    original_query: Optional[str] = None

# Input model for Method Selector Tool
class MethodSelectorInput(BaseModel):
    """Input structure for the method selector tool."""
    variables: Variables# Uses the Variables model identified by QueryInterpreter
    dataset_analysis: DatasetAnalysis # Uses the DatasetAnalysis model
    dataset_description: Optional[str] = None
    original_query: Optional[str] = None
    # Note: is_rct is expected inside inputs.variables

# --- Models for Method Validator Tool --- 

class MethodInfo(BaseModel):
    """Information about the selected causal inference method."""
    selected_method: Optional[str] = None
    method_name: Optional[str] = None # Often a title-cased version for display
    method_justification: Optional[str] = None
    method_assumptions: Optional[List[str]] = Field(default_factory=list)
    # Add alternative methods if it should be part of the standard info passed around
    alternative_methods: Optional[List[str]] = Field(default_factory=list)

class MethodValidatorInput(BaseModel):
    """Input structure for the method validator tool."""
    method_info: MethodInfo
    variables: Variables
    dataset_analysis: DatasetAnalysis
    dataset_description: Optional[str] = None
    original_query: Optional[str] = None

# --- Model for Method Executor Tool --- 

class MethodExecutorInput(BaseModel):
    """Input structure for the method executor tool."""
    method: str = Field(..., description="The causal method name (use recommended method if validation failed).")
    variables: Variables # Contains T, O, C, etc.
    dataset_path: str 
    dataset_analysis: DatasetAnalysis
    dataset_description: Optional[str] = None
    # Include validation_info from validator output if needed by estimator or LLM assist later?
    validation_info: Optional[Any] = None 
    original_query: Optional[str] = None
# --- Model for Explanation Generator Tool --- 

class ExplainerInput(BaseModel):
    """Input structure for the explanation generator tool."""
    # Based on expected output from method_executor_tool and validator
    method_info: MethodInfo 
    validation_info: Optional[Dict[str, Any]] = None # From validator tool
    variables: Variables
    results: Dict[str, Any] # Numerical results from executor
    dataset_analysis: DatasetAnalysis
    dataset_description: Optional[str] = None
    # Add original query if needed for explanation context
    original_query: Optional[str] = None 

# Add other shared models/schemas below as needed. 

class FormattedOutput(BaseModel):
    """
    Structured output containing the final formatted results and explanations
    from a causal analysis run.
    """
    query: str = Field(description="The original user query.")
    method_used: str = Field(description="The user-friendly name of the causal inference method used.")
    causal_effect: Optional[float] = Field(None, description="The point estimate of the causal effect.")
    standard_error: Optional[float] = Field(None, description="The standard error of the causal effect estimate.")
    confidence_interval: Optional[Tuple[Optional[float], Optional[float]]] = Field(None, description="The confidence interval for the causal effect (e.g., 95% CI).")
    p_value: Optional[float] = Field(None, description="The p-value associated with the causal effect estimate.")
    summary: str = Field(description="A concise summary paragraph interpreting the main findings.")
    method_explanation: Optional[str] = Field("", description="Explanation of the causal inference method used.")
    interpretation_guide: Optional[str] = Field("", description="Guidance on how to interpret the results.")
    limitations: Optional[List[str]] = Field(default_factory=list, description="List of limitations or potential issues with the analysis.")
    assumptions: Optional[str] = Field("", description="Discussion of the key assumptions underlying the method and their validity.")
    practical_implications: Optional[str] = Field("", description="Discussion of the practical implications or significance of the findings.")
    # Optionally add dataset_analysis and dataset_description if they should be part of the final structure
    # dataset_analysis: Optional[DatasetAnalysis] = None # Example if using DatasetAnalysis model
    # dataset_description: Optional[str] = None

    # This model itself doesn't include workflow_state, as it represents the *content*
    # The tool using this component will add the workflow_state separately. 

class LLMParameterDetails(BaseModel):
    parameter_name: str = Field(description="The full parameter name as found in the model results.")
    estimate: float
    p_value: float
    conf_int_low: float
    conf_int_high: float
    std_err: float
    reasoning: Optional[str] = Field(None, description="Brief reasoning for selecting this parameter and its values.")

class LLMTreatmentEffectResults(BaseModel):
    effects: Optional[Dict[str, LLMParameterDetails]] = Field(description="Dictionary where keys are treatment level names (e.g., 'LevelA', 'LevelB' if multi-level) or a generic key like 'treatment_effect' for binary/continuous treatments. Values are the statistical details for that effect.")
    all_parameters_successfully_identified: Optional[bool] = Field(description="True if all expected treatment effect parameters were identified and their values extracted, False otherwise.")
    overall_reasoning: Optional[str] = Field(None, description="Overall reasoning for the extraction process or if issues were encountered.")

class RelevantParamInfo(BaseModel):
    param_name: str = Field(description="The exact parameter name as it appears in the statsmodels results.")
    param_index: int = Field(description="The index of this parameter in the original list of parameter names.")

class LLMIdentifiedRelevantParams(BaseModel):
    identified_params: List[RelevantParamInfo] = Field(description="A list of parameters identified as relevant to the query or representing all treatment effects for a general query.")
    all_parameters_successfully_identified: bool = Field(description="True if LLM is confident it identified all necessary params based on query type (e.g., all levels for a general query).") 

Architecture
============

This document provides a comprehensive overview of the Causal AI Scientist (CAIS) architecture, detailing the autonomous agent system, component interactions, and design patterns that enable automated causal inference analysis.

.. contents::
   :local:
   :depth: 3

System Overview
---------------

CAIS is an autonomous agent system that combines Large Language Models (LLMs) with causal inference methods to automatically analyze datasets and provide causal insights. The system follows a multi-stage workflow where each component specializes in a specific aspect of the causal analysis pipeline.

**Core Design Principles:**

* **Modularity**: Each component has a single responsibility and well-defined interfaces
* **Extensibility**: New causal methods and LLM providers can be easily integrated
* **Reproducibility**: Deterministic analysis with comprehensive logging and state management
* **Robustness**: Error handling and validation at each stage of the pipeline

High-Level Architecture
-----------------------

.. mermaid::

   graph TB
       subgraph "User Interface Layer"
           CLI[CLI Interface]
           API[Python API]
           NB[Jupyter Notebooks]
       end
       
       subgraph "Agent Orchestration Layer"
           AGENT[CausalAgent]
           EXEC[AgentExecutor]
           MEM[ConversationMemory]
       end
       
       subgraph "Tool Layer"
           IP[InputParserTool]
           DA[DatasetAnalyzerTool]
           QI[QueryInterpreterTool]
           MS[MethodSelectorTool]
           MV[MethodValidatorTool]
           ME[MethodExecutorTool]
           EG[ExplanationGeneratorTool]
           OF[OutputFormatterTool]
       end
       
       subgraph "Component Layer"
           IPC[InputParser]
           DAC[DatasetAnalyzer]
           QIC[QueryInterpreter]
           DT[DecisionTree]
           DTLLM[DecisionTreeLLM]
           MVC[MethodValidator]
           EGC[ExplanationGenerator]
           OFC[OutputFormatter]
           SM[StateManager]
       end
       
       subgraph "Method Layer"
           EXP[Experimental Methods]
           QUASI[Quasi-Experimental]
           OBS[Observational Methods]
       end
       
       subgraph "Infrastructure Layer"
           LLM[LLM Providers]
           DATA[Data Storage]
           SYNTH[Synthetic Data]
       end
       
       CLI --> AGENT
       API --> AGENT
       NB --> AGENT
       
       AGENT --> EXEC
       EXEC --> MEM
       EXEC --> IP
       
       IP --> IPC
       DA --> DAC
       QI --> QIC
       MS --> DT
       MS --> DTLLM
       MV --> MVC
       ME --> EXP
       ME --> QUASI
       ME --> OBS
       EG --> EGC
       OF --> OFC
       
       DT --> LLM
       DTLLM --> LLM
       DAC --> DATA
       SYNTH --> DATA

Agent Workflow
--------------

The CAIS agent follows a strict 8-step workflow, where each step is handled by a specialized tool that wraps a corresponding component:

**1. Input Parsing**
   * **Tool**: ``input_parser_tool``
   * **Component**: ``InputParser``
   * **Purpose**: Parse user query and extract initial variable hints
   * **Output**: Structured query information and dataset path

**2. Dataset Analysis**
   * **Tool**: ``dataset_analyzer_tool``
   * **Component**: ``DatasetAnalyzer``
   * **Purpose**: Analyze dataset structure, variables, and basic statistics
   * **Output**: Dataset characteristics and variable information

**3. Query Interpretation**
   * **Tool**: ``query_interpreter_tool``
   * **Component**: ``QueryInterpreter``
   * **Purpose**: Identify treatment, outcome, and other causal variables
   * **Output**: Structured variable assignments and analysis context

**4. Method Selection**
   * **Tool**: ``method_selector_tool``
   * **Component**: ``DecisionTree`` + ``DecisionTreeLLM``
   * **Purpose**: Select appropriate causal inference method using decision tree logic
   * **Output**: Recommended method with justification

**5. Method Validation**
   * **Tool**: ``method_validator_tool``
   * **Component**: ``MethodValidator``
   * **Purpose**: Validate method selection and check assumptions
   * **Output**: Validated method configuration

**6. Method Execution**
   * **Tool**: ``method_executor_tool``
   * **Component**: Method-specific implementations
   * **Purpose**: Execute the selected causal inference method
   * **Output**: Statistical results and diagnostics

**7. Explanation Generation**
   * **Tool**: ``explanation_generator_tool``
   * **Component**: ``ExplanationGenerator``
   * **Purpose**: Generate human-readable explanations of results
   * **Output**: Interpreted results with context

**8. Output Formatting**
   * **Tool**: ``output_formatter_tool``
   * **Component**: ``OutputFormatter``
   * **Purpose**: Format final results for user consumption
   * **Output**: Structured final output

Component Architecture
----------------------

Core Components
~~~~~~~~~~~~~~~

**CausalAgent (``causal_agent/agent.py``)**

The main orchestrator that coordinates the entire analysis workflow:

.. code-block:: python

   class CausalAgent:
       """Main agent class that orchestrates causal analysis workflow"""
       
       def __init__(self, llm: BaseChatModel):
           self.llm = llm
           self.tools = self._initialize_tools()
           self.executor = self._create_executor()
       
       def run_analysis(self, query: str, dataset_path: str) -> Dict[str, Any]:
           """Execute the complete causal analysis workflow"""

**Key Features:**
* LangChain-based agent architecture
* Tool binding and execution management
* Memory management for conversation context
* Error handling and retry logic

**StateManager (``causal_agent/components/state_manager.py``)**

Manages workflow state and data flow between components:

.. code-block:: python

   @dataclass
   class WorkflowState:
       current_step: str
       completed_steps: List[str]
       data: Dict[str, Any]
       errors: List[str]
       
   def create_workflow_state_update(step: str, data: Dict) -> Dict:
       """Create standardized state update for workflow tracking"""

Analysis Components
~~~~~~~~~~~~~~~~~~~

**InputParser (``causal_agent/components/input_parser.py``)**

Parses user queries and extracts initial variable hints:

* Natural language processing of causal questions
* Extraction of potential treatment and outcome variables
* Dataset path validation and preprocessing
* Query normalization and standardization

**DatasetAnalyzer (``causal_agent/components/dataset_analyzer.py``)**

Analyzes dataset structure and characteristics:

* Column type detection and validation
* Missing value analysis
* Basic statistical summaries
* Data quality assessment
* Variable relationship detection

**QueryInterpreter (``causal_agent/components/query_interpreter.py``)**

Maps user queries to specific causal variables:

* Treatment variable identification
* Outcome variable identification
* Covariate selection
* Instrumental variable detection
* Time variable identification for panel data

**DecisionTree (``causal_agent/components/decision_tree.py``)**

Rule-based method selection logic:

.. code-block:: python

   def rule_based_select_method(
       variables: Variables,
       dataset_analysis: DatasetAnalysis,
       excluded_methods: Optional[List[str]] = None
   ) -> Dict[str, Any]:
       """
       Select causal method using rule-based decision tree logic
       
       Decision Flow:
       1. Check if RCT data -> Experimental methods
       2. Check for time variation -> Difference-in-Differences
       3. Check for instruments -> Instrumental Variables
       4. Check for discontinuity -> Regression Discontinuity
       5. Default to observational methods
       """

**DecisionTreeLLM (``causal_agent/components/decision_tree_llm.py``)**

LLM-enhanced method selection with reasoning:

* Contextual method selection using LLM reasoning
* Assumption checking and validation
* Alternative method suggestions
* Detailed justification generation

Tool Architecture
-----------------

The tool layer provides LangChain-compatible interfaces to the core components. Each tool follows a consistent pattern:

**Tool Structure Pattern:**

.. code-block:: python

   from langchain_core.tools import tool
   from causal_agent.models import ToolInputModel, ToolOutputModel
   
   @tool(args_schema=ToolInputModel)
   def example_tool(input_param: str) -> Dict[str, Any]:
       """
       Tool description for LLM understanding
       
       Args:
           input_param: Description of input parameter
           
       Returns:
           Structured output dictionary
       """
       # Input validation
       # Component execution
       # Output formatting
       # State management
       return structured_output

**Tool Responsibilities:**

* **Input Validation**: Ensure inputs match expected schema
* **Component Delegation**: Call appropriate component functions
* **Output Standardization**: Format outputs for next tool in chain
* **Error Handling**: Graceful error handling and reporting
* **State Updates**: Update workflow state for tracking

Method Implementation Architecture
----------------------------------

Causal inference methods are organized into three categories:

**Experimental Methods (``causal_agent/methods/experimental/``)**

For randomized controlled trials and experimental data:

* **Difference in Means**: Simple treatment effect estimation
* **Randomized Controlled Trials**: Full RCT analysis with diagnostics

**Quasi-Experimental Methods (``causal_agent/methods/quasi_experimental/``)**

For natural experiments and quasi-experimental designs:

* **Difference-in-Differences**: Panel data with treatment timing variation
* **Instrumental Variables**: Using instruments to address endogeneity
* **Regression Discontinuity**: Exploiting discontinuous treatment assignment

**Observational Methods (``causal_agent/methods/observational/``)**

For observational data with selection concerns:

* **Propensity Score Matching**: Matching on propensity scores
* **Propensity Score Weighting**: Inverse probability weighting
* **Backdoor Adjustment**: Controlling for confounders
* **Linear Regression**: Parametric causal effect estimation

**Method Interface Pattern:**

.. code-block:: python

   class CausalMethod:
       """Base class for all causal inference methods"""
       
       def __init__(self, **kwargs):
           self.method_name = self.__class__.__name__
           self.assumptions = self._define_assumptions()
       
       def estimate(self, data: pd.DataFrame, variables: Variables) -> Results:
           """Execute the causal inference method"""
           
       def diagnose(self, data: pd.DataFrame, variables: Variables) -> Diagnostics:
           """Run diagnostic tests for method assumptions"""
           
       def interpret(self, results: Results) -> Interpretation:
           """Generate interpretation of results"""

LLM Integration Architecture
----------------------------

CAIS integrates with multiple LLM providers through a unified interface:

**LLM Provider Support:**

* **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo
* **Anthropic**: Claude-3 (Haiku, Sonnet, Opus)
* **Google**: Gemini Pro, Gemini Pro Vision
* **Local Models**: Via Ollama or similar frameworks

**Configuration Management (``causal_agent/config.py``)**

.. code-block:: python

   def get_llm_client(
       provider: Optional[str] = None,
       model: Optional[str] = None,
       temperature: float = 0.0,
       **kwargs
   ) -> BaseChatModel:
       """
       Factory function for LLM client creation
       
       Supports multiple providers with consistent interface
       """

**Prompt Engineering Patterns**

The system uses structured prompts for different analysis phases:

* **Variable Identification Prompts**: Extract causal variables from queries
* **Method Selection Prompts**: Reason about appropriate methods
* **Result Interpretation Prompts**: Generate explanations
* **Assumption Checking Prompts**: Validate method assumptions

**Prompt Template Structure:**

.. code-block:: python

   VARIABLE_IDENTIFICATION_PROMPT = """
   You are an expert in causal inference. Your task is to identify the {variable_type} 
   variable in a dataset for causal analysis.
   
   User Query: {query}
   Dataset Description: {description}
   Available Variables: {column_info}
   
   Based on the information provided, determine which variable serves as the {variable_type}.
   
   Return your response as a valid JSON object:
   {{ "{variable_type}_variable": "COLUMN_NAME_OR_NULL" }}
   """

**Response Processing Pipeline:**

1. **Prompt Construction**: Build context-specific prompts
2. **LLM Invocation**: Call LLM with structured prompts
3. **Response Parsing**: Extract structured information from responses
4. **Validation**: Validate LLM outputs against expected schemas
5. **Error Handling**: Retry logic for malformed responses

Data Flow Architecture
----------------------

**Data Models (``causal_agent/models.py``)**

The system uses Pydantic models for type safety and validation:

.. code-block:: python

   @dataclass
   class Variables:
       treatment_variable: Optional[str]
       outcome_variable: Optional[str]
       covariates: List[str]
       time_variable: Optional[str]
       instrument_variable: Optional[str]
       running_variable: Optional[str]
       cutoff_value: Optional[float]
       is_rct: Optional[bool]
   
   @dataclass
   class DatasetAnalysis:
       column_info: Dict[str, Any]
       summary_stats: Dict[str, Any]
       missing_values: Dict[str, int]
       data_types: Dict[str, str]
       n_observations: int
       n_variables: int

**State Management Flow:**

.. mermaid::

   graph LR
       INPUT[User Input] --> PARSE[Parse Input]
       PARSE --> ANALYZE[Analyze Dataset]
       ANALYZE --> INTERPRET[Interpret Query]
       INTERPRET --> SELECT[Select Method]
       SELECT --> VALIDATE[Validate Method]
       VALIDATE --> EXECUTE[Execute Method]
       EXECUTE --> EXPLAIN[Generate Explanation]
       EXPLAIN --> FORMAT[Format Output]
       
       PARSE -.-> STATE[(Workflow State)]
       ANALYZE -.-> STATE
       INTERPRET -.-> STATE
       SELECT -.-> STATE
       VALIDATE -.-> STATE
       EXECUTE -.-> STATE
       EXPLAIN -.-> STATE
       FORMAT -.-> STATE

**Error Handling Strategy:**

* **Validation Errors**: Input validation with clear error messages
* **LLM Errors**: Retry logic with exponential backoff
* **Method Errors**: Graceful fallback to alternative methods
* **Data Errors**: Comprehensive data quality checks

Testing Architecture
--------------------

**Test Organization:**

.. code-block:: text

   tests/
   ├── unit/                    # Unit tests for individual components
   │   ├── components/          # Component-specific tests
   │   ├── methods/             # Method implementation tests
   │   └── tools/               # Tool interface tests
   ├── integration/             # Integration tests for workflows
   │   ├── test_agent_workflows.py
   │   └── test_llm_integration.py
   ├── end_to_end/             # Full pipeline tests
   │   └── test_complete_workflows.py
   └── performance/            # Performance and scalability tests
       └── test_method_performance.py

**Testing Strategies:**

* **Unit Testing**: Individual component functionality
* **Integration Testing**: Component interaction and data flow
* **End-to-End Testing**: Complete workflow validation
* **Performance Testing**: Scalability and efficiency metrics
* **LLM Testing**: Mock LLM responses for deterministic testing

**Synthetic Data Testing:**

The system includes comprehensive synthetic data generation for testing:

* **Method-Specific Datasets**: Tailored to test specific causal methods
* **Edge Case Generation**: Datasets that test boundary conditions
* **Assumption Violation Testing**: Data that violates method assumptions
* **Performance Benchmarking**: Large-scale datasets for performance testing

Extension Points
----------------

**Adding New Causal Methods:**

1. **Implement Method Class**: Extend base ``CausalMethod`` class
2. **Add to Decision Tree**: Update decision logic in ``DecisionTree``
3. **Create Tests**: Comprehensive test suite for new method
4. **Update Documentation**: Method-specific documentation

**Adding New LLM Providers:**

1. **Extend Config**: Add provider configuration in ``config.py``
2. **Implement Client**: Create provider-specific client wrapper
3. **Test Integration**: Validate with existing prompts and workflows
4. **Update Documentation**: Provider-specific setup instructions

**Adding New Data Sources:**

1. **Extend DatasetAnalyzer**: Add support for new data formats
2. **Update Validation**: Extend data validation logic
3. **Test Compatibility**: Ensure compatibility with existing methods
4. **Document Integration**: Usage examples and limitations

Performance Considerations
--------------------------

**Optimization Strategies:**

* **Caching**: LLM response caching for repeated queries
* **Parallel Processing**: Concurrent execution where possible
* **Memory Management**: Efficient data handling for large datasets
* **Method Selection**: Fast rule-based filtering before LLM reasoning

**Scalability Patterns:**

* **Batch Processing**: Support for multiple dataset analysis
* **Resource Management**: Memory and compute resource optimization
* **Error Recovery**: Robust error handling for production use
* **Monitoring**: Comprehensive logging and metrics collection

Security and Privacy
--------------------

**Data Protection:**

* **Local Processing**: Option for local-only analysis
* **API Key Management**: Secure handling of LLM provider credentials
* **Data Anonymization**: Support for anonymized dataset analysis
* **Audit Logging**: Comprehensive audit trails for compliance

**LLM Security:**

* **Prompt Injection Protection**: Input sanitization and validation
* **Response Validation**: Structured output validation
* **Rate Limiting**: Respect provider rate limits and quotas
* **Error Handling**: Secure error messages without data leakage

This architecture enables CAIS to provide robust, scalable, and extensible automated causal inference capabilities while maintaining high standards for reproducibility and reliability.
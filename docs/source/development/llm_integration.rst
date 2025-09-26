LLM Integration
===============

This document provides comprehensive guidance on LLM integration patterns, prompt engineering strategies, and response processing techniques used throughout the CAIS system.

.. contents::
   :local:
   :depth: 3

Overview
--------

CAIS leverages Large Language Models (LLMs) at multiple stages of the causal analysis workflow to provide intelligent reasoning, variable identification, method selection, and result interpretation. The system is designed to work with multiple LLM providers while maintaining consistent behavior and reliability.

**Key Integration Points:**

* **Variable Identification**: Extract causal variables from natural language queries
* **Method Selection**: Reason about appropriate causal inference methods
* **Assumption Checking**: Validate method assumptions using domain knowledge
* **Result Interpretation**: Generate human-readable explanations of statistical results
* **Error Recovery**: Provide intelligent fallback strategies when methods fail

LLM Provider Architecture
-------------------------

Supported Providers
~~~~~~~~~~~~~~~~~~~

CAIS supports multiple LLM providers through a unified interface:

.. code-block:: python

   # causal_agent/config.py
   
   SUPPORTED_PROVIDERS = {
       "openai": {
           "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
           "client_class": "ChatOpenAI"
       },
       "anthropic": {
           "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
           "client_class": "ChatAnthropic"
       },
       "google": {
           "models": ["gemini-pro", "gemini-pro-vision"],
           "client_class": "ChatGoogleGenerativeAI"
       },
       "ollama": {
           "models": ["llama2", "mistral", "codellama"],
           "client_class": "ChatOllama"
       }
   }

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

The LLM client factory provides consistent configuration across providers:

.. code-block:: python

   def get_llm_client(
       provider: Optional[str] = None,
       model: Optional[str] = None,
       temperature: float = 0.0,
       max_tokens: Optional[int] = None,
       **kwargs
   ) -> BaseChatModel:
       """
       Factory function for creating LLM clients with consistent configuration.
       
       Args:
           provider: LLM provider name (openai, anthropic, google, ollama)
           model: Specific model name within provider
           temperature: Sampling temperature (0.0 for deterministic)
           max_tokens: Maximum tokens in response
           **kwargs: Provider-specific configuration options
           
       Returns:
           Configured LLM client instance
       """
       # Environment variable fallbacks
       provider = provider or os.getenv("LLM_PROVIDER", "openai")
       model = model or os.getenv("LLM_MODEL", "gpt-4")
       
       # Provider-specific client creation
       if provider == "openai":
           return ChatOpenAI(
               model=model,
               temperature=temperature,
               max_tokens=max_tokens,
               api_key=os.getenv("OPENAI_API_KEY"),
               **kwargs
           )
       elif provider == "anthropic":
           return ChatAnthropic(
               model=model,
               temperature=temperature,
               max_tokens=max_tokens,
               api_key=os.getenv("ANTHROPIC_API_KEY"),
               **kwargs
           )
       # ... additional providers

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

LLM configuration is managed through environment variables:

.. code-block:: bash

   # Basic configuration
   export LLM_PROVIDER=openai
   export LLM_MODEL=gpt-4
   export OPENAI_API_KEY=your_api_key_here
   
   # Advanced configuration
   export LLM_TEMPERATURE=0.0
   export LLM_MAX_TOKENS=2000
   export LLM_TIMEOUT=30
   
   # Provider-specific settings
   export ANTHROPIC_API_KEY=your_anthropic_key
   export GOOGLE_API_KEY=your_google_key

Prompt Engineering Patterns
----------------------------

Core Prompt Structure
~~~~~~~~~~~~~~~~~~~~~

All CAIS prompts follow a consistent structure for reliability and maintainability:

.. code-block:: python

   PROMPT_TEMPLATE = """
   You are an expert in {domain}. Your task is to {task_description}.
   
   Context:
   {context_information}
   
   Input Data:
   {input_data}
   
   Instructions:
   {specific_instructions}
   
   Output Format:
   {output_format_specification}
   
   Examples:
   {examples_if_applicable}
   """

**Template Components:**

* **Role Definition**: Establish expertise and context
* **Task Description**: Clear statement of what needs to be accomplished
* **Context Information**: Relevant background and constraints
* **Input Data**: Structured data for analysis
* **Specific Instructions**: Detailed guidance for the task
* **Output Format**: Exact specification of expected response format
* **Examples**: Concrete examples when helpful

Variable Identification Prompts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Treatment Variable Identification:**

.. code-block:: python

   TREATMENT_VAR_IDENTIFICATION_PROMPT = """
   You are an expert in causal inference. Your task is to identify the **treatment variable** 
   in a dataset to perform causal analysis that answers the user's query.
   
   User Query:
   {query}
   
   Dataset Description:
   {description}
   
   Available Variables:
   {column_info}
   
   The treatment variable is the intervention, policy, or exposure whose causal effect 
   we want to estimate. It should be:
   - Clearly mentioned or implied in the user's query
   - Present in the available variables
   - Conceptually meaningful as a treatment/intervention
   
   If multiple variables could serve as treatment, select the one most directly 
   related to the user's causal question.
   
   If no clear treatment variable can be identified, return null.
   
   Return your response as a valid JSON object:
   {{ "treatment_variable": "COLUMN_NAME_OR_NULL" }}
   """

**Outcome Variable Identification:**

.. code-block:: python

   OUTCOME_VAR_IDENTIFICATION_PROMPT = """
   You are an expert in causal inference. Your task is to identify the **outcome variable** 
   in a dataset to perform causal analysis that answers the user's query.
   
   User Query:
   {query}
   
   Dataset Description:
   {description}
   
   Available Variables:
   {column_info}
   
   The outcome variable is the dependent variable whose value we believe is causally 
   affected by the treatment. It should be:
   - The main outcome of interest mentioned in the query
   - Present in the available variables
   - Measured after or contemporaneously with the treatment
   
   Common outcome patterns in queries:
   - "effect of X on Y" → Y is the outcome
   - "impact of X on Y" → Y is the outcome
   - "does X cause Y" → Y is the outcome
   
   Return your response as a valid JSON object:
   {{ "outcome_variable": "COLUMN_NAME_OR_NULL" }}
   """

Method Selection Prompts
~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Tree Reasoning:**

.. code-block:: python

   METHOD_SELECTION_REASONING_PROMPT = """
   You are an expert in causal inference method selection. Analyze the dataset and 
   variables to recommend the most appropriate causal inference method.
   
   Dataset Analysis:
   {dataset_analysis}
   
   Identified Variables:
   - Treatment: {treatment_variable}
   - Outcome: {outcome_variable}
   - Covariates: {covariates}
   - Time Variable: {time_variable}
   - Instrument: {instrument_variable}
   - Running Variable: {running_variable}
   - Is RCT: {is_rct}
   
   Available Methods:
   {available_methods}
   
   Selection Criteria:
   1. **Experimental Methods** (RCT, Difference in Means):
      - Use when is_rct=true or treatment is randomly assigned
      - Strongest causal identification
   
   2. **Quasi-Experimental Methods**:
      - **Difference-in-Differences**: Time variation + treatment timing variation
      - **Instrumental Variables**: Valid instrument available
      - **Regression Discontinuity**: Running variable with cutoff
   
   3. **Observational Methods**:
      - **Propensity Score Methods**: Rich set of covariates
      - **Backdoor Adjustment**: Sufficient covariates to block confounding
      - **Linear Regression**: Simple baseline method
   
   Consider:
   - Data structure and available variables
   - Method assumptions and their plausibility
   - Strength of causal identification
   - Sample size and statistical power
   
   Return your analysis as JSON:
   {{
       "recommended_method": "method_name",
       "confidence": 0.0-1.0,
       "reasoning": "detailed explanation",
       "assumptions": ["list of key assumptions"],
       "alternatives": ["alternative methods"],
       "concerns": ["potential issues"]
   }}
   """

Result Interpretation Prompts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statistical Results Interpretation:**

.. code-block:: python

   RESULT_INTERPRETATION_PROMPT = """
   You are an expert in causal inference and statistical interpretation. 
   Provide a clear, comprehensive interpretation of causal analysis results.
   
   Analysis Details:
   - Method Used: {method_name}
   - Treatment Variable: {treatment_variable}
   - Outcome Variable: {outcome_variable}
   - Sample Size: {sample_size}
   
   Statistical Results:
   - Effect Estimate: {effect_estimate}
   - Standard Error: {standard_error}
   - 95% Confidence Interval: {confidence_interval}
   - P-value: {p_value}
   
   Diagnostic Tests:
   {diagnostic_results}
   
   Method Assumptions:
   {method_assumptions}
   
   Provide interpretation covering:
   
   1. **Effect Size and Direction**:
      - Magnitude and practical significance
      - Direction of causal effect
      - Units and scale interpretation
   
   2. **Statistical Significance**:
      - P-value interpretation
      - Confidence interval meaning
      - Statistical vs practical significance
   
   3. **Assumption Assessment**:
      - How well assumptions are satisfied
      - Diagnostic test results
      - Reliability of causal interpretation
   
   4. **Limitations and Caveats**:
      - Method-specific limitations
      - Potential sources of bias
      - Generalizability concerns
   
   5. **Practical Implications**:
      - Real-world meaning of results
      - Policy or decision implications
      - Recommendations for action
   
   Format as clear, accessible explanation suitable for non-experts while 
   maintaining statistical rigor.
   """

Response Processing Architecture
-------------------------------

Structured Output Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~

CAIS uses structured output parsing to ensure reliable LLM responses:

.. code-block:: python

   from typing import Dict, Any, Optional
   import json
   import re
   from pydantic import BaseModel, ValidationError
   
   class LLMResponseParser:
       """Parser for structured LLM responses with validation and error handling"""
       
       def __init__(self, expected_schema: Optional[BaseModel] = None):
           self.expected_schema = expected_schema
       
       def parse_json_response(self, response: str) -> Dict[str, Any]:
           """
           Parse JSON response from LLM with error handling and validation.
           
           Args:
               response: Raw LLM response string
               
           Returns:
               Parsed and validated JSON object
               
           Raises:
               ValueError: If response cannot be parsed or validated
           """
           try:
               # Extract JSON from response (handle markdown formatting)
               json_str = self._extract_json(response)
               
               # Parse JSON
               parsed = json.loads(json_str)
               
               # Validate against schema if provided
               if self.expected_schema:
                   validated = self.expected_schema(**parsed)
                   return validated.dict()
               
               return parsed
               
           except (json.JSONDecodeError, ValidationError) as e:
               raise ValueError(f"Failed to parse LLM response: {e}")
       
       def _extract_json(self, response: str) -> str:
           """Extract JSON from potentially formatted response"""
           # Remove markdown code blocks
           response = re.sub(r'```json\s*', '', response)
           response = re.sub(r'```\s*$', '', response)
           
           # Find JSON object
           json_match = re.search(r'\{.*\}', response, re.DOTALL)
           if json_match:
               return json_match.group(0)
           
           # If no JSON found, try the entire response
           return response.strip()

Response Validation Schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define Pydantic schemas for structured validation:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List, Optional
   
   class VariableIdentificationResponse(BaseModel):
       """Schema for variable identification responses"""
       treatment_variable: Optional[str] = Field(None, description="Identified treatment variable")
       outcome_variable: Optional[str] = Field(None, description="Identified outcome variable")
       covariates: List[str] = Field(default_factory=list, description="Identified covariates")
       confidence: float = Field(ge=0.0, le=1.0, description="Confidence in identification")
       reasoning: str = Field(description="Explanation of identification logic")
   
   class MethodSelectionResponse(BaseModel):
       """Schema for method selection responses"""
       recommended_method: str = Field(description="Recommended causal method")
       confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendation")
       reasoning: str = Field(description="Detailed reasoning for selection")
       assumptions: List[str] = Field(description="Key method assumptions")
       alternatives: List[str] = Field(default_factory=list, description="Alternative methods")
       concerns: List[str] = Field(default_factory=list, description="Potential concerns")
   
   class ResultInterpretationResponse(BaseModel):
       """Schema for result interpretation responses"""
       effect_interpretation: str = Field(description="Interpretation of effect size")
       significance_assessment: str = Field(description="Statistical significance assessment")
       assumption_evaluation: str = Field(description="Method assumption evaluation")
       limitations: List[str] = Field(description="Analysis limitations")
       practical_implications: str = Field(description="Practical implications")

Error Handling and Retry Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement robust error handling for LLM interactions:

.. code-block:: python

   import time
   import logging
   from typing import Dict, Any, Callable
   from functools import wraps
   
   logger = logging.getLogger(__name__)
   
   def llm_retry(max_retries: int = 3, backoff_factor: float = 2.0):
       """Decorator for LLM calls with exponential backoff retry logic"""
       
       def decorator(func: Callable) -> Callable:
           @wraps(func)
           def wrapper(*args, **kwargs):
               last_exception = None
               
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   
                   except Exception as e:
                       last_exception = e
                       
                       if attempt < max_retries - 1:
                           wait_time = backoff_factor ** attempt
                           logger.warning(
                               f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                               f"Retrying in {wait_time} seconds..."
                           )
                           time.sleep(wait_time)
                       else:
                           logger.error(f"LLM call failed after {max_retries} attempts: {e}")
               
               raise last_exception
           
           return wrapper
       return decorator
   
   class LLMClient:
       """Wrapper for LLM clients with error handling and validation"""
       
       def __init__(self, llm_client, parser: LLMResponseParser):
           self.llm = llm_client
           self.parser = parser
       
       @llm_retry(max_retries=3)
       def call_with_validation(
           self, 
           prompt: str, 
           expected_schema: Optional[BaseModel] = None
       ) -> Dict[str, Any]:
           """
           Call LLM with automatic retry and response validation.
           
           Args:
               prompt: Formatted prompt string
               expected_schema: Pydantic schema for response validation
               
           Returns:
               Validated response dictionary
           """
           try:
               # Call LLM
               response = self.llm.invoke(prompt)
               response_text = response.content if hasattr(response, 'content') else str(response)
               
               # Parse and validate response
               if expected_schema:
                   self.parser.expected_schema = expected_schema
               
               parsed_response = self.parser.parse_json_response(response_text)
               
               logger.info(f"LLM call successful: {len(response_text)} characters")
               return parsed_response
               
           except Exception as e:
               logger.error(f"LLM call failed: {e}")
               raise

Prompt Optimization Strategies
------------------------------

Few-Shot Learning
~~~~~~~~~~~~~~~~~

Use examples to improve LLM performance on specific tasks:

.. code-block:: python

   FEW_SHOT_VARIABLE_IDENTIFICATION = """
   You are an expert in causal inference variable identification.
   
   Here are examples of correct variable identification:
   
   Example 1:
   Query: "What is the effect of education on income?"
   Variables: education_years, annual_income, age, gender, experience
   Response: {{"treatment_variable": "education_years", "outcome_variable": "annual_income"}}
   
   Example 2:
   Query: "Does smoking cause lung cancer?"
   Variables: smoking_status, cancer_diagnosis, age, gender, family_history
   Response: {{"treatment_variable": "smoking_status", "outcome_variable": "cancer_diagnosis"}}
   
   Example 3:
   Query: "Impact of minimum wage on employment"
   Variables: min_wage_policy, employment_rate, state, year, population
   Response: {{"treatment_variable": "min_wage_policy", "outcome_variable": "employment_rate"}}
   
   Now identify variables for this query:
   Query: {query}
   Variables: {variables}
   Response:
   """

Chain-of-Thought Reasoning
~~~~~~~~~~~~~~~~~~~~~~~~~~

Encourage step-by-step reasoning for complex decisions:

.. code-block:: python

   CHAIN_OF_THOUGHT_METHOD_SELECTION = """
   You are selecting a causal inference method. Think through this step-by-step:
   
   Step 1: Analyze the data structure
   - Is this experimental or observational data?
   - What variables are available?
   - What is the sample size?
   
   Step 2: Consider identification strategies
   - Is there random assignment?
   - Are there instruments available?
   - Is there time/policy variation?
   - Are there sufficient covariates?
   
   Step 3: Evaluate method assumptions
   - Which methods have plausible assumptions?
   - What are the key threats to identification?
   - How can assumptions be tested?
   
   Step 4: Select the best method
   - Which method provides strongest identification?
   - What are the trade-offs?
   - Are there good alternatives?
   
   Dataset: {dataset_info}
   Variables: {variables}
   
   Work through each step and provide your reasoning:
   """

Prompt Versioning and A/B Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement systematic prompt improvement:

.. code-block:: python

   class PromptManager:
       """Manager for prompt versioning and A/B testing"""
       
       def __init__(self):
           self.prompts = {}
           self.active_versions = {}
           self.performance_metrics = {}
       
       def register_prompt(
           self, 
           prompt_name: str, 
           version: str, 
           template: str,
           metadata: Dict[str, Any] = None
       ):
           """Register a prompt version"""
           if prompt_name not in self.prompts:
               self.prompts[prompt_name] = {}
           
           self.prompts[prompt_name][version] = {
               'template': template,
               'metadata': metadata or {},
               'created_at': time.time()
           }
       
       def get_prompt(self, prompt_name: str, version: str = None) -> str:
           """Get prompt template by name and version"""
           if version is None:
               version = self.active_versions.get(prompt_name, 'latest')
           
           return self.prompts[prompt_name][version]['template']
       
       def set_active_version(self, prompt_name: str, version: str):
           """Set active version for a prompt"""
           self.active_versions[prompt_name] = version
       
       def record_performance(
           self, 
           prompt_name: str, 
           version: str, 
           success: bool, 
           metrics: Dict[str, Any]
       ):
           """Record performance metrics for prompt version"""
           key = f"{prompt_name}:{version}"
           if key not in self.performance_metrics:
               self.performance_metrics[key] = []
           
           self.performance_metrics[key].append({
               'success': success,
               'metrics': metrics,
               'timestamp': time.time()
           })

Integration with Decision Tree
------------------------------

LLM-Enhanced Decision Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine rule-based logic with LLM reasoning:

.. code-block:: python

   class DecisionTreeLLMEngine:
       """LLM-enhanced decision tree for method selection"""
       
       def __init__(self, llm_client: LLMClient):
           self.llm = llm_client
           self.rule_based_engine = RuleBasedDecisionTree()
       
       def select_method(
           self, 
           variables: Variables, 
           dataset_analysis: DatasetAnalysis,
           context: Dict[str, Any] = None
       ) -> Dict[str, Any]:
           """
           Select method using combined rule-based and LLM reasoning.
           
           Args:
               variables: Identified causal variables
               dataset_analysis: Dataset characteristics
               context: Additional context for decision
               
           Returns:
               Method selection with reasoning
           """
           # First, get rule-based recommendation
           rule_based_result = self.rule_based_engine.select_method(
               variables, dataset_analysis
           )
           
           # If rule-based selection is confident, use it
           if rule_based_result['confidence'] > 0.8:
               return rule_based_result
           
           # Otherwise, use LLM for enhanced reasoning
           llm_result = self._llm_method_selection(
               variables, dataset_analysis, rule_based_result, context
           )
           
           # Combine results
           return self._combine_recommendations(rule_based_result, llm_result)
       
       def _llm_method_selection(
           self, 
           variables: Variables, 
           dataset_analysis: DatasetAnalysis,
           rule_based_result: Dict[str, Any],
           context: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Use LLM for method selection reasoning"""
           
           prompt = self._build_method_selection_prompt(
               variables, dataset_analysis, rule_based_result, context
           )
           
           response = self.llm.call_with_validation(
               prompt, MethodSelectionResponse
           )
           
           return response
       
       def _combine_recommendations(
           self, 
           rule_based: Dict[str, Any], 
           llm_based: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Combine rule-based and LLM recommendations"""
           
           # If both agree, high confidence
           if rule_based['method'] == llm_based['recommended_method']:
               return {
                   'method': rule_based['method'],
                   'confidence': min(rule_based['confidence'] + 0.2, 1.0),
                   'reasoning': f"Both rule-based and LLM reasoning agree: {llm_based['reasoning']}",
                   'assumptions': rule_based['assumptions'],
                   'alternatives': llm_based['alternatives']
               }
           
           # If they disagree, use LLM with lower confidence
           else:
               return {
                   'method': llm_based['recommended_method'],
                   'confidence': llm_based['confidence'] * 0.8,
                   'reasoning': f"LLM override of rule-based selection: {llm_based['reasoning']}",
                   'assumptions': llm_based['assumptions'],
                   'alternatives': [rule_based['method']] + llm_based['alternatives']
               }

Performance Optimization
------------------------

Caching Strategies
~~~~~~~~~~~~~~~~~~

Implement intelligent caching for LLM responses:

.. code-block:: python

   import hashlib
   from typing import Dict, Any, Optional
   
   class LLMResponseCache:
       """Cache for LLM responses to reduce API calls and improve performance"""
       
       def __init__(self, max_size: int = 1000):
           self.cache = {}
           self.max_size = max_size
           self.access_times = {}
       
       def _generate_key(self, prompt: str, model: str, temperature: float) -> str:
           """Generate cache key from prompt and parameters"""
           content = f"{prompt}:{model}:{temperature}"
           return hashlib.md5(content.encode()).hexdigest()
       
       def get(
           self, 
           prompt: str, 
           model: str, 
           temperature: float
       ) -> Optional[Dict[str, Any]]:
           """Get cached response if available"""
           key = self._generate_key(prompt, model, temperature)
           
           if key in self.cache:
               self.access_times[key] = time.time()
               return self.cache[key]
           
           return None
       
       def set(
           self, 
           prompt: str, 
           model: str, 
           temperature: float, 
           response: Dict[str, Any]
       ):
           """Cache response with LRU eviction"""
           key = self._generate_key(prompt, model, temperature)
           
           # Evict oldest if at capacity
           if len(self.cache) >= self.max_size:
               oldest_key = min(self.access_times.keys(), key=self.access_times.get)
               del self.cache[oldest_key]
               del self.access_times[oldest_key]
           
           self.cache[key] = response
           self.access_times[key] = time.time()

Batch Processing
~~~~~~~~~~~~~~~~

Optimize for multiple queries:

.. code-block:: python

   class BatchLLMProcessor:
       """Process multiple LLM requests efficiently"""
       
       def __init__(self, llm_client: LLMClient, batch_size: int = 5):
           self.llm = llm_client
           self.batch_size = batch_size
       
       def process_batch(
           self, 
           prompts: List[str], 
           schemas: List[BaseModel] = None
       ) -> List[Dict[str, Any]]:
           """Process multiple prompts in batches"""
           results = []
           
           for i in range(0, len(prompts), self.batch_size):
               batch = prompts[i:i + self.batch_size]
               batch_schemas = schemas[i:i + self.batch_size] if schemas else [None] * len(batch)
               
               # Process batch concurrently
               batch_results = self._process_concurrent_batch(batch, batch_schemas)
               results.extend(batch_results)
           
           return results
       
       def _process_concurrent_batch(
           self, 
           prompts: List[str], 
           schemas: List[BaseModel]
       ) -> List[Dict[str, Any]]:
           """Process batch of prompts concurrently"""
           import concurrent.futures
           
           with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
               futures = [
                   executor.submit(self.llm.call_with_validation, prompt, schema)
                   for prompt, schema in zip(prompts, schemas)
               ]
               
               results = []
               for future in concurrent.futures.as_completed(futures):
                   try:
                       result = future.result()
                       results.append(result)
                   except Exception as e:
                       logger.error(f"Batch processing error: {e}")
                       results.append({"error": str(e)})
               
               return results

Monitoring and Debugging
------------------------

LLM Call Logging
~~~~~~~~~~~~~~~~

Comprehensive logging for debugging and monitoring:

.. code-block:: python

   class LLMCallLogger:
       """Logger for LLM interactions with detailed metrics"""
       
       def __init__(self, log_level: str = "INFO"):
           self.logger = logging.getLogger("llm_calls")
           self.logger.setLevel(getattr(logging, log_level))
           
           # Metrics tracking
           self.call_count = 0
           self.total_tokens = 0
           self.total_cost = 0.0
           self.error_count = 0
       
       def log_call(
           self, 
           prompt: str, 
           response: str, 
           model: str,
           tokens_used: int = None,
           cost: float = None,
           duration: float = None,
           success: bool = True
       ):
           """Log LLM call with metrics"""
           self.call_count += 1
           
           if success:
               self.logger.info(
                   f"LLM Call #{self.call_count} - Model: {model}, "
                   f"Tokens: {tokens_used}, Duration: {duration:.2f}s"
               )
           else:
               self.error_count += 1
               self.logger.error(
                   f"LLM Call #{self.call_count} FAILED - Model: {model}, "
                   f"Error in response processing"
               )
           
           # Update metrics
           if tokens_used:
               self.total_tokens += tokens_used
           if cost:
               self.total_cost += cost
           
           # Log detailed information at debug level
           self.logger.debug(f"Prompt: {prompt[:200]}...")
           self.logger.debug(f"Response: {response[:200]}...")
       
       def get_metrics(self) -> Dict[str, Any]:
           """Get aggregated metrics"""
           return {
               "total_calls": self.call_count,
               "successful_calls": self.call_count - self.error_count,
               "error_rate": self.error_count / max(self.call_count, 1),
               "total_tokens": self.total_tokens,
               "total_cost": self.total_cost,
               "average_tokens_per_call": self.total_tokens / max(self.call_count, 1)
           }

Testing LLM Integration
-----------------------

Mock LLM Responses
~~~~~~~~~~~~~~~~~~

Create deterministic tests using mock responses:

.. code-block:: python

   class MockLLMClient:
       """Mock LLM client for testing with predefined responses"""
       
       def __init__(self, responses: Dict[str, str]):
           self.responses = responses
           self.call_count = 0
       
       def invoke(self, prompt: str) -> str:
           """Return predefined response based on prompt pattern"""
           self.call_count += 1
           
           # Match prompt to predefined response
           for pattern, response in self.responses.items():
               if pattern in prompt:
                   return response
           
           # Default response if no pattern matches
           return '{"error": "No mock response defined for this prompt"}'
   
   # Example usage in tests
   mock_responses = {
       "identify the treatment variable": '{"treatment_variable": "education"}',
       "identify the outcome variable": '{"outcome_variable": "income"}',
       "select causal method": '{"recommended_method": "linear_regression", "confidence": 0.8}'
   }
   
   mock_llm = MockLLMClient(mock_responses)

Integration Testing
~~~~~~~~~~~~~~~~~~~

Test LLM integration within the full workflow:

.. code-block:: python

   def test_llm_integration_workflow():
       """Test complete workflow with LLM integration"""
       
       # Use mock LLM for deterministic testing
       mock_llm = MockLLMClient(STANDARD_MOCK_RESPONSES)
       
       # Create agent with mock LLM
       agent = CausalAgent(llm=mock_llm)
       
       # Run analysis
       result = agent.run_analysis(
           query="What is the effect of education on income?",
           dataset_path="test_data.csv"
       )
       
       # Verify LLM was called appropriately
       assert mock_llm.call_count > 0
       assert "effect_estimate" in result
       assert result["method_used"] in EXPECTED_METHODS

Best Practices
--------------

Prompt Design
~~~~~~~~~~~~~

* **Be Specific**: Provide clear, unambiguous instructions
* **Use Examples**: Include few-shot examples for complex tasks
* **Structure Output**: Specify exact output format (JSON, etc.)
* **Handle Edge Cases**: Address potential ambiguities and edge cases
* **Validate Assumptions**: Make domain assumptions explicit

Error Handling
~~~~~~~~~~~~~~

* **Graceful Degradation**: Provide fallback strategies when LLM fails
* **Retry Logic**: Implement exponential backoff for transient failures
* **Input Validation**: Validate inputs before sending to LLM
* **Output Validation**: Validate LLM outputs against expected schemas
* **Logging**: Comprehensive logging for debugging and monitoring

Performance
~~~~~~~~~~~

* **Caching**: Cache responses for repeated queries
* **Batch Processing**: Process multiple requests efficiently
* **Model Selection**: Use appropriate model size for task complexity
* **Temperature Control**: Use low temperature for deterministic tasks
* **Token Management**: Optimize prompts for token efficiency

Security
~~~~~~~~

* **Input Sanitization**: Sanitize user inputs to prevent prompt injection
* **API Key Management**: Secure handling of API credentials
* **Data Privacy**: Avoid sending sensitive data to external LLMs
* **Rate Limiting**: Respect provider rate limits and quotas
* **Error Messages**: Avoid exposing sensitive information in error messages

This comprehensive LLM integration framework enables CAIS to leverage the power of large language models while maintaining reliability, performance, and security standards required for production causal inference systems.
Testing Framework
=================

This document provides comprehensive guidance on the testing framework used in CAIS, covering unit tests, integration tests, performance tests, and testing best practices for causal inference systems.

.. contents::
   :local:
   :depth: 3

Overview
--------

The CAIS testing framework is designed to ensure reliability, accuracy, and robustness of the autonomous causal inference system. It encompasses multiple testing levels and strategies:

**Testing Levels:**

* **Unit Tests**: Individual component functionality
* **Integration Tests**: Component interactions and workflows
* **End-to-End Tests**: Complete analysis pipelines
* **Performance Tests**: Scalability and efficiency
* **LLM Integration Tests**: Language model interactions

**Testing Strategies:**

* **Synthetic Data Testing**: Known ground truth validation
* **Real Data Testing**: Realistic scenario validation
* **Assumption Violation Testing**: Robustness under violations
* **Edge Case Testing**: Boundary condition handling
* **Regression Testing**: Preventing performance degradation

Test Organization
-----------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── __init__.py
   ├── conftest.py                    # Shared pytest configuration
   ├── base.py                        # Base test classes and utilities
   │
   ├── unit/                          # Unit tests
   │   ├── __init__.py
   │   ├── causal_agent/             # Agent core tests
   │   │   ├── test_agent.py
   │   │   ├── test_config.py
   │   │   └── test_models.py
   │   ├── components/               # Component tests
   │   │   ├── test_dataset_analyzer.py
   │   │   ├── test_decision_tree.py
   │   │   ├── test_input_parser.py
   │   │   ├── test_query_interpreter.py
   │   │   └── test_method_validator.py
   │   ├── methods/                  # Method implementation tests
   │   │   ├── experimental/
   │   │   ├── quasi_experimental/
   │   │   └── observational/
   │   ├── tools/                    # Tool interface tests
   │   │   ├── test_dataset_analyzer_tool.py
   │   │   ├── test_method_selector_tool.py
   │   │   └── test_method_executor_tool.py
   │   └── synthetic/                # Synthetic data tests
   │       ├── test_generator.py
   │       └── test_validation.py
   │
   ├── integration/                   # Integration tests
   │   ├── __init__.py
   │   ├── test_agent_workflows.py
   │   ├── test_llm_integration.py
   │   ├── test_method_integration.py
   │   └── test_data_flow.py
   │
   ├── end_to_end/                   # End-to-end tests
   │   ├── __init__.py
   │   ├── test_complete_workflows.py
   │   ├── test_real_datasets.py
   │   └── test_user_scenarios.py
   │
   ├── performance/                  # Performance tests
   │   ├── __init__.py
   │   ├── test_scalability.py
   │   ├── test_memory_usage.py
   │   └── test_method_performance.py
   │
   └── fixtures/                     # Test fixtures and data
       ├── __init__.py
       ├── data/                     # Test datasets
       ├── mock_llm_responses.py     # Mock LLM responses
       ├── synthetic_data.py         # Synthetic data fixtures
       └── shared_datasets.py        # Shared test datasets

Base Test Infrastructure
------------------------

Base Test Classes
~~~~~~~~~~~~~~~~~

Provide common functionality for all test types:

.. code-block:: python

   # tests/base.py
   
   import pytest
   import pandas as pd
   import numpy as np
   from typing import Dict, Any, Optional
   from unittest.mock import Mock, patch
   
   from causal_agent.models import Variables, DatasetAnalysis
   from causal_agent.synthetic.generator import DataGenerationConfig
   
   class BaseTestCase:
       """Base class for all CAIS tests with common utilities"""
       
       def setup_method(self):
           """Setup run before each test method"""
           np.random.seed(42)  # Ensure reproducible tests
           self.test_data_dir = Path("tests/fixtures/data")
           self.mock_responses = {}
       
       def create_sample_variables(
           self, 
           treatment: str = "treatment",
           outcome: str = "outcome",
           covariates: Optional[List[str]] = None,
           **kwargs
       ) -> Variables:
           """Create sample Variables object for testing"""
           return Variables(
               treatment_variable=treatment,
               outcome_variable=outcome,
               covariates=covariates or ["X1", "X2", "X3"],
               **kwargs
           )
       
       def create_sample_dataset_analysis(
           self, 
           n_observations: int = 1000,
           n_variables: int = 5
       ) -> DatasetAnalysis:
           """Create sample DatasetAnalysis for testing"""
           return DatasetAnalysis(
               column_info={
                   "treatment": {"type": "binary", "unique_values": 2},
                   "outcome": {"type": "continuous", "mean": 5.0, "std": 2.0},
                   "X1": {"type": "continuous", "mean": 0.0, "std": 1.0},
                   "X2": {"type": "continuous", "mean": 0.0, "std": 1.0},
                   "X3": {"type": "continuous", "mean": 0.0, "std": 1.0}
               },
               summary_stats={
                   "n_observations": n_observations,
                   "n_variables": n_variables,
                   "treatment_prevalence": 0.5
               },
               missing_values={col: 0 for col in ["treatment", "outcome", "X1", "X2", "X3"]},
               data_types={
                   "treatment": "int64",
                   "outcome": "float64",
                   "X1": "float64",
                   "X2": "float64",
                   "X3": "float64"
               },
               n_observations=n_observations,
               n_variables=n_variables
           )
       
       def assert_valid_causal_results(self, results: Dict[str, Any]):
           """Assert that causal analysis results have required structure"""
           required_fields = [
               'effect_estimate', 'standard_error', 'confidence_interval',
               'p_value', 'method', 'assumptions'
           ]
           
           for field in required_fields:
               assert field in results, f"Missing required field: {field}"
           
           # Type checks
           assert isinstance(results['effect_estimate'], (int, float))
           assert isinstance(results['standard_error'], (int, float))
           assert isinstance(results['confidence_interval'], (list, tuple))
           assert isinstance(results['p_value'], (int, float))
           assert isinstance(results['method'], str)
           assert isinstance(results['assumptions'], list)
           
           # Value checks
           assert results['standard_error'] > 0
           assert 0 <= results['p_value'] <= 1
           assert len(results['confidence_interval']) == 2
       
       def assert_effect_recovery(
           self, 
           estimated_effect: float, 
           true_effect: float, 
           tolerance: float = 0.5
       ):
           """Assert that estimated effect is close to true effect"""
           bias = abs(estimated_effect - true_effect)
           assert bias <= tolerance, (
               f"Effect estimate {estimated_effect} differs from true effect "
               f"{true_effect} by {bias}, exceeding tolerance {tolerance}"
           )
   
   class MockLLMTestCase(BaseTestCase):
       """Base class for tests requiring LLM mocking"""
       
       def setup_method(self):
           super().setup_method()
           self.mock_llm_responses = {
               "treatment_variable": '{"treatment_variable": "treatment"}',
               "outcome_variable": '{"outcome_variable": "outcome"}',
               "method_selection": '{"recommended_method": "linear_regression", "confidence": 0.8}',
               "result_interpretation": '{"interpretation": "Test interpretation"}'
           }
       
       def create_mock_llm(self, responses: Optional[Dict[str, str]] = None):
           """Create mock LLM client with predefined responses"""
           responses = responses or self.mock_llm_responses
           
           mock_llm = Mock()
           
           def mock_invoke(prompt):
               # Match prompt to response based on keywords
               for keyword, response in responses.items():
                   if keyword in prompt.lower():
                       mock_response = Mock()
                       mock_response.content = response
                       return mock_response
               
               # Default response
               mock_response = Mock()
               mock_response.content = '{"error": "No mock response defined"}'
               return mock_response
           
           mock_llm.invoke = mock_invoke
           return mock_llm

Pytest Configuration
~~~~~~~~~~~~~~~~~~~~

Central configuration for all tests:

.. code-block:: python

   # tests/conftest.py
   
   import pytest
   import pandas as pd
   import numpy as np
   from pathlib import Path
   from typing import Dict, Any
   
   from causal_agent.synthetic.generator import *
   from tests.fixtures.mock_llm_responses import STANDARD_MOCK_RESPONSES
   
   # Test data directory
   TEST_DATA_DIR = Path(__file__).parent / "fixtures" / "data"
   
   @pytest.fixture(scope="session")
   def test_data_dir():
       """Path to test data directory"""
       return TEST_DATA_DIR
   
   @pytest.fixture
   def sample_rct_data():
       """Generate sample RCT data for testing"""
       config = DataGenerationConfig(
           n_observations=500,
           n_continuous_covars=3,
           n_binary_covars=2,
           true_effect=1.5,
           seed=42
       )
       generator = RCTDataGenerator(config)
       data = generator.generate_data()
       true_params = generator.get_true_parameters()
       return data, true_params
   
   @pytest.fixture
   def sample_observational_data():
       """Generate sample observational data for testing"""
       config = DataGenerationConfig(
           n_observations=800,
           true_effect=1.2,
           seed=42
       )
       generator = PropensityScoreGenerator(config)
       data = generator.generate_data()
       true_params = generator.get_true_parameters()
       return data, true_params
   
   @pytest.fixture
   def mock_llm_client():
       """Mock LLM client with standard responses"""
       from tests.base import MockLLMTestCase
       test_case = MockLLMTestCase()
       test_case.setup_method()
       return test_case.create_mock_llm(STANDARD_MOCK_RESPONSES)
   
   @pytest.fixture
   def temp_data_file(tmp_path):
       """Create temporary CSV file for testing"""
       data = pd.DataFrame({
           'treatment': [0, 1, 0, 1, 0, 1],
           'outcome': [1.0, 2.5, 1.2, 2.8, 0.9, 2.3],
           'X1': [0.1, 0.5, -0.2, 0.8, -0.1, 0.6],
           'X2': [1.0, 1.5, 0.8, 1.2, 0.9, 1.4]
       })
       
       file_path = tmp_path / "test_data.csv"
       data.to_csv(file_path, index=False)
       return str(file_path)
   
   # Pytest markers for test categorization
   def pytest_configure(config):
       """Configure custom pytest markers"""
       config.addinivalue_line(
           "markers", "unit: mark test as unit test"
       )
       config.addinivalue_line(
           "markers", "integration: mark test as integration test"
       )
       config.addinivalue_line(
           "markers", "e2e: mark test as end-to-end test"
       )
       config.addinivalue_line(
           "markers", "performance: mark test as performance test"
       )
       config.addinivalue_line(
           "markers", "slow: mark test as slow running"
       )
       config.addinivalue_line(
           "markers", "llm: mark test as requiring LLM integration"
       )

Unit Testing
------------

Component Unit Tests
~~~~~~~~~~~~~~~~~~~~

Test individual components in isolation:

.. code-block:: python

   # tests/unit/components/test_dataset_analyzer.py
   
   import pytest
   import pandas as pd
   import numpy as np
   from tests.base import BaseTestCase
   
   from causal_agent.components.dataset_analyzer import DatasetAnalyzer
   
   @pytest.mark.unit
   class TestDatasetAnalyzer(BaseTestCase):
       """Test suite for DatasetAnalyzer component"""
       
       def setup_method(self):
           super().setup_method()
           self.analyzer = DatasetAnalyzer()
       
       def test_analyze_basic_dataset(self):
           """Test basic dataset analysis functionality"""
           data = pd.DataFrame({
               'treatment': [0, 1, 0, 1, 0, 1],
               'outcome': [1.0, 2.5, 1.2, 2.8, 0.9, 2.3],
               'covariate': [0.1, 0.5, -0.2, 0.8, -0.1, 0.6]
           })
           
           analysis = self.analyzer.analyze_dataset(data)
           
           # Check structure
           assert 'column_info' in analysis
           assert 'summary_stats' in analysis
           assert 'missing_values' in analysis
           assert 'data_types' in analysis
           
           # Check content
           assert len(analysis['column_info']) == 3
           assert analysis['summary_stats']['n_observations'] == 6
           assert analysis['summary_stats']['n_variables'] == 3
       
       def test_missing_value_detection(self):
           """Test missing value detection"""
           data = pd.DataFrame({
               'treatment': [0, 1, np.nan, 1, 0, 1],
               'outcome': [1.0, 2.5, 1.2, np.nan, 0.9, 2.3],
               'covariate': [0.1, 0.5, -0.2, 0.8, -0.1, 0.6]
           })
           
           analysis = self.analyzer.analyze_dataset(data)
           
           assert analysis['missing_values']['treatment'] == 1
           assert analysis['missing_values']['outcome'] == 1
           assert analysis['missing_values']['covariate'] == 0
       
       def test_data_type_detection(self):
           """Test data type detection"""
           data = pd.DataFrame({
               'binary_var': [0, 1, 0, 1, 0, 1],
               'continuous_var': [1.1, 2.5, 1.2, 2.8, 0.9, 2.3],
               'categorical_var': ['A', 'B', 'A', 'C', 'B', 'A'],
               'integer_var': [1, 2, 3, 4, 5, 6]
           })
           
           analysis = self.analyzer.analyze_dataset(data)
           
           # Check that types are correctly identified
           assert 'binary' in analysis['column_info']['binary_var']['type']
           assert 'continuous' in analysis['column_info']['continuous_var']['type']
           assert 'categorical' in analysis['column_info']['categorical_var']['type']
       
       def test_statistical_summaries(self):
           """Test statistical summary generation"""
           np.random.seed(42)
           data = pd.DataFrame({
               'var1': np.random.normal(5, 2, 1000),
               'var2': np.random.uniform(0, 10, 1000)
           })
           
           analysis = self.analyzer.analyze_dataset(data)
           
           # Check that summaries are reasonable
           var1_info = analysis['column_info']['var1']
           assert abs(var1_info['mean'] - 5) < 0.5  # Should be close to true mean
           assert abs(var1_info['std'] - 2) < 0.5   # Should be close to true std
           
           var2_info = analysis['column_info']['var2']
           assert 0 <= var2_info['min'] <= 1
           assert 9 <= var2_info['max'] <= 10
       
       def test_edge_cases(self):
           """Test edge cases and error handling"""
           # Empty dataset
           empty_data = pd.DataFrame()
           with pytest.raises(ValueError):
               self.analyzer.analyze_dataset(empty_data)
           
           # Single row
           single_row = pd.DataFrame({'var': [1]})
           analysis = self.analyzer.analyze_dataset(single_row)
           assert analysis['summary_stats']['n_observations'] == 1
           
           # All missing values
           all_missing = pd.DataFrame({'var': [np.nan, np.nan, np.nan]})
           analysis = self.analyzer.analyze_dataset(all_missing)
           assert analysis['missing_values']['var'] == 3

Method Unit Tests
~~~~~~~~~~~~~~~~~

Test causal inference method implementations:

.. code-block:: python

   # tests/unit/methods/experimental/test_diff_in_means.py
   
   import pytest
   import numpy as np
   from tests.base import BaseTestCase
   
   from causal_agent.methods.experimental.diff_in_means.estimator import estimate_diff_in_means
   from causal_agent.models import Variables
   
   @pytest.mark.unit
   class TestDiffInMeans(BaseTestCase):
       """Test suite for Difference in Means method"""
       
       def test_basic_estimation(self, sample_rct_data):
           """Test basic difference in means estimation"""
           data, true_params = sample_rct_data
           
           variables = Variables(
               treatment_variable='treatment',
               outcome_variable='outcome',
               covariates=['X1', 'X2', 'X3'],
               is_rct=True
           )
           
           results = estimate_diff_in_means(data, variables)
           
           # Check result structure
           self.assert_valid_causal_results(results)
           
           # Check effect recovery
           true_effect = true_params['true_effect']
           estimated_effect = results['effect_estimate']
           self.assert_effect_recovery(estimated_effect, true_effect, tolerance=0.3)
       
       def test_statistical_inference(self, sample_rct_data):
           """Test statistical inference components"""
           data, true_params = sample_rct_data
           
           variables = Variables(
               treatment_variable='treatment',
               outcome_variable='outcome',
               is_rct=True
           )
           
           results = estimate_diff_in_means(data, variables)
           
           # Check confidence interval
           ci = results['confidence_interval']
           effect = results['effect_estimate']
           assert ci[0] < effect < ci[1], "Effect should be within confidence interval"
           
           # Check p-value for significant effect
           if abs(true_params['true_effect']) > 0.5:  # Should be detectable
               assert results['p_value'] < 0.05, "Should detect significant effect"
       
       def test_balanced_vs_unbalanced(self):
           """Test with balanced vs unbalanced treatment assignment"""
           np.random.seed(42)
           n = 1000
           
           # Balanced treatment
           balanced_treatment = np.concatenate([np.zeros(n//2), np.ones(n//2)])
           np.random.shuffle(balanced_treatment)
           
           # Unbalanced treatment (20% treated)
           unbalanced_treatment = np.concatenate([np.zeros(int(0.8*n)), np.ones(int(0.2*n))])
           
           for treatment in [balanced_treatment, unbalanced_treatment]:
               outcome = 2 + 1.5 * treatment + np.random.normal(0, 1, len(treatment))
               data = pd.DataFrame({'treatment': treatment, 'outcome': outcome})
               
               variables = Variables(
                   treatment_variable='treatment',
                   outcome_variable='outcome',
                   is_rct=True
               )
               
               results = estimate_diff_in_means(data, variables)
               
               # Should recover true effect regardless of balance
               self.assert_effect_recovery(results['effect_estimate'], 1.5, tolerance=0.3)
       
       def test_input_validation(self):
           """Test input validation and error handling"""
           data = pd.DataFrame({
               'treatment': [0, 1, 0, 1],
               'outcome': [1, 2, 1, 2]
           })
           
           # Missing treatment variable
           variables_bad = Variables(
               treatment_variable='missing_var',
               outcome_variable='outcome'
           )
           
           with pytest.raises(ValueError):
               estimate_diff_in_means(data, variables_bad)
           
           # Non-binary treatment
           data_continuous_treatment = pd.DataFrame({
               'treatment': [0.1, 0.5, 0.8, 0.9],
               'outcome': [1, 2, 1, 2]
           })
           
           variables = Variables(
               treatment_variable='treatment',
               outcome_variable='outcome'
           )
           
           # Should handle or warn about non-binary treatment
           results = estimate_diff_in_means(data_continuous_treatment, variables)
           assert 'warning' in results or 'effect_estimate' in results

Tool Unit Tests
~~~~~~~~~~~~~~~

Test tool interfaces and LangChain integration:

.. code-block:: python

   # tests/unit/tools/test_method_selector_tool.py
   
   import pytest
   from unittest.mock import Mock, patch
   from tests.base import MockLLMTestCase
   
   from causal_agent.tools.method_selector_tool import method_selector_tool
   from causal_agent.models import Variables, DatasetAnalysis
   
   @pytest.mark.unit
   class TestMethodSelectorTool(MockLLMTestCase):
       """Test suite for Method Selector Tool"""
       
       def test_tool_basic_functionality(self):
           """Test basic method selection functionality"""
           variables = self.create_sample_variables(is_rct=True)
           dataset_analysis = self.create_sample_dataset_analysis()
           
           result = method_selector_tool.func(
               variables=variables,
               dataset_analysis=dataset_analysis,
               dataset_description="RCT dataset",
               original_query="What is the effect of treatment?"
           )
           
           # Check result structure
           assert 'method_info' in result
           assert 'reasoning' in result
           assert 'confidence' in result
           
           # For RCT data, should select experimental method
           method_info = result['method_info']
           assert method_info['method'] in ['diff_in_means', 'randomized_controlled_trial']
       
       def test_method_selection_logic(self):
           """Test method selection logic for different scenarios"""
           base_variables = self.create_sample_variables()
           dataset_analysis = self.create_sample_dataset_analysis()
           
           # Test RCT scenario
           rct_variables = Variables(**{**base_variables.__dict__, 'is_rct': True})
           result = method_selector_tool.func(
               variables=rct_variables,
               dataset_analysis=dataset_analysis
           )
           assert result['method_info']['method'] in ['diff_in_means', 'randomized_controlled_trial']
           
           # Test IV scenario
           iv_variables = Variables(**{
               **base_variables.__dict__, 
               'instrument_variable': 'instrument',
               'is_rct': False
           })
           result = method_selector_tool.func(
               variables=iv_variables,
               dataset_analysis=dataset_analysis
           )
           assert result['method_info']['method'] == 'instrumental_variable'
           
           # Test observational scenario
           obs_variables = Variables(**{**base_variables.__dict__, 'is_rct': False})
           result = method_selector_tool.func(
               variables=obs_variables,
               dataset_analysis=dataset_analysis
           )
           assert result['method_info']['method'] in [
               'propensity_score_matching', 'backdoor_adjustment', 'linear_regression'
           ]
       
       def test_excluded_methods(self):
           """Test method exclusion functionality"""
           variables = self.create_sample_variables(is_rct=False)
           dataset_analysis = self.create_sample_dataset_analysis()
           
           # Exclude propensity score methods
           excluded = ['propensity_score_matching', 'propensity_score_weighting']
           
           result = method_selector_tool.func(
               variables=variables,
               dataset_analysis=dataset_analysis,
               excluded_methods=excluded
           )
           
           selected_method = result['method_info']['method']
           assert selected_method not in excluded
       
       @patch('causal_agent.tools.method_selector_tool.get_llm_client')
       def test_llm_integration(self, mock_get_llm):
           """Test LLM integration for enhanced reasoning"""
           # Setup mock LLM
           mock_llm = self.create_mock_llm({
               'method selection': '{"recommended_method": "linear_regression", "confidence": 0.9}'
           })
           mock_get_llm.return_value = mock_llm
           
           variables = self.create_sample_variables()
           dataset_analysis = self.create_sample_dataset_analysis()
           
           result = method_selector_tool.func(
               variables=variables,
               dataset_analysis=dataset_analysis
           )
           
           # Should have used LLM reasoning
           assert 'reasoning' in result
           assert result['confidence'] > 0

Integration Testing
-------------------

Workflow Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~

Test component interactions and data flow:

.. code-block:: python

   # tests/integration/test_agent_workflows.py
   
   import pytest
   from tests.base import MockLLMTestCase
   
   from causal_agent.agent import run_causal_analysis
   from causal_agent.tools import *
   
   @pytest.mark.integration
   class TestAgentWorkflows(MockLLMTestCase):
       """Test complete agent workflows"""
       
       def test_rct_analysis_workflow(self, sample_rct_data, temp_data_file):
           """Test complete RCT analysis workflow"""
           data, true_params = sample_rct_data
           
           # Save data to temporary file
           data.to_csv(temp_data_file, index=False)
           
           # Run analysis
           result = run_causal_analysis(
               query="What is the effect of treatment on outcome?",
               dataset_path=temp_data_file,
               dataset_description="Randomized controlled trial data"
           )
           
           # Check that analysis completed successfully
           assert 'error' not in result
           assert 'results' in result
           assert 'effect_estimate' in result['results']['results']
           
           # Check that correct method was used
           method_used = result['results']['results']['method_used']
           assert method_used in ['diff_in_means', 'randomized_controlled_trial']
           
           # Check effect recovery
           estimated_effect = result['results']['results']['effect_estimate']
           true_effect = true_params['true_effect']
           self.assert_effect_recovery(estimated_effect, true_effect, tolerance=0.5)
       
       def test_observational_analysis_workflow(self, sample_observational_data, temp_data_file):
           """Test observational data analysis workflow"""
           data, true_params = sample_observational_data
           data.to_csv(temp_data_file, index=False)
           
           result = run_causal_analysis(
               query="What is the causal effect of treatment on outcome?",
               dataset_path=temp_data_file,
               dataset_description="Observational study with covariates"
           )
           
           assert 'error' not in result
           method_used = result['results']['results']['method_used']
           assert method_used in [
               'propensity_score_matching', 'backdoor_adjustment', 'linear_regression'
           ]
       
       def test_workflow_error_handling(self, temp_data_file):
           """Test workflow error handling"""
           # Create problematic data
           problematic_data = pd.DataFrame({
               'var1': [1, 2, 3],
               'var2': [4, 5, 6]
           })
           problematic_data.to_csv(temp_data_file, index=False)
           
           result = run_causal_analysis(
               query="What is the effect of treatment on outcome?",
               dataset_path=temp_data_file
           )
           
           # Should handle gracefully
           assert isinstance(result, dict)
           # May contain error or fallback results
       
       def test_tool_chain_integration(self):
           """Test that tools work together correctly"""
           # Test tool chain: input_parser -> dataset_analyzer -> query_interpreter
           
           # Step 1: Input parsing
           input_result = input_parser_tool.func(
               "What is the effect of education on income? Dataset: test_data.csv"
           )
           
           assert 'dataset_path' in input_result
           assert 'original_query' in input_result
           
           # Step 2: Dataset analysis (would need actual file)
           # This tests the interface compatibility
           
           # Step 3: Query interpretation
           from causal_agent.models import QueryInfo
           query_info = QueryInfo(
               query_text=input_result['original_query'],
               potential_treatments=['education'],
               potential_outcomes=['income']
           )
           
           # Test that outputs are compatible with next tool inputs
           assert hasattr(query_info, 'query_text')
           assert hasattr(query_info, 'potential_treatments')

LLM Integration Tests
~~~~~~~~~~~~~~~~~~~~~

Test LLM interactions and prompt effectiveness:

.. code-block:: python

   # tests/integration/test_llm_integration.py
   
   import pytest
   from unittest.mock import Mock, patch
   from tests.base import MockLLMTestCase
   
   from causal_agent.components.decision_tree_llm import DecisionTreeLLMEngine
   from causal_agent.config import get_llm_client
   
   @pytest.mark.integration
   @pytest.mark.llm
   class TestLLMIntegration(MockLLMTestCase):
       """Test LLM integration components"""
       
       def test_variable_identification_prompts(self):
           """Test variable identification with LLM"""
           mock_llm = self.create_mock_llm({
               'treatment variable': '{"treatment_variable": "education_years"}',
               'outcome variable': '{"outcome_variable": "annual_income"}'
           })
           
           with patch('causal_agent.config.get_llm_client', return_value=mock_llm):
               from causal_agent.components.query_interpreter import QueryInterpreter
               
               interpreter = QueryInterpreter()
               
               # Test treatment identification
               query = "What is the effect of education on income?"
               columns = ["education_years", "annual_income", "age", "gender"]
               
               # This would call LLM internally
               # result = interpreter.identify_treatment_variable(query, columns)
               # assert result == "education_years"
       
       def test_method_selection_reasoning(self):
           """Test LLM-enhanced method selection"""
           mock_responses = {
               'method selection': '''
               {
                   "recommended_method": "propensity_score_matching",
                   "confidence": 0.85,
                   "reasoning": "Dataset has rich covariates and observational design",
                   "assumptions": ["no unmeasured confounders", "overlap"],
                   "alternatives": ["backdoor_adjustment", "linear_regression"]
               }
               '''
           }
           
           mock_llm = self.create_mock_llm(mock_responses)
           
           with patch('causal_agent.config.get_llm_client', return_value=mock_llm):
               engine = DecisionTreeLLMEngine(mock_llm)
               
               variables = self.create_sample_variables(is_rct=False)
               dataset_analysis = self.create_sample_dataset_analysis()
               
               result = engine.select_method(variables, dataset_analysis)
               
               assert result['method'] == 'propensity_score_matching'
               assert result['confidence'] == 0.85
               assert 'reasoning' in result
       
       def test_prompt_robustness(self):
           """Test prompt robustness with various inputs"""
           # Test with different response formats
           problematic_responses = [
               '{"treatment_variable": null}',  # Null response
               '{"treatment_variable": ""}',    # Empty response
               'Invalid JSON response',         # Invalid JSON
               '{"wrong_field": "value"}',      # Wrong field
           ]
           
           for response in problematic_responses:
               mock_llm = self.create_mock_llm({'treatment': response})
               
               # Should handle gracefully without crashing
               with patch('causal_agent.config.get_llm_client', return_value=mock_llm):
                   # Test component that uses LLM
                   pass  # Implementation would test actual component
       
       @pytest.mark.slow
       def test_real_llm_integration(self):
           """Test with real LLM (if API key available)"""
           try:
               llm = get_llm_client()
               
               # Simple test prompt
               response = llm.invoke("What is 2+2?")
               
               # Should get some response
               assert response is not None
               assert hasattr(response, 'content') or isinstance(response, str)
               
           except Exception as e:
               pytest.skip(f"Real LLM test skipped: {e}")

End-to-End Testing
------------------

Complete Workflow Tests
~~~~~~~~~~~~~~~~~~~~~~~

Test complete analysis pipelines:

.. code-block:: python

   # tests/end_to_end/test_complete_workflows.py
   
   import pytest
   import pandas as pd
   from pathlib import Path
   from tests.base import BaseTestCase
   
   from causal_agent.agent import run_causal_analysis
   
   @pytest.mark.e2e
   class TestCompleteWorkflows(BaseTestCase):
       """End-to-end tests for complete analysis workflows"""
       
       def test_education_income_analysis(self, test_data_dir):
           """Test complete education-income analysis"""
           # Create realistic education-income dataset
           np.random.seed(42)
           n = 1000
           
           # Generate realistic education-income data
           education = np.random.choice([12, 14, 16, 18, 20], n, p=[0.3, 0.2, 0.3, 0.15, 0.05])
           age = np.random.normal(35, 10, n)
           experience = np.maximum(0, age - education - 6)
           
           # Income with realistic relationship
           income = (
               20000 +  # Base income
               2000 * education +  # Education premium
               500 * experience +  # Experience premium
               np.random.normal(0, 5000, n)  # Noise
           )
           
           data = pd.DataFrame({
               'education_years': education,
               'annual_income': income,
               'age': age,
               'experience_years': experience
           })
           
           # Save to temporary file
           data_path = test_data_dir / "education_income.csv"
           data.to_csv(data_path, index=False)
           
           # Run complete analysis
           result = run_causal_analysis(
               query="What is the effect of education on income?",
               dataset_path=str(data_path),
               dataset_description="Education and income dataset with age and experience"
           )
           
           # Validate results
           assert 'error' not in result
           assert 'results' in result
           
           results = result['results']['results']
           assert 'effect_estimate' in results
           assert 'method_used' in results
           
           # Education should have positive effect on income
           assert results['effect_estimate'] > 0
           
           # Should use appropriate method for observational data
           assert results['method_used'] in [
               'linear_regression', 'backdoor_adjustment', 'propensity_score_matching'
           ]
       
       def test_medical_treatment_analysis(self, test_data_dir):
           """Test medical treatment effectiveness analysis"""
           # Generate medical trial data
           np.random.seed(42)
           n = 500
           
           # Patient characteristics
           age = np.random.normal(50, 15, n)
           severity = np.random.uniform(1, 10, n)
           
           # Random treatment assignment (RCT)
           treatment = np.random.binomial(1, 0.5, n)
           
           # Outcome with treatment effect
           recovery_time = (
               10 +  # Base recovery time
               0.1 * age +  # Age effect
               0.5 * severity -  # Severity effect
               3 * treatment +  # Treatment effect
               np.random.normal(0, 2, n)  # Noise
           )
           
           data = pd.DataFrame({
               'treatment_received': treatment,
               'recovery_days': recovery_time,
               'patient_age': age,
               'disease_severity': severity
           })
           
           data_path = test_data_dir / "medical_trial.csv"
           data.to_csv(data_path, index=False)
           
           result = run_causal_analysis(
               query="Does the treatment reduce recovery time?",
               dataset_path=str(data_path),
               dataset_description="Randomized clinical trial of new treatment"
           )
           
           # Validate RCT analysis
           assert 'error' not in result
           results = result['results']['results']
           
           # Should detect treatment effect
           assert results['effect_estimate'] < 0  # Negative = reduces recovery time
           assert results['p_value'] < 0.05  # Should be significant
           
           # Should use experimental method
           assert results['method_used'] in ['diff_in_means', 'randomized_controlled_trial']
       
       def test_policy_evaluation_analysis(self, test_data_dir):
           """Test policy evaluation with difference-in-differences"""
           # Generate panel data for policy evaluation
           np.random.seed(42)
           n_states = 20
           n_years = 5
           
           data_list = []
           
           # Treatment states (policy implemented in year 3)
           treated_states = np.random.choice(n_states, n_states//2, replace=False)
           
           for state in range(n_states):
               for year in range(n_years):
                   # State and time effects
                   state_effect = np.random.normal(0, 1)
                   time_effect = np.random.normal(0, 0.5)
                   
                   # Treatment indicator
                   is_treated = state in treated_states
                   is_post = year >= 2  # Policy starts in year 3 (0-indexed)
                   treatment = 1 if (is_treated and is_post) else 0
                   
                   # Outcome (e.g., unemployment rate)
                   outcome = (
                       5 +  # Base rate
                       state_effect +  # State fixed effect
                       time_effect +  # Time trend
                       -1.5 * treatment +  # Policy effect
                       np.random.normal(0, 0.5)  # Noise
                   )
                   
                   data_list.append({
                       'state_id': state,
                       'year': year,
                       'policy_implemented': treatment,
                       'unemployment_rate': outcome,
                       'treated_state': int(is_treated),
                       'post_policy': int(is_post)
                   })
           
           data = pd.DataFrame(data_list)
           data_path = test_data_dir / "policy_evaluation.csv"
           data.to_csv(data_path, index=False)
           
           result = run_causal_analysis(
               query="What is the effect of the policy on unemployment?",
               dataset_path=str(data_path),
               dataset_description="State-level panel data with policy implementation"
           )
           
           # Validate DiD analysis
           assert 'error' not in result
           results = result['results']['results']
           
           # Should detect policy effect
           assert results['effect_estimate'] < 0  # Policy reduces unemployment
           
           # Should use difference-in-differences
           assert results['method_used'] == 'difference_in_differences'

Performance Testing
-------------------

Scalability Tests
~~~~~~~~~~~~~~~~~

Test system performance with large datasets:

.. code-block:: python

   # tests/performance/test_scalability.py
   
   import pytest
   import time
   import psutil
   import pandas as pd
   import numpy as np
   from tests.base import BaseTestCase
   
   from causal_agent.agent import run_causal_analysis
   from causal_agent.synthetic.generator import RCTDataGenerator, DataGenerationConfig
   
   @pytest.mark.performance
   @pytest.mark.slow
   class TestScalability(BaseTestCase):
       """Test system scalability and performance"""
       
       def test_large_dataset_performance(self, tmp_path):
           """Test performance with large datasets"""
           dataset_sizes = [1000, 5000, 10000, 50000]
           performance_results = []
           
           for n in dataset_sizes:
               # Generate large dataset
               config = DataGenerationConfig(
                   n_observations=n,
                   n_continuous_covars=5,
                   true_effect=1.0,
                   seed=42
               )
               generator = RCTDataGenerator(config)
               data = generator.generate_data()
               
               # Save to file
               data_path = tmp_path / f"large_data_{n}.csv"
               data.to_csv(data_path, index=False)
               
               # Measure performance
               start_time = time.time()
               start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
               
               result = run_causal_analysis(
                   query="What is the effect of treatment on outcome?",
                   dataset_path=str(data_path),
                   dataset_description=f"Large RCT dataset with {n} observations"
               )
               
               end_time = time.time()
               end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
               
               # Record performance
               performance_results.append({
                   'n_observations': n,
                   'execution_time': end_time - start_time,
                   'memory_usage': end_memory - start_memory,
                   'success': 'error' not in result
               })
               
               # Basic performance assertions
               assert result is not None
               assert end_time - start_time < 300  # Should complete within 5 minutes
               
               print(f"Dataset size {n}: {end_time - start_time:.2f}s, "
                     f"{end_memory - start_memory:.1f}MB")
           
           # Check that performance scales reasonably
           # Time should scale sub-quadratically
           for i in range(1, len(performance_results)):
               prev = performance_results[i-1]
               curr = performance_results[i]
               
               size_ratio = curr['n_observations'] / prev['n_observations']
               time_ratio = curr['execution_time'] / prev['execution_time']
               
               # Time should not scale worse than O(n^2)
               assert time_ratio < size_ratio ** 2
       
       def test_high_dimensional_performance(self, tmp_path):
           """Test performance with high-dimensional data"""
           covariate_counts = [5, 20, 50, 100]
           
           for n_covars in covariate_counts:
               config = DataGenerationConfig(
                   n_observations=1000,
                   n_continuous_covars=n_covars,
                   true_effect=1.0,
                   seed=42
               )
               generator = RCTDataGenerator(config)
               data = generator.generate_data()
               
               data_path = tmp_path / f"high_dim_{n_covars}.csv"
               data.to_csv(data_path, index=False)
               
               start_time = time.time()
               
               result = run_causal_analysis(
                   query="What is the effect of treatment on outcome?",
                   dataset_path=str(data_path),
                   dataset_description=f"High-dimensional dataset with {n_covars} covariates"
               )
               
               end_time = time.time()
               
               # Should handle high-dimensional data
               assert 'error' not in result
               assert end_time - start_time < 120  # Should complete within 2 minutes
               
               print(f"Covariates {n_covars}: {end_time - start_time:.2f}s")
       
       def test_memory_usage_limits(self, tmp_path):
           """Test memory usage stays within reasonable limits"""
           # Generate moderately large dataset
           config = DataGenerationConfig(
               n_observations=20000,
               n_continuous_covars=10,
               true_effect=1.0,
               seed=42
           )
           generator = RCTDataGenerator(config)
           data = generator.generate_data()
           
           data_path = tmp_path / "memory_test.csv"
           data.to_csv(data_path, index=False)
           
           # Monitor memory usage
           initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
           
           result = run_causal_analysis(
               query="What is the effect of treatment on outcome?",
               dataset_path=str(data_path)
           )
           
           peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
           memory_increase = peak_memory - initial_memory
           
           # Memory usage should be reasonable
           assert memory_increase < 1000  # Less than 1GB increase
           assert 'error' not in result
           
           print(f"Memory increase: {memory_increase:.1f}MB")

Method Performance Tests
~~~~~~~~~~~~~~~~~~~~~~~~

Test individual method performance:

.. code-block:: python

   # tests/performance/test_method_performance.py
   
   import pytest
   import time
   import numpy as np
   from tests.base import BaseTestCase
   
   from causal_agent.methods.experimental.diff_in_means.estimator import estimate_diff_in_means
   from causal_agent.methods.observational.propensity_score_matching.estimator import estimate_propensity_score_matching
   from causal_agent.models import Variables
   
   @pytest.mark.performance
   class TestMethodPerformance(BaseTestCase):
       """Test performance of individual causal methods"""
       
       def test_diff_in_means_performance(self):
           """Test difference in means performance across dataset sizes"""
           sizes = [1000, 5000, 10000, 25000]
           
           for n in sizes:
               # Generate data
               np.random.seed(42)
               treatment = np.random.binomial(1, 0.5, n)
               outcome = 2 + 1.5 * treatment + np.random.normal(0, 1, n)
               
               data = pd.DataFrame({
                   'treatment': treatment,
                   'outcome': outcome
               })
               
               variables = Variables(
                   treatment_variable='treatment',
                   outcome_variable='outcome',
                   is_rct=True
               )
               
               # Time execution
               start_time = time.time()
               result = estimate_diff_in_means(data, variables)
               end_time = time.time()
               
               execution_time = end_time - start_time
               
               # Should be fast for simple method
               assert execution_time < 1.0  # Less than 1 second
               assert 'effect_estimate' in result
               
               print(f"Diff in means (n={n}): {execution_time:.3f}s")
       
       def test_propensity_score_performance(self):
           """Test propensity score method performance"""
           sizes = [1000, 2000, 5000]  # Smaller sizes due to complexity
           
           for n in sizes:
               # Generate observational data
               np.random.seed(42)
               X1 = np.random.normal(0, 1, n)
               X2 = np.random.normal(0, 1, n)
               
               # Treatment with selection
               treatment_prob = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
               treatment = np.random.binomial(1, treatment_prob)
               
               # Outcome with confounding
               outcome = 2 + 1.5 * treatment + 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n)
               
               data = pd.DataFrame({
                   'treatment': treatment,
                   'outcome': outcome,
                   'X1': X1,
                   'X2': X2
               })
               
               variables = Variables(
                   treatment_variable='treatment',
                   outcome_variable='outcome',
                   covariates=['X1', 'X2'],
                   is_rct=False
               )
               
               # Time execution
               start_time = time.time()
               result = estimate_propensity_score_matching(data, variables)
               end_time = time.time()
               
               execution_time = end_time - start_time
               
               # Should complete within reasonable time
               assert execution_time < 30.0  # Less than 30 seconds
               assert 'effect_estimate' in result
               
               print(f"Propensity score (n={n}): {execution_time:.3f}s")

Test Automation and CI/CD
--------------------------

GitHub Actions Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automated testing in continuous integration:

.. code-block:: yaml

   # .github/workflows/tests.yml
   
   name: Tests
   
   on:
     push:
       branches: [ main, develop ]
     pull_request:
       branches: [ main ]
   
   jobs:
     unit-tests:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: [3.8, 3.9, "3.10", "3.11"]
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python ${{ matrix.python-version }}
         uses: actions/setup-python@v4
         with:
           python-version: ${{ matrix.python-version }}
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements.txt
           pip install -e .
           pip install pytest pytest-cov pytest-xdist
       
       - name: Run unit tests
         run: |
           pytest tests/unit/ -v --cov=causal_agent --cov-report=xml --cov-report=html -n auto
       
       - name: Upload coverage to Codecov
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml
           flags: unittests
           name: codecov-umbrella
   
     integration-tests:
       runs-on: ubuntu-latest
       needs: unit-tests
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: "3.10"
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements.txt
           pip install -e .
           pip install pytest pytest-cov
       
       - name: Run integration tests
         run: |
           pytest tests/integration/ -v --cov=causal_agent --cov-append
       
       - name: Run end-to-end tests
         run: |
           pytest tests/end_to_end/ -v -m "not slow"
   
     performance-tests:
       runs-on: ubuntu-latest
       needs: integration-tests
       if: github.event_name == 'push' && github.ref == 'refs/heads/main'
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: "3.10"
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements.txt
           pip install -e .
           pip install pytest psutil
       
       - name: Run performance tests
         run: |
           pytest tests/performance/ -v -m "not slow"

Test Coverage and Quality
~~~~~~~~~~~~~~~~~~~~~~~~~

Maintain high test coverage and quality:

.. code-block:: python

   # scripts/test_coverage.py
   
   import subprocess
   import sys
   from pathlib import Path
   
   def run_coverage_analysis():
       """Run comprehensive coverage analysis"""
       
       # Run tests with coverage
       result = subprocess.run([
           "pytest", 
           "tests/",
           "--cov=causal_agent",
           "--cov-report=html",
           "--cov-report=term",
           "--cov-report=xml",
           "--cov-fail-under=85"  # Require 85% coverage
       ], capture_output=True, text=True)
       
       print(result.stdout)
       if result.stderr:
           print("STDERR:", result.stderr)
       
       # Check coverage requirements
       if result.returncode != 0:
           print("Coverage requirements not met!")
           sys.exit(1)
       
       # Generate coverage badge
       generate_coverage_badge()
   
   def generate_coverage_badge():
       """Generate coverage badge for README"""
       try:
           import coverage
           cov = coverage.Coverage()
           cov.load()
           
           total_coverage = cov.report()
           
           # Create badge (simplified)
           badge_color = "green" if total_coverage >= 90 else "yellow" if total_coverage >= 80 else "red"
           
           print(f"Coverage: {total_coverage:.1f}% ({badge_color})")
           
       except ImportError:
           print("Coverage package not available for badge generation")
   
   if __name__ == "__main__":
       run_coverage_analysis()

Best Practices
--------------

Test Design Principles
~~~~~~~~~~~~~~~~~~~~~~

* **Isolation**: Tests should be independent and not affect each other
* **Reproducibility**: Use fixed seeds and deterministic data generation
* **Clarity**: Test names and structure should clearly indicate what is being tested
* **Completeness**: Cover normal cases, edge cases, and error conditions
* **Performance**: Tests should run efficiently to enable frequent execution

Data Management
~~~~~~~~~~~~~~~

* **Synthetic Data**: Use synthetic data with known ground truth for validation
* **Fixtures**: Create reusable test fixtures for common data scenarios
* **Cleanup**: Properly clean up temporary files and resources
* **Versioning**: Version test datasets to ensure consistency across environments

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

* **Automated Execution**: Run tests automatically on code changes
* **Multiple Environments**: Test across different Python versions and OS
* **Coverage Tracking**: Monitor and maintain high test coverage
* **Performance Monitoring**: Track performance regressions over time
* **Quality Gates**: Prevent merging code that doesn't meet quality standards

The comprehensive testing framework ensures that CAIS maintains high reliability, accuracy, and performance standards while enabling confident development and deployment of new features and methods.
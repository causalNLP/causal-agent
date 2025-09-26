Extending Methods
================

This guide provides comprehensive instructions for extending CAIS with new causal inference methods, including integration with the decision tree logic, LLM components, and testing frameworks.

.. contents::
   :local:
   :depth: 3

Overview
--------

Adding a new causal inference method to CAIS involves several components:

1. **Method Implementation**: Core statistical implementation
2. **Decision Tree Integration**: Logic for when to select the method
3. **LLM Integration**: Prompts and reasoning for method selection
4. **Testing**: Comprehensive test suite including synthetic data
5. **Documentation**: Method-specific documentation and examples

The system is designed to make method extension straightforward while maintaining consistency and reliability.

Method Implementation Structure
-------------------------------

Base Method Interface
~~~~~~~~~~~~~~~~~~~~~

All causal inference methods inherit from the base ``CausalMethod`` class:

.. code-block:: python

   # causal_agent/methods/causal_method.py
   from abc import ABC, abstractmethod
   from typing import Dict, List, Any, Optional
   import pandas as pd
   
   class CausalMethod(ABC):
       """Base class for all causal inference methods"""
       
       def __init__(self, **kwargs):
           self.method_name = self.__class__.__name__.lower()
           self.assumptions = self._define_assumptions()
           self.diagnostics = self._define_diagnostics()
       
       @abstractmethod
       def estimate(self, data: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
           """Execute the causal inference method"""
           pass
       
       @abstractmethod
       def diagnose(self, data: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
           """Run diagnostic tests for method assumptions"""
           pass
       
       @abstractmethod
       def _define_assumptions(self) -> List[str]:
           """Define the assumptions required for this method"""
           pass
       
       @abstractmethod
       def _define_diagnostics(self) -> List[str]:
           """Define available diagnostic tests"""
           pass

Method Categories
~~~~~~~~~~~~~~~~~

Methods are organized into three categories based on their use cases:

**Experimental Methods** (``causal_agent/methods/experimental/``)
   For randomized controlled trials and experimental data

**Quasi-Experimental Methods** (``causal_agent/methods/quasi_experimental/``)
   For natural experiments and quasi-experimental designs

**Observational Methods** (``causal_agent/methods/observational/``)
   For observational data with selection concerns

Step-by-Step Method Implementation
----------------------------------

Step 1: Create Method Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new method file in the appropriate category directory:

.. code-block:: python

   # causal_agent/methods/observational/my_new_method.py
   """
   My New Method for causal inference.
   
   This method implements [brief description of the method and when to use it].
   """
   
   import logging
   from typing import Dict, List, Any, Optional
   import pandas as pd
   import numpy as np
   from scipy import stats
   
   from causal_agent.methods.causal_method import CausalMethod
   from causal_agent.models import Variables
   
   logger = logging.getLogger(__name__)
   
   class MyNewMethod(CausalMethod):
       """
       Implementation of My New Method for causal inference.
       
       This method is appropriate when:
       - [Condition 1]
       - [Condition 2]
       - [Condition 3]
       
       Assumptions:
       - [Assumption 1]
       - [Assumption 2]
       - [Assumption 3]
       """
       
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.method_name = "my_new_method"
       
       def _define_assumptions(self) -> List[str]:
           """Define the assumptions required for this method"""
           return [
               "assumption 1: detailed description",
               "assumption 2: detailed description", 
               "assumption 3: detailed description"
           ]
       
       def _define_diagnostics(self) -> List[str]:
           """Define available diagnostic tests"""
           return [
               "diagnostic_test_1",
               "diagnostic_test_2",
               "diagnostic_test_3"
           ]
       
       def estimate(self, data: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
           """
           Execute My New Method for causal effect estimation.
           
           Args:
               data: Dataset containing all necessary variables
               variables: Identified causal variables (treatment, outcome, covariates)
               
           Returns:
               Dictionary containing:
               - effect_estimate: Point estimate of causal effect
               - confidence_interval: Confidence interval for effect
               - standard_error: Standard error of estimate
               - p_value: Statistical significance
               - additional_results: Method-specific results
           """
           try:
               # Extract variables
               treatment = variables.treatment_variable
               outcome = variables.outcome_variable
               covariates = variables.covariates or []
               
               # Validate inputs
               self._validate_inputs(data, variables)
               
               # Implement your method's core logic here
               # This is a template - replace with actual implementation
               
               # Example structure:
               # 1. Data preprocessing
               X = data[covariates] if covariates else pd.DataFrame()
               T = data[treatment]
               Y = data[outcome]
               
               # 2. Method-specific estimation
               effect_estimate = self._compute_effect(Y, T, X)
               standard_error = self._compute_standard_error(Y, T, X)
               
               # 3. Statistical inference
               confidence_interval = self._compute_confidence_interval(
                   effect_estimate, standard_error
               )
               p_value = self._compute_p_value(effect_estimate, standard_error)
               
               # 4. Additional method-specific results
               additional_results = self._compute_additional_results(Y, T, X)
               
               return {
                   "effect_estimate": effect_estimate,
                   "standard_error": standard_error,
                   "confidence_interval": confidence_interval,
                   "p_value": p_value,
                   "method": self.method_name,
                   "assumptions": self.assumptions,
                   "additional_results": additional_results,
                   "sample_size": len(data),
                   "treatment_variable": treatment,
                   "outcome_variable": outcome,
                   "covariates": covariates
               }
               
           except Exception as e:
               logger.error(f"Error in {self.method_name} estimation: {e}")
               raise
       
       def diagnose(self, data: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
           """
           Run diagnostic tests for method assumptions.
           
           Args:
               data: Dataset for diagnostic testing
               variables: Identified causal variables
               
           Returns:
               Dictionary containing diagnostic test results
           """
           try:
               diagnostics = {}
               
               # Extract variables
               treatment = variables.treatment_variable
               outcome = variables.outcome_variable
               covariates = variables.covariates or []
               
               X = data[covariates] if covariates else pd.DataFrame()
               T = data[treatment]
               Y = data[outcome]
               
               # Implement diagnostic tests
               diagnostics["diagnostic_test_1"] = self._diagnostic_test_1(Y, T, X)
               diagnostics["diagnostic_test_2"] = self._diagnostic_test_2(Y, T, X)
               diagnostics["diagnostic_test_3"] = self._diagnostic_test_3(Y, T, X)
               
               # Overall assessment
               diagnostics["overall_assessment"] = self._assess_assumptions(diagnostics)
               
               return diagnostics
               
           except Exception as e:
               logger.error(f"Error in {self.method_name} diagnostics: {e}")
               raise
       
       def _validate_inputs(self, data: pd.DataFrame, variables: Variables):
           """Validate inputs for the method"""
           if variables.treatment_variable not in data.columns:
               raise ValueError(f"Treatment variable {variables.treatment_variable} not found")
           
           if variables.outcome_variable not in data.columns:
               raise ValueError(f"Outcome variable {variables.outcome_variable} not found")
           
           # Add method-specific validation
           # e.g., check for required covariates, data types, etc.
       
       def _compute_effect(self, Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> float:
           """Compute the causal effect estimate"""
           # Implement your method's core estimation logic
           pass
       
       def _compute_standard_error(self, Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> float:
           """Compute standard error of the effect estimate"""
           # Implement standard error calculation
           pass
       
       def _compute_confidence_interval(self, effect: float, se: float, alpha: float = 0.05) -> tuple:
           """Compute confidence interval"""
           z_score = stats.norm.ppf(1 - alpha/2)
           lower = effect - z_score * se
           upper = effect + z_score * se
           return (lower, upper)
       
       def _compute_p_value(self, effect: float, se: float) -> float:
           """Compute p-value for null hypothesis of no effect"""
           z_stat = effect / se if se > 0 else 0
           return 2 * (1 - stats.norm.cdf(abs(z_stat)))
       
       def _compute_additional_results(self, Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> Dict[str, Any]:
           """Compute method-specific additional results"""
           return {}
       
       def _diagnostic_test_1(self, Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> Dict[str, Any]:
           """Implement first diagnostic test"""
           # Return test results with interpretation
           return {
               "test_statistic": 0.0,
               "p_value": 1.0,
               "interpretation": "Test interpretation",
               "passed": True
           }
       
       def _diagnostic_test_2(self, Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> Dict[str, Any]:
           """Implement second diagnostic test"""
           return {
               "test_statistic": 0.0,
               "p_value": 1.0,
               "interpretation": "Test interpretation",
               "passed": True
           }
       
       def _diagnostic_test_3(self, Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> Dict[str, Any]:
           """Implement third diagnostic test"""
           return {
               "test_statistic": 0.0,
               "p_value": 1.0,
               "interpretation": "Test interpretation", 
               "passed": True
           }
       
       def _assess_assumptions(self, diagnostics: Dict[str, Any]) -> str:
           """Provide overall assessment of assumption validity"""
           # Analyze diagnostic results and provide summary
           passed_tests = sum(1 for test in diagnostics.values() 
                             if isinstance(test, dict) and test.get("passed", False))
           total_tests = len([k for k in diagnostics.keys() if k != "overall_assessment"])
           
           if passed_tests == total_tests:
               return "All diagnostic tests passed. Method assumptions appear satisfied."
           elif passed_tests >= total_tests * 0.7:
               return "Most diagnostic tests passed. Method may be appropriate with caution."
           else:
               return "Multiple diagnostic tests failed. Consider alternative methods."

Step 2: Create Method-Specific Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the estimator, diagnostics, and LLM assist modules:

.. code-block:: python

   # causal_agent/methods/observational/my_new_method/estimator.py
   """Core estimation logic for My New Method"""
   
   def estimate_my_new_method(data, variables, **kwargs):
       """Main estimation function"""
       method = MyNewMethod(**kwargs)
       return method.estimate(data, variables)

.. code-block:: python

   # causal_agent/methods/observational/my_new_method/diagnostics.py
   """Diagnostic tests for My New Method"""
   
   def run_diagnostics(data, variables, **kwargs):
       """Run all diagnostic tests"""
       method = MyNewMethod(**kwargs)
       return method.diagnose(data, variables)

.. code-block:: python

   # causal_agent/methods/observational/my_new_method/llm_assist.py
   """LLM assistance for My New Method"""
   
   def get_method_explanation(variables, dataset_analysis):
       """Generate explanation for why this method was selected"""
       return f"My New Method was selected because..."

Step 3: Update Decision Tree Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your method to the decision tree in ``causal_agent/components/decision_tree.py``:

.. code-block:: python

   # Add method constant
   MY_NEW_METHOD = "my_new_method"
   
   # Add to method assumptions
   METHOD_ASSUMPTIONS = {
       # ... existing methods ...
       MY_NEW_METHOD: [
           "assumption 1: detailed description",
           "assumption 2: detailed description",
           "assumption 3: detailed description"
       ]
   }
   
   # Add to decision logic
   def rule_based_select_method(variables, dataset_analysis, excluded_methods=None):
       """Select causal method using rule-based logic"""
       
       # ... existing logic ...
       
       # Add your method's selection criteria
       if _should_use_my_new_method(variables, dataset_analysis):
           if MY_NEW_METHOD not in (excluded_methods or []):
               return {
                   "method": MY_NEW_METHOD,
                   "confidence": 0.8,
                   "reasoning": "My New Method selected because [criteria]",
                   "assumptions": METHOD_ASSUMPTIONS[MY_NEW_METHOD],
                   "alternatives": ["alternative_method_1", "alternative_method_2"]
               }
   
   def _should_use_my_new_method(variables, dataset_analysis):
       """Determine if My New Method should be used"""
       # Implement your selection criteria
       # Example:
       if (variables.treatment_variable and 
           variables.outcome_variable and
           len(variables.covariates or []) >= 3 and
           not variables.is_rct):
           return True
       return False

Step 4: Update Method Executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your method to the method executor in ``causal_agent/tools/method_executor_tool.py``:

.. code-block:: python

   # Import your method
   from causal_agent.methods.observational.my_new_method.estimator import estimate_my_new_method
   
   # Add to method mapping
   METHOD_FUNCTIONS = {
       # ... existing methods ...
       "my_new_method": estimate_my_new_method,
   }

Step 5: Create LLM Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add prompts for your method in ``causal_agent/prompts/``:

.. code-block:: python

   # causal_agent/prompts/my_new_method_prompts.py
   
   MY_NEW_METHOD_SELECTION_PROMPT = """
   You are evaluating whether My New Method is appropriate for causal analysis.
   
   Dataset Analysis:
   {dataset_analysis}
   
   Variables:
   - Treatment: {treatment_variable}
   - Outcome: {outcome_variable}
   - Covariates: {covariates}
   
   My New Method is appropriate when:
   - [Condition 1]
   - [Condition 2]
   - [Condition 3]
   
   Based on the dataset and variables, assess whether My New Method should be used.
   
   Return your response as JSON:
   {{
       "recommended": true/false,
       "confidence": 0.0-1.0,
       "reasoning": "explanation",
       "concerns": ["list of concerns if any"]
   }}
   """
   
   MY_NEW_METHOD_INTERPRETATION_PROMPT = """
   Interpret the results from My New Method analysis.
   
   Results:
   - Effect Estimate: {effect_estimate}
   - Standard Error: {standard_error}
   - P-value: {p_value}
   - Confidence Interval: {confidence_interval}
   
   Diagnostic Tests:
   {diagnostics}
   
   Provide a clear interpretation of these results, including:
   1. The estimated causal effect and its meaning
   2. Statistical significance and confidence
   3. Assessment of method assumptions
   4. Limitations and caveats
   
   Format as structured explanation suitable for non-experts.
   """

Step 6: Create Comprehensive Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a comprehensive test suite for your method:

.. code-block:: python

   # tests/unit/methods/test_my_new_method.py
   """Tests for My New Method implementation"""
   
   import pytest
   import pandas as pd
   import numpy as np
   from causal_agent.methods.observational.my_new_method import MyNewMethod
   from causal_agent.models import Variables
   
   class TestMyNewMethod:
       """Test suite for My New Method"""
       
       @pytest.fixture
       def sample_data(self):
           """Create sample data for testing"""
           np.random.seed(42)
           n = 1000
           
           # Generate covariates
           X1 = np.random.normal(0, 1, n)
           X2 = np.random.normal(0, 1, n)
           
           # Generate treatment (with selection based on covariates)
           treatment_prob = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
           T = np.random.binomial(1, treatment_prob, n)
           
           # Generate outcome (with treatment effect)
           Y = 2 + 0.5 * X1 + 0.3 * X2 + 1.5 * T + np.random.normal(0, 1, n)
           
           return pd.DataFrame({
               'treatment': T,
               'outcome': Y,
               'covariate1': X1,
               'covariate2': X2
           })
       
       @pytest.fixture
       def variables(self):
           """Create variables specification"""
           return Variables(
               treatment_variable='treatment',
               outcome_variable='outcome',
               covariates=['covariate1', 'covariate2'],
               is_rct=False
           )
       
       def test_method_initialization(self):
           """Test method initialization"""
           method = MyNewMethod()
           assert method.method_name == "my_new_method"
           assert len(method.assumptions) > 0
           assert len(method.diagnostics) > 0
       
       def test_estimate_basic(self, sample_data, variables):
           """Test basic estimation functionality"""
           method = MyNewMethod()
           results = method.estimate(sample_data, variables)
           
           # Check required output fields
           required_fields = [
               'effect_estimate', 'standard_error', 'confidence_interval',
               'p_value', 'method', 'assumptions'
           ]
           for field in required_fields:
               assert field in results
           
           # Check data types
           assert isinstance(results['effect_estimate'], (int, float))
           assert isinstance(results['standard_error'], (int, float))
           assert isinstance(results['confidence_interval'], (list, tuple))
           assert isinstance(results['p_value'], (int, float))
           
           # Check reasonable values
           assert results['standard_error'] > 0
           assert 0 <= results['p_value'] <= 1
           assert len(results['confidence_interval']) == 2
       
       def test_diagnostics(self, sample_data, variables):
           """Test diagnostic functionality"""
           method = MyNewMethod()
           diagnostics = method.diagnose(sample_data, variables)
           
           # Check that diagnostics are returned
           assert isinstance(diagnostics, dict)
           assert 'overall_assessment' in diagnostics
           
           # Check individual diagnostic tests
           for diagnostic in method.diagnostics:
               assert diagnostic in diagnostics
               assert isinstance(diagnostics[diagnostic], dict)
               assert 'passed' in diagnostics[diagnostic]
       
       def test_input_validation(self, sample_data):
           """Test input validation"""
           method = MyNewMethod()
           
           # Test missing treatment variable
           variables_missing_treatment = Variables(
               treatment_variable='missing_var',
               outcome_variable='outcome',
               covariates=['covariate1']
           )
           
           with pytest.raises(ValueError):
               method.estimate(sample_data, variables_missing_treatment)
       
       def test_edge_cases(self, variables):
           """Test edge cases and error handling"""
           method = MyNewMethod()
           
           # Test with minimal data
           minimal_data = pd.DataFrame({
               'treatment': [0, 1],
               'outcome': [1.0, 2.0],
               'covariate1': [0.0, 1.0],
               'covariate2': [0.0, 1.0]
           })
           
           # Should handle minimal data gracefully
           results = method.estimate(minimal_data, variables)
           assert 'effect_estimate' in results
       
       def test_synthetic_data_scenarios(self):
           """Test with various synthetic data scenarios"""
           # Test with different effect sizes
           for true_effect in [0, 0.5, 1.0, 2.0]:
               data = self._generate_data_with_effect(true_effect)
               variables = Variables(
                   treatment_variable='treatment',
                   outcome_variable='outcome',
                   covariates=['covariate1', 'covariate2']
               )
               
               method = MyNewMethod()
               results = method.estimate(data, variables)
               
               # Check that estimate is reasonable
               assert isinstance(results['effect_estimate'], (int, float))
               # Add more specific checks based on your method
       
       def _generate_data_with_effect(self, true_effect, n=500):
           """Generate synthetic data with known effect size"""
           np.random.seed(42)
           
           X1 = np.random.normal(0, 1, n)
           X2 = np.random.normal(0, 1, n)
           T = np.random.binomial(1, 0.5, n)
           Y = X1 + X2 + true_effect * T + np.random.normal(0, 1, n)
           
           return pd.DataFrame({
               'treatment': T,
               'outcome': Y,
               'covariate1': X1,
               'covariate2': X2
           })

Step 7: Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

Create integration tests that test your method within the full workflow:

.. code-block:: python

   # tests/integration/test_my_new_method_integration.py
   """Integration tests for My New Method"""
   
   import pytest
   from causal_agent.agent import run_causal_analysis
   from causal_agent.components.decision_tree import rule_based_select_method
   
   class TestMyNewMethodIntegration:
       """Integration tests for My New Method"""
       
       def test_method_selection_integration(self, sample_dataset_path):
           """Test that method is selected appropriately"""
           # Create query that should trigger your method
           query = "What is the effect of treatment on outcome controlling for covariates?"
           
           result = run_causal_analysis(
               query=query,
               dataset_path=sample_dataset_path,
               dataset_description="Observational study with treatment and covariates"
           )
           
           # Check that your method was selected
           assert result['results']['results']['method_used'] == 'my_new_method'
       
       def test_full_workflow_with_method(self, sample_dataset_path):
           """Test complete workflow using your method"""
           query = "Analyze the causal effect of treatment on outcome"
           
           result = run_causal_analysis(
               query=query,
               dataset_path=sample_dataset_path
           )
           
           # Check that analysis completed successfully
           assert 'error' not in result
           assert 'effect_estimate' in result['results']['results']
           assert 'explanation' in result

Step 8: Create Synthetic Data for Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create synthetic data generators specific to your method:

.. code-block:: python

   # causal_agent/synthetic/my_new_method_data.py
   """Synthetic data generation for My New Method testing"""
   
   import numpy as np
   import pandas as pd
   from causal_agent.synthetic.generator import DataGenerator
   
   class MyNewMethodDataGenerator(DataGenerator):
       """Generate synthetic data suitable for My New Method"""
       
       def __init__(self, n_observations=1000, true_effect=1.0, **kwargs):
           super().__init__(n_observations=n_observations, true_effect=true_effect, **kwargs)
           self.method = "my_new_method"
       
       def generate_data(self) -> pd.DataFrame:
           """Generate synthetic data with known causal structure"""
           np.random.seed(self.seed)
           
           # Generate covariates
           X = np.random.multivariate_normal(
               mean=self.mean,
               cov=self.covar,
               size=self.n_observations
           )
           
           # Generate treatment with selection based on covariates
           treatment_logits = 0.5 * X[:, 0] + 0.3 * X[:, 1]
           treatment_probs = 1 / (1 + np.exp(-treatment_logits))
           T = np.random.binomial(1, treatment_probs)
           
           # Generate outcome with treatment effect
           Y = (2.0 + 
                0.5 * X[:, 0] + 
                0.3 * X[:, 1] + 
                self.true_effect * T + 
                np.random.normal(0, 1, self.n_observations))
           
           # Create DataFrame
           data = pd.DataFrame({
               'treatment': T,
               'outcome': Y
           })
           
           # Add covariates
           for i in range(self.n_continuous_covars):
               data[f'covariate_{i+1}'] = X[:, i]
           
           # Add binary covariates
           for i in range(self.n_binary_covars):
               data[f'binary_covariate_{i+1}'] = np.random.binomial(1, 0.5, self.n_observations)
           
           self.data = data
           return data
       
       def get_true_parameters(self) -> Dict[str, Any]:
           """Return true parameters for validation"""
           return {
               'true_effect': self.true_effect,
               'treatment_variable': 'treatment',
               'outcome_variable': 'outcome',
               'covariates': [f'covariate_{i+1}' for i in range(self.n_continuous_covars)] +
                           [f'binary_covariate_{i+1}' for i in range(self.n_binary_covars)]
           }

Step 9: Documentation
~~~~~~~~~~~~~~~~~~~~~

Create comprehensive documentation for your method:

.. code-block:: rst

   # docs/source/methods/observational/my_new_method.rst
   
   My New Method
   =============
   
   Overview
   --------
   
   My New Method is a causal inference technique designed for [specific use case].
   It is particularly useful when [conditions] and provides [benefits].
   
   When to Use
   -----------
   
   My New Method is recommended when:
   
   * **Condition 1**: Detailed explanation
   * **Condition 2**: Detailed explanation  
   * **Condition 3**: Detailed explanation
   
   The CAIS decision tree will automatically select this method when these conditions are met.
   
   Assumptions
   -----------
   
   This method relies on the following key assumptions:
   
   1. **Assumption 1**: Detailed explanation and implications
   2. **Assumption 2**: Detailed explanation and implications
   3. **Assumption 3**: Detailed explanation and implications
   
   Diagnostic Tests
   ----------------
   
   CAIS automatically runs the following diagnostic tests:
   
   * **Test 1**: What it checks and how to interpret
   * **Test 2**: What it checks and how to interpret
   * **Test 3**: What it checks and how to interpret
   
   Implementation Details
   ----------------------
   
   The method is implemented using [statistical approach/library] and follows
   the standard workflow:
   
   1. **Data Preprocessing**: Variable extraction and validation
   2. **Estimation**: Core causal effect estimation
   3. **Inference**: Standard errors and confidence intervals
   4. **Diagnostics**: Assumption testing and validation
   
   Example Usage
   -------------
   
   .. code-block:: python
   
      from causal_agent import run_causal_analysis
      
      # The agent will automatically select My New Method when appropriate
      result = run_causal_analysis(
          query="What is the effect of treatment on outcome?",
          dataset_path="data.csv",
          dataset_description="Observational study with covariates"
      )
      
      print(f"Effect estimate: {result['results']['results']['effect_estimate']}")
      print(f"Method used: {result['results']['results']['method_used']}")
   
   Interpretation
   --------------
   
   Results from My New Method should be interpreted as:
   
   * **Effect Estimate**: [Interpretation]
   * **Confidence Interval**: [Interpretation]
   * **P-value**: [Interpretation]
   
   Limitations
   -----------
   
   * **Limitation 1**: Description and implications
   * **Limitation 2**: Description and implications
   * **Limitation 3**: Description and implications
   
   References
   ----------
   
   * Author, A. (Year). "Title of Paper". *Journal Name*.
   * Author, B. (Year). "Another Paper". *Journal Name*.

Testing Your Implementation
---------------------------

Comprehensive Testing Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test method within the full workflow
3. **Synthetic Data Tests**: Validate with known ground truth
4. **Edge Case Tests**: Test boundary conditions and error handling
5. **Performance Tests**: Ensure scalability and efficiency

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run unit tests for your method
   pytest tests/unit/methods/test_my_new_method.py -v
   
   # Run integration tests
   pytest tests/integration/test_my_new_method_integration.py -v
   
   # Run all tests
   pytest tests/ -k "my_new_method" -v
   
   # Run with coverage
   pytest tests/ -k "my_new_method" --cov=causal_agent.methods.observational.my_new_method

Validation Checklist
~~~~~~~~~~~~~~~~~~~~~

Before submitting your method implementation, ensure:

- [ ] Method class inherits from ``CausalMethod``
- [ ] All required methods are implemented (``estimate``, ``diagnose``, etc.)
- [ ] Decision tree logic includes your method
- [ ] Method executor includes your method
- [ ] Comprehensive test suite with >90% coverage
- [ ] Synthetic data generator for testing
- [ ] Documentation with examples
- [ ] LLM prompts for method selection and interpretation
- [ ] Integration tests pass
- [ ] Performance is acceptable for typical datasets

Best Practices
--------------

Code Quality
~~~~~~~~~~~~

* **Type Hints**: Use comprehensive type hints for all functions
* **Documentation**: Include detailed docstrings with examples
* **Error Handling**: Implement robust error handling and validation
* **Logging**: Add appropriate logging for debugging and monitoring
* **Code Style**: Follow PEP 8 and project conventions

Statistical Rigor
~~~~~~~~~~~~~~~~~

* **Assumptions**: Clearly document all statistical assumptions
* **Diagnostics**: Implement comprehensive diagnostic tests
* **Validation**: Test with synthetic data with known ground truth
* **Robustness**: Test with various data scenarios and edge cases
* **Interpretation**: Provide clear guidance for result interpretation

Integration
~~~~~~~~~~~

* **Decision Logic**: Implement clear criteria for method selection
* **LLM Integration**: Create effective prompts for automated reasoning
* **Error Recovery**: Handle failures gracefully with fallback options
* **Performance**: Optimize for typical dataset sizes and use cases

Common Pitfalls
---------------

* **Incomplete Validation**: Not testing edge cases or error conditions
* **Poor Integration**: Method works in isolation but fails in workflow
* **Unclear Selection Criteria**: Decision tree logic is ambiguous
* **Missing Diagnostics**: Not implementing assumption checking
* **Performance Issues**: Method doesn't scale to realistic datasets
* **Documentation Gaps**: Missing examples or unclear explanations

Getting Help
------------

If you encounter issues while implementing your method:

1. **Review Existing Methods**: Look at similar implementations for patterns
2. **Check Tests**: Ensure your tests cover all functionality
3. **Validate Integration**: Test within the full agent workflow
4. **Ask for Review**: Submit draft implementation for feedback
5. **Consult Documentation**: Review architecture and design patterns

Contributing your method to CAIS helps expand the system's capabilities and benefits the entire causal inference community. Following these guidelines ensures your implementation is robust, well-tested, and seamlessly integrated with the existing system.
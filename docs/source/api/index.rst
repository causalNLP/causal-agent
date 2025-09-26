API Reference
=============

Complete API documentation for all CAIS modules, classes, and functions. This reference provides detailed information about parameters, return values, and usage examples for every public interface.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules/index

API Overview
------------

The CAIS API is organized into several key modules:

.. autosummary::
   :toctree: modules
   :template: autosummary/module.rst
   :recursive:

   causal_agent

**Core Components**
   * :class:`causal_agent.agent.CausalAgent`: Main interface for causal analysis
   * :class:`causal_agent.components.dataset_analyzer.DatasetAnalyzer`: Data validation and preprocessing
   * :class:`causal_agent.components.decision_tree.DecisionTree`: Automatic method selection logic
   * :class:`causal_agent.components.explanation_generator.ExplanationGenerator`: Result formatting and explanation

**Causal Methods**
   * :mod:`causal_agent.methods.diff_in_means`: Simple difference in means estimation
   * :mod:`causal_agent.methods.difference_in_differences`: Difference-in-differences implementation
   * :mod:`causal_agent.methods.instrumental_variable`: Instrumental variable methods
   * :mod:`causal_agent.methods.regression_discontinuity`: Regression discontinuity design
   * :mod:`causal_agent.methods.propensity_score`: Propensity score matching and weighting
   * :mod:`causal_agent.methods.backdoor_adjustment`: Backdoor adjustment methods
   * :mod:`causal_agent.methods.linear_regression`: Linear regression for causal inference

**Tools and Utilities**
   * :mod:`causal_agent.tools`: Analysis tools and method execution
   * :mod:`causal_agent.utils`: Helper functions and utilities
   * :mod:`causal_agent.synthetic`: Synthetic data generation
   * :mod:`causal_agent.prompts`: LLM prompt templates

Quick Reference
---------------

**Basic Usage**

.. code-block:: python

   from causal_agent import CausalAgent
   
   agent = CausalAgent()
   result = agent.analyze(data, treatment, outcome)

**Method-Specific Analysis**

.. code-block:: python

   from causal_agent.methods import DifferenceInDifferences
   
   did = DifferenceInDifferences()
   result = did.estimate(data, treatment, outcome, time_var)

**Custom Configuration**

.. code-block:: python

   from causal_agent import CausalAgent, Config
   
   config = Config(llm_provider="openai", model="gpt-4")
   agent = CausalAgent(config=config)

Cross-References and Links
--------------------------

**Related Documentation Sections:**

* :doc:`../getting_started/index` - Get started with CAIS
* :doc:`../user_guide/index` - Comprehensive usage guide  
* :doc:`../methods/index` - Causal inference method details
* :doc:`../tutorials/index` - Interactive tutorials and examples

**External References:**

* `DoWhy Documentation <https://www.pywhy.org/dowhy/>`_ - Related causal inference library
* `Scikit-learn API <https://scikit-learn.org/stable/modules/classes.html>`_ - Machine learning API reference
* `Pandas API <https://pandas.pydata.org/docs/reference/>`_ - Data manipulation reference

Navigation Tips
---------------

* Use the search function to find specific methods or parameters
* Click on class/function names to see detailed documentation
* Follow cross-references to related methods and concepts
* Check the source code links for implementation details
* Use the breadcrumb navigation to understand module hierarchy
* Refer to the :doc:`../development/contributing` guide for code standards
Code Examples
=============

Quick, focused examples demonstrating specific CAIS features and common use cases. These examples are designed to be copy-and-paste ready for your own projects.

.. toctree::
   :maxdepth: 2

   dataset_method_gallery
   decision_path_comparisons
   basic_usage_examples
   data_preparation_examples
   method_specific_examples
   configuration_examples
   integration_examples

Example Categories
------------------

**Dataset Properties and Method Selection Gallery**
   Visual guide showing how different dataset characteristics lead to different method selections:
   
   * Decision tree navigation examples
   * Method exclusion explanations
   * Side-by-side comparisons
   * Interactive decision framework

**Decision Path Comparisons**
   Side-by-side comparisons of how similar datasets with slight variations lead to different method selections:
   
   * Randomized vs. observational data
   * Cross-sectional vs. panel data
   * Strong vs. weak instruments
   * Good vs. poor covariate overlap

**Basic Usage Examples**
   Simple, straightforward examples for common tasks:
   
   * Loading data and running basic analysis
   * Interpreting CAIS output
   * Saving and loading results
   * Basic visualization of results

**Data Preparation Examples**
   Preparing your data for causal analysis:
   
   * Handling missing values
   * Creating treatment and control groups
   * Time series data preparation
   * Merging multiple datasets

**Method-Specific Examples**
   Examples for each causal inference method:
   
   * RCT analysis with CAIS
   * Difference-in-Differences implementation
   * Instrumental Variables analysis
   * Regression Discontinuity Design
   * Propensity Score methods

**Configuration Examples**
   Customizing CAIS for your needs:
   
   * LLM provider configuration
   * Custom method selection
   * Batch processing setup
   * Performance optimization

**Integration Examples**
   Using CAIS with other tools and workflows:
   
   * Pandas DataFrame integration
   * Scikit-learn pipeline integration
   * Apache Spark for big data
   * MLflow for experiment tracking

Quick Start Examples
--------------------

**Basic Analysis**

.. code-block:: python

   from causal_agent import CausalAgent
   import pandas as pd
   
   # Load your data
   data = pd.read_csv('your_data.csv')
   
   # Initialize CAIS
   agent = CausalAgent()
   
   # Run analysis
   result = agent.analyze(
       data=data,
       treatment='treatment_column',
       outcome='outcome_column'
   )
   
   # View results
   print(result.summary())

**Method-Specific Analysis**

.. code-block:: python

   from causal_agent.methods import DifferenceInDifferences
   
   # Use specific method
   did = DifferenceInDifferences()
   result = did.estimate(
       data=data,
       treatment='policy_implemented',
       outcome='outcome_measure',
       time_var='year',
       unit_var='state'
   )

**Batch Processing**

.. code-block:: python

   from causal_agent import CausalAgent
   
   agent = CausalAgent()
   
   # Process multiple datasets
   datasets = ['data1.csv', 'data2.csv', 'data3.csv']
   results = []
   
   for dataset in datasets:
       data = pd.read_csv(dataset)
       result = agent.analyze(data, 'treatment', 'outcome')
       results.append(result)

**Custom Configuration**

.. code-block:: python

   from causal_agent import CausalAgent, Config
   
   # Custom configuration
   config = Config(
       llm_provider='openai',
       model='gpt-4',
       temperature=0.1,
       max_tokens=1000
   )
   
   agent = CausalAgent(config=config)

Example Datasets
----------------

All examples include sample datasets that you can use to practice:

* **Synthetic RCT Data**: Simulated randomized experiment
* **Policy Evaluation Data**: Government policy implementation
* **Marketing Campaign Data**: Advertising effectiveness study
* **Medical Treatment Data**: Healthcare intervention analysis
* **Education Program Data**: Educational intervention evaluation

Usage Tips
----------

* **Copy and Modify**: All examples are designed to be easily adapted
* **Error Handling**: Examples include proper error handling patterns
* **Best Practices**: Follow the patterns shown for robust analysis
* **Documentation**: Each example includes detailed comments
* **Testing**: Examples include basic validation and testing approaches

Contributing Examples
---------------------

We welcome community contributions of new examples! See our :doc:`../../development/contributing` guide for:

* Example submission guidelines
* Code style requirements
* Documentation standards
* Review process
Quickstart Tutorial
==================

Get up and running with Causal Agent in under 10 minutes! This tutorial will walk you through your first causal analysis using the Causal AI Scientist.

.. contents:: Quick Navigation
   :local:
   :depth: 2

Overview
--------

In this quickstart, you'll learn how to:

1. Set up Causal Agent with your API key
2. Load a sample dataset
3. Run your first causal analysis
4. Interpret the results

Prerequisites
-------------

Before starting, make sure you have:

* Causal Agent installed (see :doc:`installation`)
* An OpenAI API key (or other supported LLM provider)
* Basic familiarity with Python

Step 1: Setup and Configuration
-------------------------------

First, let's set up your environment and API key:

.. code-block:: python

   import os
   from causal_agent import run_causal_analysis
   
   # Set your API key (replace with your actual key)
   os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
   
   # Alternatively, create a .env file with your API key
   # OPENAI_API_KEY=your-openai-api-key-here

Step 2: Prepare Your Data
-------------------------

For this tutorial, we'll use a sample dataset about job training programs. You can use your own data or download our sample:

.. code-block:: python

   import pandas as pd
   
   # Option 1: Use Causal Agent sample data
   from causal_agent.synthetic import load_sample_data
   
   # Load a sample job training dataset
   data = load_sample_data('job_training')
   data.to_csv('job_training_data.csv', index=False)
   
   # Option 2: Use your own data
   # data = pd.read_csv('your_dataset.csv')

**Sample Data Structure:**

.. code-block:: python

   # Let's examine the data structure
   print(data.head())
   print(f"Dataset shape: {data.shape}")
   print(f"Columns: {list(data.columns)}")

Expected output:

.. code-block:: text

   Dataset shape: (1000, 8)
   Columns: ['participant_id', 'job_training', 'age', 'education', 'prior_income', 'post_income', 'employment_status', 'region']

Step 3: Run Your First Analysis
-------------------------------

Now let's run a causal analysis to answer: *"Does participating in job training increase income?"*

.. code-block:: python

   # Run causal analysis
   result = run_causal_analysis(
       query="Does participating in job training increase income?",
       dataset_path="job_training_data.csv",
       dataset_description="""
       This dataset contains information about individuals who may or may not have 
       participated in a job training program. It includes demographic information 
       (age, education), employment history (prior_income, employment_status), 
       treatment status (job_training), outcome (post_income), and geographic 
       information (region).
       """
   )
   
   print("Analysis complete!")

Step 4: Understanding the Results
---------------------------------

Causal Agent returns a comprehensive result object. Let's explore what it contains:

.. code-block:: python

   # Print the main results
   print("=== CAUSAL ANALYSIS RESULTS ===")
   print(f"Query: {result['query']}")
   print(f"Method Used: {result['results']['results']['method_used']}")
   print(f"Treatment Variable: {result['results']['variables']['treatment_variable']}")
   print(f"Outcome Variable: {result['results']['variables']['outcome_variable']}")
   print(f"Causal Effect: {result['results']['results']['effect_estimate']}")
   print(f"Standard Error: {result['results']['results']['standard_error']}")
   print(f"P-value: {result['results']['results']['p_value']}")
   
   # Print the interpretation
   print("\n=== INTERPRETATION ===")
   print(result['explanation']['final_explanation_text'])

**Sample Output:**

.. code-block:: text

   === CAUSAL ANALYSIS RESULTS ===
   Query: Does participating in job training increase income?
   Method Used: Propensity Score Matching
   Treatment Variable: job_training
   Outcome Variable: post_income
   Causal Effect: 2847.32
   Standard Error: 423.18
   P-value: 0.001
   
   === INTERPRETATION ===
   The analysis suggests that participating in job training increases income by 
   approximately $2,847 on average. This effect is statistically significant 
   (p < 0.05), indicating that job training has a positive causal impact on 
   post-training income levels.

Step 5: Exploring Different Queries
-----------------------------------

Try different causal questions with the same dataset:

.. code-block:: python

   # Different causal questions
   queries = [
       "What is the effect of education level on income?",
       "Does age affect the likelihood of participating in job training?",
       "How does region influence employment outcomes?"
   ]
   
   for query in queries:
       print(f"\n--- Analyzing: {query} ---")
       result = run_causal_analysis(
           query=query,
           dataset_path="job_training_data.csv",
           dataset_description="Job training dataset with demographic and outcome variables"
       )
       
       print(f"Method: {result['results']['results']['method_used']}")
       print(f"Effect: {result['results']['results']['effect_estimate']}")

Step 6: Working with Your Own Data
----------------------------------

To analyze your own dataset, follow this template:

.. code-block:: python

   # Template for your own analysis
   your_result = run_causal_analysis(
       query="Your causal question here",
       dataset_path="path/to/your/data.csv",
       dataset_description="""
       Describe your dataset here:
       - What does each row represent?
       - What are the key variables?
       - What is the context/domain?
       - Any important data collection details?
       """
   )
   
   # Examine results
   print(f"Method selected: {your_result['results']['method_used']}")
   print(f"Treatment: {your_result['results']['treatment_variable']}")
   print(f"Outcome: {your_result['results']['outcome_variable']}")
   print(f"Effect: {your_result['results']['effect_estimate']}")

Common Use Cases
----------------

Here are some example queries you can try with different types of data:

**Education Research:**

.. code-block:: python

   query = "Does class size reduction improve student test scores?"
   # Dataset should have: class_size, test_scores, student demographics

**Healthcare:**

.. code-block:: python

   query = "What is the effect of a new treatment on patient recovery time?"
   # Dataset should have: treatment_received, recovery_days, patient characteristics

**Economics:**

.. code-block:: python

   query = "Does minimum wage increase affect employment rates?"
   # Dataset should have: min_wage_policy, employment_rate, regional controls

**Marketing:**

.. code-block:: python

   query = "How does email marketing affect customer purchase behavior?"
   # Dataset should have: email_received, purchase_amount, customer demographics

Understanding Method Selection
------------------------------

Causal Agent automatically selects the most appropriate causal inference method based on your data characteristics:

.. code-block:: python

   # Check what method was selected and why
   print(f"Selected Method: {result['results']['method_used']}")
   print(f"Method Reasoning: {result['results'].get('method_reasoning', 'Not available')}")
   
   # Common methods Causal Agent might select:
   # - Randomized Controlled Trial (RCT) analysis
   # - Propensity Score Matching
   # - Difference-in-Differences (DiD)
   # - Instrumental Variables (IV)
   # - Regression Discontinuity Design (RDD)
   # - Linear Regression with controls

Next Steps
----------

Congratulations! You've completed your first causal analysis with Causal Agent. Here's what to explore next:

**Immediate Next Steps:**

1. **Try the detailed tutorial:** :doc:`first_analysis` - Learn more about interpreting results
2. **Explore different datasets:** Use the sample datasets in the ``data/`` directory
3. **Learn about methods:** :doc:`../methods/index` - Understand when each method is used

**Advanced Usage:**

1. **User Guide:** :doc:`../user_guide/index` - Advanced configuration and batch processing
2. **Tutorials:** :doc:`../tutorials/index` - Domain-specific examples and case studies
3. **API Reference:** :doc:`../api/index` - Complete function documentation

**Getting Help:**

- **Troubleshooting:** See the troubleshooting section below
- **Community:** Join our `GitHub Discussions <https://github.com/causalNLP/causal-agent/discussions>`_
- **Issues:** Report bugs on `GitHub Issues <https://github.com/causalNLP/causal-agent/issues>`_

Troubleshooting Quick Fixes
---------------------------

**API Key Issues:**

.. code-block:: python

   # Verify your API key is set
   import os
   print("API Key set:", "OPENAI_API_KEY" in os.environ)

**Import Errors:**

.. code-block:: bash

   # Reinstall if needed
   pip install --upgrade causal-agent

**Data Format Issues:**

.. code-block:: python

   # Ensure your data is in CSV format with proper headers
   data = pd.read_csv('your_data.csv')
   print(data.dtypes)  # Check data types
   print(data.isnull().sum())  # Check for missing values

Ready for more? Continue to :doc:`first_analysis` for a deeper dive into causal analysis concepts!
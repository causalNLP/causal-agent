Basic Usage
===========

This guide covers the fundamental workflows for conducting causal analysis with CAIS. Whether you're new to causal inference or experienced with other tools, this section will help you get started with common analysis patterns.

Core Workflow
-------------

CAIS follows a structured workflow that guides you through the causal analysis process:

1. **Input Parsing**: Understanding your research question and dataset
2. **Dataset Analysis**: Examining data structure and variable types
3. **Query Interpretation**: Identifying treatment, outcome, and control variables
4. **Method Selection**: Choosing the appropriate causal inference method
5. **Method Validation**: Checking assumptions and prerequisites
6. **Method Execution**: Running the analysis with diagnostics
7. **Result Interpretation**: Generating explanations and insights
8. **Output Formatting**: Presenting results in a structured format

Python API Usage
-----------------

Single Analysis
~~~~~~~~~~~~~~~

The most common use case is analyzing a single dataset with a specific causal question:

.. code-block:: python

    from causal_agent import run_causal_analysis
    
    # Basic analysis
    result = run_causal_analysis(
        query="What is the effect of job training on earnings?",
        dataset_path="data/lalonde_data.csv",
        dataset_description="LaLonde job training experiment data"
    )
    
    # Access key results
    effect_estimate = result['results']['results']['effect_estimate']
    method_used = result['results']['results']['method_used']
    treatment_var = result['results']['variables']['treatment_variable']
    outcome_var = result['results']['variables']['outcome_variable']
    
    print(f"Method: {method_used}")
    print(f"Effect of {treatment_var} on {outcome_var}: {effect_estimate}")

Understanding Results
~~~~~~~~~~~~~~~~~~~~

CAIS returns a structured dictionary with comprehensive analysis results:

.. code-block:: python

    # Example result structure
    {
        'results': {
            'results': {
                'effect_estimate': 1794.34,
                'standard_error': 632.85,
                'confidence_interval': [554.95, 3033.73],
                'p_value': 0.0045,
                'method_used': 'propensity_score_matching'
            },
            'variables': {
                'treatment_variable': 'treat',
                'outcome_variable': 're78',
                'covariates': ['age', 'education', 'black', 'hispanic', 'married']
            },
            'diagnostics': {
                'balance_statistics': {...},
                'assumption_checks': {...}
            },
            'explanation': "The analysis found a significant positive effect..."
        }
    }

Command Line Interface
----------------------

For quick analyses or integration into scripts, use the CLI:

Single Analysis
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Basic command
    causal_agent run data/lalonde_data.csv "What is the effect of job training on earnings?"
    
    # With dataset description
    causal_agent run data/lalonde_data.csv \
        "What is the effect of job training on earnings?" \
        --desc "LaLonde job training experiment with treatment and control groups"
    
    # Specify LLM provider and model
    causal_agent run data/lalonde_data.csv \
        "What is the effect of job training on earnings?" \
        --llm-provider anthropic \
        --llm-name claude-3-5-sonnet-latest

Common Analysis Patterns
------------------------

Experimental Data (RCT)
~~~~~~~~~~~~~~~~~~~~~~~

When you have randomized controlled trial data:

.. code-block:: python

    result = run_causal_analysis(
        query="What is the treatment effect in this randomized experiment?",
        dataset_path="data/rct_data.csv",
        dataset_description="Randomized controlled trial with treatment and control groups"
    )

Observational Data
~~~~~~~~~~~~~~~~~~

For observational studies where you need to control for confounders:

.. code-block:: python

    result = run_causal_analysis(
        query="What is the effect of education on income, controlling for background factors?",
        dataset_path="data/observational_data.csv", 
        dataset_description="Survey data with education, income, and demographic variables"
    )

Time Series / Panel Data
~~~~~~~~~~~~~~~~~~~~~~~~

For difference-in-differences or other temporal analyses:

.. code-block:: python

    result = run_causal_analysis(
        query="What was the effect of the policy change on outcomes over time?",
        dataset_path="data/panel_data.csv",
        dataset_description="Panel data with pre/post policy implementation periods"
    )

Instrumental Variables
~~~~~~~~~~~~~~~~~~~~~

When you have an instrument for causal identification:

.. code-block:: python

    result = run_causal_analysis(
        query="What is the effect of education on wages using distance to college as an instrument?",
        dataset_path="data/iv_data.csv",
        dataset_description="Data with education, wages, and distance to college as instrument"
    )

Regression Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~

For sharp cutoff designs:

.. code-block:: python

    result = run_causal_analysis(
        query="What is the effect of the scholarship program on test scores?",
        dataset_path="data/rdd_data.csv",
        dataset_description="Student data with test scores and scholarship eligibility cutoff"
    )

Working with Results
--------------------

Extracting Key Information
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get the main causal effect estimate
    effect = result['results']['results']['effect_estimate']
    se = result['results']['results']['standard_error']
    ci = result['results']['results']['confidence_interval']
    
    # Check statistical significance
    p_value = result['results']['results']['p_value']
    is_significant = p_value < 0.05
    
    # Get variable information
    variables = result['results']['variables']
    treatment = variables['treatment_variable']
    outcome = variables['outcome_variable']
    covariates = variables.get('covariates', [])

Interpreting Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Access diagnostic information
    diagnostics = result['results']['diagnostics']
    
    # For propensity score methods
    if 'balance_statistics' in diagnostics:
        balance = diagnostics['balance_statistics']
        print("Covariate balance after matching:")
        for var, stats in balance.items():
            print(f"  {var}: standardized difference = {stats['std_diff']:.3f}")
    
    # For IV methods  
    if 'first_stage_f_stat' in diagnostics:
        f_stat = diagnostics['first_stage_f_stat']
        print(f"First stage F-statistic: {f_stat:.2f}")
        if f_stat < 10:
            print("Warning: Weak instrument (F < 10)")

Error Handling
--------------

CAIS provides informative error messages when issues occur:

.. code-block:: python

    result = run_causal_analysis(
        query="What is the effect of X on Y?",
        dataset_path="data/problematic_data.csv"
    )
    
    # Check for errors
    if 'error' in result:
        print(f"Analysis failed: {result['error']}")
    else:
        # Process successful results
        effect = result['results']['results']['effect_estimate']

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Missing Variables**
    If CAIS can't identify treatment or outcome variables, be more specific in your query:
    
    .. code-block:: python
    
        # Instead of: "What causes what?"
        # Use: "What is the effect of education on income?"

**Data Format Issues**
    Ensure your CSV has proper headers and numeric variables are correctly formatted:
    
    .. code-block:: python
    
        import pandas as pd
        df = pd.read_csv("data.csv")
        print(df.dtypes)  # Check variable types
        print(df.head())  # Check data format

**Method Selection Issues**
    If the automatic method selection isn't appropriate, the explanation will indicate why certain methods were chosen or rejected.

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

1. **Clean Variable Names**: Use descriptive, consistent variable names
2. **Handle Missing Data**: Address missing values before analysis
3. **Check Data Types**: Ensure treatment variables are properly coded (0/1 for binary)
4. **Document Your Data**: Provide clear dataset descriptions

Query Formulation
~~~~~~~~~~~~~~~~~

1. **Be Specific**: Clearly state treatment and outcome variables
2. **Use Causal Language**: Frame questions in terms of effects and causation
3. **Provide Context**: Include relevant background information in dataset descriptions

Result Interpretation
~~~~~~~~~~~~~~~~~~~~

1. **Check Assumptions**: Review diagnostic tests and assumption checks
2. **Consider Effect Size**: Look beyond statistical significance to practical significance
3. **Validate Results**: Compare with domain knowledge and alternative methods
4. **Document Decisions**: Keep track of analysis choices for reproducibility

Next Steps
----------

- For more advanced features and customization options, see :doc:`advanced_usage`
- To process multiple datasets efficiently, see :doc:`batch_processing`  
- For LLM provider setup and configuration, see :doc:`configuration`
- For detailed method documentation, see :doc:`../methods/index`
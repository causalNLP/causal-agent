Advanced Usage
==============

This guide covers advanced features and customization options for power users who need more control over the causal analysis process. Learn how to fine-tune method selection, customize analysis parameters, and integrate CAIS into complex workflows.

Advanced Configuration
----------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

CAIS can be configured through environment variables for consistent behavior across analyses:

.. code-block:: bash

    # LLM Configuration
    export LLM_PROVIDER="anthropic"
    export LLM_MODEL="claude-3-5-sonnet-latest"
    export ANTHROPIC_API_KEY="your-api-key"
    
    # Analysis Configuration
    export CAIS_DEFAULT_CONFIDENCE_LEVEL="0.95"
    export CAIS_VERBOSE_LOGGING="true"

You can also set these in a `.env` file in your project directory:

.. code-block:: bash

    # .env file
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o
    OPENAI_API_KEY=your-api-key-here
    CAIS_DEFAULT_CONFIDENCE_LEVEL=0.95

Programmatic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

For more dynamic configuration, you can set environment variables programmatically:

.. code-block:: python

    import os
    from causal_agent import run_causal_analysis
    
    # Configure for specific analysis
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = "claude-3-5-sonnet-latest"
    
    result = run_causal_analysis(
        query="What is the effect of treatment on outcome?",
        dataset_path="data.csv"
    )

Method Selection Control
------------------------

Understanding Automatic Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CAIS uses a decision tree to automatically select appropriate causal inference methods based on:

- Data structure (cross-sectional, panel, time series)
- Variable types (binary, continuous, categorical)
- Available instruments or discontinuities
- Experimental design indicators

You can examine the method selection reasoning:

.. code-block:: python

    result = run_causal_analysis(
        query="What is the effect of education on income?",
        dataset_path="data.csv"
    )
    
    # Check which method was selected and why
    method_info = result['results']['method_info']
    print(f"Selected method: {method_info['method_name']}")
    print(f"Reasoning: {method_info['selection_reasoning']}")

Influencing Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While CAIS automatically selects methods, you can influence selection through query phrasing:

.. code-block:: python

    # Suggest instrumental variable approach
    result = run_causal_analysis(
        query="What is the effect of education on wages using distance to college as an instrument?",
        dataset_path="data.csv"
    )
    
    # Suggest difference-in-differences
    result = run_causal_analysis(
        query="What was the effect of the policy change over time comparing treated and control regions?",
        dataset_path="panel_data.csv"
    )
    
    # Suggest regression discontinuity
    result = run_causal_analysis(
        query="What is the effect of the program for those just above the eligibility cutoff?",
        dataset_path="rdd_data.csv"
    )

Custom Analysis Workflows
-------------------------

Multi-Method Comparison
~~~~~~~~~~~~~~~~~~~~~~

Compare results across different causal inference methods:

.. code-block:: python

    import pandas as pd
    from causal_agent import run_causal_analysis
    
    def compare_methods(dataset_path, base_query, method_hints):
        """Compare causal estimates across different methods."""
        results = {}
        
        for method_name, query_hint in method_hints.items():
            full_query = f"{base_query} {query_hint}"
            result = run_causal_analysis(
                query=full_query,
                dataset_path=dataset_path
            )
            
            if 'error' not in result:
                results[method_name] = {
                    'effect': result['results']['results']['effect_estimate'],
                    'se': result['results']['results']['standard_error'],
                    'method': result['results']['results']['method_used']
                }
        
        return pd.DataFrame(results).T
    
    # Example usage
    method_hints = {
        'matching': 'using propensity score matching',
        'regression': 'using regression adjustment',
        'weighting': 'using inverse probability weighting'
    }
    
    comparison = compare_methods(
        "data/observational_data.csv",
        "What is the effect of treatment on outcome",
        method_hints
    )
    print(comparison)

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~

Test the robustness of your causal conclusions:

.. code-block:: python

    def sensitivity_analysis(dataset_path, query, perturbations):
        """Run sensitivity analysis with different data perturbations."""
        import numpy as np
        import pandas as pd
        
        base_result = run_causal_analysis(query=query, dataset_path=dataset_path)
        base_effect = base_result['results']['results']['effect_estimate']
        
        results = {'base': base_effect}
        
        # Load and modify data for sensitivity tests
        df = pd.read_csv(dataset_path)
        
        for name, perturbation_func in perturbations.items():
            # Apply perturbation
            df_modified = perturbation_func(df.copy())
            temp_path = f"temp_{name}.csv"
            df_modified.to_csv(temp_path, index=False)
            
            # Run analysis on modified data
            result = run_causal_analysis(query=query, dataset_path=temp_path)
            if 'error' not in result:
                results[name] = result['results']['results']['effect_estimate']
            
            # Clean up
            import os
            os.remove(temp_path)
        
        return results
    
    # Example perturbations
    perturbations = {
        'drop_5pct': lambda df: df.sample(frac=0.95),
        'add_noise': lambda df: df.assign(
            outcome=df['outcome'] + np.random.normal(0, df['outcome'].std() * 0.1, len(df))
        )
    }
    
    sensitivity_results = sensitivity_analysis(
        "data.csv",
        "What is the effect of treatment on outcome?",
        perturbations
    )

Integration Patterns
--------------------

Jupyter Notebook Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For interactive analysis and visualization:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    from causal_agent import run_causal_analysis
    
    # Run analysis
    result = run_causal_analysis(
        query="What is the effect of education on income?",
        dataset_path="data.csv"
    )
    
    # Extract and visualize results
    effect = result['results']['results']['effect_estimate']
    ci = result['results']['results']['confidence_interval']
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar([0], [effect], yerr=[[effect - ci[0]], [ci[1] - effect]], 
                fmt='o', capsize=5, capthick=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Causal Effect Estimate')
    ax.set_title('Treatment Effect with 95% Confidence Interval')
    plt.show()

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

Integrate CAIS into data processing pipelines:

.. code-block:: python

    from typing import Dict, Any, List
    import pandas as pd
    
    class CausalAnalysisPipeline:
        """Pipeline for automated causal analysis."""
        
        def __init__(self, llm_provider: str = "openai"):
            import os
            os.environ["LLM_PROVIDER"] = llm_provider
        
        def analyze_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze a single dataset."""
            result = run_causal_analysis(
                query=dataset_info['query'],
                dataset_path=dataset_info['path'],
                dataset_description=dataset_info.get('description')
            )
            
            return {
                'dataset_id': dataset_info['id'],
                'query': dataset_info['query'],
                'effect_estimate': result['results']['results']['effect_estimate'],
                'method_used': result['results']['results']['method_used'],
                'significant': result['results']['results']['p_value'] < 0.05
            }
        
        def batch_analyze(self, datasets: List[Dict[str, Any]]) -> pd.DataFrame:
            """Analyze multiple datasets."""
            results = []
            for dataset_info in datasets:
                try:
                    result = self.analyze_dataset(dataset_info)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'dataset_id': dataset_info['id'],
                        'error': str(e)
                    })
            
            return pd.DataFrame(results)
    
    # Usage
    pipeline = CausalAnalysisPipeline(llm_provider="anthropic")
    
    datasets = [
        {
            'id': 'study1',
            'path': 'data/study1.csv',
            'query': 'What is the effect of treatment A on outcome Y?',
            'description': 'RCT data from study 1'
        },
        {
            'id': 'study2', 
            'path': 'data/study2.csv',
            'query': 'What is the effect of intervention B on metric Z?',
            'description': 'Observational data from study 2'
        }
    ]
    
    results_df = pipeline.batch_analyze(datasets)

Custom Data Preprocessing
-------------------------

Data Validation and Cleaning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement custom data validation before analysis:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from typing import Tuple, List
    
    def validate_and_clean_data(df: pd.DataFrame, 
                               treatment_col: str,
                               outcome_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean data for causal analysis."""
        warnings = []
        df_clean = df.copy()
        
        # Check for missing values
        missing_treatment = df_clean[treatment_col].isna().sum()
        missing_outcome = df_clean[outcome_col].isna().sum()
        
        if missing_treatment > 0:
            warnings.append(f"Dropping {missing_treatment} rows with missing treatment")
            df_clean = df_clean.dropna(subset=[treatment_col])
        
        if missing_outcome > 0:
            warnings.append(f"Dropping {missing_outcome} rows with missing outcome")
            df_clean = df_clean.dropna(subset=[outcome_col])
        
        # Validate treatment variable
        unique_treatments = df_clean[treatment_col].nunique()
        if unique_treatments != 2:
            warnings.append(f"Treatment variable has {unique_treatments} unique values, expected 2")
        
        # Check for outliers in outcome
        Q1 = df_clean[outcome_col].quantile(0.25)
        Q3 = df_clean[outcome_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_clean[outcome_col] < (Q1 - 1.5 * IQR)) | 
                   (df_clean[outcome_col] > (Q3 + 1.5 * IQR))).sum()
        
        if outliers > 0:
            warnings.append(f"Found {outliers} potential outliers in outcome variable")
        
        return df_clean, warnings
    
    # Usage before analysis
    df = pd.read_csv("data.csv")
    df_clean, warnings = validate_and_clean_data(df, 'treatment', 'outcome')
    
    for warning in warnings:
        print(f"Warning: {warning}")
    
    # Save cleaned data
    df_clean.to_csv("data_cleaned.csv", index=False)
    
    # Run analysis on cleaned data
    result = run_causal_analysis(
        query="What is the effect of treatment on outcome?",
        dataset_path="data_cleaned.csv"
    )

Performance Optimization
------------------------

Caching Results
~~~~~~~~~~~~~~

Cache analysis results to avoid recomputation:

.. code-block:: python

    import hashlib
    import json
    import os
    from functools import wraps
    
    def cache_analysis(cache_dir: str = "analysis_cache"):
        """Decorator to cache analysis results."""
        os.makedirs(cache_dir, exist_ok=True)
        
        def decorator(func):
            @wraps(func)
            def wrapper(query: str, dataset_path: str, **kwargs):
                # Create cache key
                cache_key = hashlib.md5(
                    f"{query}_{dataset_path}_{json.dumps(kwargs, sort_keys=True)}".encode()
                ).hexdigest()
                
                cache_file = os.path.join(cache_dir, f"{cache_key}.json")
                
                # Check if cached result exists
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                
                # Run analysis and cache result
                result = func(query, dataset_path, **kwargs)
                
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                return result
            return wrapper
        return decorator
    
    # Apply caching to analysis function
    @cache_analysis()
    def cached_causal_analysis(query: str, dataset_path: str, **kwargs):
        return run_causal_analysis(query, dataset_path, **kwargs)
    
    # Usage - subsequent calls with same parameters will use cache
    result1 = cached_causal_analysis(
        "What is the effect of treatment on outcome?",
        "data.csv"
    )
    result2 = cached_causal_analysis(  # This will use cached result
        "What is the effect of treatment on outcome?", 
        "data.csv"
    )



Best Practices for Advanced Usage
---------------------------------

1. **Version Control**: Track analysis configurations and results
2. **Documentation**: Document custom workflows and parameter choices
3. **Testing**: Validate analyses on known datasets before production use
4. **Monitoring**: Log analysis performance and error rates
5. **Reproducibility**: Use fixed random seeds and version pinning

Next Steps
----------

- For batch processing workflows, see :doc:`batch_processing`
- For LLM provider configuration, see :doc:`configuration`
- For method-specific details, see :doc:`../methods/index`
- For developer documentation, see :doc:`../development/index`
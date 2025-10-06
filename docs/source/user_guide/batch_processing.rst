Batch Processing
================

CAIS provides powerful batch processing capabilities for analyzing multiple datasets efficiently. This guide covers command-line batch processing, programmatic batch analysis, and best practices for large-scale causal inference workflows.

Command Line Batch Processing
------------------------------

Basic Batch Analysis
~~~~~~~~~~~~~~~~~~~~

The simplest way to process multiple datasets is using the CLI batch command with a metadata CSV file:

.. code-block:: bash

    causal_agent batch metadata.csv data_folder/ results.json

Metadata CSV Format
~~~~~~~~~~~~~~~~~~~

Create a CSV file with the following columns:

.. code-block:: text

    natural_language_query,data_description,data_files,method,answer
    "What is the effect of job training on earnings?","LaLonde job training data","lalonde_data.csv","propensity_score","positive"
    "What is the effect of education on income?","Survey data with demographics","survey_data.csv","regression","positive"
    "What was the impact of the policy change?","Panel data pre/post policy","policy_data.csv","difference_in_differences","significant"

Required columns:
- ``natural_language_query``: The causal question to analyze
- ``data_description``: Description of the dataset
- ``data_files``: Filename of the dataset (relative to data folder)

Optional columns:
- ``method``: Expected method (for validation)
- ``answer``: Expected result (for comparison)

Example Batch Command
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Basic batch processing
    causal_agent batch studies_metadata.csv /path/to/data/ batch_results.json
    
    # With specific LLM configuration
    causal_agent batch studies_metadata.csv /path/to/data/ batch_results.json \
        --llm-provider anthropic \
        --llm-name claude-3-5-sonnet-latest
    
    # Process subset of studies
    head -10 studies_metadata.csv > subset_metadata.csv
    causal_agent batch subset_metadata.csv /path/to/data/ subset_results.json

Next Steps
----------

- For LLM provider configuration and optimization, see :doc:`configuration`
- For understanding method selection and validation, see :doc:`../methods/index`
- For advanced customization options, see :doc:`advanced_usage`

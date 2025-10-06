Configuration
=============

This guide covers how to configure CAIS for different LLM providers, customize analysis parameters, and optimize performance for your specific use case. Proper configuration ensures reliable and efficient causal analysis workflows.

LLM Provider Configuration
---------------------------

CAIS supports multiple Large Language Model providers, each with different capabilities and pricing models. Choose the provider that best fits your needs in terms of performance, cost, and availability.

Supported Providers
~~~~~~~~~~~~~~~~~~~

**OpenAI**, **Anthropic**, **Google Gemini**, **Together AI**


OpenAI Configuration
~~~~~~~~~~~~~~~~~~~~

Set up OpenAI as your LLM provider:

.. code-block:: bash

    # Environment variables
    export LLM_PROVIDER="openai"
    export LLM_MODEL="gpt-4o-mini"  # or gpt-4o, gpt-4, gpt-3.5-turbo
    export OPENAI_API_KEY="your-openai-api-key"

.. code-block:: python

    # Python configuration
    import os
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "gpt-4o-mini"
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    from causal_agent import run_causal_analysis
    
    result = run_causal_analysis(
        query="What is the effect of treatment on outcome?",
        dataset_path="data.csv"
    )

.. code-block:: bash

    # CLI usage
    causal_agent run data.csv "What is the effect of treatment on outcome?" \
        --llm-provider openai \
        --llm-name gpt-4o-mini

Anthropic Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Configure Anthropic Claude models:

.. code-block:: bash

    # Environment variables
    export LLM_PROVIDER="anthropic"
    export LLM_MODEL="claude-3-5-sonnet-latest"  # or claude-3-haiku-latest, claude-3-opus-latest
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

.. code-block:: python

    # Python configuration
    import os
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = "claude-3-5-sonnet-latest"
    os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"


Together AI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Set up Together AI:

.. code-block:: bash

    # Environment variables
    export LLM_PROVIDER="together"
    export LLM_MODEL="deepseek-ai/DeepSeek-V3"  # or other available models
    export TOGETHER_API_KEY="your-together-api-key"

Environment Configuration
-------------------------

Using .env Files
~~~~~~~~~~~~~~~~

Create a `.env` file in your project directory for persistent configuration:

.. code-block:: bash

    # .env file
    LLM_PROVIDER=anthropic
    LLM_MODEL=claude-3-5-sonnet-latest
    ANTHROPIC_API_KEY=your-api-key-here
    


.. code-block:: python

    # No need to set environment variables manually
    from causal_agent import run_causal_analysis
    
    result = run_causal_analysis(
        query="What is the effect of treatment on outcome?",
        dataset_path="data.csv"
    )


Next Steps
----------

- For basic usage patterns, see :doc:`basic_usage`
- For advanced customization, see :doc:`advanced_usage`
- For batch processing setup, see :doc:`batch_processing`
- For deployment considerations, see :doc:`../deployment/index`
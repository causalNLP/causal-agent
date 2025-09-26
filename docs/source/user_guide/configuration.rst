Configuration
=============

This guide covers how to configure CAIS for different LLM providers, customize analysis parameters, and optimize performance for your specific use case. Proper configuration ensures reliable and efficient causal analysis workflows.

LLM Provider Configuration
---------------------------

CAIS supports multiple Large Language Model providers, each with different capabilities and pricing models. Choose the provider that best fits your needs in terms of performance, cost, and availability.

Supported Providers
~~~~~~~~~~~~~~~~~~~

**OpenAI**
- Models: GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- Best for: General purpose, reliable performance
- Pricing: Pay-per-token

**Anthropic**
- Models: Claude-3.5-Sonnet, Claude-3-Haiku, Claude-3-Opus
- Best for: Complex reasoning, safety-focused applications
- Pricing: Pay-per-token

**Google Gemini**
- Models: Gemini-2.5-Flash, Gemini-1.5-Pro
- Best for: Multimodal capabilities, cost-effective
- Pricing: Pay-per-token with generous free tier

**DeepSeek**
- Models: DeepSeek-Chat, DeepSeek-Coder
- Best for: Cost-effective, good performance
- Pricing: Very competitive rates

**Together AI**
- Models: Various open-source models including DeepSeek-V3
- Best for: Open-source models, flexible deployment
- Pricing: Competitive rates

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

Google Gemini Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up Google Gemini:

.. code-block:: bash

    # Environment variables
    export LLM_PROVIDER="gemini"
    export LLM_MODEL="gemini-2.5-flash"  # or gemini-1.5-pro
    export GEMINI_API_KEY="your-gemini-api-key"

.. code-block:: python

    # Python configuration
    import os
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini-2.5-flash"
    os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

DeepSeek Configuration
~~~~~~~~~~~~~~~~~~~~~~

Configure DeepSeek models:

.. code-block:: bash

    # Environment variables
    export LLM_PROVIDER="deepseek"
    export LLM_MODEL="deepseek-chat"
    export DEEPSEEK_API_KEY="your-deepseek-api-key"

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
    
    # Optional analysis settings
    CAIS_DEFAULT_CONFIDENCE_LEVEL=0.95
    CAIS_VERBOSE_LOGGING=false
    CAIS_MAX_RETRIES=3

CAIS will automatically load these variables when imported:

.. code-block:: python

    # No need to set environment variables manually
    from causal_agent import run_causal_analysis
    
    result = run_causal_analysis(
        query="What is the effect of treatment on outcome?",
        dataset_path="data.csv"
    )

Multiple Environment Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manage different configurations for different environments:

.. code-block:: bash

    # .env.development
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini
    OPENAI_API_KEY=dev-api-key
    CAIS_VERBOSE_LOGGING=true
    
    # .env.production
    LLM_PROVIDER=anthropic
    LLM_MODEL=claude-3-5-sonnet-latest
    ANTHROPIC_API_KEY=prod-api-key
    CAIS_VERBOSE_LOGGING=false

.. code-block:: python

    # Load specific environment
    from dotenv import load_dotenv
    import os
    
    # Load development configuration
    load_dotenv('.env.development')
    
    # Or load production configuration
    # load_dotenv('.env.production')
    
    from causal_agent import run_causal_analysis

Advanced Configuration Options
------------------------------

Model Parameters
~~~~~~~~~~~~~~~~

Customize model behavior through environment variables:

.. code-block:: bash

    # Model temperature (creativity vs consistency)
    export LLM_TEMPERATURE=0.0  # More deterministic (default)
    # export LLM_TEMPERATURE=0.3  # More creative
    
    # Maximum tokens for responses
    export LLM_MAX_TOKENS=4000
    
    # Request timeout (seconds)
    export LLM_TIMEOUT=60

.. code-block:: python

    # Programmatic configuration
    from causal_agent.config import get_llm_client
    
    # Get LLM client with custom parameters
    llm = get_llm_client(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.1,
        max_tokens=2000,
        timeout=30
    )

Analysis Configuration
~~~~~~~~~~~~~~~~~~~~~~

Configure analysis behavior:

.. code-block:: bash

    # Default confidence level for confidence intervals
    export CAIS_DEFAULT_CONFIDENCE_LEVEL=0.95
    
    # Enable verbose logging for debugging
    export CAIS_VERBOSE_LOGGING=true
    
    # Maximum retry attempts for failed analyses
    export CAIS_MAX_RETRIES=3
    
    # Timeout for individual analysis steps (seconds)
    export CAIS_STEP_TIMEOUT=120

Performance Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize performance for your use case:

.. code-block:: bash

    # Enable result caching
    export CAIS_ENABLE_CACHING=true
    export CAIS_CACHE_DIR="./analysis_cache"
    
    # Parallel processing settings
    export CAIS_MAX_WORKERS=4
    export CAIS_BATCH_SIZE=10
    
    # Memory management
    export CAIS_MEMORY_THRESHOLD=0.8  # Trigger cleanup at 80% memory usage

Provider-Specific Optimization
------------------------------

OpenAI Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Optimize for cost vs performance
    import os
    
    # Cost-optimized setup
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "gpt-4o-mini"  # Most cost-effective
    os.environ["LLM_TEMPERATURE"] = "0"      # Reduce variability
    
    # Performance-optimized setup
    os.environ["LLM_MODEL"] = "gpt-4o"       # Best performance
    os.environ["LLM_MAX_TOKENS"] = "4000"    # Allow longer responses

Anthropic Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configure for complex reasoning tasks
    import os
    
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = "claude-3-5-sonnet-latest"  # Best reasoning
    os.environ["LLM_TEMPERATURE"] = "0"                   # Consistent results
    
    # For faster, simpler tasks
    os.environ["LLM_MODEL"] = "claude-3-haiku-latest"     # Faster, cheaper

Configuration Validation
------------------------

Validate Configuration
~~~~~~~~~~~~~~~~~~~~~~

Test your configuration before running analyses:

.. code-block:: python

    from causal_agent.config import get_llm_client
    import os
    
    def validate_configuration():
        """Validate LLM configuration."""
        try:
            provider = os.getenv("LLM_PROVIDER", "openai")
            model = os.getenv("LLM_MODEL")
            
            print(f"Testing configuration: {provider} - {model}")
            
            # Test LLM client initialization
            llm = get_llm_client()
            
            # Test simple query
            response = llm.invoke("Hello, can you respond with 'Configuration test successful'?")
            print(f"Response: {response.content}")
            
            print("✓ Configuration validation successful")
            return True
            
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return False
    
    # Run validation
    if validate_configuration():
        print("Ready to run causal analyses!")

Configuration Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common configuration issues and solutions:

.. code-block:: python

    import os
    from causal_agent.config import get_llm_client
    
    def diagnose_configuration():
        """Diagnose common configuration issues."""
        
        print("CAIS Configuration Diagnosis")
        print("=" * 40)
        
        # Check environment variables
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        
        print(f"LLM Provider: {provider or 'Not set (will default to openai)'}")
        print(f"LLM Model: {model or 'Not set (will use provider default)'}")
        
        # Check API keys
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "together": os.getenv("TOGETHER_API_KEY")
        }
        
        print("\nAPI Keys:")
        for provider_name, key in api_keys.items():
            status = "✓ Set" if key else "✗ Not set"
            print(f"  {provider_name}: {status}")
        
        # Test LLM initialization
        print("\nTesting LLM initialization...")
        try:
            llm = get_llm_client()
            print("✓ LLM client initialized successfully")
        except Exception as e:
            print(f"✗ LLM initialization failed: {e}")
            
            # Provide specific guidance
            if "API key" in str(e).lower():
                print("  → Check that the correct API key is set for your provider")
            elif "provider" in str(e).lower():
                print("  → Check that LLM_PROVIDER is set to a supported provider")
            elif "model" in str(e).lower():
                print("  → Check that LLM_MODEL is set to a valid model for your provider")
    
    # Run diagnosis
    diagnose_configuration()

Production Configuration
------------------------

Security Best Practices
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Use environment variables, never hardcode API keys
    export OPENAI_API_KEY="sk-..."
    
    # Use separate API keys for different environments
    export OPENAI_API_KEY_DEV="sk-dev-..."
    export OPENAI_API_KEY_PROD="sk-prod-..."
    
    # Restrict API key permissions when possible
    # (e.g., read-only keys for analysis-only applications)

.. code-block:: python

    # Load API keys from secure sources
    import os
    from pathlib import Path
    
    def load_secure_config():
        """Load configuration from secure sources."""
        
        # Option 1: Load from secure file
        config_file = Path.home() / ".cais" / "config"
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
        
        # Option 2: Load from cloud secret manager
        # (implementation depends on your cloud provider)
        
        # Option 3: Load from encrypted configuration
        # (implementation depends on your encryption method)

Monitoring and Logging
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import logging
    import os
    from datetime import datetime
    
    def setup_production_logging():
        """Set up comprehensive logging for production."""
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_file = log_dir / f"cais_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        # Set specific log levels
        logging.getLogger("causal_agent").setLevel(logging.INFO)
        logging.getLogger("langchain").setLevel(logging.WARNING)  # Reduce noise
        
        return logging.getLogger(__name__)
    
    # Usage
    logger = setup_production_logging()
    logger.info("Starting causal analysis application")

Cost Optimization
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from typing import Dict, Any
    
    def cost_optimized_config():
        """Configure CAIS for cost optimization."""
        
        # Use most cost-effective models
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-4o-mini"  # Most cost-effective OpenAI model
        
        # Reduce temperature for more deterministic (and cheaper) responses
        os.environ["LLM_TEMPERATURE"] = "0"
        
        # Enable caching to avoid repeated API calls
        os.environ["CAIS_ENABLE_CACHING"] = "true"
        
        # Set reasonable token limits
        os.environ["LLM_MAX_TOKENS"] = "2000"
    
    def performance_optimized_config():
        """Configure CAIS for maximum performance."""
        
        # Use highest-performance models
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["LLM_MODEL"] = "claude-3-5-sonnet-latest"
        
        # Allow longer responses for complex analyses
        os.environ["LLM_MAX_TOKENS"] = "4000"
        
        # Enable parallel processing
        os.environ["CAIS_MAX_WORKERS"] = "8"

Configuration Templates
-----------------------

Development Template
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # .env.development
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini
    OPENAI_API_KEY=your-dev-api-key
    
    # Development settings
    CAIS_VERBOSE_LOGGING=true
    CAIS_ENABLE_CACHING=true
    CAIS_CACHE_DIR=./dev_cache
    CAIS_MAX_RETRIES=1  # Fail fast in development

Production Template
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # .env.production
    LLM_PROVIDER=anthropic
    LLM_MODEL=claude-3-5-sonnet-latest
    ANTHROPIC_API_KEY=your-prod-api-key
    
    # Production settings
    CAIS_VERBOSE_LOGGING=false
    CAIS_ENABLE_CACHING=true
    CAIS_CACHE_DIR=/var/cache/cais
    CAIS_MAX_RETRIES=3
    CAIS_MAX_WORKERS=4

Testing Template
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # .env.testing
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini
    OPENAI_API_KEY=your-test-api-key
    
    # Testing settings
    CAIS_VERBOSE_LOGGING=true
    CAIS_ENABLE_CACHING=false  # Always run fresh for tests
    CAIS_MAX_RETRIES=1
    CAIS_STEP_TIMEOUT=30  # Shorter timeouts for tests

Next Steps
----------

- For basic usage patterns, see :doc:`basic_usage`
- For advanced customization, see :doc:`advanced_usage`
- For batch processing setup, see :doc:`batch_processing`
- For deployment considerations, see :doc:`../deployment/index`
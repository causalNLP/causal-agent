Installation Guide
==================

This guide will help you install causal-agent in your preferred environment. Choose the installation method that best fits your needs.

.. contents:: Installation Options
   :local:
   :depth: 2

Prerequisites
-------------

Before installing causal-agent, ensure you have:

* **Python 3.10 or higher** (Python 3.10 is recommended)
* **Git** (for development installation)
* An **LLM API key** (OpenAI, Anthropic, Google, etc.)

Quick Start (Recommended)
-------------------------

For most users, we recommend using pip with a virtual environment:

.. code-block:: bash

   # Create and activate virtual environment
   python -m venv causal_agent_env
   source causal_agent_env/bin/activate  # On Windows: causal_agent_env\Scripts\activate
   
   # Install causal-agent
   pip install causal-agent
   
   # Verify installation
   python -c "import causal_agent; print('causal-aget=nt installed successfully!')"
   
   # Test basic functionality
   python -c "from causal_agent import run_causal_analysis; print('API imported successfully!')"

Installation Methods
--------------------

Method 1: pip (PyPI)
~~~~~~~~~~~~~~~~~~~~

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install causal-agent

For development features (latest from GitHub):

.. code-block:: bash

   pip install git+https://github.com/causalNLP/causal-agent.git

Method 2: Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a dedicated conda environment with Python 3.10:

.. code-block:: bash

   # Create conda environment
   conda create -n causal_agent python=3.10
   conda activate causal_agent
   
   # Install causal-agent
   pip install causal-agent
   
   # Or install from requirements file
   git clone https://github.com/causalNLP/causal-agent.git
   cd causal-agent
   pip install -r requirements.txt
   pip install -e .




Method 4: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/causalNLP/causal-agent.git
   cd causal-agent
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements.txt
   
   # Run tests to verify installation
   pytest tests/

Configuration Setup
-------------------

After installation, you need to configure your LLM provider:

1. **Copy the example configuration:**

   .. code-block:: bash

      cp .env.example .env

2. **Edit the .env file with your API keys:**

   .. code-block:: bash

      # OpenAI Configuration
      OPENAI_API_KEY=your_openai_api_key_here
      
      # Anthropic Configuration (optional)
      ANTHROPIC_API_KEY=your_anthropic_api_key_here
      
      # Google Configuration (optional)
      GOOGLE_API_KEY=your_google_api_key_here

3. **Verify configuration:**

   .. code-block:: python

      from causal_agent import run_causal_analysis
      
      # Test with a simple example
      result = run_causal_analysis(
          query="Test query",
          dataset_path="path/to/test/data.csv",
          dataset_description="Test dataset"
      )

Environment-Specific Instructions
---------------------------------

Google Colab
~~~~~~~~~~~~

Install causal-agent in Google Colab:

.. code-block:: bash

   !pip install causal-agent
   
   # Set up API key
   import os
   from google.colab import userdata
   os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
   
   # Import and use
   from causal_agent import run_causal_analysis

Jupyter Notebook
~~~~~~~~~~~~~~~~

Install causal-agent in your Jupyter environment:

.. code-block:: bash

   # Install in your current environment
   pip install causal-agent
   
   # Or create a new kernel
   python -m ipykernel install --user --name causal_agent --display-name "Causal Agent Environment"





Verification
------------

Verify your installation with these tests:

**Basic Import Test:**

.. code-block:: python

   import causal_agent
   print(f"Causal Agent version: {causal_agent.__version__}")

**CLI Test:**

.. code-block:: bash

   causal-agent --help

**API Test:**

.. code-block:: python

   from causal_agent import run_causal_analysis
   
   # This should not raise any import errors
   print("Causal Agent API imported successfully!")

**Complete Example:**

.. code-block:: python

   from causal_agent import run_causal_analysis
   import os
   
   # Set API key
   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
   
   # Run a simple analysis
   result = run_causal_analysis(
       query="What is the effect of education on income?",
       dataset_path="your_data.csv",
       dataset_description="Dataset containing education and income data"
   )
   
   # Print results
   print(f"Method used: {result['results']['method_used']}")
   print(f"Effect estimate: {result['results']['effect_estimate']}")
   print(f"Interpretation: {result['results']['interpretation']}")

Next Steps
----------

After successful installation:

1. **Complete the quickstart tutorial:** :doc:`quickstart`
2. **Try your first analysis:** :doc:`first_analysis`
3. **Explore the user guide:** :doc:`../user_guide/index`
4. **Check out tutorials:** :doc:`../tutorials/index`


Getting Additional Help
~~~~~~~~~~~~~~~~~~~~~~~

If you're still experiencing issues:

1. **Check the FAQ:** Visit our `GitHub Wiki <https://github.com/causalNLP/causal-agent/wiki>`_
2. **Search existing issues:** `GitHub Issues <https://github.com/causalNLP/causal-agent/issues>`_
3. **Report a bug:** Create a new issue with:
   - Your operating system and Python version
   - Complete error message
   - Minimal code example that reproduces the issue
   - Your dataset structure (without sensitive data)

**When reporting issues, include:**

.. code-block:: python

   # System information
   import sys
   import causal_agent
   print(f"Python version: {sys.version}")
   print(f"Causal Agent version: {causal_agent.__version__}")
   print(f"Operating system: {sys.platform}")
   
   # Error details
   # Include the full error traceback
   # Provide a minimal example that reproduces the issue
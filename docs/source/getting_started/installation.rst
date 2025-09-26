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
   python -m ipykernel install --user --name causal_agent --display-name "CAIS Environment"





Verification
------------

Verify your installation with these tests:

**Basic Import Test:**

.. code-block:: python

   import causal_agent
   print(f"CAIS version: {causal_agent.__version__}")

**CLI Test:**

.. code-block:: bash

   causal-agent --help

**API Test:**

.. code-block:: python

   from causal_agent import run_causal_analysis
   
   # This should not raise any import errors
   print("CAIS API imported successfully!")

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

Need help? See our troubleshooting section below or visit our `GitHub Issues <https://github.com/causalNLP/causal-agent/issues>`_.

.. _troubleshooting-section:

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Python Version Issues**

.. code-block:: bash

   # Error: "Python 3.10 or higher is required"
   # Solution: Check your Python version
   python --version
   
   # Install Python 3.10 if needed
   # On macOS with Homebrew:
   brew install python@3.10
   
   # On Ubuntu/Debian:
   sudo apt install python3.10

**Package Installation Failures**

.. code-block:: bash

   # Error: "Failed building wheel for [package]"
   # Solution: Update pip and setuptools
   pip install --upgrade pip setuptools wheel
   
   # Then retry installation
   pip install causal-agent

**Virtual Environment Issues**

.. code-block:: bash

   # Error: "No module named 'causal_agent'"
   # Solution: Ensure virtual environment is activated
   source your_env/bin/activate  # Linux/macOS
   your_env\Scripts\activate     # Windows
   
   # Verify installation in the correct environment
   which python
   pip list | grep causal-agent

**Permission Errors**

.. code-block:: bash

   # Error: "Permission denied" during installation
   # Solution: Use --user flag or virtual environment
   pip install --user causal-agent
   
   # Or create a virtual environment (recommended)
   python -m venv causal_env
   source causal_env/bin/activate
   pip install causal-agent

API Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**API Key Not Found**

.. code-block:: python

   # Error: "OpenAI API key not found"
   # Solution: Set environment variable
   import os
   os.environ['OPENAI_API_KEY'] = 'your-key-here'
   
   # Or create .env file:
   # OPENAI_API_KEY=your-key-here

**API Key Invalid**

.. code-block:: python

   # Error: "Invalid API key"
   # Solution: Verify your API key
   import openai
   openai.api_key = 'your-key-here'
   
   # Test the key
   try:
       openai.models.list()
       print("API key is valid")
   except Exception as e:
       print(f"API key issue: {e}")

**Rate Limiting**

.. code-block:: python

   # Error: "Rate limit exceeded"
   # Solution: Add delays between requests
   import time
   
   # For batch processing, add delays
   for query in queries:
       result = run_causal_analysis(query, data, description)
       time.sleep(1)  # Wait 1 second between requests

Import and Runtime Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Module Import Errors**

.. code-block:: python

   # Error: "No module named 'causal_agent'"
   # Solution: Check installation
   import sys
   print(sys.path)
   
   # Reinstall if necessary
   pip uninstall causal-agent
   pip install causal-agent

**Dependency Conflicts**

.. code-block:: bash

   # Error: "Conflicting dependencies"
   # Solution: Create fresh environment
   conda create -n causal_agent_clean python=3.10
   conda activate causal_agent_clean
   pip install causal-agent

**Memory Issues**

.. code-block:: python

   # Error: "Out of memory" with large datasets
   # Solution: Process data in chunks
   import pandas as pd
   
   # For large datasets, sample first
   data = pd.read_csv('large_dataset.csv')
   sample_data = data.sample(n=10000)  # Use 10k rows
   sample_data.to_csv('sample_data.csv', index=False)

Data Format Issues
~~~~~~~~~~~~~~~~~~

**CSV Reading Errors**

.. code-block:: python

   # Error: "Unable to read CSV file"
   # Solution: Check file format and encoding
   import pandas as pd
   
   # Try different encodings
   try:
       data = pd.read_csv('data.csv', encoding='utf-8')
   except UnicodeDecodeError:
       data = pd.read_csv('data.csv', encoding='latin-1')

**Missing Data Issues**

.. code-block:: python

   # Error: "Too much missing data"
   # Solution: Clean data before analysis
   data = pd.read_csv('data.csv')
   
   # Check missing data
   print(data.isnull().sum())
   
   # Remove columns with >50% missing
   threshold = len(data) * 0.5
   data = data.dropna(thresh=threshold, axis=1)

**Data Type Issues**

.. code-block:: python

   # Error: "Invalid data types"
   # Solution: Convert data types
   data['numeric_column'] = pd.to_numeric(data['numeric_column'], errors='coerce')
   data['date_column'] = pd.to_datetime(data['date_column'], errors='coerce')

Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows Issues**

.. code-block:: batch

   REM Error: "Microsoft Visual C++ 14.0 is required"
   REM Solution: Install Visual Studio Build Tools
   REM Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
   REM Or use conda for problematic packages
   conda install -c conda-forge causal-agent

**macOS Issues**

.. code-block:: bash

   # Error: "Command line tools not found"
   # Solution: Install Xcode command line tools
   xcode-select --install
   
   # Error: "Architecture mismatch" (M1/M2 Macs)
   # Solution: Use conda with proper architecture
   conda create -n causal_agent python=3.10
   conda activate causal_agent
   pip install causal-agent

**Linux Issues**

.. code-block:: bash

   # Error: "Package dependencies not found"
   # Solution: Install system dependencies
   sudo apt update
   sudo apt install python3-dev python3-pip build-essential
   
   # For CentOS/RHEL:
   sudo yum install python3-devel gcc gcc-c++

Docker Issues
~~~~~~~~~~~~~

**Docker Build Failures**

.. code-block:: bash

   # Error: "Docker build failed"
   # Solution: Check Docker version and resources
   docker --version
   docker system df  # Check disk space
   
   # Clean up if needed
   docker system prune -a

**Container Permission Issues**

.. code-block:: bash

   # Error: "Permission denied in container"
   # Solution: Run with proper user permissions
   docker run -it --rm \
     --user $(id -u):$(id -g) \
     -v $(pwd):/workspace \
     causalagent/cais:latest

Performance Issues
~~~~~~~~~~~~~~~~~~

**Slow Analysis**

.. code-block:: python

   # Issue: Analysis takes too long
   # Solution: Optimize dataset size and complexity
   
   # 1. Sample large datasets
   if len(data) > 50000:
       data = data.sample(n=10000)
   
   # 2. Remove unnecessary columns
   important_cols = ['treatment', 'outcome'] + control_variables
   data = data[important_cols]
   
   # 3. Use faster LLM models
   os.environ['LLM_MODEL'] = 'gpt-3.5-turbo'  # Faster than gpt-4

**High API Costs**

.. code-block:: python

   # Issue: High API usage costs
   # Solutions:
   
   # 1. Use cheaper models
   os.environ['LLM_MODEL'] = 'gpt-3.5-turbo'
   
   # 2. Reduce dataset description length
   short_description = "Brief dataset description with key variables only"
   
   # 3. Cache results for repeated analyses
   import pickle
   
   # Save results
   with open('analysis_results.pkl', 'wb') as f:
       pickle.dump(result, f)

Getting Additional Help
~~~~~~~~~~~~~~~~~~~~~~~

If you're still experiencing issues:

1. **Check the FAQ:** Visit our `GitHub Wiki <https://github.com/causalNLP/causal-agent/wiki>`_
2. **Search existing issues:** `GitHub Issues <https://github.com/causalNLP/causal-agent/issues>`_
3. **Ask the community:** `GitHub Discussions <https://github.com/causalNLP/causal-agent/discussions>`_
4. **Report a bug:** Create a new issue with:
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
   print(f"CAIS version: {causal_agent.__version__}")
   print(f"Operating system: {sys.platform}")
   
   # Error details
   # Include the full error traceback
   # Provide a minimal example that reproduces the issue
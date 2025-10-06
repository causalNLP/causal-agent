Causal Agent Documentation
========================================

Welcome to the Causal-Agent is an intelligent system that helps researchers and practitioners conduct rigorous causal inference analysis by automatically selecting appropriate causal inference methods, validating assumptions, and providing interpretable results.

.. note::
   Causal Agent is designed to democratize causal inference by making advanced statistical methods accessible to users with varying levels of statistical expertise.

What is causal-agent?
-------------

The Causal Agent is an AI-powered tool that:

* **Automatically selects** the most appropriate causal inference method based on your data and research question
* **Validates assumptions** required for causal identification and provides diagnostic feedback
* **Generates interpretable results** with clear explanations of causal effects and their limitations
* **Supports multiple methods** including RCTs, Difference-in-Differences, Instrumental Variables, Regression Discontinuity, and Propensity Score methods
* **Provides theoretical guidance** to help users understand the causal inference process

Key Features
------------

🔍 **Intelligent Method Selection**
   Uses decision tree logic to recommend the best causal inference approach for your specific research context.

📊 **Comprehensive Method Support**
   Supports experimental, quasi-experimental, and observational study designs with automated diagnostics.

🤖 **LLM-Powered Analysis**
   Leverages large language models to provide contextual explanations and interpret results in plain language.

📈 **Robust Validation**
   Automatically checks assumptions and provides diagnostic tests to ensure the validity of causal conclusions.

🎯 **Domain Flexibility**
   Works across various domains including economics, healthcare, education, and social sciences.

Quick Start
-----------

Get started with Causal Agent in just a few steps:

1. **Install Causal Agent**: ``pip install causal-agent``
2. **Load your data**: Import your dataset in CSV or pandas DataFrame format
3. **Run analysis**: Let Causal Agent automatically select and execute the appropriate causal inference method
4. **Interpret results**: Review the generated causal effect estimates and explanations

.. code-block:: python

   from causal_agent import CausalAgent
   
   # Initialize the agent
   agent = CausalAgent()
   
   # Run causal analysis
   result = agent.analyze(
       data="your_dataset.csv",
       treatment="treatment_variable",
       outcome="outcome_variable"
   )
   
   # View results
   print(result.summary())

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/index

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials & Examples

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Causal Inference Methods

   methods/index

.. toctree::
   :maxdepth: 1
   :caption: Theoretical Background

   theory/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/index

.. toctree::
   :maxdepth: 1
   :caption: About

   about/index

Community & Support
-------------------

* **GitHub Repository**: `causal-ai-scientist <https://github.com/your-org/causal-ai-scientist>`_
* **Issue Tracker**: Report bugs and request features
* **Discussions**: Join the community discussions
* **Contributing**: See our :doc:`development/contributing` guide

Citation
--------

If you use Causal Agent in your research, please cite:

.. code-block:: bibtex

   @software{causal_ai_scientist,
     title={Causal AI Scientist: Automated Causal Inference Analysis},
     author={Causal Agent Team},
     year={2024},
     url={https://github.com/your-org/causal-ai-scientist}
   }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
Interactive Notebooks
====================

Learn CAIS through interactive Jupyter notebooks that you can run locally or in cloud environments. These notebooks provide step-by-step tutorials with executable code and detailed explanations.

.. toctree::
   :maxdepth: 2

   education_analysis_tutorial
   healthcare_analysis_tutorial
   economics_analysis_tutorial

Notebook Categories
-------------------

**Beginner Notebooks**
   * Getting Started with CAIS
   * Your First Causal Analysis
   * Understanding CAIS Output
   * Common Data Preparation Steps

**Intermediate Notebooks**
   * Comparing Different Methods
   * Assumption Testing and Validation
   * Handling Missing Data
   * Sensitivity Analysis

**Advanced Notebooks**
   * Custom Method Implementation
   * Large-Scale Batch Processing
   * Integration with ML Pipelines
   * Performance Optimization

Running the Notebooks
---------------------

**Quick Start - Cloud Environments**
   * **Binder**: |binder-badge| Launch all notebooks in an interactive environment
   * **Google Colab**: Click the "Open in Colab" button in each notebook
   * **GitHub Codespaces**: Open the repository in a cloud development environment

**Local Installation**
   1. Install CAIS: ``pip install causal-agent``
   2. Install Jupyter: ``pip install jupyter notebook``
   3. Download notebooks: :download:`All Notebooks (ZIP) <../_static/cais_tutorial_notebooks.zip>`
   4. Run: ``jupyter notebook``

**Individual Downloads**
   * :download:`Education Analysis Tutorial <../_static/education_analysis_tutorial_tutorial.zip>`
   * :download:`Healthcare Analysis Tutorial <../_static/healthcare_analysis_tutorial_tutorial.zip>`
   * :download:`Economics Analysis Tutorial <../_static/economics_analysis_tutorial_tutorial.zip>`

**Requirements**
   * Python 3.8+
   * CAIS package
   * Standard data science libraries (pandas, numpy, matplotlib, seaborn, scipy)
   * LLM provider API key (for full functionality)

.. |binder-badge| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/causal-ai-scientist/causal-ai-scientist/main?filepath=docs/source/tutorials/notebooks/
   :alt: Launch Binder

Notebook Features
-----------------

* **Interactive Code**: Modify and run code cells to experiment
* **Rich Visualizations**: Charts, plots, and diagnostic graphics
* **Downloadable Data**: Practice datasets included with each notebook
* **Solution Cells**: Compare your results with expected outcomes
* **Extension Exercises**: Additional challenges to deepen learning
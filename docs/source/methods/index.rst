Causal Inference Methods
========================

CAIS supports a comprehensive range of causal inference methods, from experimental designs to observational studies. This section provides detailed documentation for each method, including when to use them, their assumptions, and implementation details.

.. toctree::
   :maxdepth: 2
   :caption: Method Documentation

   overview
   decision_tree
   experimental/index
   quasi_experimental/index
   observational/index

Method Selection Guide
----------------------

Not sure which method to use? CAIS can automatically select the most appropriate method based on your data and research design. However, understanding the different approaches will help you make informed decisions and interpret results correctly.

**Start Here**
   * :doc:`overview` - Introduction to causal inference and CAIS methods
   * :doc:`decision_tree` - Interactive guide to method selection

**Method Categories**

**Experimental Methods** (:doc:`experimental/index`)
   Gold standard for causal inference when randomization is possible.
   
   * Randomized Controlled Trials (RCT)
   * A/B Testing
   * Field Experiments

**Quasi-Experimental Methods** (:doc:`quasi_experimental/index`)
   Leverage natural experiments and policy changes for causal identification.
   
   * Difference-in-Differences (DiD)
   * Instrumental Variables (IV)
   * Regression Discontinuity (RDD)

**Observational Methods** (:doc:`observational/index`)
   Extract causal insights from observational data with careful identification strategies.
   
   * Propensity Score Matching
   * Propensity Score Weighting
   * Backdoor Adjustment
   * Linear Regression (with controls)

Method Comparison
-----------------

.. list-table:: Method Comparison Overview
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Data Requirements
     - Key Assumptions
     - Strength of Evidence
     - Common Use Cases
   * - RCT
     - Randomized treatment
     - No spillovers
     - Highest
     - Medical trials, A/B tests
   * - DiD
     - Panel data, treatment timing
     - Parallel trends
     - High
     - Policy evaluation
   * - IV
     - Valid instrument
     - Exclusion restriction
     - High
     - Natural experiments
   * - RDD
     - Continuous assignment variable
     - Continuity at cutoff
     - High
     - Threshold-based policies
   * - Propensity Score
     - Rich covariates
     - Unconfoundedness
     - Medium
     - Observational studies

Choosing the Right Method
-------------------------

The choice of causal inference method depends on several factors:

1. **Research Design**: Experimental vs. observational data
2. **Data Structure**: Cross-sectional, panel, or time series
3. **Treatment Assignment**: Random, rule-based, or endogenous
4. **Available Variables**: Instruments, covariates, time dimensions
5. **Assumptions**: Which identifying assumptions are plausible

CAIS automatically evaluates these factors and recommends the most appropriate method, but understanding the trade-offs helps you make informed decisions about your analysis strategy.
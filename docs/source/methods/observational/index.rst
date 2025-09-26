Observational Methods
====================

Observational methods extract causal insights from non-experimental data by controlling for confounding variables and making identifying assumptions about selection processes.

.. toctree::
   :maxdepth: 2

   propensity_score_matching
   propensity_score_weighting
   backdoor_adjustment
   linear_regression

Overview
--------

Observational methods are used when experimental or quasi-experimental designs are not available. These methods rely on the assumption that all confounding variables are observed and controlled for, making causal identification possible through statistical adjustment.

**Key Advantages:**
* Can be applied to any observational dataset
* Utilize existing data sources
* Cost-effective compared to experiments
* Allow for large sample sizes

**Key Limitations:**
* Strong unconfoundedness assumption required
* Vulnerable to omitted variable bias
* Selection on unobservables cannot be ruled out
* Require rich covariate data

Method Details
--------------

**Propensity Score Matching**
   Matches treated and control units with similar propensity scores.
   
   * **When to use**: Rich covariate data, clear treatment definition
   * **Key assumption**: Unconfoundedness given observed covariates
   * **Strengths**: Intuitive matching logic, balances covariates
   * **Limitations**: Curse of dimensionality, common support issues

**Propensity Score Weighting**
   Reweights observations to balance treatment and control groups.
   
   * **When to use**: When matching is not feasible or desirable
   * **Key assumption**: Unconfoundedness and positivity
   * **Strengths**: Uses all observations, flexible weighting schemes
   * **Limitations**: Sensitive to extreme weights, model dependence

**Backdoor Adjustment**
   Controls for confounders identified through causal graphs.
   
   * **When to use**: When causal graph is well-understood
   * **Key assumption**: Backdoor criterion satisfied
   * **Strengths**: Principled confounder selection
   * **Limitations**: Requires causal graph knowledge

**Linear Regression**
   Controls for confounders through linear regression adjustment.
   
   * **When to use**: Linear relationships, continuous outcomes
   * **Key assumption**: Correct functional form, no omitted variables
   * **Strengths**: Simple, interpretable, widely understood
   * **Limitations**: Strong functional form assumptions

Implementation in CAIS
-----------------------

CAIS automatically selects and implements appropriate observational methods:

.. code-block:: python

   from causal_agent import CausalAgent
   
   # CAIS selects best observational method
   agent = CausalAgent()
   result = agent.analyze(
       data=observational_data,
       treatment='treatment_variable',
       outcome='outcome_variable',
       covariates=['covar1', 'covar2', 'covar3']
   )

**Automatic Method Selection**
   CAIS chooses methods based on:
   * Data characteristics (sample size, covariate richness)
   * Treatment assignment patterns
   * Outcome variable type
   * User preferences and constraints

**Covariate Selection**
   * Automatic confounder detection
   * Causal graph-based selection when available
   * Statistical significance-based inclusion
   * Domain knowledge integration

Assumption Validation
---------------------

**Unconfoundedness**
   * Cannot be directly tested
   * Sensitivity analyses for unobserved confounding
   * Placebo tests using pre-treatment outcomes
   * Comparison with experimental benchmarks when available

**Positivity/Common Support**
   * Propensity score distribution overlap
   * Trimming observations outside common support
   * Diagnostic plots and statistics
   * Sensitivity to support restrictions

**Correct Specification**
   * Model specification tests
   * Functional form diagnostics
   * Residual analysis
   * Cross-validation approaches

Balance Assessment
------------------

**Covariate Balance**
   * Standardized mean differences
   * Variance ratios
   * Kolmogorov-Smirnov tests
   * Graphical balance assessment

**Propensity Score Balance**
   * Propensity score distribution comparison
   * Stratification balance tests
   * Matching quality diagnostics
   * Weighting effectiveness measures

Best Practices
--------------

**Study Design**
   * Collect rich covariate data
   * Include pre-treatment outcomes when possible
   * Consider multiple comparison groups
   * Document data collection process

**Analysis**
   * Check balance before and after adjustment
   * Conduct sensitivity analyses
   * Use multiple methods for robustness
   * Report all diagnostic results

**Interpretation**
   * Acknowledge unconfoundedness assumption
   * Discuss potential sources of bias
   * Consider external validity
   * Report confidence intervals and uncertainty

Sensitivity Analysis
--------------------

**Unobserved Confounding**
   * Rosenbaum bounds for matched samples
   * Imbens sensitivity analysis
   * Simulation-based approaches
   * Benchmarking against known confounders

**Model Specification**
   * Alternative functional forms
   * Different covariate sets
   * Various matching/weighting schemes
   * Robustness to outliers

**Sample Restrictions**
   * Different common support definitions
   * Trimming strategies
   * Subgroup analyses
   * Temporal stability

Common Challenges
-----------------

**Data Quality Issues**
   * Missing covariate data
   * Measurement error in variables
   * Inconsistent variable definitions
   * Sample selection issues

**Methodological Challenges**
   * Curse of dimensionality
   * Extreme propensity scores
   * Poor covariate balance
   * Model dependence

**Interpretation Issues**
   * Distinguishing correlation from causation
   * Communicating uncertainty
   * Addressing skepticism about assumptions
   * Policy relevance of estimates

Advanced Topics
---------------

**Machine Learning Methods**
   * Targeted maximum likelihood estimation (TMLE)
   * Double machine learning
   * Causal forests
   * Neural network-based methods

**Multiple Treatments**
   * Generalized propensity scores
   * Multiple treatment matching
   * Dose-response relationships
   * Treatment interaction effects

**Time-Varying Treatments**
   * Marginal structural models
   * G-computation
   * Inverse probability weighting over time
   * Sequential ignorability
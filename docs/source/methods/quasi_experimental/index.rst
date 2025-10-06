Quasi-Experimental Methods
==========================

Quasi-experimental methods leverage natural experiments, policy changes, and other sources of exogenous variation to identify causal effects when randomization is not possible.

.. toctree::
   :maxdepth: 2

   difference_in_differences
   instrumental_variables
   regression_discontinuity
   synthetic_control

Overview
--------

Quasi-experimental methods bridge the gap between experimental and observational studies by exploiting natural sources of variation that approximate random assignment. These methods are particularly valuable for policy evaluation and situations where controlled experiments are not feasible.

**Key Advantages:**
* Can be applied to observational data
* Leverage natural experiments
* Strong causal identification when assumptions hold
* Useful for policy evaluation

**Key Limitations:**
* Rely on stronger identifying assumptions
* Assumptions often untestable
* May have limited external validity
* Require specific data structures

Method Details
--------------

**Difference-in-Differences (DiD)**
   Compares changes over time between treatment and control groups.
   
   * **When to use**: Panel data with treatment timing variation
   * **Key assumption**: Parallel trends in absence of treatment
   * **Strengths**: Controls for time-invariant confounders
   * **Limitations**: Requires parallel trends assumption

**Instrumental Variables (IV)**
   Uses an instrument that affects treatment but not outcome directly.
   
   * **When to use**: When you have a valid instrument
   * **Key assumption**: Exclusion restriction and relevance
   * **Strengths**: Handles endogenous treatment assignment
   * **Limitations**: Requires strong, often untestable assumptions

**Regression Discontinuity (RDD)**
   Exploits arbitrary cutoffs in treatment assignment rules.
   
   * **When to use**: Treatment assigned based on continuous variable cutoff
   * **Key assumption**: Continuity of potential outcomes at cutoff
   * **Strengths**: Transparent identification strategy
   * **Limitations**: Local treatment effects only

**Synthetic Control**
   Creates synthetic control units using weighted combinations of control units.
   
   * **When to use**: Few treated units, many potential controls
   * **Key assumption**: No unobserved confounders affecting outcome trends
   * **Strengths**: Transparent counterfactual construction
   * **Limitations**: Requires good pre-treatment fit

Implementation in Causal Agent
-----------------------

Causal Agent automatically detects quasi-experimental designs and applies appropriate methods:

.. code-block:: python

   from causal_agent import CausalAgent
   
   # Causal Agent detects DiD design from panel structure
   agent = CausalAgent()
   result = agent.analyze(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_measure',
       time_var='year',
       unit_var='state'
   )

**Automatic Method Selection**
   Causal Agent identifies quasi-experimental opportunities by analyzing:
   * Data structure (panel, cross-sectional, time series)
   * Treatment assignment patterns
   * Available instrumental variables
   * Discontinuities in assignment rules

**Diagnostic Testing**
   * Parallel trends testing for DiD
   * Instrument strength and validity tests for IV
   * Continuity tests for RDD
   * Pre-treatment fit assessment for synthetic control

Assumption Validation
---------------------

**Difference-in-Differences**
   * **Parallel Trends**: Test pre-treatment trends
   * **No Anticipation**: Check for pre-treatment effects
   * **Stable Composition**: Ensure consistent units over time

**Instrumental Variables**
   * **Relevance**: First-stage F-statistic > 10
   * **Exclusion Restriction**: Theoretical justification required
   * **Monotonicity**: No defiers assumption

**Regression Discontinuity**
   * **Continuity**: Test for jumps in covariates at cutoff
   * **No Manipulation**: McCrary density test
   * **Bandwidth Selection**: Optimal bandwidth procedures

**Synthetic Control**
   * **Pre-treatment Fit**: RMSPE in pre-treatment period
   * **Placebo Tests**: Test on non-treated units
   * **Robustness**: Vary donor pool and predictors

Best Practices
--------------

**Method Selection**
   * Choose method based on data structure and identification strategy
   * Consider multiple methods when possible for robustness
   * Understand the specific assumptions of each method
   * Validate assumptions as thoroughly as possible

**Implementation**
   * Use appropriate standard errors (clustered, robust)
   * Conduct sensitivity analyses
   * Report diagnostic test results
   * Consider effect heterogeneity

**Interpretation**
   * Understand the specific estimand (LATE, ATT, etc.)
   * Consider external validity carefully
   * Report limitations and assumptions clearly
   * Discuss policy implications appropriately

Common Pitfalls
---------------

**Difference-in-Differences**
   * Assuming parallel trends without testing
   * Ignoring treatment effect heterogeneity over time
   * Using inappropriate standard errors
   * Misinterpreting dynamic treatment effects

**Instrumental Variables**
   * Weak instruments leading to bias
   * Violating exclusion restriction
   * Misinterpreting LATE as ATE
   * Inadequate first-stage diagnostics

**Regression Discontinuity**
   * Choosing bandwidth inappropriately
   * Ignoring manipulation of running variable
   * Extrapolating beyond local effects
   * Misspecifying functional form

**Synthetic Control**
   * Poor pre-treatment fit
   * Inadequate placebo testing
   * Overfitting with too many predictors
   * Inappropriate donor pool selection
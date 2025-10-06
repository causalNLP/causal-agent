Instrumental Variables (IV)
===========================

Instrumental Variables (IV) is a powerful method for estimating causal effects when treatment assignment is endogenous. IV uses a third variable (the instrument) that affects treatment but not the outcome directly, allowing identification of causal effects even with unmeasured confounding.

When to Use IV
---------------

**Ideal Conditions:**
- Treatment assignment is endogenous (correlated with unobserved factors)
- A valid instrument is available that affects treatment but not outcome directly
- You want to estimate causal effects despite unmeasured confounding
- The instrument has sufficient strength (relevance)

**Common Applications:**
- Returns to education (using compulsory schooling laws as instruments)
- Labor supply effects (using tax policy changes as instruments)
- Program evaluation with non-random participation
- Medical treatments with physician preference as instrument
- Geographic variation in policy implementation

**Not Suitable When:**
- No valid instrument is available
- Instrument is weak (F-statistic < 10)
- Exclusion restriction is implausible
- Treatment is already exogenous (randomized)

Theoretical Background
----------------------

The IV Framework
~~~~~~~~~~~~~~~~

IV addresses the problem of endogenous treatment assignment. Consider the structural equation:

.. math::

   Y_i = \\alpha + \\tau D_i + \\epsilon_i

Where :math:`D_i` (treatment) is correlated with :math:`\\epsilon_i` (unobserved factors), making OLS biased.

**The IV Solution:**
Use an instrument :math:`Z_i` that satisfies:
1. **Relevance**: :math:`Cov(Z_i, D_i) \\neq 0`
2. **Exclusion Restriction**: :math:`Cov(Z_i, \\epsilon_i) = 0`

**Two-Stage Least Squares (2SLS):**

*First Stage:*

.. math::

   D_i = \\pi_0 + \\pi_1 Z_i + \\nu_i

*Second Stage:*

.. math::

   Y_i = \\alpha + \\tau \\hat{D_i} + \\epsilon_i

Where :math:`\\hat{D_i}` is the predicted treatment from the first stage.

**Reduced Form:**

.. math::

   Y_i = \\gamma_0 + \\gamma_1 Z_i + u_i

**Wald Estimator:**

.. math::

   \\hat{\\tau}_{IV} = \\frac{\\gamma_1}{\\pi_1} = \\frac{Cov(Y_i, Z_i)}{Cov(D_i, Z_i)}

Key Assumptions
---------------

1. **Relevance (First-Stage Strength)**
   
   **Definition**: The instrument must be correlated with the endogenous treatment.
   
   **Mathematical**: :math:`Cov(Z_i, D_i) \\neq 0` or :math:`\\pi_1 \\neq 0`
   
   **Testing**: First-stage F-statistic should be > 10 (preferably > 20)
   
   **Why it matters**: Weak instruments lead to biased and imprecise estimates.

2. **Exclusion Restriction (Exogeneity)**
   
   **Definition**: The instrument affects the outcome only through its effect on treatment.
   
   **Mathematical**: :math:`Cov(Z_i, \\epsilon_i) = 0`
   
   **Testing**: Generally untestable, requires theoretical justification
   
   **Why it matters**: Violations lead to biased causal estimates.

3. **Monotonicity (No Defiers)**
   
   **Definition**: The instrument affects treatment in the same direction for all units.
   
   **Mathematical**: :math:`D_i(Z_i = 1) \\geq D_i(Z_i = 0)$ for all i
   
   **Why it matters**: Ensures LATE interpretation is meaningful.
   
   **Testing**: Examine first-stage heterogeneity across subgroups.

4. **Independence**
   
   **Definition**: The instrument is as-good-as-randomly assigned.
   
   **Mathematical**: :math:`Z_i \\perp (Y_i(0), Y_i(1), D_i(0), D_i(1))$
   
   **Testing**: Check balance of covariates across instrument values.

Types of IV Estimands
---------------------

Local Average Treatment Effect (LATE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: IV estimates the treatment effect for compliers - units whose treatment status is affected by the instrument.

**Mathematical**: :math:`LATE = E[Y_i(1) - Y_i(0) | D_i(1) > D_i(0)]$

**Population Groups:**
- **Compliers**: :math:`D_i(Z_i=1) = 1, D_i(Z_i=0) = 0`
- **Always-takers**: :math:`D_i(Z_i=1) = 1, D_i(Z_i=0) = 1`  
- **Never-takers**: :math:`D_i(Z_i=1) = 0, D_i(Z_i=0) = 0$
- **Defiers**: :math:`D_i(Z_i=1) = 0, D_i(Z_i=0) = 1$ (ruled out by monotonicity)

**Interpretation**: LATE may differ from ATE if treatment effects are heterogeneous.

Implementation in Causal Agent
----------------------

Basic IV Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causal_agent import CausalAgent
   
   # Causal Agent automatically detects IV setup
   agent = CausalAgent()
   result = agent.analyze(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument='compulsory_schooling_law'
   )
   
   print(f"IV Treatment Effect: {result.ate}")
   print(f"95% Confidence Interval: {result.confidence_interval}")
   print(f"First-stage F-statistic: {result.first_stage_f}")

IV with Multiple Instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple instruments are available:

.. code-block:: python

   # Multiple instruments
   result = agent.analyze(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument=['compulsory_schooling', 'distance_to_college', 'tuition_changes']
   )

IV with Covariates
~~~~~~~~~~~~~~~~~~

Including exogenous covariates can improve precision:

.. code-block:: python

   # IV with controls
   result = agent.analyze(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument='compulsory_schooling_law',
       covariates=['age', 'gender', 'race', 'region']
   )

Diagnostic Tests and Validation
-------------------------------

First-Stage Strength
~~~~~~~~~~~~~~~~~~~~

Test whether instruments are sufficiently strong:

.. code-block:: python

   # First-stage diagnostics
   first_stage = agent.first_stage_diagnostics(
       data=iv_data,
       treatment='education_years',
       instrument='compulsory_schooling_law',
       covariates=['age', 'gender']
   )
   
   print(f"First-stage F-statistic: {first_stage.f_stat}")
   print(f"Partial R-squared: {first_stage.partial_r2}")

**Benchmarks:**
- F-statistic > 10 (minimum threshold)
- F-statistic > 20 (preferred threshold)
- Effective F-statistic for multiple instruments

Overidentification Tests
~~~~~~~~~~~~~~~~~~~~~~~~

When you have more instruments than endogenous variables:

.. code-block:: python

   # Test overidentifying restrictions
   overid_test = agent.overidentification_test(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument=['compulsory_schooling', 'distance_to_college']
   )
   
   print(f"Hansen J-statistic: {overid_test.j_stat}")
   print(f"P-value: {overid_test.p_value}")

**Interpretation**: 
- Null hypothesis: All instruments are valid
- Rejection suggests at least one instrument violates exclusion restriction
- Cannot identify which instrument is invalid

Endogeneity Tests
~~~~~~~~~~~~~~~~~

Test whether IV is actually needed (Hausman test):

.. code-block:: python

   # Test for endogeneity
   endogeneity_test = agent.endogeneity_test(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument='compulsory_schooling_law'
   )
   
   print(f"Hausman test p-value: {endogeneity_test.p_value}")

**Interpretation**:
- Null hypothesis: Treatment is exogenous (OLS is consistent)
- Rejection suggests endogeneity and IV is needed
- Failure to reject doesn't prove exogeneity

Weak Instrument Robust Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When instruments may be weak:

.. code-block:: python

   # Weak-IV robust confidence intervals
   robust_ci = agent.weak_iv_robust_inference(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument='compulsory_schooling_law'
   )
   
   print(f"Anderson-Rubin CI: {robust_ci.ar_ci}")
   print(f"Conditional LR CI: {robust_ci.clr_ci}")

Common IV Applications
----------------------

Returns to Education
~~~~~~~~~~~~~~~~~~~~

**Research Question**: What is the causal effect of education on wages?

**Endogeneity Problem**: More able individuals get more education and higher wages

**Instruments**:
- Compulsory schooling laws
- Distance to college
- Quarter of birth (Angrist & Krueger)
- College tuition changes

.. code-block:: python

   # Returns to education IV
   result = agent.analyze(
       data=education_data,
       treatment='years_education',
       outcome='log_hourly_wage',
       instrument='compulsory_schooling_age',
       covariates=['age', 'age_squared', 'gender', 'race']
   )

Labor Supply
~~~~~~~~~~~~

**Research Question**: How does unearned income affect labor supply?

**Endogeneity Problem**: Unearned income may be correlated with unobserved preferences

**Instruments**:
- Lottery winnings
- Inheritance
- Spouse's income shocks
- Tax policy changes

Program Evaluation
~~~~~~~~~~~~~~~~~~

**Research Question**: What is the effect of job training programs on earnings?

**Endogeneity Problem**: Participants self-select into programs

**Instruments**:
- Random assignment to program eligibility
- Geographic variation in program availability
- Caseworker assignment
- Waiting list randomization

Best Practices
--------------

Instrument Selection
~~~~~~~~~~~~~~~~~~~~

**Strong Theoretical Foundation**:
- Understand the economic mechanism linking instrument to treatment
- Ensure exclusion restriction is plausible
- Consider potential violations and their implications

**Empirical Validation**:
- Test first-stage strength (F > 10, preferably > 20)
- Examine instrument balance across observable characteristics
- Consider multiple instruments when available

**Transparency**:
- Clearly explain instrument choice and validity arguments
- Discuss potential threats to exclusion restriction
- Report all diagnostic test results

Analysis Approach
~~~~~~~~~~~~~~~~~

**Specification Testing**:
- Always report first-stage results
- Test overidentifying restrictions when possible
- Consider weak-instrument robust inference
- Examine heterogeneity in first-stage effects

**Robustness Checks**:
- Use alternative instruments when available
- Vary the set of control variables
- Test sensitivity to sample restrictions
- Compare IV to OLS estimates

**Interpretation**:
- Remember IV estimates LATE, not ATE
- Discuss external validity carefully
- Consider who the compliers are
- Report both statistical and economic significance

Common Pitfalls and Solutions
-----------------------------

**Pitfall**: Using weak instruments
**Solution**: Test first-stage strength and use weak-IV robust methods

**Pitfall**: Violating exclusion restriction
**Solution**: Provide strong theoretical justification and test when possible

**Pitfall**: Misinterpreting LATE as ATE
**Solution**: Discuss complier population and external validity

**Pitfall**: Ignoring heterogeneous treatment effects
**Solution**: Explore first-stage and reduced-form heterogeneity

**Pitfall**: Over-relying on statistical tests
**Solution**: Combine statistical evidence with economic reasoning

Example: Returns to Education
-----------------------------

**Research Question**: What is the causal return to an additional year of education?

**Data**: Individual-level data with education, wages, and compulsory schooling laws

**Endogeneity**: More able individuals get more education and earn higher wages

**Instrument**: Changes in compulsory schooling laws across states and birth cohorts

**Analysis**:

.. code-block:: python

   # IV analysis of returns to education
   result = agent.analyze(
       data=education_data,
       treatment='years_education',
       outcome='log_hourly_wage',
       instrument='compulsory_schooling_years',
       covariates=['age', 'age_squared', 'gender', 'race', 'state', 'birth_year']
   )
   
   # First-stage diagnostics
   first_stage = agent.first_stage_diagnostics(
       data=education_data,
       treatment='years_education',
       instrument='compulsory_schooling_years'
   )
   
   print(f"IV Estimate: {result.ate:.3f}")
   print(f"OLS Estimate: {result.ols_comparison:.3f}")
   print(f"First-stage F: {first_stage.f_stat:.1f}")

**Results Interpretation**:
- IV estimate: 8.5% return per year of education (95% CI: [6.2%, 10.8%])
- OLS estimate: 6.2% return per year of education
- First-stage F-statistic: 15.3 (instrument is sufficiently strong)
- IV > OLS suggests ability bias in OLS or heterogeneous returns

Advanced IV Methods
-------------------

Limited Information Maximum Likelihood (LIML)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternative to 2SLS that performs better with weak instruments:

.. code-block:: python

   # LIML estimation
   result = agent.analyze(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument='compulsory_schooling_law',
       method='liml'
   )

Continuously Updated GMM (CUE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GMM estimator that's more robust to weak instruments:

.. code-block:: python

   # CUE-GMM estimation
   result = agent.analyze(
       data=iv_data,
       treatment='education_years',
       outcome='log_wages',
       instrument='compulsory_schooling_law',
       method='cue_gmm'
   )

Control Function Approach
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternative to 2SLS for nonlinear models:

.. code-block:: python

   # Control function for binary outcomes
   result = agent.analyze(
       data=iv_data,
       treatment='education_years',
       outcome='employed',  # binary outcome
       instrument='compulsory_schooling_law',
       method='control_function'
   )

Further Reading
---------------

**Foundational Papers**:
- Wright, P.G. (1928). "The Tariff on Animal and Vegetable Oils"
- Angrist, J.D. & Krueger, A.B. (1991). "Does Compulsory School Attendance Affect Schooling and Earnings?"
- Imbens, G.W. & Angrist, J.D. (1994). "Identification and Estimation of Local Average Treatment Effects"

**Modern Developments**:
- Stock, J.H. & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression"
- Andrews, D.W.K., Stock, J.H. & Sun, L. (2019). "Weak Instruments in Instrumental Variables Regression"
- Mogstad, M. & Torgovitsky, A. (2018). "Identification and Extrapolation of Causal Effects with Instrumental Variables"

**Practical Guides**:
- Angrist, J.D. & Pischke, J.S. (2008). "Mostly Harmless Econometrics"
- Murray, M.P. (2006). "Avoiding Invalid Instruments and Coping with Weak Instruments"
Difference-in-Differences (DiD)
===============================

Difference-in-Differences (DiD) is a quasi-experimental method that estimates causal effects by comparing changes in outcomes over time between treatment and control groups. It's particularly powerful for evaluating policy interventions and natural experiments.

When to Use DiD
----------------

**Ideal Conditions:**
- Panel data with repeated observations over time
- Treatment occurs at different times for different units
- Clear before/after treatment periods
- Plausible parallel trends assumption

**Common Applications:**
- Policy evaluation (minimum wage, healthcare reforms)
- Program evaluation (job training, education interventions)
- Natural experiments (policy changes, external shocks)
- Business interventions (store openings, marketing campaigns)

**Not Suitable When:**
- Only cross-sectional data available
- Treatment and control groups have different trends pre-treatment
- Treatment timing is endogenous
- Significant spillover effects between units

Theoretical Background
----------------------

The DiD Logic
~~~~~~~~~~~~~

DiD exploits variation in treatment timing to identify causal effects. The key insight is that by comparing changes over time, we can difference out time-invariant confounders.

**Basic Setup:**
- **Treatment Group**: Units that receive treatment at time T
- **Control Group**: Units that never receive treatment (or receive it later)
- **Pre-Period**: Time before treatment (t < T)
- **Post-Period**: Time after treatment (t ≥ T)

**The DiD Estimator:**

.. math::

   \\hat{\\tau}_{DiD} = (\\bar{Y}_{T,post} - \\bar{Y}_{T,pre}) - (\\bar{Y}_{C,post} - \\bar{Y}_{C,pre})

Where:
- :math:`\\bar{Y}_{T,post}` = Average outcome for treated units post-treatment
- :math:`\\bar{Y}_{T,pre}` = Average outcome for treated units pre-treatment  
- :math:`\\bar{Y}_{C,post}` = Average outcome for control units post-treatment
- :math:`\\bar{Y}_{C,pre}` = Average outcome for control units pre-treatment

**Regression Specification:**

.. math::

   Y_{it} = \\alpha + \\beta_1 Treat_i + \\beta_2 Post_t + \\tau (Treat_i \\times Post_t) + \\epsilon_{it}

Where:
- :math:`Treat_i` = 1 if unit i is in treatment group
- :math:`Post_t` = 1 if time period is post-treatment
- :math:`\\tau` = DiD treatment effect estimate

Key Assumptions
---------------

1. **Parallel Trends Assumption**
   
   **Definition**: In the absence of treatment, treatment and control groups would have followed parallel trends.
   
   **Mathematical**: :math:`E[Y_{1t}^0 - Y_{1t'}^0] = E[Y_{0t}^0 - Y_{0t'}^0]` for all t, t'
   
   **Why it matters**: This is the core identifying assumption that allows causal interpretation.
   
   **Testing**: Examine pre-treatment trends between groups.

2. **No Anticipation Effects**
   
   **Definition**: Treatment effects only occur after treatment implementation.
   
   **Why it matters**: If units anticipate treatment and change behavior beforehand, the parallel trends assumption is violated.
   
   **Testing**: Look for treatment effects in periods just before treatment.

3. **Stable Unit Treatment Value Assumption (SUTVA)**
   
   **Definition**: No spillover effects between units.
   
   **Why it matters**: If treatment of some units affects outcomes of control units, estimates will be biased.
   
   **Considerations**: Geographic proximity, network effects, general equilibrium effects.

4. **Stable Composition**
   
   **Definition**: The composition of treatment and control groups remains stable over time.
   
   **Why it matters**: If different types of units enter/exit groups over time, trends may not be comparable.
   
   **Testing**: Check for changes in observable characteristics over time.

Implementation in CAIS
----------------------

Basic DiD Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causal_agent import CausalAgent
   
   # CAIS automatically detects DiD design
   agent = CausalAgent()
   result = agent.analyze(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_variable',
       time_var='year',
       unit_var='state_id'
   )
   
   print(f"DiD Treatment Effect: {result.ate}")
   print(f"95% Confidence Interval: {result.confidence_interval}")

DiD with Covariates
~~~~~~~~~~~~~~~~~~~

Including time-varying covariates can improve precision and address concerns about parallel trends:

.. code-block:: python

   # DiD with time-varying controls
   result = agent.analyze(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_variable',
       time_var='year',
       unit_var='state_id',
       covariates=['unemployment_rate', 'gdp_per_capita']
   )

Staggered Treatment Timing
~~~~~~~~~~~~~~~~~~~~~~~~~~

When treatment occurs at different times for different units:

.. code-block:: python

   # Staggered DiD (handles multiple treatment timing)
   result = agent.analyze(
       data=staggered_data,
       treatment='treatment_date',
       outcome='outcome_variable',
       time_var='year',
       unit_var='unit_id',
       method='difference_in_differences'
   )

Diagnostic Tests and Validation
-------------------------------

Parallel Trends Testing
~~~~~~~~~~~~~~~~~~~~~~~

The most important diagnostic for DiD is testing the parallel trends assumption:

.. code-block:: python

   # Test parallel trends assumption
   trends_test = agent.test_parallel_trends(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_variable',
       time_var='year',
       unit_var='state_id',
       pre_periods=5
   )
   
   print(f"Parallel trends p-value: {trends_test.p_value}")

**What to look for:**
- Non-significant differences in pre-treatment trends
- Parallel visual trends in event study plots
- No systematic divergence before treatment

Event Study Analysis
~~~~~~~~~~~~~~~~~~~~

Event studies show treatment effects over time and help validate parallel trends:

.. code-block:: python

   # Event study plot
   event_study = agent.event_study(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_variable',
       time_var='year',
       unit_var='state_id',
       leads=3,  # periods before treatment
       lags=5    # periods after treatment
   )

**Interpretation:**
- Pre-treatment coefficients should be close to zero
- Treatment effects may evolve over time post-treatment
- Confidence intervals help assess statistical significance

Placebo Tests
~~~~~~~~~~~~~

Test DiD on fake treatment dates or outcomes that shouldn't be affected:

.. code-block:: python

   # Placebo test with fake treatment date
   placebo_test = agent.placebo_test(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_variable',
       time_var='year',
       unit_var='state_id',
       fake_treatment_date='2015'  # before actual treatment
   )

Advanced DiD Methods
--------------------

Two-Way Fixed Effects (TWFE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard DiD regression with unit and time fixed effects:

.. math::

   Y_{it} = \\alpha_i + \\lambda_t + \\tau D_{it} + X_{it}'\\beta + \\epsilon_{it}

Where:
- :math:`\\alpha_i` = Unit fixed effects
- :math:`\\lambda_t` = Time fixed effects  
- :math:`D_{it}` = Treatment indicator
- :math:`X_{it}` = Time-varying covariates

**Advantages**: Controls for unit and time-invariant factors
**Limitations**: Can be biased with heterogeneous treatment effects and staggered timing

Callaway and Sant'Anna (2021) Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Addresses bias in TWFE with staggered treatment adoption:

.. code-block:: python

   # Use CS method for staggered DiD
   result = agent.analyze(
       data=staggered_data,
       treatment='treatment_date',
       outcome='outcome_variable', 
       time_var='year',
       unit_var='unit_id',
       method='callaway_santanna'
   )

**Key Features:**
- Handles treatment effect heterogeneity
- Provides group-time average treatment effects
- Aggregates to overall treatment effect

Synthetic DiD
~~~~~~~~~~~~~

Combines DiD with synthetic control methods:

.. code-block:: python

   # Synthetic DiD approach
   result = agent.analyze(
       data=panel_data,
       treatment='policy_implemented',
       outcome='outcome_variable',
       time_var='year', 
       unit_var='unit_id',
       method='synthetic_did'
   )

Best Practices
--------------

Design and Data Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data Structure:**
- Balanced panel preferred but not required
- Sufficient pre-treatment periods for trend testing
- Clear treatment timing definition
- Adequate sample size in both groups

**Treatment Definition:**
- Binary treatment indicator clearly defined
- Treatment timing precisely measured
- Consider treatment intensity if relevant
- Document any treatment reversals

Analysis Approach
~~~~~~~~~~~~~~~~~

**Specification Testing:**
- Test parallel trends assumption thoroughly
- Conduct event studies to examine dynamics
- Use appropriate standard errors (clustered by unit)
- Consider alternative specifications as robustness checks

**Robustness Checks:**
- Vary the sample period
- Exclude units close to treatment boundary
- Test different control groups
- Use alternative outcome measures

**Reporting:**
- Present event study plots
- Report parallel trends test results
- Discuss assumption plausibility
- Consider effect heterogeneity

Common Pitfalls and Solutions
-----------------------------

**Pitfall**: Assuming parallel trends without testing
**Solution**: Always test pre-treatment trends and conduct event studies

**Pitfall**: Using inappropriate standard errors
**Solution**: Cluster standard errors at the unit level (or higher level of treatment variation)

**Pitfall**: Ignoring treatment effect heterogeneity
**Solution**: Use modern DiD methods (CS, Sun & Abraham) for staggered timing

**Pitfall**: Misinterpreting dynamic effects
**Solution**: Use event studies to understand how effects evolve over time

**Pitfall**: Inadequate control group
**Solution**: Carefully justify control group selection and test robustness

Example: Minimum Wage Policy Evaluation
---------------------------------------

**Research Question**: What is the effect of minimum wage increases on employment?

**Data**: State-level panel data, 2010-2020
- Treatment: States that increased minimum wage in 2015
- Control: States that did not increase minimum wage
- Outcome: Employment rate

**Analysis**:

.. code-block:: python

   # DiD analysis of minimum wage policy
   result = agent.analyze(
       data=min_wage_data,
       treatment='min_wage_increase_2015',
       outcome='employment_rate',
       time_var='year',
       unit_var='state',
       covariates=['gdp_growth', 'unemployment_rate']
   )
   
   # Test parallel trends
   trends_test = agent.test_parallel_trends(
       data=min_wage_data,
       treatment='min_wage_increase_2015',
       outcome='employment_rate',
       time_var='year',
       unit_var='state',
       pre_periods=5
   )
   
   print(f"Treatment Effect: {result.ate:.3f}")
   print(f"Parallel Trends p-value: {trends_test.p_value:.3f}")

**Interpretation**: 
The minimum wage increase led to a X percentage point change in employment rates (95% CI: [Y, Z]). The parallel trends assumption is supported by the pre-treatment trend analysis (p = 0.XX).

Extensions and Related Methods
------------------------------

**Triple Differences (DDD)**
- Adds a third dimension of variation
- Useful when parallel trends may be violated
- Controls for group-specific time trends

**Synthetic Control**
- Alternative when few treated units
- Creates synthetic control from donor pool
- More transparent counterfactual construction

**Regression Discontinuity in Time**
- Exploits sharp changes in treatment probability over time
- Useful for policy changes with specific implementation dates
- Requires continuity assumption around cutoff

Further Reading
---------------

**Foundational Papers**:
- Ashenfelter, O. & Card, D. (1985). "Using the Longitudinal Structure of Earnings to Estimate the Effect of Training Programs"
- Card, D. & Krueger, A.B. (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania"

**Modern Developments**:
- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with Multiple Time Periods"
- Sun, L. & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects"
- Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing"

**Practical Guides**:
- Angrist, J.D. & Pischke, J.S. (2008). "Mostly Harmless Econometrics"
- Cunningham, S. (2021). "Causal Inference: The Mixtape"
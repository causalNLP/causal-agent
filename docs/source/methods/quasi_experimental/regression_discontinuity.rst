Regression Discontinuity Design (RDD)
====================================

Regression Discontinuity Design (RDD) is a quasi-experimental method that exploits arbitrary cutoffs in treatment assignment rules to identify causal effects. RDD compares units just above and below a threshold to estimate local treatment effects.

When to Use RDD
----------------

**Ideal Conditions:**
- Treatment assignment is determined by a continuous variable (running variable) crossing a threshold
- Assignment rule is strictly enforced and known
- Units cannot precisely manipulate the running variable around the cutoff
- Sufficient observations near the cutoff

**Common Applications:**
- Educational interventions (test score cutoffs for remedial programs)
- Financial aid eligibility (income thresholds)
- Policy interventions (age cutoffs, geographic boundaries)
- Medical treatments (clinical thresholds for treatment)
- Electoral systems (vote share thresholds)

**Not Suitable When:**
- Treatment assignment is not based on a clear cutoff
- Running variable can be easily manipulated
- Insufficient observations near the cutoff
- Multiple simultaneous cutoffs exist

Theoretical Background
----------------------

The RDD Framework
~~~~~~~~~~~~~~~~~

**Basic Setup:**
- **Running Variable (X)**: Continuous variable determining treatment assignment
- **Cutoff (c)**: Threshold value where treatment assignment changes
- **Treatment Assignment**: :math:`D_i = 1` if :math:`X_i \\geq c`, :math:`D_i = 0$ if :math:`X_i < c$

**Sharp RDD:**
Treatment assignment is a deterministic function of the running variable:

.. math::

   D_i = \\begin{cases} 
   1 & \\text{if } X_i \\geq c \\\\
   0 & \\text{if } X_i < c
   \\end{cases}

**Fuzzy RDD:**
Treatment probability changes discontinuously at the cutoff, but assignment is not deterministic:

.. math::

   P(D_i = 1 | X_i) = \\begin{cases}
   g_1(X_i) & \\text{if } X_i \\geq c \\\\
   g_0(X_i) & \\text{if } X_i < c
   \\end{cases}

Where :math:`g_1(c) \\neq g_0(c)$.

**RDD Estimand:**
The treatment effect at the cutoff:

.. math::

   \\tau_{RDD} = E[Y_i(1) - Y_i(0) | X_i = c]

**Sharp RDD Estimation:**

.. math::

   \\hat{\\tau}_{RDD} = \\lim_{x \\to c^+} E[Y_i | X_i = x] - \\lim_{x \\to c^-} E[Y_i | X_i = x]

Key Assumptions
---------------

1. **Continuity of Potential Outcomes**
   
   **Definition**: Potential outcomes are continuous at the cutoff in the absence of treatment.
   
   **Mathematical**: :math:`\\lim_{x \\to c^+} E[Y_i(0) | X_i = x] = \\lim_{x \\to c^-} E[Y_i(0) | X_i = x]$
   
   **Why it matters**: This is the core identifying assumption that allows causal interpretation.
   
   **Testing**: Check for discontinuities in covariates at the cutoff.

2. **No Precise Manipulation**
   
   **Definition**: Units cannot precisely control their value of the running variable around the cutoff.
   
   **Why it matters**: If units can manipulate assignment, selection bias is reintroduced.
   
   **Testing**: McCrary density test for discontinuities in running variable density.

3. **No Other Discontinuities**
   
   **Definition**: No other treatments or interventions change discontinuously at the same cutoff.
   
   **Why it matters**: Other discontinuous changes would confound the treatment effect.
   
   **Testing**: Examine institutional rules and policy changes around the cutoff.

Types of RDD
------------

Sharp RDD
~~~~~~~~~

**Characteristics:**
- Treatment assignment is deterministic based on running variable
- All units above cutoff receive treatment, all below do not
- Simpler analysis and interpretation

**Estimation:**
- Compare outcomes just above and below cutoff
- Use local linear regression or other nonparametric methods
- Focus on observations within optimal bandwidth

Fuzzy RDD
~~~~~~~~~

**Characteristics:**
- Treatment probability changes at cutoff but assignment is not deterministic
- Some units above cutoff don't receive treatment (non-compliance)
- Some units below cutoff receive treatment (always-takers)

**Estimation:**
- Use instrumental variables approach
- Running variable above/below cutoff as instrument for treatment
- Estimates Local Average Treatment Effect (LATE) for compliers

**Two-Stage Approach:**
*First Stage:* :math:`D_i = \\alpha_0 + \\alpha_1 \\mathbf{1}(X_i \\geq c) + f(X_i) + \\epsilon_i$
*Second Stage:* :math:`Y_i = \\beta_0 + \\tau \\hat{D_i} + g(X_i) + u_i$

Implementation in Causal Agent
----------------------

Sharp RDD Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causal_agent import CausalAgent
   
   # Causal Agent automatically detects RDD design
   agent = CausalAgent()
   result = agent.analyze(
       data=rdd_data,
       treatment='above_cutoff',
       outcome='test_score',
       running_var='prior_score',
       cutoff_value=70
   )
   
   print(f"RDD Treatment Effect: {result.ate}")
   print(f"95% Confidence Interval: {result.confidence_interval}")
   print(f"Bandwidth used: {result.bandwidth}")

Fuzzy RDD Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fuzzy RDD with imperfect compliance
   result = agent.analyze(
       data=rdd_data,
       treatment='actually_treated',  # actual treatment received
       outcome='test_score',
       running_var='prior_score',
       cutoff_value=70,
       method='fuzzy_rdd'
   )
   
   print(f"LATE Estimate: {result.late}")
   print(f"First-stage jump: {result.first_stage_jump}")

Bandwidth Selection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom bandwidth selection
   result = agent.analyze(
       data=rdd_data,
       treatment='above_cutoff',
       outcome='test_score',
       running_var='prior_score',
       cutoff_value=70,
       bandwidth_method='optimal',  # or 'cross_validation', 'rule_of_thumb'
       bandwidth_value=5.0  # manual bandwidth
   )

Diagnostic Tests and Validation
-------------------------------

Manipulation Testing
~~~~~~~~~~~~~~~~~~~~

Test whether units can precisely manipulate the running variable:

.. code-block:: python

   # McCrary density test
   manipulation_test = agent.mccrary_test(
       data=rdd_data,
       running_var='prior_score',
       cutoff_value=70
   )
   
   print(f"McCrary test p-value: {manipulation_test.p_value}")
   print(f"Density discontinuity: {manipulation_test.discontinuity}")

**What to look for:**
- Non-significant p-value (no evidence of manipulation)
- Smooth density around the cutoff
- No unusual bunching just above or below cutoff

Covariate Balance Testing
~~~~~~~~~~~~~~~~~~~~~~~~~

Check for discontinuities in predetermined characteristics:

.. code-block:: python

   # Test balance of covariates at cutoff
   balance_test = agent.covariate_balance_rdd(
       data=rdd_data,
       covariates=['age', 'gender', 'socioeconomic_status'],
       running_var='prior_score',
       cutoff_value=70
   )
   
   print("Covariate balance results:")
   for var, result in balance_test.items():
       print(f"{var}: discontinuity = {result.discontinuity:.3f}, p = {result.p_value:.3f}")

**Interpretation:**
- Non-significant discontinuities support validity
- Significant jumps suggest potential confounding
- Pattern of imbalances may indicate manipulation

Bandwidth Sensitivity
~~~~~~~~~~~~~~~~~~~~~

Test robustness to bandwidth choice:

.. code-block:: python

   # Sensitivity to bandwidth selection
   bandwidth_sensitivity = agent.bandwidth_sensitivity(
       data=rdd_data,
       treatment='above_cutoff',
       outcome='test_score',
       running_var='prior_score',
       cutoff_value=70,
       bandwidth_range=[2, 3, 4, 5, 6, 7, 8]
   )
   
   print("Bandwidth sensitivity results:")
   for bw, estimate in bandwidth_sensitivity.items():
       print(f"Bandwidth {bw}: Effect = {estimate.effect:.3f} (SE = {estimate.se:.3f})")

Placebo Cutoff Tests
~~~~~~~~~~~~~~~~~~~~

Test for treatment effects at fake cutoffs:

.. code-block:: python

   # Placebo tests at alternative cutoffs
   placebo_tests = agent.placebo_cutoff_tests(
       data=rdd_data,
       treatment='above_cutoff',
       outcome='test_score',
       running_var='prior_score',
       true_cutoff=70,
       placebo_cutoffs=[65, 67.5, 72.5, 75]
   )
   
   print("Placebo test results:")
   for cutoff, result in placebo_tests.items():
       print(f"Cutoff {cutoff}: Effect = {result.effect:.3f}, p = {result.p_value:.3f}")

**Interpretation:**
- Non-significant effects at placebo cutoffs support validity
- Significant effects suggest confounding or model misspecification

Functional Form Testing
~~~~~~~~~~~~~~~~~~~~~~~

Test sensitivity to polynomial order and functional form:

.. code-block:: python

   # Test different polynomial orders
   functional_form_test = agent.functional_form_sensitivity(
       data=rdd_data,
       treatment='above_cutoff',
       outcome='test_score',
       running_var='prior_score',
       cutoff_value=70,
       polynomial_orders=[1, 2, 3, 4]
   )

Best Practices
--------------

Design and Data Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Running Variable Selection:**
- Choose variables that determine treatment assignment
- Ensure precise measurement around cutoff
- Document assignment rules clearly
- Consider multiple running variables if relevant

**Sample Size Planning:**
- Focus observations near the cutoff
- Ensure adequate power for local effects
- Consider optimal sample allocation
- Plan for potential manipulation

**Data Quality:**
- Verify assignment rule implementation
- Check for measurement error in running variable
- Document any exceptions or overrides
- Collect rich covariate data for validation

Analysis Implementation
~~~~~~~~~~~~~~~~~~~~~~~

**Bandwidth Selection:**
- Use data-driven optimal bandwidth methods
- Report sensitivity to bandwidth choice
- Consider different bandwidths for different outcomes
- Balance bias-variance tradeoff

**Functional Form:**
- Start with local linear regression
- Test sensitivity to polynomial order
- Consider nonparametric methods
- Avoid overfitting with high-order polynomials

**Standard Errors:**
- Use robust standard errors
- Consider clustering if appropriate
- Account for bandwidth selection uncertainty
- Report confidence intervals

Validation and Robustness
~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption Testing:**
- Always conduct manipulation tests
- Check covariate balance at cutoff
- Test for other discontinuities
- Examine institutional details

**Sensitivity Analysis:**
- Vary bandwidth systematically
- Test different functional forms
- Exclude observations very close to cutoff
- Use alternative estimation methods

**Transparency:**
- Report all diagnostic tests
- Show graphical evidence
- Discuss institutional context
- Acknowledge limitations

Common Pitfalls and Solutions
-----------------------------

**Pitfall**: Using inappropriate bandwidth
**Solution**: Use optimal bandwidth methods and test sensitivity

**Pitfall**: Ignoring manipulation possibilities
**Solution**: Always conduct McCrary tests and examine institutional incentives

**Pitfall**: Overfitting with high-order polynomials
**Solution**: Use local linear regression and test functional form sensitivity

**Pitfall**: Misinterpreting local effects as global
**Solution**: Clearly state that RDD estimates local effects at the cutoff

**Pitfall**: Inadequate sample size near cutoff
**Solution**: Focus data collection near cutoff and conduct power analysis

Example: Educational Remediation Program
----------------------------------------

**Research Question**: What is the effect of mandatory tutoring on student achievement?

**Setting**: Students with test scores below 70 are required to attend tutoring
- Running Variable: Prior test score (0-100)
- Cutoff: Score of 70
- Treatment: Mandatory tutoring participation
- Outcome: End-of-year test score

**Analysis**:

.. code-block:: python

   # Sharp RDD analysis
   result = agent.analyze(
       data=education_rdd,
       treatment='mandatory_tutoring',
       outcome='end_year_score',
       running_var='prior_test_score',
       cutoff_value=70
   )
   
   # Validation tests
   manipulation_test = agent.mccrary_test(
       data=education_rdd,
       running_var='prior_test_score',
       cutoff_value=70
   )
   
   balance_test = agent.covariate_balance_rdd(
       data=education_rdd,
       covariates=['age', 'gender', 'free_lunch'],
       running_var='prior_test_score',
       cutoff_value=70
   )
   
   print(f"RDD Treatment Effect: {result.ate:.2f} points")
   print(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
   print(f"McCrary test p-value: {manipulation_test.p_value:.3f}")

**Results Interpretation**:
Students just below the cutoff (required to attend tutoring) scored X points higher on the end-of-year test compared to students just above the cutoff. The McCrary test shows no evidence of score manipulation (p = 0.XX).

Advanced RDD Methods
--------------------

Multi-Cutoff RDD
~~~~~~~~~~~~~~~~

When multiple cutoffs exist:

.. code-block:: python

   # Multiple cutoffs analysis
   result = agent.analyze(
       data=multi_cutoff_data,
       treatment='treatment_intensity',
       outcome='outcome_var',
       running_var='score',
       cutoff_values=[50, 70, 85],
       method='multi_cutoff_rdd'
   )

Geographic RDD
~~~~~~~~~~~~~~

Using geographic boundaries as cutoffs:

.. code-block:: python

   # Geographic discontinuity
   result = agent.analyze(
       data=geographic_data,
       treatment='policy_exposure',
       outcome='outcome_var',
       running_var='distance_to_boundary',
       cutoff_value=0,
       method='geographic_rdd'
   )

Regression Kink Design
~~~~~~~~~~~~~~~~~~~~~~

When treatment intensity (rather than probability) changes at cutoff:

.. code-block:: python

   # Regression kink design
   result = agent.analyze(
       data=kink_data,
       treatment='treatment_intensity',
       outcome='outcome_var',
       running_var='eligibility_score',
       cutoff_value=75,
       method='regression_kink'
   )

Further Reading
---------------

**Foundational Papers**:
- Thistlethwaite, D.L. & Campbell, D.T. (1960). "Regression-Discontinuity Analysis: An Alternative to the Ex Post Facto Experiment"
- Hahn, J., Todd, P. & Van der Klaauw, W. (2001). "Identification and Estimation of Treatment Effects with a Regression-Discontinuity Design"
- Imbens, G.W. & Lemieux, T. (2008). "Regression Discontinuity Designs: A Guide to Practice"

**Modern Developments**:
- Calonico, S., Cattaneo, M.D. & Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs"
- Cattaneo, M.D., Idrobo, N. & Titiunik, R. (2019). "A Practical Introduction to Regression Discontinuity Designs: Foundations"
- Kolesár, M. & Rothe, C. (2018). "Inference in Regression Discontinuity Designs with a Discrete Running Variable"

**Practical Guides**:
- Lee, D.S. & Lemieux, T. (2010). "Regression Discontinuity Designs in Economics"
- Jacob, R., Zhu, P., Somers, M.A. & Bloom, H. (2012). "A Practical Guide to Regression Discontinuity"
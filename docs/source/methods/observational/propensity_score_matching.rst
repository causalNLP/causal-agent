Propensity Score Matching
=========================

Propensity Score Matching (PSM) is a statistical technique that estimates causal effects by matching treated and control units with similar propensity scores - the probability of receiving treatment given observed covariates. This method helps reduce selection bias in observational studies.

When to Use PSM
----------------

**Ideal Conditions:**
- Rich set of observed covariates that predict treatment assignment
- Clear binary treatment definition
- Sufficient overlap in propensity scores between treatment groups
- Reasonable sample size in both treatment and control groups

**Common Applications:**
- Program evaluation (job training, education programs)
- Medical treatment effectiveness studies
- Policy impact assessment
- Marketing intervention analysis
- Social program evaluation

**Not Suitable When:**
- Limited covariate information available
- Poor overlap between treatment and control groups
- Continuous or multi-valued treatments (use GPS instead)
- Strong selection on unobservables suspected

Theoretical Background
----------------------

The Propensity Score
~~~~~~~~~~~~~~~~~~~~

**Definition**: The propensity score is the probability of receiving treatment given observed covariates:

.. math::

   e(X_i) = P(D_i = 1 | X_i)

Where:
- :math:`D_i` = Treatment indicator (1 if treated, 0 if control)
- :math:`X_i` = Vector of observed covariates for unit i

**Balancing Property**: 
If the propensity score is correctly specified, then:

.. math::

   D_i \\perp X_i | e(X_i)

This means that within strata of the propensity score, treatment assignment is independent of covariates.

**Unconfoundedness Assumption**:
The key identifying assumption is that treatment assignment is unconfounded given observed covariates:

.. math::

   (Y_i^1, Y_i^0) \\perp D_i | X_i

Where :math:`Y_i^1` and :math:`Y_i^0` are potential outcomes under treatment and control.

Rosenbaum and Rubin (1983) Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key Result**: If treatment assignment is unconfounded given X, then it is also unconfounded given the propensity score e(X).

**Implication**: Instead of matching on the full vector of covariates X (which suffers from the curse of dimensionality), we can match on the scalar propensity score.

**Average Treatment Effect on the Treated (ATT)**:

.. math::

   ATT = E[Y^1 - Y^0 | D = 1] = E[Y^1 | D = 1] - E[Y^0 | D = 1]

Since we don't observe :math:`Y^0` for treated units, PSM estimates:

.. math::

   \\hat{ATT} = E[Y | D = 1] - E[Y | D = 0, e(X) = e(X_{treated})]

Key Assumptions
---------------

1. **Unconfoundedness (Selection on Observables)**
   
   **Definition**: All variables that affect both treatment assignment and outcomes are observed.
   
   **Mathematical**: :math:`(Y^1, Y^0) \\perp D | X`
   
   **Why it matters**: This is the core identifying assumption that cannot be tested.
   
   **Implications**: Omitted variable bias if important confounders are unobserved.

2. **Common Support (Overlap)**
   
   **Definition**: For each value of covariates, there is a positive probability of being in both treatment and control groups.
   
   **Mathematical**: :math:`0 < P(D = 1 | X) < 1$ for all X
   
   **Testing**: Examine propensity score distributions across groups.
   
   **Why it matters**: Without overlap, no valid comparisons can be made.

3. **Stable Unit Treatment Value Assumption (SUTVA)**
   
   **Definition**: No spillover effects between units and treatment is well-defined.
   
   **Components**: 
   - No interference between units
   - No hidden variations of treatment
   
   **Why it matters**: Violations lead to biased treatment effect estimates.

Types of Matching
-----------------

Nearest Neighbor Matching
~~~~~~~~~~~~~~~~~~~~~~~~~

**Method**: Match each treated unit to the control unit(s) with the closest propensity score.

**Variants**:
- 1:1 matching (each treated unit matched to one control)
- 1:k matching (each treated unit matched to k controls)
- With/without replacement

**Advantages**: Simple, intuitive, preserves sample size
**Disadvantages**: May result in poor matches if few controls available

Caliper Matching
~~~~~~~~~~~~~~~~

**Method**: Only match units if their propensity scores are within a specified distance (caliper).

**Caliper Choice**: Common rule is 0.2 × standard deviation of propensity score

**Advantages**: Ensures match quality, avoids bad matches
**Disadvantages**: May discard observations, reduces sample size

Kernel Matching
~~~~~~~~~~~~~~~

**Method**: Match each treated unit to a weighted average of all control units, with weights inversely related to distance.

**Weight Function**: Common choices include Gaussian, Epanechnikov, uniform kernels

**Advantages**: Uses all control observations, smooth weighting
**Disadvantages**: May include poor matches, computationally intensive

Stratification Matching
~~~~~~~~~~~~~~~~~~~~~~~

**Method**: Divide propensity score range into strata and compare within strata.

**Implementation**: Typically use 5 strata (quintiles) of propensity score

**Advantages**: Simple, uses all observations, easy to implement
**Disadvantages**: Coarse matching, may not achieve balance within strata

Implementation in Causal Agent
----------------------

Basic PSM Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causal_agent import CausalAgent
   
   # Causal Agent automatically implements PSM
   agent = CausalAgent()
   result = agent.analyze(
       data=observational_data,
       treatment='program_participation',
       outcome='earnings',
       covariates=['age', 'education', 'experience', 'gender'],
       method='propensity_score_matching'
   )
   
   print(f"ATT Estimate: {result.att}")
   print(f"95% Confidence Interval: {result.confidence_interval}")
   print(f"Number of matched pairs: {result.n_matched}")

Customizing Matching Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Specify matching parameters
   result = agent.analyze(
       data=observational_data,
       treatment='program_participation',
       outcome='earnings',
       covariates=['age', 'education', 'experience', 'gender'],
       method='propensity_score_matching',
       matching_options={
           'method': 'nearest_neighbor',
           'n_neighbors': 2,  # 1:2 matching
           'caliper': 0.1,    # caliper width
           'replace': False   # matching without replacement
       }
   )

Propensity Score Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom propensity score model
   result = agent.analyze(
       data=observational_data,
       treatment='program_participation',
       outcome='earnings',
       covariates=['age', 'education', 'experience', 'gender'],
       method='propensity_score_matching',
       ps_model='logistic',  # or 'probit', 'random_forest'
       ps_formula='age + education + I(age**2) + education*experience'
   )

Diagnostic Tests and Validation
-------------------------------

Propensity Score Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, estimate and evaluate the propensity score model:

.. code-block:: python

   # Propensity score diagnostics
   ps_diagnostics = agent.propensity_score_diagnostics(
       data=observational_data,
       treatment='program_participation',
       covariates=['age', 'education', 'experience', 'gender']
   )
   
   print(f"Pseudo R-squared: {ps_diagnostics.pseudo_r2}")
   print(f"Model AUC: {ps_diagnostics.auc}")

**What to look for**:
- Reasonable predictive power (Pseudo R² between 0.1-0.3)
- Good discrimination (AUC > 0.7)
- No perfect prediction (avoid propensity scores of 0 or 1)

Common Support Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~

Check overlap in propensity score distributions:

.. code-block:: python

   # Common support analysis
   support_analysis = agent.common_support_analysis(
       data=observational_data,
       treatment='program_participation',
       covariates=['age', 'education', 'experience', 'gender']
   )
   
   print(f"Observations on common support: {support_analysis.n_on_support}")
   print(f"Percentage on support: {support_analysis.pct_on_support}")

**Visual Inspection**:
- Histogram of propensity scores by treatment group
- Density plots showing overlap
- Box plots comparing distributions

Balance Assessment
~~~~~~~~~~~~~~~~~~

The key test is whether matching achieves covariate balance:

.. code-block:: python

   # Balance assessment before and after matching
   balance_results = agent.balance_assessment(
       data=observational_data,
       treatment='program_participation',
       covariates=['age', 'education', 'experience', 'gender'],
       method='propensity_score_matching'
   )
   
   print("Balance before matching:")
   print(balance_results.before_matching)
   print("Balance after matching:")
   print(balance_results.after_matching)

**Balance Metrics**:
- **Standardized Mean Difference**: |difference in means| / pooled standard deviation
- **Variance Ratio**: ratio of variances between treatment groups
- **t-test p-values**: test of mean differences

**Acceptable Balance**:
- Standardized differences < 0.1 (strict) or < 0.25 (lenient)
- Variance ratios between 0.5 and 2.0
- Non-significant t-tests (though less important than effect sizes)

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Test robustness to unobserved confounding:

.. code-block:: python

   # Rosenbaum bounds sensitivity analysis
   sensitivity_results = agent.rosenbaum_bounds(
       data=matched_data,
       treatment='program_participation',
       outcome='earnings'
   )
   
   print(f"Gamma values tested: {sensitivity_results.gamma_range}")
   print(f"Critical gamma: {sensitivity_results.critical_gamma}")

**Interpretation**:
- Gamma = 1: No unobserved confounding
- Higher gamma: More robust to unobserved confounding
- Critical gamma: Level of confounding needed to change conclusions

Best Practices
--------------

Propensity Score Model Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Variable Selection**:
- Include all variables that predict treatment assignment
- Include variables that predict outcomes (even if not related to treatment)
- Avoid including variables affected by treatment (post-treatment variables)
- Consider interactions and non-linear terms

**Model Choice**:
- Logistic regression is most common and usually sufficient
- Consider machine learning methods (random forest, boosting) for complex relationships
- Focus on balance achievement rather than model fit statistics

**Specification Testing**:
- Test different functional forms
- Include interaction terms where theoretically justified
- Use cross-validation to avoid overfitting

Matching Implementation
~~~~~~~~~~~~~~~~~~~~~~~

**Method Selection**:
- Start with 1:1 nearest neighbor matching
- Use calipers to ensure match quality
- Consider kernel matching for efficiency
- Stratification for simplicity and transparency

**Quality Control**:
- Always check balance after matching
- Examine individual matches for quality
- Consider trimming extreme propensity scores
- Report number of observations dropped

**Standard Errors**:
- Use robust standard errors
- Account for matching uncertainty
- Bootstrap for complex matching procedures
- Cluster at appropriate level if relevant

Analysis and Reporting
~~~~~~~~~~~~~~~~~~~~~~

**Effect Estimation**:
- Report ATT as primary estimand
- Consider ATE if appropriate and feasible
- Examine effect heterogeneity across subgroups
- Test for treatment effect heterogeneity by propensity score

**Robustness Checks**:
- Try different matching methods
- Vary caliper width and number of matches
- Test different propensity score specifications
- Conduct sensitivity analysis for unobserved confounding

**Transparency**:
- Report all diagnostic results
- Show propensity score distributions
- Present balance tables before and after matching
- Discuss limitations and assumptions clearly

Common Pitfalls and Solutions
-----------------------------

**Pitfall**: Poor propensity score model specification
**Solution**: Focus on achieving balance rather than model fit; include relevant predictors

**Pitfall**: Ignoring common support violations
**Solution**: Always check overlap; trim observations outside common support

**Pitfall**: Accepting poor balance after matching
**Solution**: Iterate on propensity score specification until good balance is achieved

**Pitfall**: Not conducting sensitivity analysis
**Solution**: Always test robustness to unobserved confounding using Rosenbaum bounds

**Pitfall**: Misinterpreting ATT as ATE
**Solution**: Be clear about the estimand and its policy relevance

Example: Job Training Program Evaluation
----------------------------------------

**Research Question**: What is the effect of a job training program on earnings?

**Data**: Observational data with program participants and non-participants
- Treatment: Program participation (binary)
- Outcome: Annual earnings 2 years post-program
- Covariates: Age, education, prior earnings, unemployment duration, gender, race

**Analysis**:

.. code-block:: python

   # PSM analysis of job training program
   result = agent.analyze(
       data=training_data,
       treatment='program_participation',
       outcome='earnings_2yr',
       covariates=['age', 'education', 'prior_earnings', 
                  'unemployment_duration', 'gender', 'race'],
       method='propensity_score_matching'
   )
   
   # Check balance
   balance = agent.balance_assessment(
       data=training_data,
       treatment='program_participation',
       covariates=['age', 'education', 'prior_earnings', 
                  'unemployment_duration', 'gender', 'race'],
       method='propensity_score_matching'
   )
   
   # Sensitivity analysis
   sensitivity = agent.rosenbaum_bounds(
       data=result.matched_data,
       treatment='program_participation',
       outcome='earnings_2yr'
   )
   
   print(f"ATT Estimate: ${result.att:,.0f}")
   print(f"95% CI: [${result.ci_lower:,.0f}, ${result.ci_upper:,.0f}]")
   print(f"Critical Gamma: {sensitivity.critical_gamma}")

**Results Interpretation**:
- Program participants earned $X more annually than matched non-participants
- Results are robust to unobserved confounding up to Gamma = Y
- Good covariate balance achieved after matching (all standardized differences < 0.1)

Extensions and Related Methods
------------------------------

**Propensity Score Weighting**
- Alternative to matching that uses all observations
- Weights observations by inverse propensity scores
- More efficient but potentially less robust

**Doubly Robust Methods**
- Combine propensity score methods with outcome regression
- Consistent if either propensity score or outcome model is correct
- Examples: AIPW, TMLE

**Machine Learning Propensity Scores**
- Use random forests, boosting, or neural networks
- Can capture complex relationships
- Focus on balance rather than prediction accuracy

**Multiple Treatment PSM**
- Generalized propensity scores for multiple treatments
- Multinomial logit for propensity score estimation
- More complex balance assessment

Further Reading
---------------

**Foundational Papers**:
- Rosenbaum, P.R. & Rubin, D.B. (1983). "The Central Role of the Propensity Score in Observational Studies for Causal Effects"
- Rosenbaum, P.R. & Rubin, D.B. (1985). "Constructing a Control Group Using Multivariate Matched Sampling Methods"
- Dehejia, R.H. & Wahba, S. (1999). "Causal Effects in Nonexperimental Studies: Reevaluating the Evaluation of Training Programs"

**Modern Developments**:
- Imbens, G.W. (2004). "Nonparametric Estimation of Average Treatment Effects Under Exogeneity"
- Stuart, E.A. (2010). "Matching Methods for Causal Inference: A Review and a Look Forward"
- Austin, P.C. (2011). "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding"

**Practical Guides**:
- Caliendo, M. & Kopeinig, S. (2008). "Some Practical Guidance for the Implementation of Propensity Score Matching"
- Thoemmes, F.J. & Kim, E.S. (2011). "A Systematic Review of Propensity Score Methods in the Social Sciences"
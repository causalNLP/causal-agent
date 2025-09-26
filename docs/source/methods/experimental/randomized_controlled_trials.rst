Randomized Controlled Trials (RCT)
===================================

Randomized Controlled Trials (RCTs) are the gold standard for causal inference. By randomly assigning units to treatment and control groups, RCTs eliminate selection bias and provide unbiased estimates of causal effects.

.. raw:: html

   <div class="expandable-section" data-collapsed="false">
       <h4 class="expandable-title">Quick Overview: When CAIS Recommends RCTs</h4>
       <div class="expandable-content">

CAIS automatically recommends RCT analysis when it detects:

* **Random assignment variable** in your dataset
* **Clear treatment/control groups** with balanced allocation  
* **Post-treatment outcome measurements**
* **No evidence of systematic selection bias**

The decision tree prioritizes RCTs as the **highest confidence method** because randomization eliminates the need for additional identifying assumptions.

.. raw:: html

       </div>
   </div>

When to Use RCTs
-----------------

**Ideal Conditions:**
- Randomization is feasible and ethical
- You have control over treatment assignment
- Units can be clearly assigned to treatment/control
- Outcome can be measured after treatment

**Common Applications:**
- Medical trials testing new treatments
- Educational interventions in schools
- Policy pilot programs
- Product feature testing (A/B tests)
- Social program evaluations

**Not Suitable When:**
- Randomization is unethical (e.g., harmful treatments)
- Treatment assignment cannot be controlled
- Spillover effects are likely
- Long-term outcomes are needed but follow-up is limited

Theoretical Background
----------------------

The Potential Outcomes Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RCTs are grounded in the potential outcomes framework (Rubin Causal Model):

- **Y₁ᵢ**: Potential outcome for unit i under treatment
- **Y₀ᵢ**: Potential outcome for unit i under control  
- **Individual Treatment Effect**: τᵢ = Y₁ᵢ - Y₀ᵢ

The fundamental problem is that we can only observe one potential outcome for each unit. RCTs solve this by using randomization to create comparable groups.

.. raw:: html

   <div class="expandable-section" data-collapsed="true">
       <h5 class="expandable-title">Mathematical Details: Why Randomization Enables Causal Identification</h5>
       <div class="expandable-content">

The key insight is that randomization makes treatment assignment independent of potential outcomes. This allows us to identify causal effects without observing the counterfactual.

**Average Treatment Effect (ATE):**

.. math::

   ATE = E[Y₁ᵢ - Y₀ᵢ] = E[Y₁ᵢ] - E[Y₀ᵢ]

Under randomization:

.. math::

   ATE = E[Yᵢ|Dᵢ=1] - E[Yᵢ|Dᵢ=0]

Where Dᵢ is the treatment indicator.

.. raw:: html

       </div>
   </div>

Why Randomization Works
~~~~~~~~~~~~~~~~~~~~~~~

Randomization ensures that:

1. **Treatment assignment is independent of potential outcomes**: (Y₁ᵢ, Y₀ᵢ) ⊥ Dᵢ
2. **Treatment and control groups are balanced in expectation** on all characteristics
3. **Selection bias is eliminated** by design
4. **Causal identification is achieved** without additional assumptions

Key Assumptions
---------------

RCTs require relatively few assumptions compared to observational methods:

1. **Stable Unit Treatment Value Assumption (SUTVA)**
   
   - No spillover effects between units
   - Treatment is well-defined and consistent across units
   - One unit's treatment doesn't affect another's outcome

2. **Randomization Integrity**
   
   - Treatment assignment follows the randomization protocol
   - No systematic deviations from random assignment
   - Assignment mechanism is known and controlled

3. **No Attrition Bias**
   
   - Missing outcomes are not systematically related to treatment
   - Attrition rates are similar across treatment groups
   - Missing data mechanism is ignorable

Types of RCT Analysis
---------------------

Intent-to-Treat (ITT) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: Compares outcomes based on original treatment assignment, regardless of actual treatment received.

**When to use**: 
- Primary analysis for policy evaluation
- When non-compliance is expected
- For regulatory approval (medical trials)

**Advantages**:
- Preserves randomization
- Estimates policy-relevant effects
- Avoids selection bias from non-compliance

**Limitations**:
- May underestimate biological/direct effects
- Includes dilution from non-compliance

Treatment-on-Treated (TOT) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: Estimates effect of actually receiving treatment, accounting for non-compliance.

**Methods**:
- Instrumental Variables (using randomization as instrument)
- Per-protocol analysis (with caution)
- Complier Average Causal Effect (CACE)

**When to use**:
- Understanding biological/direct effects
- When compliance rates are low
- For mechanism understanding

**Caution**: Can introduce selection bias if non-compliance is related to outcomes.

Implementation in CAIS
----------------------

Basic RCT Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causal_agent import CausalAgent
   
   # Simple RCT analysis
   agent = CausalAgent()
   result = agent.analyze(
       data=rct_data,
       treatment='randomized_treatment',
       outcome='outcome_variable',
       is_rct=True
   )
   
   print(f"Average Treatment Effect: {result.ate}")
   print(f"95% Confidence Interval: {result.confidence_interval}")

RCT with Covariates
~~~~~~~~~~~~~~~~~~~

Including baseline covariates can improve precision:

.. code-block:: python

   # RCT with covariates for precision
   result = agent.analyze(
       data=rct_data,
       treatment='randomized_treatment',
       outcome='outcome_variable',
       covariates=['age', 'gender', 'baseline_score'],
       is_rct=True
   )

Handling Non-Compliance
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Treatment-on-treated analysis
   result = agent.analyze(
       data=rct_data,
       treatment='actual_treatment_received',
       outcome='outcome_variable',
       instrument='randomized_assignment',
       method='instrumental_variable'
   )

Diagnostic Tests and Validation
-------------------------------

Balance Checks
~~~~~~~~~~~~~~

Verify that randomization created balanced groups:

.. code-block:: python

   # Check balance on baseline characteristics
   balance_results = agent.check_balance(
       data=rct_data,
       treatment='randomized_treatment',
       covariates=['age', 'gender', 'baseline_score']
   )

**What to look for**:
- Similar means across treatment groups
- Non-significant t-tests for differences
- Standardized differences < 0.1

Randomization Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Test whether treatment assignment appears random:

.. code-block:: python

   # Test randomization integrity
   randomization_test = agent.test_randomization(
       data=rct_data,
       treatment='randomized_treatment',
       covariates=['age', 'gender', 'baseline_score']
   )

**Red flags**:
- Systematic patterns in treatment assignment
- Significant predictability from baseline characteristics
- Unequal treatment group sizes (without stratification)

Attrition Analysis
~~~~~~~~~~~~~~~~~~

Check for differential attrition between groups:

.. code-block:: python

   # Analyze missing data patterns
   attrition_analysis = agent.analyze_attrition(
       data=rct_data,
       treatment='randomized_treatment',
       outcome='outcome_variable'
   )

**Concerns**:
- Different attrition rates by treatment group
- Attrition related to baseline characteristics
- Systematic patterns in missing data

Best Practices
--------------

Design Phase
~~~~~~~~~~~~

**Sample Size Planning**:
- Conduct power analysis for adequate sample size
- Account for expected attrition rates
- Consider multiple testing if analyzing subgroups

**Randomization Strategy**:
- Use proper randomization procedures (computer-generated)
- Consider stratified randomization for balance
- Block randomization for sequential enrollment
- Document randomization protocol clearly

**Outcome Measurement**:
- Pre-specify primary and secondary outcomes
- Use validated measurement instruments
- Plan timing of outcome measurement
- Consider multiple outcome measures

Implementation Phase
~~~~~~~~~~~~~~~~~~~~

**Maintaining Integrity**:
- Protect randomization sequence until assignment
- Train staff on protocol adherence
- Monitor for protocol deviations
- Document any changes or issues

**Data Collection**:
- Collect rich baseline data before randomization
- Standardize data collection procedures
- Monitor data quality throughout study
- Plan for missing data and attrition

Analysis Phase
~~~~~~~~~~~~~~

**Primary Analysis**:
- Conduct intent-to-treat analysis as primary
- Report confidence intervals, not just p-values
- Use appropriate statistical tests for data type
- Account for multiple testing if relevant

**Secondary Analysis**:
- Explore effect heterogeneity responsibly
- Conduct sensitivity analyses
- Consider treatment-on-treated analysis if relevant
- Report all analyses conducted

**Reporting**:
- Follow CONSORT guidelines for reporting
- Report both statistical and practical significance
- Discuss limitations and generalizability
- Make data and code available when possible

Common Pitfalls and Solutions
-----------------------------

**Pitfall**: Post-randomization exclusions
**Solution**: Include all randomized participants in ITT analysis

**Pitfall**: Changing outcomes after seeing data
**Solution**: Pre-specify all outcomes and analysis plans

**Pitfall**: Ignoring non-compliance
**Solution**: Report both ITT and TOT analyses when relevant

**Pitfall**: Over-interpreting subgroup analyses
**Solution**: Pre-specify subgroups and adjust for multiple testing

**Pitfall**: Assuming external validity
**Solution**: Discuss generalizability limitations explicitly

Example: Educational Intervention RCT
-------------------------------------

**Research Question**: Does a new math curriculum improve student test scores?

**Design**:
- Randomize 100 classrooms to new curriculum (treatment) or standard curriculum (control)
- Measure math test scores at end of school year
- Collect baseline student and teacher characteristics

**Analysis**:

.. code-block:: python

   # Primary ITT analysis
   result = agent.analyze(
       data=education_rct,
       treatment='new_curriculum',
       outcome='math_test_score',
       covariates=['baseline_score', 'teacher_experience'],
       is_rct=True
   )
   
   print(f"ITT Effect: {result.ate:.2f} points")
   print(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
   print(f"P-value: {result.p_value:.3f}")

**Interpretation**: 
The new curriculum increased math test scores by an average of X points (95% CI: [Y, Z]), representing the effect of being assigned to the new curriculum regardless of implementation fidelity.

Further Reading
---------------

**Classic References**:
- Fisher, R.A. (1935). "The Design of Experiments"
- Rubin, D.B. (1974). "Estimating Causal Effects of Treatments"
- Holland, P.W. (1986). "Statistics and Causal Inference"

**Modern Applications**:
- Duflo, E., Glennerster, R., & Kremer, M. (2007). "Using Randomization in Development Economics Research"
- Gerber, A.S. & Green, D.P. (2012). "Field Experiments: Design, Analysis, and Interpretation"

**Reporting Guidelines**:
- CONSORT Statement: http://www.consort-statement.org/
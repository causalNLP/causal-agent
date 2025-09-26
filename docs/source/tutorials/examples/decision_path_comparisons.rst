Decision Path Comparisons: Similar Datasets, Different Methods
==============================================================

This document provides side-by-side comparisons of how CAIS selects different methods for similar datasets with slight variations in characteristics. Understanding these comparisons helps illustrate the decision tree logic and method selection criteria.

Overview
--------

Small changes in dataset characteristics can lead to dramatically different method selections. This document shows:

- How minor data differences affect method choice
- Why certain methods are preferred over others
- What happens when key assumptions are violated
- How to interpret method selection decisions

Comparison 1: Randomized vs. Observational Education Data
---------------------------------------------------------

**Scenario**: Evaluating the impact of a tutoring program on student test scores

Randomized Version
~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Students randomly assigned to tutoring program
- Balanced baseline characteristics
- Rich covariate information available
- Perfect compliance with assignment

.. mermaid::

   flowchart TD
       A[Randomized Tutoring Study] --> B{Is this randomized?}
       B -->|Yes ✓| C{Are covariates available?}
       C -->|Yes ✓| D[Linear Regression<br/>with Covariates]
       
       style A fill:#e3f2fd
       style B fill:#e8f5e8
       style C fill:#fff3e0
       style D fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Linear Regression with Covariates
   
   Reasoning:
   ✓ Randomization ensures causal identification
   ✓ Covariates improve precision (reduce standard errors)
   ✓ No selection bias concerns
   ✓ Straightforward interpretation
   
   Expected Results:
   - Unbiased treatment effect estimate
   - Narrow confidence intervals (high precision)
   - Clear causal interpretation

Observational Version
~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Students self-select into tutoring program
- Systematic differences in baseline characteristics
- Same rich covariate information available
- Good overlap in covariate distributions

.. mermaid::

   flowchart TD
       A[Observational Tutoring Study] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|No ✗| G{Rich covariates?}
       G -->|Yes ✓| H{Good overlap?}
       H -->|Yes ✓| I[Propensity Score<br/>Matching]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#ffebee
       style G fill:#fff3e0
       style H fill:#fff3e0
       style I fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Propensity Score Matching
   
   Reasoning:
   ❌ No randomization (selection bias present)
   ✓ Rich covariates available for matching
   ✓ Good covariate overlap enables valid matches
   ✓ Can control for observed confounders
   
   Expected Results:
   - Potentially biased if unobserved confounders exist
   - Wider confidence intervals (less precision)
   - Requires strong unconfoundedness assumption

**Side-by-Side Comparison**:

.. list-table:: Randomized vs. Observational Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Randomized Version
     - Observational Version
   * - Method Selected
     - Linear Regression + Covariates
     - Propensity Score Matching
   * - Identification
     - Randomization
     - Unconfoundedness assumption
   * - Bias Risk
     - None (randomized)
     - Possible (unobserved confounders)
   * - Precision
     - High (uses all data)
     - Lower (matched sample only)
   * - Assumptions
     - Minimal
     - Strong (no unmeasured confounding)

---

Comparison 2: Cross-Sectional vs. Panel Policy Data
---------------------------------------------------

**Scenario**: Evaluating the impact of minimum wage increases on employment

Cross-Sectional Version
~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Single time point after policy implementation
- States with and without minimum wage increases
- Rich economic and demographic controls
- No pre-policy baseline data

.. mermaid::

   flowchart TD
       A[Cross-Sectional Policy Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|No ✗| G{Rich covariates?}
       G -->|Yes ✓| H{Good overlap?}
       H -->|Yes ✓| I[Propensity Score<br/>Methods]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#ffebee
       style G fill:#fff3e0
       style H fill:#fff3e0
       style I fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Propensity Score Methods
   
   Reasoning:
   ❌ No randomization (policy endogenous)
   ❌ No panel data (single time point)
   ❌ No clear running variable
   ✓ Rich covariates for matching/weighting
   
   Limitations:
   ⚠️ Cannot control for unobserved state characteristics
   ⚠️ Policy adoption may be endogenous
   ⚠️ Strong unconfoundedness assumption required

Panel Version
~~~~~~~~~~~~~

**Dataset Characteristics**:
- Multiple time periods before and after policy
- Staggered implementation across states
- Same rich controls available
- Clear treatment timing variation

.. mermaid::

   flowchart TD
       A[Panel Policy Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|Yes ✓| D{Treatment timing varies?}
       D -->|Yes ✓| E[Difference-in-Differences]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#fff3e0
       style D fill:#fff3e0
       style E fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Difference-in-Differences
   
   Reasoning:
   ❌ No randomization (policy endogenous)
   ✓ Panel data with timing variation
   ✓ Can control for time-invariant confounders
   ✓ Exploits policy timing for identification
   
   Advantages:
   ✓ Controls for unobserved state characteristics
   ✓ Handles time trends
   ✓ More credible identification than cross-sectional

**Side-by-Side Comparison**:

.. list-table:: Cross-Sectional vs. Panel Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Cross-Sectional Version
     - Panel Version
   * - Method Selected
     - Propensity Score Methods
     - Difference-in-Differences
   * - Identification
     - Unconfoundedness
     - Parallel trends
   * - Controls For
     - Observed confounders only
     - Time-invariant unobservables
   * - Key Assumption
     - No unmeasured confounding
     - Parallel trends
   * - Credibility
     - Lower (strong assumptions)
     - Higher (weaker assumptions)

---

Comparison 3: Sharp vs. Fuzzy Discontinuity
-------------------------------------------

**Scenario**: Evaluating scholarship program effects on college enrollment

Sharp Discontinuity Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Test score determines scholarship eligibility
- Sharp cutoff at score = 1200
- All students above cutoff get scholarship
- No students below cutoff get scholarship

.. mermaid::

   flowchart TD
       A[Sharp RDD Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable with cutoff?}
       D -->|Yes ✓| E{Sharp discontinuity?}
       E -->|Yes ✓| F[Sharp Regression<br/>Discontinuity]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#fff3e0
       style E fill:#fff3e0
       style F fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Sharp Regression Discontinuity
   
   Reasoning:
   ✓ Clear running variable (test score)
   ✓ Sharp cutoff at 1200
   ✓ Treatment probability jumps from 0 to 1
   ✓ Local randomization around cutoff
   
   Implementation:
   - Compare students just above/below cutoff
   - Estimate local treatment effect
   - Check continuity assumptions

Fuzzy Discontinuity Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Same test score running variable
- Same cutoff at score = 1200
- Scholarship probability increases but doesn't reach 100%
- Some students below cutoff still get scholarships

.. mermaid::

   flowchart TD
       A[Fuzzy RDD Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable with cutoff?}
       D -->|Yes ✓| E{Sharp discontinuity?}
       E -->|No ✗| F{Fuzzy discontinuity?}
       F -->|Yes ✓| G[Fuzzy Regression<br/>Discontinuity]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#fff3e0
       style E fill:#ffebee
       style F fill:#fff3e0
       style G fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Fuzzy Regression Discontinuity
   
   Reasoning:
   ✓ Clear running variable (test score)
   ✓ Discontinuous jump in treatment probability
   ❌ Treatment probability doesn't reach 100%
   ✓ Can use IV approach with cutoff as instrument
   
   Implementation:
   - First stage: cutoff predicts scholarship probability
   - Second stage: predicted scholarship affects enrollment
   - Estimate local average treatment effect (LATE)

**Side-by-Side Comparison**:

.. list-table:: Sharp vs. Fuzzy RDD Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Sharp RDD
     - Fuzzy RDD
   * - Method Selected
     - Sharp RDD
     - Fuzzy RDD (IV approach)
   * - Treatment Assignment
     - Deterministic at cutoff
     - Probabilistic at cutoff
   * - Identification
     - Direct comparison
     - Instrumental variables
   * - Interpretation
     - Average treatment effect
     - Local average treatment effect
   * - Complexity
     - Simpler
     - More complex (two-stage)

---

Comparison 4: Strong vs. Weak Instrument
----------------------------------------

**Scenario**: Evaluating the effect of education on earnings

Strong Instrument Version
~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Distance to college as instrument for education
- Strong first-stage relationship (F > 50)
- Credible exclusion restriction
- Large sample size

.. mermaid::

   flowchart TD
       A[Strong IV Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|No ✗| F{Continuous treatment}
       F --> G{Instrumental variable?}
       G -->|Yes ✓| H{Strong instrument?}
       H -->|Yes ✓| I[Instrumental Variables]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#ffebee
       style F fill:#fff3e0
       style G fill:#fff3e0
       style H fill:#fff3e0
       style I fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Instrumental Variables
   
   Reasoning:
   ✓ Strong first-stage relationship (F = 52.3)
   ✓ Credible exclusion restriction
   ✓ Handles unmeasured confounding
   ✓ Large sample provides adequate power
   
   Expected Results:
   - Consistent estimates
   - Reasonable precision
   - Valid inference

Weak Instrument Version
~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Same distance to college instrument
- Weak first-stage relationship (F < 10)
- Same exclusion restriction
- Same sample size

.. mermaid::

   flowchart TD
       A[Weak IV Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|No ✗| F{Continuous treatment}
       F --> G{Instrumental variable?}
       G -->|Yes ✓| H{Strong instrument?}
       H -->|No ✗| I[⚠️ Weak Instrument<br/>Consider Alternatives]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#ffebee
       style F fill:#fff3e0
       style G fill:#fff3e0
       style H fill:#ffebee
       style I fill:#fff3e0

**Agent Decision**:

.. code-block:: text

   ⚠️ Weak Instrument Detected: Consider Alternatives
   
   Problems with Weak IV:
   ❌ First-stage F-statistic = 8.2 (< 10 threshold)
   ❌ Biased estimates in finite samples
   ❌ Invalid inference (confidence intervals too narrow)
   ❌ Sensitive to small violations of exclusion restriction
   
   Recommended Alternatives:
   1. Find stronger instruments
   2. Use limited information maximum likelihood (LIML)
   3. Consider observational methods with rich controls
   4. Collect more data to improve first-stage power

**Side-by-Side Comparison**:

.. list-table:: Strong vs. Weak IV Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Strong Instrument
     - Weak Instrument
   * - First-Stage F
     - 52.3 (strong)
     - 8.2 (weak)
   * - Method Selected
     - Standard IV
     - Alternative methods recommended
   * - Bias Risk
     - Low
     - High (finite sample bias)
   * - Inference
     - Valid
     - Invalid (undersized tests)
   * - Sensitivity
     - Robust
     - Highly sensitive

---

Comparison 5: Good vs. Poor Covariate Overlap
---------------------------------------------

**Scenario**: Evaluating job training program effectiveness

Good Overlap Version
~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Observational data with selection bias
- Rich set of baseline characteristics
- Good overlap in covariate distributions
- Treated and control units across full covariate range

.. mermaid::

   flowchart TD
       A[Good Overlap Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|No ✗| G{Rich covariates?}
       G -->|Yes ✓| H{Good overlap?}
       H -->|Yes ✓| I[Propensity Score<br/>Matching]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#ffebee
       style G fill:#fff3e0
       style H fill:#fff3e0
       style I fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Propensity Score Matching
   
   Reasoning:
   ✓ Rich covariates available
   ✓ Excellent covariate overlap (common support)
   ✓ Can find good matches for most treated units
   ✓ Transparent balance assessment
   
   Expected Results:
   - High-quality matches
   - Good balance on observables
   - Credible causal estimates (if unconfoundedness holds)

Poor Overlap Version
~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Same observational data structure
- Same rich baseline characteristics
- Poor overlap in covariate distributions
- Treated units concentrated in one region of covariate space

.. mermaid::

   flowchart TD
       A[Poor Overlap Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|No ✗| G{Rich covariates?}
       G -->|Yes ✓| H{Good overlap?}
       H -->|No ✗| I[Propensity Score<br/>Weighting]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#ffebee
       style G fill:#fff3e0
       style H fill:#ffebee
       style I fill:#e8f5e8

**Agent Decision**:

.. code-block:: text

   🎯 Selected Method: Propensity Score Weighting
   
   Reasoning:
   ✓ Rich covariates available
   ❌ Poor covariate overlap (limited common support)
   ❌ Matching would discard many observations
   ✓ Weighting can handle poor overlap better
   
   Caveats:
   ⚠️ Extrapolation required (poor overlap)
   ⚠️ High variance in weights possible
   ⚠️ Results may be sensitive to specification

**Side-by-Side Comparison**:

.. list-table:: Good vs. Poor Overlap Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Good Overlap
     - Poor Overlap
   * - Method Selected
     - Propensity Score Matching
     - Propensity Score Weighting
   * - Common Support
     - Excellent
     - Limited
   * - Sample Usage
     - High (good matches)
     - Full sample (with weights)
   * - Extrapolation
     - Minimal
     - Substantial
   * - Variance
     - Lower
     - Higher (extreme weights)

Key Learning Points
------------------

Decision Tree Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Small changes in data characteristics can lead to dramatically different method selections:

1. **Randomization Status**: Completely changes the analysis approach
2. **Data Structure**: Panel vs. cross-sectional determines method families
3. **Instrument Strength**: Weak instruments invalidate IV approaches
4. **Overlap Quality**: Affects choice between matching and weighting

Method Hierarchy
~~~~~~~~~~~~~~~~

CAIS follows a clear hierarchy of method preferences:

1. **Experimental Methods**: Always preferred when randomization is available
2. **Natural Experiments**: RDD and strong IV are next best
3. **Quasi-Experiments**: DiD with credible parallel trends
4. **Observational Methods**: Matching/weighting with rich covariates
5. **Regression Methods**: Last resort with strong assumptions

Assumption Importance
~~~~~~~~~~~~~~~~~~~~

Different methods rely on different assumptions:

- **Randomization**: Minimal assumptions, strongest identification
- **Parallel Trends**: Moderate assumptions, good identification
- **Exclusion Restriction**: Strong assumptions, requires careful validation
- **Unconfoundedness**: Very strong assumptions, often untestable

Practical Implications
~~~~~~~~~~~~~~~~~~~~~

Understanding these comparisons helps with:

1. **Study Design**: Plan data collection to enable better methods
2. **Method Selection**: Understand why CAIS chooses specific approaches
3. **Result Interpretation**: Know the limitations of your selected method
4. **Robustness Checking**: Test sensitivity across similar methods

Next Steps
----------

1. **Apply to Your Data**: Use these comparisons to understand your method selection
2. **Design Better Studies**: Plan data collection to enable stronger methods
3. **Validate Assumptions**: Check key assumptions for your selected method
4. **Explore Alternatives**: Consider how small data changes might improve identification

**Related Resources**:
- :doc:`../case_studies/index` - Detailed case studies by domain
- :doc:`../../methods/decision_tree` - Complete decision tree documentation
- :doc:`dataset_method_gallery` - Visual method selection examples
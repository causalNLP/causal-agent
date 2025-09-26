Dataset Properties and Method Selection Gallery
===============================================

This gallery demonstrates how different dataset characteristics lead CAIS to select different causal inference methods. Each example shows the decision tree path and explains why specific methods are chosen or excluded.

Overview
--------

CAIS uses a systematic decision tree to select the most appropriate causal inference method based on your data characteristics. This gallery provides visual examples of how different data properties lead to different method selections.

**Key Decision Factors**:
- Randomization status
- Data structure (cross-sectional, panel, etc.)
- Treatment variable type (binary, continuous, categorical)
- Available instruments
- Covariate richness and overlap

Gallery Examples
----------------

Example 1: Perfect Randomized Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Randomized controlled trial
- Binary treatment assignment
- Rich baseline covariates
- Perfect compliance

.. mermaid::

   flowchart TD
       A[RCT Dataset] --> B{Is this randomized?}
       B -->|Yes ✓| C{Are covariates available?}
       C -->|Yes ✓| D[Linear Regression<br/>with Covariates]
       
       style A fill:#e3f2fd
       style B fill:#e8f5e8
       style C fill:#fff3e0
       style D fill:#e8f5e8

**Agent Decision Process**:

.. code-block:: text

   🎯 Method Selection: Linear Regression with Covariates
   
   Decision Path:
   1. Randomization check: ✅ PASSED (balanced assignment)
   2. Covariate assessment: ✅ AVAILABLE (baseline measures)
   3. Selected method: Linear regression with covariates
   
   Why this method?
   ✓ Randomization ensures causal identification
   ✓ Covariates improve precision (reduce standard errors)
   ✓ Transparent and interpretable results
   ✓ Optimal for experimental data

**Example Datasets**: Learning mindset intervention, A/B tests, clinical trials

---

Example 2: Observational Data with Rich Covariates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Non-randomized observational study
- Binary treatment
- Rich set of confounding variables
- Good covariate overlap

.. mermaid::

   flowchart TD
       A[Observational Dataset] --> B{Is this randomized?}
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

**Agent Decision Process**:

.. code-block:: text

   🎯 Method Selection: Propensity Score Matching
   
   Decision Path:
   1. Randomization check: ❌ FAILED (selection bias detected)
   2. Panel data check: ❌ NOT AVAILABLE
   3. Running variable check: ❌ NOT AVAILABLE
   4. Treatment type: ✅ BINARY
   5. Instrumental variable: ❌ NOT AVAILABLE
   6. Covariate richness: ✅ RICH COVARIATES
   7. Overlap assessment: ✅ GOOD OVERLAP
   8. Selected method: Propensity score matching
   
   Why this method?
   ✓ Handles selection bias through matching
   ✓ Rich covariates enable credible matching
   ✓ Good overlap ensures valid comparisons
   ✓ Transparent balance assessment

**Example Datasets**: Hospital treatment effects, job training programs, educational interventions

---

Example 3: Panel Data with Treatment Timing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Panel data (multiple time periods)
- Treatment timing varies across units
- Clear before/after periods
- Parallel trends plausible

.. mermaid::

   flowchart TD
       A[Panel Dataset] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|Yes ✓| D{Treatment timing varies?}
       D -->|Yes ✓| E[Difference-in-Differences]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#fff3e0
       style D fill:#fff3e0
       style E fill:#e8f5e8

**Agent Decision Process**:

.. code-block:: text

   🎯 Method Selection: Difference-in-Differences
   
   Decision Path:
   1. Randomization check: ❌ FAILED
   2. Panel data check: ✅ AVAILABLE (multiple time periods)
   3. Treatment timing: ✅ VARIES across units
   4. Selected method: Difference-in-differences
   
   Why this method?
   ✓ Exploits timing variation for identification
   ✓ Controls for time-invariant confounders
   ✓ Handles unobserved heterogeneity
   ✓ Robust to selection on observables and unobservables
   
   Key assumption: Parallel trends between treatment and control

**Example Datasets**: Policy evaluations, minimum wage studies, healthcare reforms

---

Example 4: Sharp Regression Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Continuous running variable
- Sharp cutoff for treatment assignment
- Treatment probability jumps discontinuously
- No manipulation of running variable

.. mermaid::

   flowchart TD
       A[RDD Dataset] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable with cutoff?}
       D -->|Yes ✓| E{Sharp discontinuity?}
       E -->|Yes ✓| F[Regression Discontinuity<br/>Design]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#fff3e0
       style E fill:#fff3e0
       style F fill:#e8f5e8

**Agent Decision Process**:

.. code-block:: text

   🎯 Method Selection: Regression Discontinuity Design
   
   Decision Path:
   1. Randomization check: ❌ FAILED
   2. Panel data check: ❌ NOT AVAILABLE
   3. Running variable: ✅ DETECTED (continuous assignment variable)
   4. Discontinuity: ✅ SHARP (treatment probability jumps)
   5. Selected method: Regression discontinuity design
   
   Why this method?
   ✓ Exploits discontinuous assignment rule
   ✓ Local randomization around cutoff
   ✓ Credible identification strategy
   ✓ Transparent assumptions
   
   Key assumption: Continuity of potential outcomes at cutoff

**Example Datasets**: Scholarship eligibility, policy thresholds, age-based programs

---

Example 5: Instrumental Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Endogenous treatment assignment
- Valid instrumental variable available
- Strong first-stage relationship
- Credible exclusion restriction

.. mermaid::

   flowchart TD
       A[IV Dataset] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|Yes ✓| G{Valid instrument?}
       G -->|Yes ✓| H[Instrumental Variables]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#fff3e0
       style G fill:#fff3e0
       style H fill:#e8f5e8

**Agent Decision Process**:

.. code-block:: text

   🎯 Method Selection: Instrumental Variables
   
   Decision Path:
   1. Randomization check: ❌ FAILED
   2. Panel data check: ❌ NOT AVAILABLE
   3. Running variable check: ❌ NOT AVAILABLE
   4. Treatment type: ✅ BINARY
   5. Instrumental variable: ✅ DETECTED
   6. Instrument validation: ✅ VALID (relevance + exogeneity)
   7. Selected method: Instrumental variables
   
   Why this method?
   ✓ Handles unmeasured confounding
   ✓ Valid instrument provides exogenous variation
   ✓ Strong first-stage relationship
   ✓ Credible exclusion restriction
   
   Key assumptions: Relevance, exogeneity, exclusion restriction

**Example Datasets**: Marketing campaigns with server downtime, education with distance instruments

---

Example 6: Continuous Treatment with IV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Characteristics**:
- Continuous treatment variable
- Endogeneity concerns
- Valid instrumental variable
- No clear cutoff or panel structure

.. mermaid::

   flowchart TD
       A[Continuous Treatment] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|No ✗| F{Continuous treatment}
       F --> G{Instrumental variable?}
       G -->|Yes ✓| H[Instrumental Variables<br/>Continuous Treatment]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#ffebee
       style F fill:#fff3e0
       style G fill:#fff3e0
       style H fill:#e8f5e8

**Agent Decision Process**:

.. code-block:: text

   🎯 Method Selection: IV with Continuous Treatment
   
   Decision Path:
   1. Randomization check: ❌ FAILED
   2. Panel data check: ❌ NOT AVAILABLE
   3. Running variable check: ❌ NOT AVAILABLE
   4. Treatment type: ✅ CONTINUOUS
   5. Instrumental variable: ✅ AVAILABLE
   6. Selected method: IV with continuous treatment
   
   Why this method?
   ✓ Handles continuous endogenous treatment
   ✓ Valid instrument provides identification
   ✓ Can estimate dose-response relationships
   ✓ Flexible functional form specification

**Example Datasets**: Advertising intensity, education years, healthcare dosage

Method Exclusion Examples
-------------------------

Understanding why methods are excluded is as important as understanding why they're selected.

Example 7: Why Not Difference-in-Differences?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset**: Cross-sectional observational data with rich covariates

.. code-block:: text

   ❌ Difference-in-Differences: EXCLUDED
   
   Data Requirements Not Met:
   - Requires: Panel data with multiple time periods
   - Available: Cross-sectional data (single time point)
   - Missing: Pre-treatment outcome measurements
   - Missing: Variation in treatment timing
   
   Alternative Selected: Propensity Score Matching
   - Uses available rich covariates
   - Handles selection bias through matching
   - Appropriate for cross-sectional data

---

Example 8: Why Not Regression Discontinuity?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset**: Observational data without clear assignment rule

.. code-block:: text

   ❌ Regression Discontinuity: EXCLUDED
   
   Data Requirements Not Met:
   - Requires: Continuous running variable with sharp cutoff
   - Available: Discretionary treatment assignment
   - Missing: Clear assignment rule or threshold
   - Problem: No discontinuous treatment probability
   
   Alternative Selected: Propensity Score Methods
   - Handles discretionary assignment
   - Uses observed characteristics for matching
   - Appropriate for non-rule-based assignment

---

Example 9: Why Not Instrumental Variables?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset**: Randomized experiment with perfect compliance

.. code-block:: text

   ❌ Instrumental Variables: EXCLUDED
   
   Not Needed:
   - Randomization already provides identification
   - No endogeneity concerns in experimental data
   - IV would be less efficient than direct analysis
   - Perfect compliance eliminates need for instruments
   
   Selected Method: Linear Regression with Covariates
   - Leverages randomization for identification
   - More efficient than IV approach
   - Simpler interpretation and implementation

Dataset Property Decision Matrix
-------------------------------

This matrix shows how different combinations of data characteristics lead to method selection:

.. list-table:: Method Selection Matrix
   :header-rows: 1
   :widths: 15 15 15 15 15 25

   * - Randomized
     - Panel Data
     - Running Var
     - Instrument
     - Treatment Type
     - Selected Method
   * - ✅ Yes
     - Any
     - Any
     - Any
     - Binary
     - Linear Regression + Covariates
   * - ✅ Yes
     - Any
     - Any
     - Any
     - Continuous
     - Linear Regression + Covariates
   * - ❌ No
     - ✅ Yes
     - Any
     - Any
     - Any
     - Difference-in-Differences
   * - ❌ No
     - ❌ No
     - ✅ Yes
     - Any
     - Any
     - Regression Discontinuity
   * - ❌ No
     - ❌ No
     - ❌ No
     - ✅ Yes
     - Binary
     - Instrumental Variables
   * - ❌ No
     - ❌ No
     - ❌ No
     - ✅ Yes
     - Continuous
     - IV Continuous Treatment
   * - ❌ No
     - ❌ No
     - ❌ No
     - ❌ No
     - Binary
     - Propensity Score Methods
   * - ❌ No
     - ❌ No
     - ❌ No
     - ❌ No
     - Continuous
     - Linear Regression + Controls

Common Decision Patterns
-----------------------

Pattern 1: Experimental Data Priority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rule**: Randomized experiments always preferred when available

.. code-block:: text

   Priority Hierarchy:
   1. Randomized Controlled Trial → Linear Regression + Covariates
   2. Natural Experiment (RDD/IV) → RDD or IV
   3. Quasi-Experiment (DiD) → Difference-in-Differences  
   4. Observational (Matching) → Propensity Score Methods
   5. Observational (Regression) → Linear Regression + Controls

Pattern 2: Data Structure Drives Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rule**: Method selection follows data availability hierarchy

.. code-block:: text

   Data Structure Priority:
   1. Randomization → Experimental methods
   2. Panel + Timing → Difference-in-Differences
   3. Running Variable → Regression Discontinuity
   4. Valid Instrument → Instrumental Variables
   5. Rich Covariates → Propensity Score Methods
   6. Limited Data → Linear Regression

Pattern 3: Treatment Type Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rule**: Treatment variable type affects method choice within categories

.. code-block:: text

   Treatment Type Adaptations:
   - Binary Treatment: Standard methods (matching, IV, etc.)
   - Continuous Treatment: Specialized versions (generalized PS, IV)
   - Categorical Treatment: Multinomial approaches
   - Time-Varying Treatment: Dynamic methods



Next Steps
----------

1. **Apply to Your Data**: Use the decision framework with your datasets
2. **Explore Case Studies**: See detailed examples in :doc:`../case_studies/index`
3. **Read Method Documentation**: Deep dive into specific methods in :doc:`../../methods/index`

**Related Resources**:
- :doc:`../../methods/decision_tree` - Complete decision tree documentation
- :doc:`../case_studies/index` - Detailed case studies by domain
- :doc:`../../getting_started/quickstart` - Quick start guide for CAIS
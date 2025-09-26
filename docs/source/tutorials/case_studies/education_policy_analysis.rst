Education Policy Analysis: Learning Mindset Intervention
=========================================================

This case study demonstrates how CAIS analyzes the causal impact of a growth mindset intervention on student academic performance. We'll walk through the complete agent workflow, from initial data analysis to final results interpretation, showing how the decision tree guides method selection.

Problem Statement
-----------------

**Research Question**: Does a growth mindset intervention improve student academic achievement?

**Context**: A school district implemented a brief online intervention designed to teach students that intellectual abilities can be developed. The intervention was randomly assigned to students, but we need to account for baseline differences and understand the causal mechanism.

**Policy Relevance**: Understanding whether low-cost psychological interventions can improve educational outcomes at scale.

Dataset Overview
----------------

**Source**: Learning mindset intervention study (Yeager et al., 2019)
**Sample Size**: 12,490 students across 65 schools
**Treatment**: Growth mindset intervention (binary)
**Outcome**: Standardized achievement scores
**Key Variables**:

- ``treatment``: Binary indicator for mindset intervention
- ``achievement``: Post-intervention standardized test scores  
- ``baseline_achievement``: Pre-intervention test scores
- ``school_id``: School identifier
- ``demographics``: Student background characteristics

.. code-block:: python

   import pandas as pd
   from causal_agent import CausalAgent
   
   # Load the learning mindset dataset
   df = pd.read_csv('data/all_data/learning_mindset.csv')
   
   print("Dataset shape:", df.shape)
   print("\nFirst few rows:")
   print(df.head())
   
   print("\nTreatment distribution:")
   print(df['treatment'].value_counts())

Agent Decision-Making Process
-----------------------------

Let's trace through how CAIS analyzes this dataset and selects the appropriate causal method.

Step 1: Initial Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent first examines the dataset structure and identifies key characteristics:

.. code-block:: python

   agent = CausalAgent()
   
   # Agent performs automatic data analysis
   analysis_result = agent.analyze(
       data=df,
       treatment='treatment',
       outcome='achievement',
       verbose=True  # Show decision-making process
   )

**Agent Output**:

.. code-block:: text

   🔍 CAIS Data Analysis
   =====================
   
   Dataset Characteristics:
   - Sample size: 12,490 observations
   - Treatment variable: 'treatment' (binary)
   - Outcome variable: 'achievement' (continuous)
   - Missing values: 0.3% (manageable)
   
   Treatment Assignment Analysis:
   - Treatment group: 6,320 students (50.6%)
   - Control group: 6,170 students (49.4%)
   - Assignment appears balanced
   
   Covariate Analysis:
   - Baseline achievement available: ✓
   - Demographic variables: ✓
   - School-level clustering: ✓
   
   🤔 Assessing Randomization...
   - Balance test p-values: 0.23 (baseline), 0.45 (demographics)
   - Randomization check: PASSED ✓

Step 2: Decision Tree Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent navigates the decision tree based on detected data characteristics:

.. mermaid::

   flowchart TD
       A[Learning Mindset Data] --> B{Is this randomized?}
       B -->|Yes ✓| C{Are covariates available?}
       C -->|Yes ✓| D[Linear Regression<br/>with Covariates]
       
       style A fill:#e3f2fd
       style B fill:#fff3e0
       style C fill:#fff3e0
       style D fill:#e8f5e8

**Decision Logic**:

1. **Randomization Check**: ✅ PASSED
   - Treatment assignment is balanced across observable characteristics
   - No systematic differences in baseline covariates
   - Conclusion: This is a randomized experiment

2. **Covariate Assessment**: ✅ COVARIATES AVAILABLE
   - Baseline achievement scores available
   - Rich demographic information
   - School-level identifiers for clustering
   - Conclusion: Include covariates for precision gains

3. **Method Selection**: **Linear Regression with Covariates**
   - Primary method: Leverages randomization for identification
   - Covariates: Improve precision and handle any residual imbalance
   - Clustering: Account for school-level correlation

**Agent Reasoning**:

.. code-block:: text

   🎯 Method Selection: Linear Regression with Covariates
   
   Why this method?
   ✓ Randomization ensures causal identification
   ✓ Covariates improve precision (reduce standard errors)
   ✓ Can handle clustered data structure
   ✓ Transparent and interpretable results
   
   Alternative methods considered:
   - Difference in Means: Less precise (ignores covariates)
   - Propensity Score: Unnecessary (already randomized)
   - Instrumental Variables: Not needed (no endogeneity)

Step 3: Method Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent implements the selected method with appropriate specifications:

.. code-block:: python

   # Agent automatically implements the analysis
   results = analysis_result.get_results()
   
   print("Selected Method:", results.method)
   print("Specification:", results.specification)
   print("\nResults Summary:")
   print(results.summary())

**Implementation Details**:

.. code-block:: text

   📊 Analysis Implementation
   ==========================
   
   Method: Linear Regression with Covariates
   Specification:
   - Outcome: achievement
   - Treatment: treatment (binary)
   - Covariates: baseline_achievement, demographics
   - Clustering: Robust standard errors by school_id
   - Sample: Full sample (N=12,490)
   
   Model Equation:
   achievement_i = β₀ + β₁×treatment_i + β₂×baseline_i + β₃×demographics_i + ε_i

Step 4: Results and Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causal Effect Estimate**:

.. code-block:: text

   🎯 Causal Effect Results
   ========================
   
   Treatment Effect: +0.127 standard deviations
   95% Confidence Interval: [0.089, 0.165]
   P-value: < 0.001
   
   Interpretation:
   The growth mindset intervention increases student achievement by 
   approximately 0.13 standard deviations on average. This is a 
   statistically significant and educationally meaningful effect size.

**Robustness Checks**:

The agent automatically performs several robustness checks:

.. code-block:: python

   # Agent provides robustness analysis
   robustness = analysis_result.get_robustness_checks()
   
   for check in robustness:
       print(f"{check.name}: {check.result}")

.. code-block:: text

   🔍 Robustness Checks
   ====================
   
   ✓ Balance Check: Treatment groups balanced on observables
   ✓ Sensitivity Analysis: Results stable across specifications
   ✓ Subgroup Analysis: Effects consistent across demographics
   ✓ Placebo Tests: No effects on pre-treatment outcomes
   
   Alternative Method Comparison:
   - Difference in Means: +0.134 [0.096, 0.172] ✓ Similar
   - Matching: +0.125 [0.087, 0.163] ✓ Similar
   - Conclusion: Results robust across methods

Decision Tree Walkthrough
-------------------------

Let's examine how different dataset characteristics would lead to different method selections:

Scenario Comparison: What If This Wasn't Randomized?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical Scenario**: Same data, but treatment was not randomly assigned.

.. mermaid::

   flowchart TD
       A[Non-Randomized Version] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|No ✗| G{Rich covariates?}
       G -->|Yes ✓| H{Good overlap?}
       H -->|Yes ✓| I[Propensity Score<br/>Matching]
       
       style A fill:#ffebee
       style B fill:#fff3e0
       style I fill:#f3e5f5

**Alternative Decision Path**:

If this were observational data, the agent would:

1. Check for panel structure → Not available
2. Look for regression discontinuity → No running variable
3. Assess instrumental variables → None available
4. Evaluate covariates → Rich covariates available
5. Check overlap → Good covariate overlap
6. **Select**: Propensity Score Matching

**Why Different Method?**:
- Without randomization, need to control for selection bias
- Rich covariates allow credible matching approach
- Good overlap ensures valid comparisons

Method Exclusion Examples
-------------------------

The agent also excludes inappropriate methods. Here's why certain methods weren't selected:

Difference-in-Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**: 
- No panel data structure (single post-treatment measurement)
- No variation in treatment timing
- Cannot identify parallel trends

**Agent Logic**:

.. code-block:: text

   ❌ Difference-in-Differences: EXCLUDED
   
   Reason: Insufficient data structure
   - Requires: Panel data with treatment timing variation
   - Available: Cross-sectional post-treatment data only
   - Conclusion: Cannot implement DiD design

Instrumental Variables
~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:
- No valid instruments available
- Randomization already provides identification
- Would be less efficient than direct analysis

**Agent Logic**:

.. code-block:: text

   ❌ Instrumental Variables: EXCLUDED
   
   Reason: Not needed and no valid instruments
   - Randomization provides identification
   - No instruments that satisfy exclusion restriction
   - Would reduce precision unnecessarily

Regression Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:
- No running variable with treatment cutoff
- Treatment assignment was randomized, not rule-based

**Agent Logic**:

.. code-block:: text

   ❌ Regression Discontinuity: EXCLUDED
   
   Reason: No discontinuous treatment assignment
   - Requires: Running variable with sharp cutoff
   - Available: Random assignment mechanism
   - Conclusion: RDD design not applicable

Real-World Implications
-----------------------

Policy Recommendations
~~~~~~~~~~~~~~~~~~~~~~

Based on the causal analysis:

**Effect Size**: 0.127 standard deviations
**Cost-Effectiveness**: Very high (low-cost intervention)
**Scalability**: High (online delivery possible)
**Recommendation**: Implement intervention district-wide

**Caveats and Limitations**:

1. **External Validity**: Results from specific school contexts
2. **Long-term Effects**: Only measured immediate post-treatment
3. **Mechanism**: Unclear which components drive the effect
4. **Heterogeneity**: May vary across student populations

Comparison with Alternative Approaches
--------------------------------------

Traditional Analysis vs. CAIS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Traditional Approach**:
- Researcher manually selects method
- May miss important robustness checks
- Prone to specification searching
- Limited systematic validation

**CAIS Approach**:
- Systematic method selection based on data characteristics
- Automatic robustness checking
- Transparent decision-making process
- Comprehensive sensitivity analysis

**Advantages of CAIS**:

1. **Consistency**: Same data → same method selection
2. **Transparency**: Clear reasoning for method choice
3. **Robustness**: Automatic validation and sensitivity checks
4. **Efficiency**: Rapid analysis with best practices

Side-by-Side Method Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's compare how different methods would analyze this same dataset:

.. code-block:: python

   # Compare multiple methods on same data
   methods_comparison = agent.compare_methods(
       data=df,
       treatment='treatment',
       outcome='achievement',
       methods=['linear_regression', 'diff_in_means', 'propensity_matching']
   )
   
   print(methods_comparison.summary_table())

**Results Comparison**:

.. list-table:: Method Comparison Results
   :header-rows: 1
   :widths: 25 20 25 30

   * - Method
     - Effect Size
     - 95% CI
     - Notes
   * - Linear Regression (Selected)
     - +0.127
     - [0.089, 0.165]
     - Most precise (uses covariates)
   * - Difference in Means
     - +0.134
     - [0.096, 0.172]
     - Valid but less precise
   * - Propensity Matching
     - +0.125
     - [0.087, 0.163]
     - Unnecessary (already randomized)

**Key Insights**:
- All methods give similar point estimates (good sign!)
- Linear regression most precise (narrowest confidence interval)
- Consistency across methods increases confidence in results

Learning Objectives Achieved
-----------------------------

After working through this case study, you should understand:

✅ **Decision Tree Navigation**: How data characteristics guide method selection

✅ **Randomization Benefits**: Why randomized experiments simplify causal inference

✅ **Covariate Usage**: How covariates improve precision in randomized studies

✅ **Method Exclusion Logic**: Why certain methods are inappropriate for specific data structures

✅ **Robustness Checking**: How to validate causal findings across specifications

✅ **Policy Interpretation**: How to translate causal estimates into actionable insights

Next Steps
----------

1. **Try Alternative Specifications**: Experiment with different covariate sets
2. **Explore Subgroup Effects**: Analyze heterogeneous treatment effects
3. **Compare with Observational Methods**: See how results change without randomization
4. **Read Method Documentation**: Deep dive into :doc:`../methods/experimental/randomized_controlled_trials`

**Related Case Studies**:
- :doc:`healthcare_treatment_effects` - Observational study with matching
- :doc:`economic_policy_impact` - Regression discontinuity design
- :doc:`marketing_campaign_evaluation` - Instrumental variables approach

**Download Materials**:
- `Learning Mindset Dataset <../../../data/all_data/learning_mindset.csv>`_
- `Complete Analysis Notebook <../notebooks/education_analysis_tutorial.ipynb>`_
- `Replication Code <https://github.com/cais-project/case-studies>`_
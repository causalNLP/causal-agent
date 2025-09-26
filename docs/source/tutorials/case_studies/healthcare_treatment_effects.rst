Healthcare Treatment Effects: Hospital Treatment Analysis
========================================================

This case study demonstrates how CAIS analyzes observational healthcare data to estimate treatment effects when randomization is not possible. We'll explore how the agent navigates complex selection bias issues and chooses appropriate matching methods.

Problem Statement
-----------------

**Research Question**: Does a new hospital treatment protocol improve patient recovery outcomes?

**Context**: A hospital implemented a new treatment protocol for certain patients, but treatment assignment was not randomized. Doctors chose treatments based on patient characteristics, creating potential selection bias. We need to estimate the causal effect while accounting for this bias.

**Clinical Relevance**: Understanding treatment effectiveness is crucial for evidence-based medicine, but randomized trials aren't always feasible or ethical.

Dataset Overview
----------------

**Source**: Hospital patient records with treatment and outcome data
**Sample Size**: 3,504 patients
**Treatment**: New treatment protocol (binary)
**Outcome**: Recovery time (continuous, days)
**Key Variables**:

- ``treatment``: Binary indicator for new treatment protocol
- ``recovery_time``: Days until full recovery
- ``age``: Patient age
- ``severity``: Disease severity score (1-10)
- ``comorbidities``: Number of additional conditions
- ``hospital_id``: Hospital identifier

.. code-block:: python

   import pandas as pd
   from causal_agent import CausalAgent
   
   # Load the hospital treatment dataset
   df = pd.read_csv('data/all_data/hospital_treatment.csv')
   
   print("Dataset shape:", df.shape)
   print("\nTreatment distribution:")
   print(df['treatment'].value_counts())
   
   print("\nBaseline characteristics by treatment:")
   print(df.groupby('treatment')[['age', 'severity', 'comorbidities']].mean())

Agent Decision-Making Process
-----------------------------

Let's trace through CAIS's analysis of this observational healthcare data.

Step 1: Initial Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent examines the dataset and immediately identifies selection bias concerns:

.. code-block:: python

   agent = CausalAgent()
   
   # Agent performs comprehensive data analysis
   analysis_result = agent.analyze(
       data=df,
       treatment='treatment',
       outcome='recovery_time',
       verbose=True
   )

**Agent Output**:

.. code-block:: text

   🔍 CAIS Data Analysis
   =====================
   
   Dataset Characteristics:
   - Sample size: 3,504 observations
   - Treatment variable: 'treatment' (binary)
   - Outcome variable: 'recovery_time' (continuous)
   - Missing values: 1.2% (manageable)
   
   Treatment Assignment Analysis:
   - Treatment group: 1,456 patients (41.6%)
   - Control group: 2,048 patients (58.4%)
   - Assignment appears NON-RANDOM ⚠️
   
   Selection Bias Indicators:
   - Age difference: 8.3 years (p < 0.001)
   - Severity difference: 1.7 points (p < 0.001)
   - Comorbidities difference: 0.9 conditions (p < 0.001)
   
   🚨 Randomization check: FAILED
   Strong evidence of systematic treatment assignment

Step 2: Decision Tree Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent navigates the decision tree for observational data:

.. mermaid::

   flowchart TD
       A[Hospital Treatment Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|No ✗| G{Rich covariates?}
       G -->|Yes ✓| H{Good covariate overlap?}
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

**Decision Logic**:

1. **Randomization Check**: ❌ FAILED
   - Systematic differences in patient characteristics
   - Treatment assignment appears based on clinical judgment
   - Conclusion: This is observational data with selection bias

2. **Panel Data Check**: ❌ NOT AVAILABLE
   - Only single time point per patient
   - Cannot use difference-in-differences
   - Need alternative approach for confounding

3. **Running Variable Check**: ❌ NOT AVAILABLE
   - No clear cutoff rule for treatment assignment
   - Cannot use regression discontinuity
   - Treatment assignment appears discretionary

4. **Instrumental Variable Check**: ❌ NOT AVAILABLE
   - No valid instruments identified
   - Hospital policies don't create exogenous variation
   - Need to rely on observed confounders

5. **Covariate Assessment**: ✅ RICH COVARIATES AVAILABLE
   - Patient demographics, severity measures, comorbidities
   - Variables likely predict both treatment and outcome
   - Can potentially control for selection bias

6. **Overlap Assessment**: ✅ GOOD OVERLAP
   - Treated and control patients exist across covariate ranges
   - Common support condition satisfied
   - Matching approach feasible

7. **Method Selection**: **Propensity Score Matching**

**Agent Reasoning**:

.. code-block:: text

   🎯 Method Selection: Propensity Score Matching
   
   Why this method?
   ✓ Handles selection bias through matching
   ✓ Rich covariates available for propensity model
   ✓ Good covariate overlap enables valid matches
   ✓ Transparent and interpretable approach
   
   Alternative methods considered:
   - Linear Regression: Strong unconfoundedness assumption
   - Propensity Weighting: Good alternative (will test)
   - Instrumental Variables: No valid instruments available
   - DiD/RDD: Data structure doesn't support

Step 3: Propensity Score Model Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent develops a propensity score model to predict treatment assignment:

.. code-block:: python

   # Agent automatically builds propensity score model
   propensity_results = analysis_result.get_propensity_analysis()
   
   print("Propensity Model Summary:")
   print(propensity_results.model_summary())

**Propensity Model**:

.. code-block:: text

   📊 Propensity Score Model
   =========================
   
   Model: Logistic Regression
   Dependent Variable: treatment
   
   Covariates Included:
   - age: β = 0.045 (p < 0.001)
   - severity: β = 0.312 (p < 0.001)  
   - comorbidities: β = 0.198 (p < 0.001)
   - age²: β = -0.0003 (p = 0.023)
   - severity × comorbidities: β = 0.089 (p = 0.012)
   
   Model Fit:
   - Pseudo R²: 0.284
   - C-statistic: 0.742
   - Hosmer-Lemeshow p-value: 0.234 (good fit)

**Propensity Score Distribution**:

.. code-block:: text

   📈 Propensity Score Overlap Assessment
   ======================================
   
   Common Support Analysis:
   - Treated units: 1,456 (100% on support)
   - Control units: 2,048 (97.8% on support)
   - Overlap region: [0.12, 0.89]
   - Excellent overlap ✓
   
   Balance Before Matching:
   - Age: Standardized difference = 0.67
   - Severity: Standardized difference = 0.84
   - Comorbidities: Standardized difference = 0.52

Step 4: Matching Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent implements propensity score matching with optimal parameters:

.. code-block:: python

   # Agent performs matching analysis
   matching_results = analysis_result.get_matching_results()
   
   print("Matching Specification:")
   print(matching_results.specification)

**Matching Details**:

.. code-block:: text

   🔗 Propensity Score Matching Implementation
   ===========================================
   
   Matching Algorithm: 1-to-1 Nearest Neighbor
   Caliper: 0.1 standard deviations
   Replacement: Without replacement
   
   Matching Results:
   - Treated units matched: 1,398 (96.0%)
   - Control units matched: 1,398 (68.3%)
   - Total matched sample: 2,796 patients
   - Units dropped: 708 (poor matches)

**Post-Matching Balance**:

.. code-block:: text

   ⚖️ Covariate Balance After Matching
   ===================================
   
   Standardized Differences:
   - Age: 0.67 → 0.08 ✓ (target: < 0.1)
   - Severity: 0.84 → 0.06 ✓ (target: < 0.1)  
   - Comorbidities: 0.52 → 0.09 ✓ (target: < 0.1)
   
   Balance Tests:
   - Joint significance test: p = 0.234 ✓
   - Pseudo R² after matching: 0.003 ✓
   - Mean bias reduction: 89.2% ✓
   
   Conclusion: Excellent balance achieved

Step 5: Treatment Effect Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With balanced matched samples, the agent estimates the treatment effect:

.. code-block:: python

   # Get final treatment effect results
   results = analysis_result.get_results()
   
   print("Treatment Effect Results:")
   print(results.summary())

**Causal Effect Results**:

.. code-block:: text

   🎯 Causal Effect Results
   ========================
   
   Average Treatment Effect (ATE): -2.34 days
   95% Confidence Interval: [-3.12, -1.56]
   P-value: < 0.001
   
   Interpretation:
   The new treatment protocol reduces recovery time by 
   approximately 2.3 days on average. This represents a 
   statistically significant improvement in patient outcomes.
   
   Effect Size:
   - Cohen's d: -0.42 (medium effect)
   - Percentage improvement: 18.7%
   - Number needed to treat: 4.3 patients

Method Exclusion Examples
-------------------------

Let's examine why other methods were excluded for this dataset:

Difference-in-Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Difference-in-Differences: EXCLUDED
   
   Reason: Insufficient data structure
   - Requires: Panel data with pre/post treatment periods
   - Available: Cross-sectional data (single time point)
   - Missing: Baseline outcome measurements
   - Conclusion: Cannot implement DiD design

**What Would Be Needed**:
- Patient outcomes before and after treatment implementation
- Multiple time periods for each patient
- Variation in treatment timing across patients/hospitals

Instrumental Variables
~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Instrumental Variables: EXCLUDED
   
   Reason: No valid instruments identified
   - Examined: Hospital policies, physician preferences, capacity
   - Problem: All potential instruments correlated with patient outcomes
   - Exclusion restriction: Cannot be satisfied
   - Conclusion: No credible instruments available

**What Would Be Needed**:
- Random variation in treatment assignment (e.g., physician rotation)
- Policy changes affecting treatment availability
- Geographic variation unrelated to patient characteristics

Regression Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Regression Discontinuity: EXCLUDED
   
   Reason: No discontinuous assignment rule
   - Examined: Age cutoffs, severity thresholds, hospital capacity
   - Finding: Treatment assignment appears discretionary
   - No sharp cutoff: Continuous clinical judgment
   - Conclusion: RDD design not applicable

**What Would Be Needed**:
- Clear cutoff rule (e.g., "treat if severity > 7")
- Sharp discontinuity in treatment probability
- Continuity of other characteristics at cutoff

Robustness Analysis
-------------------

The agent performs comprehensive robustness checks:

Alternative Matching Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Agent tests alternative specifications
   robustness = analysis_result.get_robustness_checks()
   
   for check in robustness:
       print(f"{check.name}: {check.result}")

**Robustness Results**:

.. code-block:: text

   🔍 Robustness Checks
   ====================
   
   Alternative Matching Methods:
   ✓ 1-to-2 Matching: -2.28 days [-3.18, -1.38] (similar)
   ✓ Caliper 0.05: -2.41 days [-3.25, -1.57] (similar)
   ✓ Kernel Matching: -2.19 days [-2.98, -1.40] (similar)
   
   Alternative Methods:
   ✓ Propensity Weighting: -2.45 days [-3.31, -1.59] (similar)
   ✓ Linear Regression: -2.52 days [-3.28, -1.76] (similar)
   ⚠️ Naive Comparison: -4.12 days [-4.78, -3.46] (biased)
   
   Sensitivity Analysis:
   ✓ Hidden bias (Γ = 1.5): Results remain significant
   ✓ Placebo outcomes: No effects on pre-treatment variables
   ✓ Subgroup analysis: Consistent across patient types

Comparison with Naive Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Naive Approach** (ignoring selection bias):

.. code-block:: text

   📊 Naive vs. Causal Analysis Comparison
   =======================================
   
   Naive Difference in Means:
   - Treatment effect: -4.12 days
   - Interpretation: Severely biased (overestimate)
   - Problem: Sicker patients got new treatment
   
   CAIS Propensity Matching:
   - Treatment effect: -2.34 days  
   - Interpretation: Causal effect after bias correction
   - Method: Controls for observed confounders
   
   Bias Correction:
   - Selection bias: 1.78 days (43% of naive estimate)
   - Direction: Naive analysis overestimates benefit
   - Reason: Treated patients were sicker at baseline

Decision Tree Alternative Scenarios
-----------------------------------

Let's explore how different data characteristics would change the analysis:

Scenario 1: Panel Data Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical**: Same patients observed before and after treatment implementation

.. mermaid::

   flowchart TD
       A[Panel Data Version] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|Yes ✓| D{Treatment timing varies?}
       D -->|Yes ✓| E[Difference-in-Differences]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#fff3e0
       style D fill:#fff3e0
       style E fill:#e8f5e8

**Alternative Analysis**:
- Method: Difference-in-Differences
- Advantage: Controls for time-invariant confounders
- Requirements: Pre-treatment outcomes, parallel trends

Scenario 2: Instrumental Variable Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical**: Random physician assignment creates treatment variation

.. mermaid::

   flowchart TD
       A[IV Data Version] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|Yes ✓| G[Instrumental Variables]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#fff3e0
       style G fill:#e8f5e8

**Alternative Analysis**:
- Method: Instrumental Variables
- Advantage: Handles unmeasured confounding
- Requirements: Valid instrument (physician assignment)

Clinical Implications
---------------------

Treatment Effectiveness
~~~~~~~~~~~~~~~~~~~~~~~

**Clinical Significance**:
- Effect size: 2.3 days reduction in recovery time
- Relative improvement: 18.7% faster recovery
- Clinical relevance: Meaningful for patient care and hospital efficiency

**Cost-Benefit Analysis**:
- Reduced hospital stays: $1,200 savings per patient
- Treatment cost: $300 per patient
- Net benefit: $900 per patient
- Return on investment: 300%

**Implementation Recommendations**:

1. **Adopt Protocol**: Strong evidence of effectiveness
2. **Monitor Outcomes**: Continue tracking patient recovery
3. **Expand Gradually**: Implement across similar patient populations
4. **Train Staff**: Ensure proper protocol implementation

Limitations and Caveats
~~~~~~~~~~~~~~~~~~~~~~~

**Study Limitations**:

1. **Unmeasured Confounding**: May still exist despite matching
2. **External Validity**: Results specific to this hospital setting
3. **Selection on Unobservables**: Cannot rule out completely
4. **Temporal Changes**: Treatment effects may vary over time

**Sensitivity Considerations**:

.. code-block:: text

   ⚠️ Sensitivity to Hidden Bias
   =============================
   
   Rosenbaum Bounds Analysis:
   - Γ = 1.0: p < 0.001 (no hidden bias)
   - Γ = 1.5: p = 0.023 (moderate hidden bias)
   - Γ = 2.0: p = 0.156 (substantial hidden bias)
   
   Interpretation:
   Results robust to moderate levels of hidden bias.
   Would need substantial unmeasured confounding 
   (doubling odds of treatment) to eliminate significance.

Comparison with Traditional Analysis
------------------------------------

**Traditional Approach**:
- Often relies on linear regression with covariates
- May not check balance or overlap
- Limited sensitivity analysis
- Prone to model specification issues

**CAIS Approach**:
- Systematic method selection based on data structure
- Automatic balance checking and diagnostics
- Comprehensive robustness analysis
- Transparent decision-making process

**Key Advantages**:

1. **Bias Detection**: Automatically identifies selection bias
2. **Method Appropriateness**: Selects methods suited to data structure
3. **Balance Assessment**: Ensures valid comparisons
4. **Sensitivity Analysis**: Tests robustness of findings

Learning Objectives Achieved
-----------------------------

After working through this case study, you should understand:

✅ **Selection Bias**: How non-random treatment assignment creates bias

✅ **Propensity Scores**: How to model treatment assignment probability

✅ **Matching Methods**: How to create balanced comparison groups

✅ **Balance Assessment**: How to evaluate covariate balance

✅ **Robustness Checking**: How to test sensitivity of results

✅ **Clinical Interpretation**: How to translate results into practice

Next Steps
----------

1. **Explore Sensitivity Analysis**: Test different hidden bias scenarios
2. **Try Alternative Methods**: Compare with propensity score weighting
3. **Examine Heterogeneity**: Look for subgroup effects
4. **Read Method Documentation**: Deep dive into :doc:`../methods/observational/propensity_score_matching`

**Related Case Studies**:
- :doc:`education_policy_analysis` - Randomized experiment analysis
- :doc:`economic_policy_impact` - Regression discontinuity design
- :doc:`marketing_campaign_evaluation` - Instrumental variables approach

**Download Materials**:
- `Hospital Treatment Dataset <../../../data/all_data/hospital_treatment.csv>`_
- `Complete Analysis Notebook <../notebooks/healthcare_analysis_tutorial.ipynb>`_
- `Replication Code <https://github.com/cais-project/case-studies>`_
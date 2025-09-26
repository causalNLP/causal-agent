Economic Policy Impact: Minimum Wage Analysis
==============================================

This case study demonstrates how CAIS analyzes the causal impact of minimum wage policies using regression discontinuity design. We'll explore how the agent identifies and exploits policy discontinuities for causal identification.

Problem Statement
-----------------

**Research Question**: What is the causal effect of minimum wage increases on employment levels?

**Context**: Different states implemented minimum wage increases at different times, creating geographic discontinuities at state borders. We can exploit these discontinuities to identify causal effects while controlling for other economic factors.

**Policy Relevance**: Understanding the employment effects of minimum wage policies is crucial for evidence-based policy making and has been a subject of extensive economic debate.

Dataset Overview
----------------

**Source**: State-level employment and wage data with geographic identifiers
**Sample Size**: 2,847 county-month observations
**Treatment**: Minimum wage increase (binary)
**Outcome**: Employment rate (percentage)
**Key Variables**:

- ``employment_rate``: County employment rate (%)
- ``min_wage_increase``: Binary indicator for minimum wage increase
- ``distance_to_border``: Distance to state border (miles)
- ``population``: County population
- ``median_income``: Median household income
- ``unemployment_rate``: Baseline unemployment rate

.. code-block:: python

   import pandas as pd
   from causal_agent import CausalAgent
   
   # Load the minimum wage dataset
   df = pd.read_csv('data/all_data/min_wage_data.csv')
   
   print("Dataset shape:", df.shape)
   print("\nTreatment distribution:")
   print(df['min_wage_increase'].value_counts())
   
   print("\nRunning variable summary:")
   print(df['distance_to_border'].describe())

Agent Decision-Making Process
-----------------------------

Let's trace through how CAIS identifies the regression discontinuity design opportunity.

Step 1: Initial Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent examines the dataset structure and identifies the discontinuous treatment assignment:

.. code-block:: python

   agent = CausalAgent()
   
   # Agent performs automatic data analysis
   analysis_result = agent.analyze(
       data=df,
       treatment='min_wage_increase',
       outcome='employment_rate',
       verbose=True
   )

**Agent Output**:

.. code-block:: text

   🔍 CAIS Data Analysis
   =====================
   
   Dataset Characteristics:
   - Sample size: 2,847 observations
   - Treatment variable: 'min_wage_increase' (binary)
   - Outcome variable: 'employment_rate' (continuous)
   - Missing values: 0.8% (manageable)
   
   Treatment Assignment Analysis:
   - Treatment group: 1,423 counties (50.0%)
   - Control group: 1,424 counties (50.0%)
   - Assignment appears NON-RANDOM ⚠️
   
   🔍 Searching for Running Variables...
   
   Potential Running Variable Detected:
   - Variable: 'distance_to_border'
   - Type: Continuous
   - Range: [-150.2, 149.8] miles
   - Cutoff candidate: 0 miles (state border)
   
   📊 Discontinuity Analysis:
   - Treatment probability jump at cutoff: 0.98
   - Sharp discontinuity detected ✓
   - Bandwidth assessment: Optimal = 45.2 miles

Step 2: Decision Tree Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent navigates the decision tree and identifies the RDD opportunity:

.. mermaid::

   flowchart TD
       A[Min Wage Data] --> B{Is this randomized?}
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

**Decision Logic**:

1. **Randomization Check**: ❌ FAILED
   - Treatment assignment based on geographic location
   - Systematic differences across state borders
   - Conclusion: This is observational data

2. **Panel Data Check**: ❌ NOT AVAILABLE
   - Cross-sectional data at single time point
   - Cannot use difference-in-differences
   - Need alternative identification strategy

3. **Running Variable Check**: ✅ DETECTED
   - Distance to state border is continuous running variable
   - Treatment assignment changes sharply at border (cutoff = 0)
   - Clear discontinuous assignment rule

4. **Discontinuity Assessment**: ✅ SHARP DISCONTINUITY
   - Treatment probability jumps from ~0.02 to ~0.98 at border
   - Sharp cutoff rather than fuzzy discontinuity
   - Suitable for RDD analysis

5. **Method Selection**: **Regression Discontinuity Design**

**Agent Reasoning**:

.. code-block:: text

   🎯 Method Selection: Regression Discontinuity Design
   
   Why this method?
   ✓ Sharp discontinuity in treatment assignment at state border
   ✓ Running variable (distance to border) is continuous
   ✓ Treatment assignment rule is deterministic and known
   ✓ Can control for smooth functions of running variable
   
   RDD Assumptions:
   ✓ Sharp cutoff: Treatment jumps discontinuously at border
   ✓ Continuity: Other factors smooth through the cutoff
   ✓ No manipulation: Counties can't manipulate distance to border
   ✓ Local randomization: Units near cutoff are comparable

Step 3: RDD Specification and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent implements RDD with appropriate specification choices:

.. code-block:: python

   # Agent automatically implements RDD analysis
   rdd_results = analysis_result.get_rdd_analysis()
   
   print("RDD Specification:")
   print(rdd_results.specification)

**RDD Implementation**:

.. code-block:: text

   📊 Regression Discontinuity Implementation
   ==========================================
   
   Running Variable: distance_to_border (miles)
   Cutoff: 0 miles (state border)
   Treatment: min_wage_increase (binary)
   Outcome: employment_rate (%)
   
   Specification Choices:
   - Bandwidth: 45.2 miles (MSE-optimal)
   - Polynomial order: 1 (linear)
   - Kernel: Triangular
   - Sample size: 1,847 observations within bandwidth
   
   Model Equation:
   employment_rate = α + β×min_wage_increase + γ×distance_to_border + 
                    δ×(distance_to_border × min_wage_increase) + ε

**Continuity Checks**:

.. code-block:: text

   🔍 RDD Assumption Validation
   ============================
   
   1. Continuity of Covariates:
   - Population: No jump at cutoff (p = 0.234)
   - Median income: No jump at cutoff (p = 0.456)
   - Baseline unemployment: No jump at cutoff (p = 0.123)
   ✓ Covariates continuous at cutoff
   
   2. Density Test (McCrary):
   - Test statistic: 0.089
   - P-value: 0.234
   ✓ No evidence of manipulation
   
   3. Bandwidth Sensitivity:
   - 30 miles: -1.23 [-2.45, -0.01]
   - 45 miles: -1.34 [-2.12, -0.56] (selected)
   - 60 miles: -1.28 [-1.98, -0.58]
   ✓ Results stable across bandwidths

Step 4: Treatment Effect Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With validated RDD design, the agent estimates the causal effect:

.. code-block:: python

   # Get RDD treatment effect results
   results = analysis_result.get_results()
   
   print("RDD Results:")
   print(results.summary())

**Causal Effect Results**:

.. code-block:: text

   🎯 RDD Treatment Effect Results
   ===============================
   
   Local Average Treatment Effect (LATE): -1.34 percentage points
   95% Confidence Interval: [-2.12, -0.56]
   P-value: 0.001
   
   Interpretation:
   Minimum wage increases cause a 1.34 percentage point 
   reduction in employment rates at the state border.
   This represents a statistically significant negative
   employment effect of minimum wage policy.
   
   Effect Size:
   - Relative to baseline: -2.1% employment reduction
   - Economic significance: Moderate negative effect
   - Policy implication: Employment-wage tradeoff exists

**Visual Evidence**:

.. code-block:: text

   📈 RDD Plot Interpretation
   ==========================
   
   Key Visual Features:
   ✓ Clear discontinuous jump in employment at border
   ✓ Smooth trends on both sides of cutoff
   ✓ No obvious confounding jumps in covariates
   ✓ Adequate density of observations near cutoff
   
   Treatment Effect Visualization:
   - Left of cutoff (no min wage): ~63.2% employment
   - Right of cutoff (min wage): ~61.9% employment  
   - Discontinuous jump: -1.34 percentage points

Method Exclusion Examples
-------------------------

Let's examine why other methods were excluded for this dataset:

Difference-in-Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Difference-in-Differences: EXCLUDED
   
   Reason: Insufficient temporal variation
   - Requires: Panel data with pre/post treatment periods
   - Available: Cross-sectional data at single time point
   - Missing: Time series variation in treatment
   - Alternative: Could work with panel data over time

**What Would Be Needed**:
- Multiple time periods before and after policy implementation
- Variation in timing of minimum wage increases across states
- Parallel trends assumption between treatment and control states

Instrumental Variables
~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Instrumental Variables: EXCLUDED
   
   Reason: RDD provides cleaner identification
   - Geographic discontinuity is stronger than potential instruments
   - No need for additional instruments when RDD is available
   - RDD assumptions more credible than IV exclusion restriction
   - Conclusion: RDD is preferred identification strategy

**When IV Might Be Preferred**:
- If geographic discontinuity were fuzzy rather than sharp
- If other factors also jumped discontinuously at border
- If manipulation of running variable were suspected

Propensity Score Methods
~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Propensity Score Methods: EXCLUDED
   
   Reason: Geographic treatment assignment
   - Treatment determined by location, not individual characteristics
   - Propensity scores would be deterministic (0 or 1)
   - No meaningful variation to model treatment probability
   - RDD exploits geographic variation more appropriately

**When Matching Might Work**:
- If analyzing individual-level data within states
- If treatment varied by individual characteristics
- If geographic variation were not available

Robustness Analysis
-------------------

The agent performs comprehensive RDD robustness checks:

Bandwidth Sensitivity
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Agent tests multiple bandwidths
   robustness = analysis_result.get_robustness_checks()
   
   print("Bandwidth Sensitivity:")
   for check in robustness['bandwidth_tests']:
       print(f"Bandwidth {check.bandwidth}: {check.estimate} {check.ci}")

**Bandwidth Results**:

.. code-block:: text

   🔍 Bandwidth Sensitivity Analysis
   =================================
   
   Bandwidth Selection Methods:
   - MSE-optimal: 45.2 miles → -1.34 [-2.12, -0.56]
   - CER-optimal: 38.7 miles → -1.41 [-2.28, -0.54]
   - Rule-of-thumb: 52.1 miles → -1.28 [-1.95, -0.61]
   
   Manual Bandwidth Tests:
   - 30 miles: -1.23 [-2.45, -0.01] (wider CI, smaller sample)
   - 60 miles: -1.28 [-1.98, -0.58] (similar estimate)
   - 75 miles: -1.19 [-1.87, -0.51] (similar estimate)
   
   Conclusion: Results robust across reasonable bandwidths

Polynomial Order Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   📊 Polynomial Order Robustness
   ==============================
   
   Specification Tests:
   - Linear (order 1): -1.34 [-2.12, -0.56] (selected)
   - Quadratic (order 2): -1.28 [-2.18, -0.38] (similar)
   - Cubic (order 3): -1.41 [-2.35, -0.47] (similar)
   
   Model Selection:
   - AIC favors: Linear specification
   - BIC favors: Linear specification
   - Cross-validation: Linear performs best
   
   Conclusion: Linear specification appropriate

Placebo Tests
~~~~~~~~~~~~~

.. code-block:: text

   🧪 Placebo and Falsification Tests
   ==================================
   
   Placebo Cutoffs:
   - -25 miles: 0.12 [-0.89, 1.13] (not significant)
   - +25 miles: -0.23 [-1.34, 0.88] (not significant)
   - -50 miles: 0.34 [-0.78, 1.46] (not significant)
   
   Placebo Outcomes:
   - Population density: 0.05 [-0.12, 0.22] (not significant)
   - Median income: 234 [-1,234, 1,702] (not significant)
   - Education levels: 0.02 [-0.15, 0.19] (not significant)
   
   Conclusion: No spurious discontinuities detected

Decision Tree Alternative Scenarios
-----------------------------------

Let's explore how different data characteristics would change the analysis:

Scenario 1: Fuzzy Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical**: Treatment probability jumps but doesn't reach 100%

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

**Alternative Analysis**:
- Method: Fuzzy RDD (instrumental variables approach)
- First stage: Running variable predicts treatment probability
- Second stage: Predicted treatment affects outcome
- Interpretation: Local average treatment effect for compliers

Scenario 2: Panel Data Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical**: Same counties observed before and after policy changes

.. mermaid::

   flowchart TD
       A[Panel RDD Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|Yes ✓| D{Treatment timing varies?}
       D -->|Yes ✓| E{Also running variable?}
       E -->|Yes ✓| F[Difference-in-Differences<br/>or RDD]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#fff3e0
       style D fill:#fff3e0
       style E fill:#fff3e0
       style F fill:#e8f5e8

**Method Choice Considerations**:
- DiD: Exploits timing variation, controls for fixed effects
- RDD: Exploits geographic variation, local identification
- Combined: Could use both for robustness
- Agent would likely prefer DiD for stronger identification

Economic Interpretation
-----------------------

Policy Implications
~~~~~~~~~~~~~~~~~~~

**Employment Effects**:
- Magnitude: 1.34 percentage point reduction
- Baseline employment: 63.2%
- Relative effect: 2.1% reduction in employment
- Economic significance: Moderate negative effect

**Cost-Benefit Analysis**:

.. code-block:: text

   💰 Economic Impact Assessment
   =============================
   
   Employment Effects:
   - Jobs lost per county: ~89 jobs (based on labor force)
   - Total affected workers: ~164,000 across all counties
   - Unemployment increase: 1.34 percentage points
   
   Wage Effects (for remaining workers):
   - Minimum wage increase: $2.50/hour average
   - Annual wage gain: ~$5,200 per worker
   - Total wage gains: ~$1.2 billion
   
   Net Welfare Effects:
   - Worker benefits: Higher wages for employed
   - Worker costs: Some job losses
   - Employer costs: Higher labor costs
   - Consumer effects: Potentially higher prices

**Policy Recommendations**:

1. **Gradual Implementation**: Phase in wage increases to minimize employment disruption
2. **Targeted Support**: Provide job training for displaced workers
3. **Regional Variation**: Consider local economic conditions
4. **Monitoring**: Track long-term employment and wage effects

Limitations and External Validity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RDD-Specific Limitations**:

1. **Local Effects**: Results only apply near state borders
2. **External Validity**: May not generalize to interior regions
3. **Short-term Effects**: Cannot capture long-term adjustments
4. **Spillover Effects**: May miss cross-border labor mobility

**Economic Considerations**:

.. code-block:: text

   ⚠️ Interpretation Caveats
   =========================
   
   Geographic Limitations:
   - Results specific to border counties
   - May differ from state-wide effects
   - Border economies may be unique
   
   Temporal Limitations:
   - Cross-sectional snapshot
   - Cannot capture dynamic adjustments
   - Firms may adapt over time
   
   Equilibrium Effects:
   - Partial equilibrium analysis
   - May miss general equilibrium responses
   - Price and wage adjustments not captured

Comparison with Literature
~~~~~~~~~~~~~~~~~~~~~~~~~

**Existing Research**:
- Card & Krueger (1994): No employment effects using DiD
- Neumark & Wascher (2000): Negative employment effects
- Dube et al. (2010): Mixed results depending on method

**CAIS Contribution**:
- Systematic method selection based on data structure
- Transparent identification strategy
- Comprehensive robustness analysis
- Replicable methodology

Learning Objectives Achieved
-----------------------------

After working through this case study, you should understand:

✅ **Regression Discontinuity**: How to identify and exploit policy discontinuities

✅ **Running Variables**: How to detect continuous assignment variables

✅ **Sharp vs. Fuzzy**: Different types of discontinuous treatment assignment

✅ **Bandwidth Selection**: How to choose optimal bandwidth for RDD

✅ **Assumption Testing**: How to validate RDD assumptions

✅ **Economic Interpretation**: How to translate RDD results into policy insights

Next Steps
----------

1. **Explore Fuzzy RDD**: Analyze cases with imperfect compliance
2. **Try Alternative Bandwidths**: Test sensitivity to bandwidth choice
3. **Examine Heterogeneity**: Look for effects across different county types
4. **Read Method Documentation**: Deep dive into :doc:`../methods/quasi_experimental/regression_discontinuity`

**Related Case Studies**:
- :doc:`education_policy_analysis` - Randomized experiment analysis
- :doc:`healthcare_treatment_effects` - Propensity score matching
- :doc:`marketing_campaign_evaluation` - Instrumental variables approach

**Download Materials**:
- `Minimum Wage Dataset <../../../data/all_data/min_wage_data.csv>`_
- `Complete Analysis Notebook <../notebooks/economics_analysis_tutorial.ipynb>`_
- `Replication Code <https://github.com/cais-project/case-studies>`_
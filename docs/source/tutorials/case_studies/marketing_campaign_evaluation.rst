Marketing Campaign Evaluation: Instrumental Variables Analysis
==============================================================

This case study demonstrates how CAIS uses instrumental variables to measure the causal impact of advertising campaigns when randomization isn't possible and selection bias is a concern. We'll explore how the agent identifies valid instruments and handles endogeneity issues.

Problem Statement
-----------------

**Research Question**: What is the causal effect of digital advertising exposure on customer purchase behavior?

**Context**: A company wants to measure the effectiveness of their digital advertising campaign, but ad exposure is not randomly assigned. Customers who see more ads may be systematically different (more engaged, higher income, different preferences), creating endogeneity bias in naive comparisons.

**Business Relevance**: Understanding true advertising effectiveness is crucial for optimal marketing budget allocation and ROI measurement.

Dataset Overview
----------------

**Source**: Customer behavior data with advertising exposure and purchase outcomes
**Sample Size**: 8,742 customers
**Treatment**: Digital advertising exposure (continuous, number of ads seen)
**Outcome**: Monthly purchase amount (continuous, dollars)
**Instrument**: Random server downtime affecting ad delivery
**Key Variables**:

- ``purchase_amount``: Monthly purchase amount ($)
- ``ad_exposure``: Number of ads seen in the month
- ``server_downtime``: Hours of random server downtime (instrument)
- ``customer_engagement``: Historical engagement score
- ``income_level``: Customer income category
- ``demographics``: Age, location, etc.

.. code-block:: python

   import pandas as pd
   from causal_agent import CausalAgent
   
   # Load the marketing campaign dataset
   df = pd.read_csv('data/all_data/billboard_impact.csv')
   
   print("Dataset shape:", df.shape)
   print("\nAd exposure distribution:")
   print(df['ad_exposure'].describe())
   
   print("\nInstrument (server downtime) distribution:")
   print(df['server_downtime'].describe())
   
   print("\nCorrelation between instrument and treatment:")
   print(f"Correlation: {df['server_downtime'].corr(df['ad_exposure']):.3f}")

Agent Decision-Making Process
-----------------------------

Let's trace through how CAIS identifies the instrumental variables opportunity and validates the instrument.

Step 1: Initial Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent examines the dataset and detects endogeneity concerns:

.. code-block:: python

   agent = CausalAgent()
   
   # Agent performs comprehensive data analysis
   analysis_result = agent.analyze(
       data=df,
       treatment='ad_exposure',
       outcome='purchase_amount',
       verbose=True
   )

**Agent Output**:

.. code-block:: text

   🔍 CAIS Data Analysis
   =====================
   
   Dataset Characteristics:
   - Sample size: 8,742 observations
   - Treatment variable: 'ad_exposure' (continuous)
   - Outcome variable: 'purchase_amount' (continuous)
   - Missing values: 1.1% (manageable)
   
   Treatment Assignment Analysis:
   - Treatment range: 0-47 ads per month
   - Mean exposure: 12.3 ads
   - Distribution: Right-skewed (some heavy users)
   
   Endogeneity Assessment:
   🚨 ENDOGENEITY CONCERNS DETECTED
   
   Selection Bias Indicators:
   - High-engagement customers see more ads (r = 0.67)
   - Higher-income customers see more ads (r = 0.43)
   - Ad exposure correlated with unobserved preferences
   
   🔍 Searching for Instrumental Variables...
   
   Potential Instrument Detected:
   - Variable: 'server_downtime'
   - Relevance: Corr(server_downtime, ad_exposure) = -0.34
   - Exogeneity: Random technical failures
   - Strength: F-statistic = 89.2 (> 10 threshold)

Step 2: Decision Tree Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent navigates the decision tree for continuous treatment with endogeneity:

.. mermaid::

   flowchart TD
       A[Marketing Campaign Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|No ✗| F{Continuous treatment}
       F --> G{Instrumental variable?}
       G -->|Yes ✓| H{Valid instrument?}
       H -->|Yes ✓| I[Instrumental Variables<br/>Continuous Treatment]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#ffebee
       style F fill:#fff3e0
       style G fill:#fff3e0
       style H fill:#fff3e0
       style I fill:#e8f5e8

**Decision Logic**:

1. **Randomization Check**: ❌ FAILED
   - Ad exposure not randomly assigned
   - Systematic differences in exposure across customers
   - Conclusion: Observational data with selection bias

2. **Panel Data Check**: ❌ NOT AVAILABLE
   - Cross-sectional data at single time point
   - Cannot use difference-in-differences
   - Need alternative identification strategy

3. **Running Variable Check**: ❌ NOT AVAILABLE
   - No clear cutoff rule for ad exposure
   - Continuous variation without discontinuities
   - Cannot use regression discontinuity

4. **Treatment Type**: ✅ CONTINUOUS TREATMENT
   - Ad exposure varies continuously (0-47 ads)
   - Not binary treatment variable
   - Need methods for continuous endogenous variables

5. **Instrumental Variable Check**: ✅ DETECTED
   - Server downtime affects ad exposure but not purchases directly
   - Random technical failures provide exogenous variation
   - Strong first-stage relationship (F > 10)

6. **Instrument Validation**: ✅ VALID INSTRUMENT
   - Relevance: Strong correlation with treatment
   - Exogeneity: Random server failures
   - Exclusion: Affects purchases only through ad exposure

7. **Method Selection**: **Instrumental Variables (Continuous Treatment)**

**Agent Reasoning**:

.. code-block:: text

   🎯 Method Selection: Instrumental Variables
   
   Why this method?
   ✓ Handles endogeneity bias from unobserved confounders
   ✓ Valid instrument available (server downtime)
   ✓ Continuous treatment requires IV approach
   ✓ Can identify local average treatment effects
   
   IV Assumptions Satisfied:
   ✓ Relevance: Server downtime strongly predicts ad exposure
   ✓ Exogeneity: Random technical failures are uncorrelated with customer characteristics
   ✓ Exclusion: Server downtime affects purchases only through reduced ad exposure
   ✓ Monotonicity: Downtime reduces ad exposure for all affected customers

Step 3: Instrument Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent performs comprehensive instrument validation:

.. code-block:: python

   # Agent automatically validates the instrument
   iv_diagnostics = analysis_result.get_iv_diagnostics()
   
   print("Instrument Validation Results:")
   print(iv_diagnostics.summary())

**Instrument Validation**:

.. code-block:: text

   📊 Instrumental Variable Validation
   ===================================
   
   1. Relevance Condition:
   First-Stage Regression: ad_exposure ~ server_downtime + controls
   - Coefficient: -0.67 (p < 0.001)
   - F-statistic: 89.2 ✓ (> 10 threshold)
   - R²: 0.23
   - Conclusion: Strong instrument ✓
   
   2. Exogeneity Assessment:
   Balance Tests (server_downtime vs. customer characteristics):
   - Customer engagement: p = 0.234 ✓
   - Income level: p = 0.456 ✓
   - Demographics: p = 0.123 ✓
   - Historical purchases: p = 0.345 ✓
   - Conclusion: Instrument appears exogenous ✓
   
   3. Exclusion Restriction:
   Reduced-Form Test: purchase_amount ~ server_downtime + controls
   - Direct effect: 0.12 (p = 0.234) - not significant ✓
   - Indirect effect (through ads): 2.34 (p < 0.001) ✓
   - Conclusion: Exclusion restriction plausible ✓

**First-Stage Results**:

.. code-block:: text

   🎯 First-Stage Regression Results
   =================================
   
   Dependent Variable: ad_exposure
   
   Key Coefficients:
   - server_downtime: -0.67 [-0.82, -0.52] (p < 0.001)
   - customer_engagement: 8.23 [7.45, 9.01] (p < 0.001)
   - income_level: 2.34 [1.89, 2.79] (p < 0.001)
   
   Model Fit:
   - R²: 0.234
   - F-statistic: 89.2 (strong instrument)
   - Observations: 8,742
   
   Interpretation:
   Each hour of server downtime reduces ad exposure by 0.67 ads on average.
   The instrument is strong and relevant.

Step 4: IV Estimation
~~~~~~~~~~~~~~~~~~~~~

With validated instrument, the agent estimates the causal effect:

.. code-block:: python

   # Get IV treatment effect results
   results = analysis_result.get_results()
   
   print("IV Results:")
   print(results.summary())

**IV Treatment Effect Results**:

.. code-block:: text

   🎯 IV Treatment Effect Results
   ==============================
   
   Two-Stage Least Squares (2SLS) Results:
   
   Second-Stage: purchase_amount ~ ad_exposure_hat + controls
   
   Causal Effect of Ad Exposure:
   - Coefficient: $3.47 per additional ad
   - 95% Confidence Interval: [$2.12, $4.82]
   - P-value: < 0.001
   
   Interpretation:
   Each additional ad exposure increases monthly purchase 
   amount by $3.47 on average. This represents the causal 
   effect for customers whose ad exposure is affected by 
   server downtime (compliers).
   
   Effect Size:
   - Baseline purchases: $127.50
   - Relative effect: 2.7% increase per ad
   - At mean exposure (12.3 ads): $42.68 total effect

Method Exclusion Examples
-------------------------

Let's examine why other methods were excluded for this dataset:

Linear Regression
~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Linear Regression: EXCLUDED
   
   Reason: Severe endogeneity bias
   - Ad exposure correlated with unobserved customer preferences
   - High-engagement customers both see more ads and purchase more
   - Naive regression would overestimate advertising effects
   - Bias direction: Positive (upward bias)
   
   Naive OLS Result: $5.23 per ad [biased upward]
   IV Result: $3.47 per ad [causal effect]
   Bias: $1.76 per ad (51% overestimate)

**Endogeneity Sources**:
- Customer engagement affects both ad exposure and purchases
- Income levels correlate with both variables
- Unobserved preferences create selection bias

Propensity Score Methods
~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Propensity Score Methods: EXCLUDED
   
   Reason: Continuous treatment variable
   - Propensity scores designed for binary treatments
   - Ad exposure varies continuously (0-47 ads)
   - Generalized propensity scores possible but complex
   - IV provides cleaner identification with valid instrument

**Alternative Approach**:
- Could discretize ad exposure into categories
- Use generalized propensity score methods
- But IV is preferred with strong instrument available

Difference-in-Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

**Why Excluded**:

.. code-block:: text

   ❌ Difference-in-Differences: EXCLUDED
   
   Reason: No temporal variation in treatment
   - Cross-sectional data at single time point
   - No before/after campaign comparison
   - No variation in campaign timing across customers
   - Would need panel data with campaign rollout variation

**What Would Be Needed**:
- Multiple time periods before and after campaign
- Staggered campaign rollout across customer segments
- Parallel trends assumption between treatment and control groups

Robustness Analysis
-------------------

The agent performs comprehensive IV robustness checks:

Instrument Strength Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Agent tests instrument strength
   robustness = analysis_result.get_robustness_checks()
   
   print("Instrument Strength Tests:")
   for test in robustness['strength_tests']:
       print(f"{test.name}: {test.result}")

**Strength Test Results**:

.. code-block:: text

   🔍 Instrument Strength Analysis
   ===============================
   
   Weak Instrument Tests:
   ✓ First-stage F-statistic: 89.2 (> 10 threshold)
   ✓ Cragg-Donald statistic: 87.4 (> critical value)
   ✓ Kleibergen-Paap statistic: 85.1 (robust to heteroskedasticity)
   
   Conclusion: Strong instrument, no weak instrument concerns
   
   Alternative Instruments (if available):
   - Geographic variation in server infrastructure
   - Random A/B testing of ad delivery algorithms
   - Seasonal variation in server capacity

Overidentification Tests
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   🧪 Overidentification and Specification Tests
   =============================================
   
   Hansen J-Test (if multiple instruments):
   - Not applicable (single instrument)
   - Would test if multiple instruments give consistent results
   
   Endogeneity Test (Hausman):
   - Test statistic: 12.34
   - P-value: 0.002
   - Conclusion: OLS and IV significantly different ✓
   - Confirms endogeneity bias in OLS
   
   Sargan Test (alternative specification):
   - Test statistic: 2.45
   - P-value: 0.234
   - Conclusion: Instrument validity not rejected ✓

Alternative IV Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   📊 Alternative IV Specifications
   ================================
   
   Different Control Sets:
   - Minimal controls: $3.52 [$2.18, $4.86] (similar)
   - Full controls: $3.47 [$2.12, $4.82] (selected)
   - Kitchen sink: $3.41 [$2.05, $4.77] (similar)
   
   Different Functional Forms:
   - Linear: $3.47 [$2.12, $4.82] (selected)
   - Log-linear: 2.8% increase per ad (similar)
   - Quadratic: Diminishing returns detected
   
   Conclusion: Results robust across specifications

Comparison with Naive Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bias Decomposition**:

.. code-block:: text

   📊 Naive vs. Causal Analysis Comparison
   =======================================
   
   Naive OLS Regression:
   - Estimated effect: $5.23 per ad
   - Problem: Endogeneity bias from customer selection
   - Direction: Overestimates true effect
   
   IV Estimation:
   - Causal effect: $3.47 per ad
   - Method: Uses random server downtime as instrument
   - Interpretation: True causal effect for compliers
   
   Bias Analysis:
   - Selection bias: $1.76 per ad (51% of naive estimate)
   - Bias direction: Positive (upward bias)
   - Reason: Engaged customers see more ads and buy more

Decision Tree Alternative Scenarios
-----------------------------------

Let's explore how different data characteristics would change the analysis:

Scenario 1: Binary Treatment Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical**: Campaign exposure as binary (exposed vs. not exposed)

.. mermaid::

   flowchart TD
       A[Binary Campaign Data] --> B{Is this randomized?}
       B -->|No ✗| C{Panel data available?}
       C -->|No ✗| D{Running variable?}
       D -->|No ✗| E{Binary treatment?}
       E -->|Yes ✓| F{Instrumental variable?}
       F -->|Yes ✓| G[Instrumental Variables<br/>Binary Treatment]
       
       style A fill:#e3f2fd
       style B fill:#ffebee
       style C fill:#ffebee
       style D fill:#ffebee
       style E fill:#fff3e0
       style F fill:#fff3e0
       style G fill:#e8f5e8

**Alternative Analysis**:
- Method: IV with binary treatment
- Interpretation: Local average treatment effect (LATE)
- Compliers: Customers affected by server downtime
- Simpler interpretation than continuous treatment case

Scenario 2: Panel Data Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hypothetical**: Same customers observed before and after campaign

.. mermaid::

   flowchart TD
       A[Panel Campaign Data] --> B{Is this randomized?}
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
- Advantage: Controls for time-invariant customer characteristics
- Requirements: Pre-campaign baseline, staggered rollout
- May be preferred if parallel trends assumption holds

Business Implications
---------------------

Marketing ROI Analysis
~~~~~~~~~~~~~~~~~~~~~~

**Advertising Effectiveness**:
- Causal effect: $3.47 per ad exposure
- Cost per ad impression: $0.15
- Return on investment: 2,213% ($3.47/$0.15)
- Break-even: 1 ad exposure per customer

**Campaign Optimization**:

.. code-block:: text

   💰 Marketing ROI Analysis
   =========================
   
   Cost-Benefit Calculation:
   - Cost per ad: $0.15
   - Revenue per ad: $3.47
   - Net profit per ad: $3.32
   - ROI: 2,213%
   
   Optimal Campaign Intensity:
   - Current average: 12.3 ads per customer
   - Current profit: $40.84 per customer
   - Potential for expansion: High ROI suggests under-investment
   
   Budget Allocation:
   - Increase ad frequency for high-value customers
   - Expand reach to similar customer segments
   - Invest in server infrastructure to reduce downtime

**Heterogeneous Effects**:

.. code-block:: python

   # Analyze effects by customer segments
   heterogeneity = analysis_result.get_heterogeneity_analysis()
   
   print("Effects by Customer Segment:")
   print(heterogeneity.by_segment())

.. code-block:: text

   📊 Heterogeneous Treatment Effects
   ==================================
   
   Effects by Customer Segment:
   - High engagement: $4.23 per ad [3.12, 5.34]
   - Medium engagement: $3.47 per ad [2.12, 4.82]
   - Low engagement: $2.15 per ad [0.89, 3.41]
   
   Effects by Income Level:
   - High income: $5.67 per ad [4.23, 7.11]
   - Medium income: $3.47 per ad [2.12, 4.82]
   - Low income: $1.89 per ad [0.45, 3.33]
   
   Targeting Implications:
   - Focus on high-engagement, high-income customers
   - Diminishing returns for low-engagement segments
   - Customize ad frequency by customer type

Limitations and Caveats
~~~~~~~~~~~~~~~~~~~~~~~

**IV-Specific Limitations**:

1. **Local Effects**: Results apply only to compliers (customers affected by server downtime)
2. **External Validity**: May not generalize to all customers
3. **Exclusion Restriction**: Assumes server downtime only affects purchases through ads
4. **Monotonicity**: Assumes downtime reduces ad exposure for all affected customers

**Business Considerations**:

.. code-block:: text

   ⚠️ Business Implementation Caveats
   ==================================
   
   Complier Population:
   - Results apply to customers affected by server issues
   - May not represent all customers equally
   - High-engagement customers more likely to be compliers
   
   Long-term Effects:
   - Analysis captures short-term purchase response
   - Long-term brand building effects not measured
   - Customer lifetime value implications unclear
   
   Competitive Response:
   - Assumes competitors don't respond to increased advertising
   - Market-level effects may differ from individual effects
   - Advertising arms race potential

Comparison with Traditional Analysis
------------------------------------

**Traditional Marketing Analytics**:
- Often relies on correlation analysis
- May not account for selection bias
- Limited causal interpretation
- Prone to overestimating advertising effects

**CAIS IV Approach**:
- Systematic identification of endogeneity issues
- Automatic instrument detection and validation
- Comprehensive robustness testing
- Clear causal interpretation with caveats

**Key Advantages**:

1. **Bias Correction**: Accounts for customer selection effects
2. **Causal Identification**: Provides true causal estimates
3. **Transparency**: Clear assumptions and limitations
4. **Robustness**: Comprehensive sensitivity analysis

Learning Objectives Achieved
-----------------------------

After working through this case study, you should understand:

✅ **Endogeneity Problems**: How selection bias affects causal inference

✅ **Instrumental Variables**: How to identify and validate instruments

✅ **IV Assumptions**: Relevance, exogeneity, and exclusion restriction

✅ **Continuous Treatment**: How IV works with non-binary treatments

✅ **LATE Interpretation**: Local average treatment effects for compliers

✅ **Business Applications**: How to translate IV results into marketing strategy

Next Steps
----------

1. **Explore Weak Instruments**: Understand problems with weak instruments
2. **Try Multiple Instruments**: Analyze overidentification tests
3. **Examine Complier Characteristics**: Understand who is affected by the instrument
4. **Read Method Documentation**: Deep dive into :doc:`../methods/quasi_experimental/instrumental_variables`

**Related Case Studies**:
- :doc:`education_policy_analysis` - Randomized experiment analysis
- :doc:`healthcare_treatment_effects` - Propensity score matching
- :doc:`economic_policy_impact` - Regression discontinuity design

**Download Materials**:
- `Marketing Campaign Dataset <../../../data/all_data/billboard_impact.csv>`_
- `Complete Analysis Notebook <../notebooks/marketing_analysis_tutorial.ipynb>`_
- `Replication Code <https://github.com/cais-project/case-studies>`_
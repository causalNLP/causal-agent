Method Selection and Decision-Making
====================================

This section explains how the CAIS agent selects the most appropriate causal inference method for your data and research question. Understanding this process helps you interpret the agent's decisions and assess the credibility of your results.

The Method Selection Challenge
-----------------------------

Choosing the right causal inference method is one of the most critical decisions in any analysis. The wrong method can lead to:

* **Biased estimates**: Results that systematically over- or under-estimate true effects
* **Invalid inferences**: Conclusions that don't hold up to scrutiny
* **Wasted resources**: Time and money spent on unreliable analysis

**Traditional Approach**: Researchers manually review methodological literature, assess data characteristics, and make subjective decisions about method appropriateness.

**CAIS Approach**: The agent systematically evaluates all available methods, tests assumptions, and selects the approach with the strongest identification strategy for your specific context.

The Agent's Decision Framework
-----------------------------

The agent uses a multi-stage decision process that considers:

1. **Identification Strength**: How convincingly does the method establish causation?
2. **Assumption Plausibility**: How likely are the method's assumptions to hold?
3. **Data Requirements**: Does your data satisfy the method's requirements?
4. **Robustness**: How sensitive are results to assumption violations?

Decision Tree Overview
~~~~~~~~~~~~~~~~~~~~~

The agent follows a structured decision tree with multiple pathways:

.. mermaid::

   flowchart TD
       A[Data Analysis] --> B{Treatment Assignment}
       B -->|Random| C[Experimental Methods]
       B -->|As-Good-As-Random| D[Quasi-Experimental Methods]
       B -->|Non-Random| E[Observational Methods]
       
       C --> C1[Randomized Controlled Trial]
       C --> C2[Natural Experiment]
       
       D --> D1{Panel Data?}
       D1 -->|Yes| D2{Policy Change?}
       D2 -->|Yes| D3[Difference-in-Differences]
       D2 -->|No| D4{Discontinuity?}
       D4 -->|Yes| D5[Regression Discontinuity]
       D4 -->|No| D6[Panel Methods]
       D1 -->|No| D7{Instrument Available?}
       D7 -->|Yes| D8[Instrumental Variables]
       D7 -->|No| D9{Discontinuity?}
       D9 -->|Yes| D5
       D9 -->|No| E
       
       E --> E1[Propensity Score Methods]
       E --> E2[Backdoor Adjustment]
       E --> E3[Linear Regression]

**Key Insight**: The agent doesn't just follow rigid rules. It uses contextual reasoning to navigate edge cases and make nuanced decisions.

Stage 1: Treatment Assignment Analysis
-------------------------------------

The first and most crucial step is understanding how treatment was assigned.

Random Assignment (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Indicators the Agent Looks For**:
   * Variables named "random_assignment", "treatment_group", "randomized"
   * Balanced baseline characteristics between treatment and control
   * Study documentation mentioning randomization
   * Equal treatment probabilities across subgroups

**Agent Decision Process**:

.. code-block:: text

   Agent: "I detect a 'randomization_indicator' variable and observe 
   balanced baseline characteristics (p-values > 0.1 for all covariates). 
   This appears to be a randomized experiment. I'll use experimental 
   methods and focus on intention-to-treat effects."

**Selected Methods**:
   * Simple difference in means
   * Regression with baseline controls (for precision)
   * Instrumental variables (if non-compliance exists)

As-Good-As-Random Assignment (Quasi-Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What the Agent Looks For**:
   * Policy changes or natural experiments
   * Discontinuities in treatment assignment
   * Temporal variation in treatment timing
   * Geographic variation in treatment rollout

**Example Scenarios**:

**Policy Change**:
.. code-block:: text

   Agent: "I see a policy implemented in 2018 with pre- and post-policy 
   data across multiple states. Some states implemented earlier than others. 
   This is a difference-in-differences setup."

**Discontinuity**:
.. code-block:: text

   Agent: "Treatment assignment changes sharply at age 65, with a clear 
   cutoff. This suggests regression discontinuity design is appropriate."

**Natural Experiment**:
.. code-block:: text

   Agent: "Treatment varies based on lottery numbers or random assignment 
   to bureaucrats. This creates as-good-as-random variation suitable for 
   instrumental variables."

Non-Random Assignment (Observational)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When This Occurs**:
   * Individuals self-select into treatment
   * Treatment assigned based on characteristics
   * No clear source of exogenous variation

**Agent Response**:
.. code-block:: text

   Agent: "Treatment appears to be chosen by individuals based on their 
   characteristics. I'll need to use methods that control for selection 
   bias, such as propensity score matching or backdoor adjustment."

Stage 2: Data Structure Analysis
-------------------------------

The agent analyzes your data structure to determine which methods are feasible.

Panel Data Detection
~~~~~~~~~~~~~~~~~~~

**What the Agent Looks For**:
   * Multiple observations per unit over time
   * Time-varying treatments or outcomes
   * Variables indicating time periods or waves

**Implications for Method Selection**:
   * Enables difference-in-differences methods
   * Allows for unit fixed effects
   * Supports dynamic treatment effect analysis

**Agent Decision**:
.. code-block:: text

   Agent: "I detect panel structure with individuals observed over 5 years. 
   This enables me to control for time-invariant confounders using fixed 
   effects methods."

Cross-Sectional Data
~~~~~~~~~~~~~~~~~~~

**Characteristics**:
   * Single observation per unit
   * No temporal variation
   * Limited to methods that don't require time dimension

**Available Methods**:
   * Instrumental variables (if instruments available)
   * Regression discontinuity (if discontinuity exists)
   * Propensity score methods
   * Linear regression with controls

Time Series Data
~~~~~~~~~~~~~~~

**Special Considerations**:
   * Temporal dependencies in data
   * Potential for dynamic effects
   * Need for time series methods

**Agent Approach**:
.. code-block:: text

   Agent: "This is time series data with a single treated unit. I'll 
   consider synthetic control methods to construct an appropriate 
   counterfactual."

Stage 3: Method Prioritization
-----------------------------

The agent ranks methods based on identification strength and assumption plausibility.

Identification Strength Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tier 1: Experimental Methods**
   * Randomized controlled trials
   * Natural experiments with random assignment

**Tier 2: Strong Quasi-Experimental Methods**
   * Sharp regression discontinuity
   * Difference-in-differences with parallel trends
   * Strong instrumental variables

**Tier 3: Moderate Quasi-Experimental Methods**
   * Fuzzy regression discontinuity
   * Difference-in-differences with concerns
   * Weak instrumental variables

**Tier 4: Observational Methods**
   * Propensity score methods with good overlap
   * Backdoor adjustment with rich controls
   * Linear regression

**Agent Prioritization Logic**:

.. code-block:: text

   Agent: "Multiple methods are applicable:
   1. RDD (Tier 2): Strong identification, clear discontinuity
   2. IV (Tier 3): Instrument available but exclusion restriction uncertain  
   3. Matching (Tier 4): Good overlap but selection on unobservables concern
   
   I'll implement RDD as primary method with IV and matching as robustness checks."

Assumption Assessment
~~~~~~~~~~~~~~~~~~~

For each potential method, the agent evaluates assumption plausibility:

**Difference-in-Differences Example**:
.. code-block:: text

   Agent: "Assessing DiD assumptions:
   ✓ Parallel trends: Pre-treatment trends similar (p=0.23)
   ✓ No spillovers: Geographically separated treatment/control
   ? Stable composition: Some migration between regions
   
   Overall assessment: STRONG identification with minor composition concerns.
   Recommendation: Include robustness check excluding high-migration areas."

**Instrumental Variables Example**:
.. code-block:: text

   Agent: "Assessing IV assumptions:
   ✓ Relevance: F-stat = 47.3 (strong instrument)
   ? Exclusion restriction: Instrument may affect outcome through other channels
   ✓ Monotonicity: Likely satisfied based on institutional context
   
   Overall assessment: MODERATE identification due to exclusion restriction concerns.
   Recommendation: Use as robustness check, not primary method."

Stage 4: Robustness and Sensitivity Analysis
-------------------------------------------

The agent doesn't just select one method—it implements a comprehensive robustness strategy.

Multiple Method Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Primary Method**: Strongest identification strategy
**Secondary Methods**: Alternative approaches for robustness
**Sensitivity Analyses**: Tests of key assumptions

**Example Strategy**:
.. code-block:: text

   Agent: "Implementing comprehensive analysis:
   
   PRIMARY: Difference-in-differences
   - Main specification with state and year fixed effects
   - Clustered standard errors at state level
   
   ROBUSTNESS CHECKS:
   - Alternative time windows (exclude early/late periods)
   - Placebo tests (fake treatment dates)
   - Synthetic control method
   - Event study specification
   
   SENSITIVITY ANALYSES:
   - Different control group definitions
   - Varying treatment intensity measures
   - Subgroup analysis by key characteristics"

Assumption Testing Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent systematically tests method assumptions:

**Pre-Analysis Tests**:
   * Balance tests for randomized experiments
   * Parallel trends tests for difference-in-differences
   * Density tests for regression discontinuity
   * Instrument strength tests for IV methods

**Post-Analysis Validation**:
   * Placebo tests with fake treatments
   * Sensitivity to specification changes
   * Outlier influence analysis
   * Cross-validation with alternative methods

**Example Testing Sequence**:
.. code-block:: text

   Agent: "Running assumption tests:
   
   1. Parallel trends test: PASSED (p=0.34)
   2. Placebo test (pre-treatment): PASSED (effect ≈ 0)
   3. Sensitivity to controls: ROBUST (effect stable)
   4. Outlier analysis: ROBUST (no influential observations)
   
   Conclusion: Strong evidence for causal interpretation."

Stage 5: Method Selection Decision
---------------------------------

The agent synthesizes all information to make final method selection.

Decision Integration
~~~~~~~~~~~~~~~~~~

The agent considers:
   * Statistical test results
   * Assumption plausibility assessments
   * Data quality and sample size
   * Robustness across specifications
   * External validity considerations

**Final Decision Process**:
.. code-block:: text

   Agent: "FINAL METHOD SELECTION:
   
   PRIMARY METHOD: Difference-in-Differences
   - Strong identification strategy
   - Assumptions well-supported by tests
   - Robust across multiple specifications
   - Good external validity for policy evaluation
   
   CONFIDENCE LEVEL: HIGH
   - Multiple assumption tests passed
   - Consistent results across robustness checks
   - Large sample size provides precise estimates
   
   LIMITATIONS:
   - Results specific to this policy context
   - Cannot rule out all unobserved confounders
   - Effect may vary across subgroups"

Communicating Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~

The agent transparently communicates the strength of identification:

**High Confidence**: Strong experimental or quasi-experimental design with assumptions well-supported by tests

**Moderate Confidence**: Good identification strategy but some assumption concerns or limited robustness checks

**Low Confidence**: Observational methods with strong assumptions or failed assumption tests

**Example Communication**:
.. code-block:: text

   Agent: "CONFIDENCE ASSESSMENT: HIGH
   
   This analysis uses a strong quasi-experimental design (difference-in-differences) 
   with well-supported assumptions. The parallel trends assumption is validated 
   through pre-treatment tests, and results are robust across multiple 
   specifications. While we cannot completely rule out all confounders, 
   the identification strategy provides credible evidence for a causal interpretation.
   
   Key strengths:
   - Clear policy discontinuity creates quasi-random variation
   - Rich data allows comprehensive assumption testing
   - Results consistent across robustness checks
   
   Remaining limitations:
   - External validity limited to similar policy contexts
   - Cannot detect very small effect heterogeneity
   - Long-term effects beyond study period unknown"

Common Decision Scenarios
------------------------

**Scenario 1: Clear Randomized Experiment**
.. code-block:: text

   Data: Randomization indicator, balanced baseline characteristics
   Agent Decision: Simple difference in means with baseline controls
   Confidence: HIGH

**Scenario 2: Policy Change with Panel Data**
.. code-block:: text

   Data: Policy implemented at different times across states
   Agent Decision: Difference-in-differences with staggered adoption methods
   Confidence: HIGH (if parallel trends hold)

**Scenario 3: Age-Based Eligibility Cutoff**
.. code-block:: text

   Data: Sharp discontinuity in treatment at specific age
   Agent Decision: Sharp regression discontinuity design
   Confidence: HIGH (if no manipulation of running variable)

**Scenario 4: Observational Data with Rich Controls**
.. code-block:: text

   Data: Self-selected treatment, many baseline characteristics
   Agent Decision: Propensity score matching with overlap assessment
   Confidence: MODERATE (depends on selection on observables assumption)

**Scenario 5: Weak Identification**
.. code-block:: text

   Data: Observational with limited controls, no clear instruments
   Agent Decision: Multiple methods with extensive sensitivity analysis
   Confidence: LOW (strong assumptions required)

Best Practices for Users
-----------------------

**Provide Context**: Include variable descriptions and study background to help the agent make better decisions

**Review Decisions**: Understand why the agent selected specific methods and assess whether the reasoning makes sense in your context

**Consider Limitations**: Pay attention to the agent's confidence assessments and stated limitations

**Validate Results**: Use the agent's analysis as a starting point, but consider additional robustness checks based on your domain knowledge

**Seek Expert Review**: For high-stakes decisions, have domain experts review the agent's method selection and results

The agent's method selection process represents a systematic approach to one of the most challenging aspects of causal inference, but human judgment remains essential for interpreting results in context and making policy decisions.
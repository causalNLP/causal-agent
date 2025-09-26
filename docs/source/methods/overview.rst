Overview of Causal Inference Methods
====================================

Causal inference is the process of determining whether and how one variable causes changes in another. Unlike correlation analysis, which only identifies statistical relationships, causal inference methods help us understand the underlying causal mechanisms and estimate the magnitude of causal effects.

CAIS (Causal AI Scientist) implements a comprehensive suite of causal inference methods, from gold-standard experimental designs to sophisticated observational study techniques. This overview introduces the key concepts and method categories available in CAIS.

What is Causal Inference?
-------------------------

Causal inference addresses the fundamental question: "What would happen to outcome Y if we changed treatment X?" This counterfactual question is challenging because we can never observe the same unit under both treatment and control conditions simultaneously.

**Key Concepts:**

* **Treatment (X)**: The intervention or exposure of interest
* **Outcome (Y)**: The variable we want to understand the causal effect on  
* **Confounders**: Variables that affect both treatment and outcome, potentially biasing estimates
* **Causal Effect**: The difference in outcomes that would result from changing treatment status

The Fundamental Problem of Causal Inference
--------------------------------------------

The core challenge in causal inference is that we cannot observe counterfactual outcomes. For any individual unit, we observe either the treated outcome or the control outcome, but never both. This is known as the "fundamental problem of causal inference."

Different causal inference methods address this problem through various identification strategies:

1. **Randomization**: Randomly assigning treatment eliminates confounding
2. **Natural Experiments**: Leveraging quasi-random variation in treatment assignment
3. **Controlling for Confounders**: Adjusting for observed variables that affect both treatment and outcome
4. **Instrumental Variables**: Using variables that affect treatment but not outcome directly

Method Categories in CAIS
-------------------------

Experimental Methods
~~~~~~~~~~~~~~~~~~~~

**Gold Standard for Causal Inference**

When randomization is possible, experimental methods provide the strongest causal evidence:

* **Randomized Controlled Trials (RCT)**: Random assignment of treatment
* **A/B Testing**: Online experiments with random user assignment
* **Field Experiments**: Real-world randomized interventions

*Advantages*: Eliminates confounding, provides unbiased causal estimates
*Limitations*: Often expensive, may not be feasible or ethical

Quasi-Experimental Methods  
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Leveraging Natural Experiments**

When randomization isn't possible, quasi-experimental methods exploit natural or policy-induced variation:

* **Difference-in-Differences (DiD)**: Compares changes over time between treatment and control groups
* **Instrumental Variables (IV)**: Uses variables that affect treatment but not outcome directly
* **Regression Discontinuity (RDD)**: Exploits arbitrary cutoffs in treatment assignment

*Advantages*: Can provide strong causal evidence without randomization
*Limitations*: Requires specific data structures and identifying assumptions

Observational Methods
~~~~~~~~~~~~~~~~~~~~~

**Extracting Causal Insights from Observational Data**

When no natural experiment exists, observational methods control for confounding through statistical adjustment:

* **Propensity Score Methods**: Match or weight units with similar treatment probabilities
* **Backdoor Adjustment**: Control for confounders that block backdoor paths
* **Linear Regression**: Estimate causal effects with appropriate controls

*Advantages*: Can be applied to many datasets, relatively straightforward
*Limitations*: Relies on strong assumptions about unobserved confounders

How CAIS Selects Methods
------------------------

CAIS automatically analyzes your data and research question to recommend the most appropriate causal inference method. The selection process considers:

**Data Characteristics:**
* Experimental vs. observational data
* Cross-sectional vs. panel structure  
* Treatment variable type (binary, continuous, categorical)
* Available variables (instruments, running variables, time dimensions)

**Identifying Assumptions:**
* Which assumptions are plausible given your research context
* Strength of identification strategy
* Robustness to assumption violations

**Research Goals:**
* Population of interest (ATE vs. ATT)
* Precision requirements
* Interpretability needs

The Decision Tree Process
-------------------------

CAIS uses a systematic decision tree to guide method selection:

1. **Is this a randomized experiment?**
   
   * Yes → Use experimental methods (RCT analysis)
   * No → Continue to observational methods

2. **What data structure do you have?**
   
   * Panel data with treatment timing → Consider Difference-in-Differences
   * Running variable with cutoff → Consider Regression Discontinuity
   * Cross-sectional → Continue to other methods

3. **Are instrumental variables available?**
   
   * Yes → Consider Instrumental Variables approach
   * No → Continue to other methods

4. **What covariates are available?**
   
   * Rich covariates with good overlap → Propensity Score methods
   * Limited covariates → Linear regression with controls
   * No covariates → Simple difference-in-means

Method Assumptions and Validity
-------------------------------

Each causal inference method relies on specific identifying assumptions. Understanding these assumptions is crucial for:

* **Method Selection**: Choosing methods with plausible assumptions
* **Sensitivity Analysis**: Testing robustness to assumption violations  
* **Result Interpretation**: Understanding the limitations of causal estimates

**Common Assumptions:**

* **Unconfoundedness**: No unmeasured confounders affect both treatment and outcome
* **Overlap/Positivity**: Units with similar characteristics can receive either treatment
* **SUTVA**: Stable Unit Treatment Value Assumption (no spillovers)
* **Parallel Trends**: Treatment and control groups follow similar trends absent treatment

Best Practices
--------------

**Before Analysis:**
1. Clearly define your causal question and target population
2. Understand your data generation process
3. Consider what assumptions are plausible in your context
4. Plan for robustness checks and sensitivity analysis

**During Analysis:**
1. Examine balance and overlap between treatment groups
2. Test key identifying assumptions where possible
3. Consider multiple methods as robustness checks
4. Report uncertainty and confidence intervals

**After Analysis:**
1. Interpret results in context of assumptions
2. Discuss limitations and potential biases
3. Consider external validity and generalizability
4. Plan follow-up studies to strengthen causal evidence

Getting Started
---------------

Ready to start your causal analysis? Here are the next steps:

1. **Installation**: :doc:`../getting_started/installation`
2. **Quick Start**: :doc:`../getting_started/quickstart` 
3. **Method Selection**: :doc:`decision_tree`
4. **Tutorials**: :doc:`../tutorials/index`

For specific method documentation, see:

* :doc:`experimental/index` - Experimental methods
* :doc:`quasi_experimental/index` - Quasi-experimental methods  
* :doc:`observational/index` - Observational methods
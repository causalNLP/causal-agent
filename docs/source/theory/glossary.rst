Glossary
========

This glossary provides definitions for statistical, causal inference, and AI agent terms used throughout the CAIS documentation. Terms are organized alphabetically with cross-references to related concepts.

A
-

**Agent**
   An autonomous AI system that can perceive its environment, make decisions, and take actions to achieve specific goals. In CAIS, the agent analyzes data, selects methods, and interprets results for causal inference.

**As-Good-As-Random Assignment**
   Treatment assignment that, while not explicitly randomized, creates quasi-random variation due to institutional rules, natural events, or policy changes. Examples include lottery-based assignment or policy discontinuities.

**Assumption Testing**
   The process of evaluating whether the key assumptions required for a causal inference method are satisfied in the data. Common tests include balance tests, parallel trends tests, and density tests.

**Average Treatment Effect (ATE)**
   The average causal effect of treatment across the entire population of interest. Represents the difference in expected outcomes between treatment and control conditions.

**Average Treatment Effect on the Treated (ATT)**
   The average causal effect of treatment among those who actually received treatment. May differ from ATE due to treatment effect heterogeneity and selection effects.

B
-

**Backdoor Adjustment**
   A method for identifying causal effects by controlling for a sufficient set of confounding variables that "block" all backdoor paths between treatment and outcome in a causal graph.

**Balance Test**
   A statistical test used to assess whether treatment and control groups have similar distributions of observed characteristics, particularly important for validating randomization in experiments.

**Bandwidth**
   In regression discontinuity design, the range of values around the cutoff point used for analysis. Optimal bandwidth balances bias (too wide) against variance (too narrow).

C
-

**Causal Effect**
   The difference in outcomes that would result from changing the treatment status of a unit, holding all else constant. Represents the impact of an intervention or policy change.

**Causal Inference**
   The process of determining whether and how one variable causes changes in another, going beyond mere correlation to establish causation.

**Common Support**
   The range of propensity scores where both treated and control units are observed. Analysis is typically restricted to this region to ensure meaningful comparisons.

**Complier**
   In instrumental variables analysis, an individual whose treatment status is affected by the instrument. The IV estimate represents the effect for compliers only.

**Confounding**
   A situation where a third variable affects both the treatment and outcome, creating a spurious association between them. Confounders must be controlled for to obtain unbiased causal estimates.

**Counterfactual**
   The outcome that would have been observed under an alternative treatment condition. Since counterfactuals are unobservable, causal inference methods aim to estimate them.

D
-

**Decision Tree**
   In CAIS, a structured algorithm that guides method selection based on data characteristics, treatment assignment mechanisms, and assumption plausibility.

**Difference-in-Differences (DiD)**
   A quasi-experimental method that compares changes in outcomes over time between treatment and control groups, controlling for time-invariant confounders and common time trends.

E
-

**Effect Heterogeneity**
   Variation in treatment effects across individuals, time periods, or contexts. Important for understanding who benefits most from interventions.

**Exclusion Restriction**
   In instrumental variables analysis, the assumption that the instrument affects the outcome only through its effect on the treatment variable, not through any other pathway.

**External Validity**
   The extent to which study results can be generalized to other populations, settings, or time periods beyond the specific study context.

F
-

**Fixed Effects**
   A method for controlling for time-invariant unobserved characteristics by including indicator variables for each unit (individual, state, etc.) in the analysis.

**Fundamental Problem of Causal Inference**
   The impossibility of observing both potential outcomes (treated and untreated) for the same unit at the same time, making direct calculation of individual treatment effects impossible.

I
-

**Identification Strategy**
   The approach used to isolate causal effects from confounding factors. Strong identification strategies rely on randomization or quasi-random variation.

**Instrumental Variable (IV)**
   A variable that affects treatment assignment but has no direct effect on the outcome except through treatment. Used to address endogeneity and selection bias.

**Internal Validity**
   The extent to which a study provides unbiased estimates of causal effects for the specific population and context studied.

L
-

**Large Language Model (LLM)**
   An AI system trained on vast amounts of text data that can understand and generate human-like text. In CAIS, LLMs help interpret data, select methods, and communicate results.

**Local Average Treatment Effect (LATE)**
   The average treatment effect for a specific subgroup, typically compliers in instrumental variables analysis. May not represent the population average effect.

M
-

**Matching**
   A method for creating comparable treatment and control groups by pairing units with similar observed characteristics, often using propensity scores.

**Monotonicity**
   In instrumental variables analysis, the assumption that the instrument affects treatment in the same direction for all individuals (no defiers).

N
-

**Natural Experiment**
   A situation where treatment assignment is determined by natural events, policy changes, or institutional rules that create quasi-random variation.

O
-

**Observational Study**
   A study where treatment assignment is not controlled by the researcher but occurs naturally, requiring methods to address selection bias and confounding.

P
-

**Parallel Trends**
   The key assumption in difference-in-differences analysis that treatment and control groups would have followed similar outcome trends in the absence of treatment.

**Potential Outcomes**
   The outcomes that would be observed under different treatment conditions. The causal effect is the difference between potential outcomes under treatment and control.

**Propensity Score**
   The probability of receiving treatment given observed characteristics. Used in matching and weighting methods to balance treatment and control groups.

Q
-

**Quasi-Experimental Design**
   A research design that exploits natural or policy-induced variation to approximate randomized experiments, providing credible causal identification.

R
-

**Randomized Controlled Trial (RCT)**
   An experimental design where treatment is randomly assigned, providing the strongest evidence for causal effects by ensuring treatment and control groups are comparable.

**Regression Discontinuity (RD)**
   A quasi-experimental method that exploits sharp cutoffs in treatment assignment based on a continuous variable (running variable) to identify causal effects.

**Robustness Check**
   Additional analyses conducted to test whether main results are sensitive to alternative specifications, samples, or assumptions.

**Running Variable**
   In regression discontinuity design, the continuous variable that determines treatment assignment at a specific cutoff point.

S
-

**Selection Bias**
   Bias that occurs when treatment and control groups differ systematically in ways that affect the outcome, beyond the treatment itself.

**Selection on Observables**
   The assumption that all variables affecting both treatment selection and outcomes are observed and can be controlled for in the analysis.

**Spillover Effects**
   Situations where treatment of some units affects the outcomes of other units, violating the stable unit treatment value assumption (SUTVA).

**Staggered Adoption**
   A research design where treatment is implemented at different times across units, allowing for difference-in-differences analysis with multiple treatment timing.

**Statistical Significance**
   The probability that an observed effect is not due to random chance, typically assessed using p-values with conventional thresholds (e.g., p < 0.05).

**Synthetic Control**
   A method that constructs a counterfactual by creating a weighted combination of control units that best matches the treated unit's pre-treatment characteristics.

T
-

**Treatment**
   The intervention, policy, or condition whose causal effect is being studied. Can be binary (treatment vs. control) or continuous (dose-response).

**Treatment Effect Heterogeneity**
   See Effect Heterogeneity.

U
-

**Unconfoundedness**
   The assumption that treatment assignment is independent of potential outcomes conditional on observed covariates. Also known as selection on observables.

V
-

**Validity**
   See Internal Validity and External Validity.

Common Acronyms
---------------

**ATE**: Average Treatment Effect
**ATT**: Average Treatment Effect on the Treated
**CAIS**: Causal AI Scientist
**CI**: Confidence Interval
**DiD**: Difference-in-Differences
**IV**: Instrumental Variables
**LATE**: Local Average Treatment Effect
**LLM**: Large Language Model
**OLS**: Ordinary Least Squares
**RCT**: Randomized Controlled Trial
**RD/RDD**: Regression Discontinuity (Design)
**SUTVA**: Stable Unit Treatment Value Assumption
**TWFE**: Two-Way Fixed Effects

Statistical Terms
-----------------

**Confidence Interval**
   A range of values that likely contains the true parameter value, typically expressed as 95% confidence intervals.

**P-value**
   The probability of observing the data (or more extreme data) if the null hypothesis of no effect were true.

**Standard Error**
   A measure of the uncertainty in a statistical estimate, used to construct confidence intervals and test statistics.

**Type I Error**
   Falsely rejecting a true null hypothesis (false positive), typically controlled at 5% level.

**Type II Error**
   Failing to reject a false null hypothesis (false negative), related to statistical power.

AI and Machine Learning Terms
-----------------------------

**Autonomous Agent**
   An AI system capable of independent decision-making and action-taking to achieve specified goals.

**Prompt Engineering**
   The practice of designing effective prompts to guide large language models toward desired outputs.

**Natural Language Processing (NLP)**
   AI techniques for understanding and generating human language, used in CAIS for interpreting research questions and communicating results.

**Machine Learning (ML)**
   Algorithms that learn patterns from data, used in CAIS for data analysis and pattern recognition.

Research Design Terms
--------------------

**Cross-sectional Data**
   Data collected at a single point in time across multiple units.

**Panel Data**
   Data that follows the same units over multiple time periods, enabling within-unit comparisons.

**Time Series Data**
   Data collected over time for a single unit or aggregate.

**Longitudinal Study**
   A study that follows subjects over time, enabling analysis of changes and causal relationships.

Policy Evaluation Terms
----------------------

**Cost-Effectiveness Analysis**
   Evaluation of interventions based on the ratio of costs to benefits or outcomes achieved.

**Implementation Fidelity**
   The degree to which an intervention is delivered as intended in the original design.

**Scalability**
   The ability of an intervention to be expanded to larger populations or different contexts while maintaining effectiveness.

**Sustainability**
   The ability of an intervention to continue producing benefits over time without ongoing external support.

This glossary serves as a reference for understanding the technical terminology used throughout the CAIS documentation and in causal inference more broadly.
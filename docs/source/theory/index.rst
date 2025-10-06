Theoretical Background
=====================

Understanding the theoretical foundations of causal inference is crucial for conducting rigorous analysis and interpreting results correctly. This section provides accessible explanations of key concepts, from basic causal thinking to advanced identification strategies.

.. toctree::
   :maxdepth: 2
   :caption: Theoretical Foundations

   causal_inference_basics
   agent_architecture
   llm_integration
   method_selection
   interpretation
   glossary

Learning Path
-------------

**For Beginners**
   Start with :doc:`causal_inference_basics` to understand fundamental concepts like causation vs. correlation, confounding, and the potential outcomes framework, with a focus on how automated analysis systems approach these challenges.

**For Understanding the Agent**
   Read :doc:`agent_architecture` to understand how the Causal Agent autonomous agent works, from data analysis to result interpretation, and :doc:`llm_integration` to learn how Large Language Models enable sophisticated decision-making in causal analysis.

**For Practitioners**
   Review :doc:`method_selection` to understand how the agent selects appropriate methods based on data characteristics and identification strategy strength.

**For All Users**
   Study :doc:`interpretation` to learn how to correctly interpret causal effect estimates and communicate results to different audiences, and consult the :doc:`glossary` for definitions of technical terms.

Key Concepts
------------

**Autonomous Agent Decision-Making**
   The Causal Agent agent combines Large Language Models with rigorous statistical methods to automatically select appropriate causal inference methods, test assumptions, and interpret results. Understanding this process helps you trust and validate the agent's decisions.

**Fundamental Problem of Causal Inference**
   We can never observe both potential outcomes for the same unit. The agent addresses this limitation by systematically identifying the best available identification strategy for your data.

**Identification Strategies**
   The agent evaluates different approaches to achieving causal identification:
   
   * Randomization (experimental methods) - highest priority when available
   * Natural experiments (quasi-experimental methods) - strong identification with testable assumptions
   * Selection on observables (observational methods) - requires stronger assumptions but often necessary

**LLM-Enhanced Analysis**
   Large Language Models enable the agent to understand context, interpret variable meanings, reason about causal relationships, and communicate results in accessible language while maintaining methodological rigor.

**Automated Assumption Testing**
   The agent systematically tests method assumptions where possible and provides transparent assessment of assumption plausibility, helping you understand the credibility of your results.

Common Pitfalls and How the Agent Addresses Them
-----------------------------------------------

* **Correlation vs. Causation**: The agent systematically evaluates treatment assignment mechanisms to distinguish causal relationships from mere correlations
* **Selection Bias**: Automated detection of selection patterns and method selection that addresses non-random treatment assignment
* **Confounding**: LLM-powered identification of potential confounders and selection of methods that control for them
* **External Validity**: Transparent discussion of generalizability and limitations of findings
* **Assumption Violations**: Systematic testing of method assumptions and sensitivity analysis when violations are detected

Agent Capabilities and Limitations
----------------------------------

**What the Agent Does Well**:
   * Systematic method selection based on data characteristics
   * Comprehensive assumption testing and validation
   * Clear communication of results and limitations
   * Consistent application of best practices

**What Requires Human Judgment**:
   * Research question formulation and context interpretation
   * External validity assessment for specific applications
   * Policy decision-making based on results
   * Ethical considerations and implementation planning

**Best Practice**: Use the agent as a sophisticated analytical tool that augments human expertise rather than replacing critical thinking and domain knowledge.
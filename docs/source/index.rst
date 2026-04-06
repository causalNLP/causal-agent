.. image:: ../../blob/main/asset/cais.png
   :width: 450px
   :align: center
   :alt: Causal AI Scientist (CAIS)

===========================================================
Causal AI Scientist (CAIS)
===========================================================


**Related resources**:
`Source Repository <https://github.com/causalNLP/causal-agent>`__ |
`Issues & Ideas <https://github.com/causalNLP/causal-agent/issues>`__ |
`CAIS on PyPI <https://pypi.org/project/causal-agent/>`__ |

Causal AI Scientist (CAIS) is an LLM-powered tool for generating data-driven answers to natural language causal queries. It takes a natural language query (for example, "Does participating in a job training program lead to higher income?"), an accompanying dataset, and the corresponding description as inputs. CAIS then frames a suitable causal estimation problem by selecting appropriate treatment and outcome variables. It finds the suitable method for causal effect estimation, implements it, runs diagnostic tests, and finally interprets the numerical results in the context of the original query

📊 What CAIS Does
-----------------

- Parses your natural language causal question
- Analyzes your dataset structure and variables
- Selects the most appropriate causal inference method:
     - Randomized Controlled Trials (RCT)
     - Difference-in-Differences (DiD)
     - Instrumental Variables (IV)
     - Regression Discontinuity Design (RDD)
     - Propensity Score Matching/Weighting
     - Linear Regression with controls
     - And more...
- Executes the analysis with proper diagnostics
- Interprets results in the context of your original question


.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card:: 🚀 Quick Start
        :shadow: md
        :link: getting_started/index
        :link-type: doc

        :octicon:`rocket;2em;sd-text-info`
        ^^^
        New to CAIS? This section will help you get started. It includes a quick start guide, installation
        instructions, and simple examples to get you up and running quickly.

    .. grid-item-card:: User Guide
        :shadow: md
        :link: user_guide/index
        :link-type: doc

        :octicon:`book;2em;sd-text-info`
        ^^^
        To come soon: A comprehensive user guide that covers all the features of CAIS, including detailed explanations.

    .. grid-item-card:: Examples
        :shadow: md
        :link: example_notebooks/nb_index
        :link-type: doc

        :octicon:`video;2em;sd-text-info`
        ^^^
        If you prefer to learn by example, we recommend to browse the examples. It covers a wide variety of problems
        that you can use to liken to your own problem.

    .. grid-item-card:: API Reference
        :shadow: md
        :link: api
        :link-type: doc

        :octicon:`code;2em;sd-text-info`
        ^^^
        The API reference contains a detailed description of the functions, modules, and objects included in CAIS.



Interactive Examples
===================

This page demonstrates the interactive features available in the CAIS documentation.

Method Selection Decision Tree
------------------------------

CAIS uses a systematic decision tree to automatically select the most appropriate causal inference method for your dataset. The decision tree is available in the :doc:`../methods/decision_tree` section.

**Quick Reference:**

The decision process follows this hierarchy:

1. **🏆 Experimental Methods** (Highest Priority) - When randomization is present
2. **🥈 Quasi-Experimental Methods** (High Priority) - Natural experiments  
3. **🥉 Observational Methods** (Medium Priority) - Statistical adjustment

**View the Complete Decision Tree:**

For the full interactive decision tree with detailed logic, see: :doc:`../methods/decision_tree`

Code Examples with Copy Functionality
--------------------------------------

All code examples in the documentation include copy-to-clipboard functionality. Try clicking the copy button on any code block:

.. code-block:: python

   # Example: Basic CAIS usage
   from causal_agent import CausalAgent
   
   # Initialize the agent
   agent = CausalAgent()
   
   # Load your dataset
   data = agent.load_data('your_dataset.csv')
   
   # Define your research question
   query = {
       'outcome': 'sales',
       'treatment': 'marketing_campaign',
       'question': 'What is the effect of the marketing campaign on sales?'
   }
   
   # Run the analysis
   result = agent.analyze(data, query)
   
   # Get the results
   print(result.summary())

.. code-block:: bash

   # Installation commands
   pip install causal-agent
   
   # Or with conda
   conda install -c conda-forge causal-agent

Expandable Sections
-------------------

Complex concepts can be hidden in expandable sections to improve readability.

.. raw:: html

   <div class="expandable-section" data-collapsed="true">
       <h4 class="expandable-title">Advanced Configuration Options</h4>
       <div class="expandable-content">

**LLM Provider Configuration**

CAIS supports multiple LLM providers. Here are the detailed configuration options:

* **OpenAI**: Requires API key and supports GPT-3.5 and GPT-4 models
* **Anthropic**: Requires API key and supports Claude models  
* **Google**: Requires API key and supports PaLM models
* **Local Models**: Supports local deployment via Ollama or similar

**Advanced Prompt Engineering**

You can customize the prompts used by CAIS for different analysis phases:

.. code-block:: python

   # Custom prompt configuration
   agent = CausalAgent(
       llm_config={
           'provider': 'openai',
           'model': 'gpt-4',
           'custom_prompts': {
               'dataset_analysis': 'Your custom dataset analysis prompt...',
               'method_selection': 'Your custom method selection prompt...',
               'result_interpretation': 'Your custom interpretation prompt...'
           }
       }
   )

**Performance Optimization**

For large datasets, consider these optimization strategies:

* Use sampling for initial exploration
* Enable caching for repeated analyses  
* Configure parallel processing for multiple queries
* Use batch processing for multiple datasets

.. raw:: html

       </div>
   </div>

Tooltips for Technical Terms
-----------------------------

Technical terms throughout the documentation include helpful tooltips. Hover over or focus on terms like :term:`causal inference`, :term:`treatment effect`, or :term:`confounding` to see their definitions.

The CAIS system uses sophisticated :term:`propensity score` methods when randomization is not available. It can also leverage :term:`instrumental variable` approaches when suitable instruments are identified in your data.

For panel data, :term:`difference-in-differences` methods are often preferred, while :term:`regression discontinuity` designs work well when treatment assignment follows a cutoff rule.

Feedback System
---------------

We value your feedback! Use the feedback widget in the bottom-right corner to:

* Rate the helpfulness of this page
* Provide detailed comments about your experience
* Report issues or suggest improvements
* Request additional examples or clarification

You can also provide quick feedback using the buttons at the bottom of each page, or click the 💬 icon next to section headers for section-specific feedback.

Interactive Code Examples
--------------------------

Some code examples include interactive elements. Click on code blocks to highlight them:

.. raw:: html

   <div class="interactive-example">

.. code-block:: python

   # Interactive example: Click to highlight
   import pandas as pd
   from causal_agent import CausalAgent
   
   # This code block is interactive
   agent = CausalAgent()
   
   # Load sample data
   data = pd.read_csv('sample_data.csv')
   
   # Define treatment and outcome
   treatment = 'policy_intervention'
   outcome = 'economic_indicator'
   
   # Run analysis
   results = agent.analyze(
       data=data,
       treatment=treatment,
       outcome=outcome,
       method='auto'  # Let CAIS choose the best method
   )

.. raw:: html

   </div>

Accessibility Features
----------------------

The interactive features include comprehensive accessibility support:

* **Keyboard Navigation**: All interactive elements can be navigated using the keyboard
* **Screen Reader Support**: ARIA labels and roles are provided for assistive technologies
* **High Contrast Mode**: Styles adapt automatically for users with high contrast preferences
* **Reduced Motion**: Animations are disabled for users who prefer reduced motion
* **Skip Links**: Quick navigation links are available for keyboard users

Try navigating this page using only your keyboard (Tab, Enter, Space, Arrow keys) to experience the accessibility features.

.. note::
   
   The interactive features on this page demonstrate the enhanced user experience available throughout the CAIS documentation. These features are designed to make complex causal inference concepts more accessible and engaging for users of all backgrounds.
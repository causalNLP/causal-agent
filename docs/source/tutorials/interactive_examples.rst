Method Selection Decision Tree
------------------------------

causal-agent uses a systematic decision tree to automatically select the most appropriate causal inference method for your dataset. The decision tree is available in the :doc:`../methods/decision_tree` section.

**Quick Reference:**

The decision process follows this hierarchy:

1. **🏆 Experimental Methods** (Highest Priority) - When randomization is present
2. **🥈 Quasi-Experimental Methods** (High Priority) - Natural experiments  
3. **🥉 Observational Methods** (Medium Priority) - Statistical adjustment

**View the Complete Decision Tree:**

For the full decision tree with detailed logic, see: :doc:`../methods/decision_tree`

Code Examples with Copy Functionality
--------------------------------------

All code examples in the documentation include copy-to-clipboard functionality. Try clicking the copy button on any code block:

.. code-block:: python

   # Example: Basic causal-agent usage
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



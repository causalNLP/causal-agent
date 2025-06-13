"""
LangChain agent for the auto_causal module.

This module configures a LangChain agent with specialized tools for causal inference,
allowing for an interactive approach to analyzing datasets and applying appropriate
causal inference methods.
"""

import logging
from typing import Dict, List, Any, Optional

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.tools import tool
# Import the callback handler
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
# Import tool rendering utility
from langchain.tools.render import render_text_description
# Import LCEL components
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseChatModel

# Import actual tools from the tools directory
from auto_causal.tools.input_parser_tool import input_parser_tool
from auto_causal.tools.dataset_analyzer_tool import dataset_analyzer_tool
from auto_causal.tools.query_interpreter_tool import query_interpreter_tool
from auto_causal.tools.method_selector_tool import method_selector_tool
from auto_causal.tools.method_validator_tool import method_validator_tool
from auto_causal.tools.method_executor_tool import method_executor_tool
from auto_causal.tools.explanation_generator_tool import explanation_generator_tool
from auto_causal.tools.output_formatter_tool import output_formatter_tool
#from auto_causal.prompts import SYSTEM_PROMPT # Assuming SYSTEM_PROMPT is defined here or imported

# Import the centralized factory function
from .config import get_llm_client 
#from .prompts import SYSTEM_PROMPT 

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Centralized LLM Client Factory (REMOVED FROM HERE) --- 
# load_dotenv() # Moved to config
# def get_llm_client(...): # Moved to config
#     ...
# --- End Removed Section --- 

def create_agent_prompt(tools: List[tool]) -> ChatPromptTemplate:
    """Create the prompt template for the causal inference agent, emphasizing workflow and data handoff.
       (This is the version required by the LCEL agent structure below)
    """
    # Get the tool descriptions
    tool_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])

    # Define the system prompt template string
    system_template = f"""
You are a causal inference expert helping users answer causal questions by following a strict workflow using specialized tools.

TOOLS:
------
You have access to the following tools:

{tool_description}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as a single, valid JSON object string. Check the tool definition for required arguments and structure.
Observation: the result of the action, often containing structured data like 'variables', 'dataset_analysis', 'method_info', etc.
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

**IMPORTANT TOOL USAGE:**
1.  **Action Input Format:** The value for 'Action Input' MUST be a single, valid JSON object string. Do NOT include any other text or formatting around the JSON string.
2.  **Argument Gathering:** You MUST gather ALL required arguments for the Action Input JSON from the initial Human input AND the 'Observation' outputs of PREVIOUS steps. Look carefully at the required arguments for the tool you are calling.
3.  **Data Handoff:** The 'Observation' from a previous step often contains structured data needed by the next tool. For example, the 'variables' output from `query_interpreter_tool` contains fields like `treatment_variable`, `outcome_variable`, `covariates`, `time_variable`, `instrument_variable`, `running_variable`, `cutoff_value`, and `is_rct`. When calling `method_selector_tool`, you MUST construct its required `variables` input argument by including **ALL** these relevant fields identified by the `query_interpreter_tool` in the previous Observation. Similarly, pass the full `dataset_analysis`, `dataset_description`, and `original_query` when required by the next tool.

IMPORTANT WORKFLOW:
-------------------
You must follow this exact workflow, selecting the appropriate tool for each step:

1. ALWAYS start with `input_parser_tool` to understand the query
2. THEN use `dataset_analyzer_tool` to analyze the dataset
3. THEN use `query_interpreter_tool` to identify variables (output includes `variables` and `dataset_analysis`)
4. THEN use `method_selector_tool` (input requires `variables` and `dataset_analysis` from previous step)
5. THEN use `method_validator_tool` (input requires `method_info` and `variables` from previous step)
6. THEN use `method_executor_tool` (input requires `method`, `variables`, `dataset_path`)
7. THEN use `explanation_generator_tool` (input requires results, method_info, variables, etc.)
8. FINALLY use `output_formatter_tool` to return the results 

REASONING PROCESS:
------------------
EXPLICITLY REASON about:
1. What step you're currently on (based on previous tool's Observation)
2. Why you're selecting a particular tool (should follow the workflow)
3. How the output of the previous tool (especially structured data like `variables`, `dataset_analysis`, `method_info`) informs the inputs required for the current tool.

Begin!
"""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history", optional=True), # Use MessagesPlaceholder
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"), # Use MessagesPlaceholder
    ])
    return prompt

def create_causal_agent(llm: BaseChatModel) -> AgentExecutor:
    """
    Create and configure the LangChain agent with causal inference tools.
    (Using explicit LCEL construction, compatible with shared LLM client)
    """
    # Define tools available to the agent
    agent_tools = [
        input_parser_tool,
        dataset_analyzer_tool,
        query_interpreter_tool,
        method_selector_tool,
        method_validator_tool,
        method_executor_tool,
        explanation_generator_tool,
        output_formatter_tool
    ]
    
    # Create the prompt using the helper
    prompt = create_agent_prompt(agent_tools)
    
    # Bind tools to the LLM (using the passed shared instance)
    llm_with_tools = llm.bind_tools(agent_tools)
    
    # Create memory
    # Consider if memory needs to be passed in or created here
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Manually construct the agent runnable using LCEL
    agent = (
        RunnablePassthrough.assign(
            # agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
            # Load memory variables, handle non-existent keys gracefully
            chat_history=lambda x: memory.load_memory_variables(x).get("chat_history", []), 
            # Add input to the chain's input dictionary if not present
            input=lambda x: x.get("input", "") # Ensure input key exists
        )
        .assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | llm_with_tools
        # Use the base ToolsAgentOutputParser directly
        | ToolsAgentOutputParser() 
    )
    
    # Create executor (should now work with the manually constructed agent)
    executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        memory=memory, # Pass the memory object
        verbose=True,
        # callbacks=[ConsoleCallbackHandler()], # Optional: for console debugging
        handle_parsing_errors=True # Let AE handle parsing errors
    )
    
    return executor

def run_causal_analysis(query: str, dataset_path: str, 
                        dataset_description: Optional[str] = None, 
                        api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Run causal analysis on a dataset based on a user query.
    
    Args:
        query: User's causal question
        dataset_path: Path to the dataset
        dataset_description: Optional textual description of the dataset
        api_key: Optional OpenAI API key (DEPRECATED - will be ignored)
        
    Returns:
        Dictionary containing the final formatted analysis results from the agent's last step.
    """
    # Log the start of the analysis
    logger.info("Starting causal analysis run...")
    
    try:
        # --- Instantiate the shared LLM client --- 
        shared_llm = get_llm_client(temperature=0) # Or read provider/model from env
        
        # --- Dependency Injection Note (REMAINS RELEVANT) --- 
        # If tools need the LLM, they must be adapted. Example using partial:
        # from functools import partial
        # from .components import input_parser 
        # # Assume input_parser.parse_input needs llm 
        # input_parser_tool_with_llm = tool(partial(input_parser.parse_input, llm=shared_llm)) 
        # Use input_parser_tool_with_llm in the tools list passed to the agent below.
        # Similar adjustments needed for decision_tree._recommend_ps_method if used.
        # --- End Note --- 

        # --- Create agent using the shared LLM --- 
        agent_executor = create_causal_agent(shared_llm) 
        
        # Construct input, including description if available
        # IMPORTANT: Agent now expects 'input' and potentially 'chat_history'
        # The input needs to contain all initial info the first tool might need.
        initial_input_dict = {
            "query": query,
            "dataset_path": dataset_path,
            "dataset_description": dataset_description
        }
        # Maybe format this into a single input string if the prompt expects {input}
        input_text = f"My question is: {query}\n"
        input_text += f"The dataset is located at: {dataset_path}\n"
        if dataset_description:
            input_text += f"Dataset Description: {dataset_description}\n"
        input_text += "Please perform the causal analysis following the workflow."
        
        # Log the constructed input text
        logger.info(f"Constructed input for agent: \n{input_text}")

        result = agent_executor.invoke({"input": input_text})
        
        # AgentExecutor returns dict. Extract the final output dictionary.
        logger.info("Causal analysis run finished.")
        
        # Ensure result is a dict and extract the 'output' part
        if isinstance(result, dict):
            final_output = result.get("output")
            if isinstance(final_output, dict):
                return final_output # Return only the dictionary from the final tool
            else:
                logger.error(f"Agent result['output'] was not a dictionary: {type(final_output)}. Returning error dict.")
                return {"error": "Agent did not produce the expected dictionary output in the 'output' key.", "raw_agent_result": result}
        else:
            logger.error(f"Agent returned non-dict type: {type(result)}. Returning error dict.")
            return {"error": "Agent did not return expected dictionary output.", "raw_output": str(result)}

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        # Return an error dictionary in case of exception too
        return {"error": f"Error: Configuration issue - {e}"} # Ensure consistent error return type
    except Exception as e:
        logger.error(f"An unexpected error occurred during causal analysis: {e}", exc_info=True)
        # Return an error dictionary in case of exception too
        return {"error": f"An unexpected error occurred: {e}"} 
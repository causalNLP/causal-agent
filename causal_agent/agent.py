"""
LangChain agent for the causal_agent module.

This module configures a LangChain agent with specialized tools for causal inference,
allowing for an interactive approach to analyzing datasets and applying appropriate
causal inference methods.

The main entry point is the `run_causal_analysis` function which orchestrates:
1. Input parsing and validation
2. Dataset analysis and variable identification  
3. Query interpretation and method selection
4. Causal method execution with diagnostics
5. Result interpretation and explanation generation

Example:
    >>> from causal_agent import run_causal_analysis
    >>> result = run_causal_analysis(
    ...     query="What is the effect of education on income?",
    ...     dataset_path="data.csv",
    ...     dataset_description="Education and income dataset"
    ... )
    >>> print(result['results']['results']['effect_estimate'])

The agent supports multiple LLM providers (OpenAI, Anthropic, Google, etc.) and
automatically selects from various causal inference methods including RCT analysis,
Difference-in-Differences, Instrumental Variables, Propensity Score methods, and more.
"""

import logging
from typing import Dict, List, Any, Optional
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_tool_calling_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.tools import tool

from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler

from langchain.tools.render import render_text_description

from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseChatModel
from langchain_anthropic.chat_models import convert_to_anthropic_tool
import os

from causal_agent.tools.input_parser_tool import input_parser_tool
from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
from causal_agent.tools.query_interpreter_tool import query_interpreter_tool
from causal_agent.tools.method_selector_tool import method_selector_tool
from causal_agent.tools.method_validator_tool import method_validator_tool
from causal_agent.tools.method_executor_tool import method_executor_tool
from causal_agent.tools.explanation_generator_tool import explanation_generator_tool
from causal_agent.tools.output_formatter_tool import output_formatter_tool
from langchain_core.output_parsers import StrOutputParser
from .config import get_llm_client 
#from .prompts import SYSTEM_PROMPT 
from langchain_core.messages import AIMessage, AIMessageChunk
import re
import json
from typing import Union
from langchain_core.output_parsers import BaseOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_anthropic.output_parsers import ToolsOutputParser
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.agents import AgentOutputParser
from langchain.agents.agent import AgentAction, AgentFinish, OutputParserException
import re
from typing import Union, List
from causal_agent.models import *

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:'"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and parse-able actions"
)


class ReActMultiInputOutputParser(AgentOutputParser):
    """Parses ReAct-style output that may contain multiple tool calls."""

    def get_format_instructions(self) -> str:
        
        return FORMAT_INSTRUCTIONS + (
            "\n\nIf you need to call more than one tool, simply repeat:\n"
            "Action: <tool_name>\n"
            "Action Input: <json or text>\n"
            "…for each tool in sequence."
        )

    @property
    def _type(self) -> str:
        return "react-multi-input"

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        print('-------------------')
        print(text)
        print('-------------------')
        # Grab every Action / Action Input block
        pattern = (
            r"Action\s*\d*\s*:[\s]*(.*?)\s*"
            r"Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*?)(?=(?:Action\s*\d*\s*:|$))"
        )
        matches = list(re.finditer(pattern, text, re.DOTALL))

        # If we found tool calls…
        if matches:
            if includes_answer:
                # both a final answer *and* tool calls is ambiguous
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )

            actions: List[AgentAction] = []
            for m in matches:
                tool_name = m.group(1).strip()
                tool_input = m.group(2).strip().strip('"')
                print('\n--------------------------')
                print(tool_input)
                print('--------------------------')
                actions.append(AgentAction(tool_name, json.loads(tool_input), text))

            return actions

        # Otherwise, if there's a final answer, finish
        if includes_answer:
            answer = text.split(FINAL_ANSWER_ACTION, 1)[1].strip()
            return AgentFinish({"output": answer}, text)

        # No calls and no final answer → figure out which error to throw
        if not re.search(r"Action\s*\d*\s*Input\s*\d*:", text):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )

        # Fallback
        raise OutputParserException(f"Could not parse LLM output: `{text}`")

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_agent_prompt(tools: List[tool]) -> ChatPromptTemplate:
    """Create the prompt template for the causal inference agent, emphasizing workflow and data handoff.
       (This is the version required by the LCEL agent structure below)
    """
    # Get the tool descriptions
    tool_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])

    # Define the system prompt template string
    system_template = """
You are a causal inference expert helping users answer causal questions by following a strict workflow using specialized tools.

Remember you always have to always generate the Thought, Action and Action Input block.
TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as a single, valid JSON object string. Check the tool definition for required arguments and structure.
Observation: the result of the action, often containing structured data like 'variables', 'dataset_analysis', 'method_info', etc.

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

DO NOT UNDER ANY CIRCUMSTANCE CALL MORE THAN ONE TOOL IN A  STEP

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

IMPORTANT RULES:
1. Do not make more than one tool call in a single step.
2. Do not include ``` in your output at all.
3. Don't use action names like default_api.dataset_analyzer_tool, instead use tool names like dataset_analyzer_tool.
4. Always start, action, and observation with a new line.
5. Don't use '\\' before double quotes
6. Don't include ```json for Action Input. Also ensure that Action Input is a valid json. DO no add any text after Action Iput.
7. You have to always choose one of the tools unless it's the final answer.
Begin!
""" 

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history", optional=True), # Use MessagesPlaceholder
        # MessagesPlaceholder("agent_scratchpad"),  

        ("human", "{input}\n Thought:{agent_scratchpad}"),
        # ("ai", "{agent_scratchpad}"),
        # MessagesPlaceholder("agent_scratchpad" ), # Use MessagesPlaceholder
        # "agent_scratchpad"
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
    # anthropic_agent_tools = [ convert_to_anthropic_tool(anthropic_tool) for anthropic_tool in agent_tools]
    # Create the prompt using the helper
    prompt = create_agent_prompt(agent_tools)
    # Bind tools to the LLM (using the passed shared instance)
    
    
    # Create memory
    # Consider if memory needs to be passed in or created here
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Manually construct the agent runnable using LCEL
    from langchain_anthropic.output_parsers import ToolsOutputParser
    from langchain.agents.output_parsers.json import JSONAgentOutputParser
    # from langchain.agents.react.output_parser import MultiActionAgentOutputParsers ReActMultiInputOutputParser
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "gemini":
        base_parser=ReActMultiInputOutputParser()
        llm_with_tools = llm.bind_tools(agent_tools)
    else:
        base_parser=ToolsAgentOutputParser()
        llm_with_tools = llm.bind_tools(agent_tools, tool_choice="any")
    agent = create_react_agent(llm_with_tools, agent_tools, prompt, output_parser=base_parser)
    
    
    # Create executor (should now work with the manually constructed agent)
    executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        memory=memory, # Pass the memory object
        verbose=True,
        callbacks=[ConsoleCallbackHandler()], # Optional: for console debugging
        handle_parsing_errors=True, # Let AE handle parsing errors
        max_retries = 100
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
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        if model_name in ['o3', 'o4-mini', 'o3-mini']:
            print('-------------------------')
            shared_llm = get_llm_client()
        else:
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
        # agent_executor = create_causal_agent(shared_llm) 
        
        # Construct input, including description if available
        # IMPORTANT: Agent now expects 'input' and potentially 'chat_history'
        # The input needs to contain all initial info the first tool might need.
        input_text = f"My question is: {query}\n"
        input_text += f"The dataset is located at: {dataset_path}\n"
        if dataset_description:
            input_text += f"Dataset Description: {dataset_description}\n"
        input_text += "Please perform the causal analysis following the workflow."
        
        # Log the constructed input text
        logger.info(f"Constructed input for agent: \n{input_text}")

        input_parsing_result = input_parser_tool(input_text)
        dataset_analysis_result = dataset_analyzer_tool.func(dataset_path=input_parsing_result["dataset_path"], dataset_description=input_parsing_result["dataset_description"], original_query=input_parsing_result["original_query"]).analysis_results
        query_info = QueryInfo(
        query_text=input_parsing_result["original_query"],
        potential_treatments=input_parsing_result["extracted_variables"].get("treatment"),
        potential_outcomes=input_parsing_result["extracted_variables"].get("outcome"),
        covariates_hints=input_parsing_result["extracted_variables"].get("covariates_mentioned"),
        instrument_hints=input_parsing_result["extracted_variables"].get("instruments_mentioned")
    )

        query_interpreter_output = query_interpreter_tool.func(query_info=query_info, dataset_analysis=dataset_analysis_result, dataset_description=input_parsing_result["dataset_description"], original_query = input_parsing_result["original_query"]).variables
        method_selector_output = method_selector_tool.func(variables=query_interpreter_output,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            original_query = input_parsing_result["original_query"],
            excluded_methods=None)
        method_info = MethodInfo(
            **method_selector_output['method_info']
        )
        method_validator_input = MethodValidatorInput(
            method_info=method_info,
            variables=query_interpreter_output,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            original_query = input_parsing_result["original_query"]
        )
        method_validator_output = method_validator_tool.func(method_validator_input)
        method_executor_input = MethodExecutorInput(
            **method_validator_output
        )
        method_executor_output = method_executor_tool.func(method_executor_input, original_query = input_parsing_result["original_query"])
        explainer_output = explanation_generator_tool.func(            method_info=method_info,
            validation_info=method_validator_output,
            variables=query_interpreter_output,
            results=method_executor_output,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            original_query = input_parsing_result["original_query"])
        result = explainer_output
        result['results']['results']["method_used"] = method_validator_output['method']
        logger.info(result)
        logger.info("Causal analysis run finished.")
        
        # Ensure result is a dict and extract the 'output' part
        if isinstance(result, dict):
            final_output = result
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
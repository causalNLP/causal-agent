"""
Core class for the CausalAgent, which orchestrates the workflow of analyzing a dataset and query,
selecting and validating methods, cleaning the dataset, executing the method, and generating explanations. 
"""

from typing import Dict, List, Any, Optional
from cais.estimator_lib import Estimators
from cais.tools.input_parser_tool import input_parser_tool
from cais.tools.dataset_analyzer_tool import dataset_analyzer_tool
from cais.tools.query_interpreter_tool import query_interpreter_tool
from cais.tools.method_selector_tool import method_selector_tool
from cais.tools.controls_selector_tool import controls_selector_tool
from cais.tools.method_validator_tool import method_validator_tool
from cais.tools.method_executor_tool import method_executor_tool
from cais.tools.explanation_generator_tool import explanation_generator_tool
from cais.tools.output_formatter_tool import output_formatter_tool

from cais.methods.linear_regression.estimator import LinearRegression
from cais.methods.regression_discontinuity.estimator import RDDRegression
from cais.methods.difference_in_differences.estimator import DiDRegression
from cais.methods.instrumental_variable.estimator import IVRegression
from cais.methods.propensity_score.matching import PropensityScoreMatching
from cais.models import Variables, MethodInfo
from cais.components.assumption_checks import IVTest, ObervationalTest, DiDTest, RDDTest

from .config import get_llm_client 
#from .prompts import SYSTEM_PROMPT 
from cais.models import *
from cais.tools.dataset_cleaner_tool import dataset_cleaner_tool
import pandas as pd
import os, logging
import re
import json

LINEAR_REGRESSION = "linear_regression"
DIFF_IN_DIFF = "difference_in_differences"
REGRESSION_DISCONTINUITY = "regression_discontinuity_design"
PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
INSTRUMENTAL_VARIABLE = "instrumental_variable"

# Temporary name conversion
convert = {
    LINEAR_REGRESSION: LinearRegression.name,
    DIFF_IN_DIFF: DiDRegression.name,
    REGRESSION_DISCONTINUITY: RDDRegression.name,
    INSTRUMENTAL_VARIABLE: IVRegression.name,
    PROPENSITY_SCORE_MATCHING: PropensityScoreMatching.name
}

os.makedirs('./logs/', exist_ok=True)
logger = logging.getLogger(__name__)

class CausalAgent():
    
    def __init__(
            self,
            dataset_path: Union[str, pd.DataFrame], # dataset path or dataframe directly
            dataset_description: Optional[str] = None, # Description of the dataset
            model_name: Optional[str] = None,
            provider: Optional[str] = None
    ):
        # Query not passed to constructor or saved so we can rerun different queries on the same dataset

        self.llm_info = {
            'model_name' : model_name,
            'provider' : provider
        }
        
        # MUST PASS
        self.llm = get_llm_client(
            provider=provider,
            model_name=model_name
        )

        # Estimator library
        self.estimators = Estimators()

        # Metadata
        self.dataset_path = dataset_path # store dataset; stop saving then rewriting
        self.cleaned_dataset_path: Optional[str] = None
        self.dataset_description = dataset_description # will need checks for None

        # Pipeline states 
        self.dataset_analysis: Optional[DatasetAnalysis] = None
        self.query_interpreter_output: Optional[QueryInterpreterOutput] = None # Unnecessary
        self.variables: Optional[Variables] = None
        self.selected_method: Optional[MethodInfo] = None

        # Outputs
        self.results: Optional[Dict[str, Any]] = None
        self.explanations: Optional[Dict[str, Any]] = None

        self.last_used_query = None

    def checkq(self, query):
        '''
        Checks if a query was passed; if not, uses the most recently used query.
        '''
        if not query:
            query = self.last_used_query
        self.last_used_query = query
        return query

    def load_dataset(self, cleaned=False):
        
        if not cleaned or not self.cleaned_dataset_path:
            if cleaned:
                print("Warning: Cleaned dataset not found. Please run clean_dataset() before loading dataset.")
            return pd.read_csv(self.dataset_path)
        else:
            return pd.read_csv(self.cleaned_dataset_path)

    def analyse_dataset(self, query=None):
        
        query = self.checkq(query)

        # Analyse dataset based on provided description 
        dataset_analysis = dataset_analyzer_tool.func(
            dataset_path=self.dataset_path,
            dataset_description=self.dataset_description,
            original_query=query,
            llm=self.llm
        ).analysis_results        
        
        # Analyse query based on dataset analysis and dataset description
        query_interpreter_output = query_interpreter_tool.func(
            dataset_analysis=dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query
        )
        
        self.dataset_analysis = dataset_analysis
        self.query_interpreter_output = query_interpreter_output
        self.variables = query_interpreter_output.variables

    def select_method(self, query=None, llm_decision=True):

        query = self.checkq(query)

        excluded = set(convert.values()) - self.estimators.names()
        method_selector_output = method_selector_tool.func(
            variables=self.variables,
            dataset_analysis=self.dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query,
            excluded_methods=excluded,
            use_decision_tree=llm_decision, # LLM Decision Tree vs. Rule-based Decision Tree
        )

        self.method_info = MethodInfo(**method_selector_output['method_info'])
        self.selected_method = self.method_info.selected_method
        return self.selected_method


    def validate_method(self, query=None):
        '''
        Do not use yet
        '''
        query = self.checkq(query)

        # TODO: Move changes from assumption_checker branch to refactoring (this branch)

        method_validator_input = MethodValidatorInput(
                method_info=self.method_info,
                variables=self.variables,
                dataset_analysis=self.dataset_analysis,
                dataset_description=self.dataset_description,
                original_query=query,
            )
        method_validator_output = method_validator_tool.func(method_validator_input)
        method_name = method_validator_output.get('method')

    def select_controls(self, query=None) -> list:

        query = self.checkq(query)

        controls_selector_output = controls_selector_tool(
            method_name=self.selected_method,
            variables=self.variables,
            dataset_analysis=self.dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query,
        )
      
        self.variables = Variables(**controls_selector_output['variables']) # refined controls; need to update after cleaning

    def clean_dataset(self, query=None):   

        query = self.checkq(query)

        cleaning_output = dataset_cleaner_tool.func(
            dataset_path=self.dataset_path,
            variables=self.variables.model_dump(),
            dataset_description=self.dataset_description,
            original_query=query,
            causal_method=self.selected_method
        )
        self.cleaned_dataset_path = cleaning_output.get("cleaned_dataset_path", self.dataset_path)
        return self.cleaned_dataset_path

    def execute_method(self, query=None, remove_cleaned=True):
        
        query = self.checkq(query)
        logger.info(f"Starting method execution. Trying to run {self.selected_method}")
        try:
            estimator = self.estimators[
                convert[self.selected_method]
            ]

            df = self.load_dataset(cleaned=True)
            df.dropna(subset=[
                self.variables.outcome_variable,
                self.variables.treatment_variable
                ] + self.variables.confounders,
                inplace=True
            ) # safety
            
            self.results = estimator(
                df=df,
                variables=self.variables,
                query=query
            ) | self.llm_info # append llm info
        except:
            method_executor_input = MethodExecutorInput(
                        method = self.selected_method,
                        variables=self.query_interpreter_output,
                        dataset_path=self.cleaned_dataset_path,
                        dataset_analysis=self.dataset_analysis,
                        dataset_description=self.dataset_description,
                        # validation_info=method_validator_output,
                        original_query = query
                    )
            logger.debug(method_executor_input)
            self.results = method_executor_tool.func(
                method_executor_input,
                original_query=query
            )

        self.explanations = explanation_generator_tool.func(
            method_info=self.method_info,
            variables=self.variables,
            results=self.results,
            dataset_analysis=self.dataset_analysis,
            validation_info=None,
            dataset_description=self.dataset_description,
            original_query=query
        )['explanation']

        if self.cleaned_dataset_path and remove_cleaned:
            if isinstance(self.load_dataset(cleaned=True), pd.DataFrame):
                os.remove(self.cleaned_dataset_path)
                self.cleaned_dataset_path=None
                logger.info("Succesfully Removed Cleaned Dataset.")

        return {
            "results" : self.results,
            "explanation": self.explanations
        }
    
    def run_analysis(self, query, llm_method_selection: Optional[bool] = True):

        logger.info("[Causal AI Scientist Stage 1] - Dataset and Query analysis")

        self.query = query

        self.analyse_dataset(
            query=query
        )
        self.select_method(
            query=query,
            llm_decision=llm_method_selection
        )
        self.select_controls(
            query=query
        )
        self.clean_dataset(
            query=query
        )
        return self.execute_method(
            query=query
        )


# ===== DEPRECIATED ======


def run_causal_analysis(query: str, dataset_path: str,
                        dataset_description: Optional[str] = None,
                        api_key: Optional[str] = None,
                        use_method_validator: bool = True) -> Dict[str, Any]:
    """
    Run causal analysis on a dataset based on a user query.
    
    Args:
        query: User's causal question
        dataset_path: Path to the dataset
        dataset_description: Optional textual description of the dataset
        api_key: Optional OpenAI API key (DEPRECATED - will be ignored)
        use_method_validator: Whether to run the method validator step
        
    Returns:
        Dictionary containing the final formatted analysis results from the agent's last step.
    """
    logger.info("Starting causal analysis run...")
    try:
        # --- Instantiate the shared LLM client --- 
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        if model_name in ['o3', 'o4-mini', 'o3-mini']:
            print('-------------------------')
            shared_llm = get_llm_client()
        else:
            shared_llm = get_llm_client(temperature=0) # Or read provider/model from env

        logger.info(f"Initializing LLM client: Provider='{os.getenv('LLM_PROVIDER')}', Model='{os.getenv('LLM_MODEL')}'")
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
        logger.debug(f"Constructed input for agent: \n{input_text}")
        
        
        logger.info("[Causal AI Scientist Stage 1] - Data Processing")
        
        input_parsing_result = input_parser_tool(input_text)
        # This just returns query, dataset_path for the csv file and dataset_description
        # and workflow state update but that's probably not needed

        dataset_analysis_result = dataset_analyzer_tool.func(dataset_path=input_parsing_result["dataset_path"], dataset_description=input_parsing_result["dataset_description"], original_query=input_parsing_result["original_query"]).analysis_results
        
        query_info = QueryInfo(
        query_text=input_parsing_result["original_query"],
        potential_treatments=input_parsing_result["extracted_variables"].get("treatment"),
        potential_outcomes=input_parsing_result["extracted_variables"].get("outcome"),
        covariates_hints=input_parsing_result["extracted_variables"].get("covariates_mentioned"),
        instrument_hints=input_parsing_result["extracted_variables"].get("instruments_mentioned")
        )

        query_interpreter_output = query_interpreter_tool.func(query_info=query_info, dataset_analysis=dataset_analysis_result, dataset_description=input_parsing_result["dataset_description"], original_query = input_parsing_result["original_query"]).variables

        # print('LOG RESULTS')
        # print(input_parsing_result['extracted_variables'])
        # print(input_parsing_result['extracted_variables'].get("treatment"))
        # print(input_parsing_result['extracted_variables'].get("outcome"))
        # print(input_parsing_result['extracted_variables'].get("covariates_mentioned"))
        # print(input_parsing_result['extracted_variables'].get("instruments_mentioned"))

        # print('QUERY INTERPRETER OUTPUT')
        # print(query_interpreter_output)


        logger.info("[Causal AI Scientist Stage 2] - Method Selection")
        
        
        method_selector_output = method_selector_tool.func(variables=query_interpreter_output,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            original_query = input_parsing_result["original_query"],
            excluded_methods=None)

        

        print('METHOD SELECTOR OUTPUT: ', method_selector_output)

        import sys
        sys.exit()

        # NEW: Select control variables based on chosen method
        method_info = MethodInfo(
            **method_selector_output['method_info']
        )

        

        logger.info("[Causal AI Scientist Stage 3] - Method Validation")
        if use_method_validator:
            method_validator_input = MethodValidatorInput(
                method_info=method_info,
                variables=query_interpreter_output,
                dataset_analysis=dataset_analysis_result,
                dataset_description=input_parsing_result["dataset_description"],
                original_query = input_parsing_result["original_query"]
            )
            method_validator_output = method_validator_tool.func(method_validator_input)
            # method_validator_output['method'] = "linear_regression"
            method_name = method_validator_output.get('method')
        else:
            method_name = method_info.selected_method
            method_validator_output = {
                "method": method_name,
                "validation_info": {
                    "original_method": method_info.selected_method,
                    "recommended_method": method_name,
                    "assumptions_valid": None,
                    "failed_assumptions": [],
                    "warnings": ["Method validation skipped by flag."],
                    "suggestions": []
                }
            }
        controls_selector_output = controls_selector_tool.func(
            method_name=method_name,
            variables=query_interpreter_output,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            original_query=input_parsing_result["original_query"]
        )
        # Update variables with selected controls
        from cais.models import Variables
        query_interpreter_output = Variables(**controls_selector_output['variables'])
        logger.info(f"Selected controls: {query_interpreter_output}")
        logger.info('Started Dataset Cleaning... ')

        original_path = dataset_analysis_result.dataset_info.file_path 
        cleaning_output = dataset_cleaner_tool.func(
            dataset_path=original_path,
            variables=query_interpreter_output.model_dump(),
            dataset_description=input_parsing_result["dataset_description"],
            original_query=input_parsing_result["original_query"],
            causal_method = method_name
        )
        cleaned_path = cleaning_output.get("cleaned_dataset_path", original_path)
        #print("----------Cleaned Dataset Path-----------")
        logger.info(cleaned_path)

        logger.info("[Causal AI Scientist Stage 4] - Execution")
        method_executor_input = MethodExecutorInput(
            method = method_name,
            variables=query_interpreter_output,
            dataset_path=cleaned_path,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            # validation_info=method_validator_output,
            original_query = input_parsing_result["original_query"]
        )
        logger.debug(method_executor_input)
        method_executor_output = method_executor_tool.func(method_executor_input, original_query = input_parsing_result["original_query"])
        explainer_output = explanation_generator_tool.func(            method_info=method_info,
            validation_info=method_validator_output,
            variables=query_interpreter_output,
            results=method_executor_output,
            dataset_analysis=dataset_analysis_result,
            dataset_description=input_parsing_result["dataset_description"],
            original_query = input_parsing_result["original_query"])
        result = explainer_output
        #result['results']['results']["method_used"] = method_validator_output.get('method')
        logger.debug(result)
        logger.info("Causal analysis run finished.")
        
        # Remove the cleaned csv
        logger.info("Removing cleaned csv.")
        os.remove(cleaned_path)

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
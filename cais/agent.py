"""
Core class for the CausalAgent, which orchestrates the workflow of analyzing a dataset and query,
selecting and validating methods, cleaning the dataset, executing the method, and generating explanations. 
"""

from typing import Dict, List, Any, Optional
from cais.library import Estimators
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
from cais.models import Variables

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
    
# Set up basic logging
os.makedirs('./logs/', exist_ok=True)
logger = logging.getLogger(__name__)


class CausalAgent():
    
    def __init__(
            self,
            dataset_path: Union[str, pd.DataFrame], # dataset path or dataframe directly
            dataset_description: Optional[str] = None # Description of the dataset
    ):
        # Query not passed to constructor or saved so we can rerun different queries on the same dataset

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
            original_query=query
        ).analysis_results        
        
        # Analyse query based on dataset analysis and dataset description
        query_interpreter_output = query_interpreter_tool.func(
            dataset_analysis=dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query
        )
        
        self.dataset_analysis = dataset_analysis
        #self.query_interpreter_output = query_interpreter_output
        self.variables = query_interpreter_output.variables

    def select_method(self, query=None):

        query = self.checkq(query)

        method_selector_output = method_selector_tool.func(
            variables=self.variables,
            dataset_analysis=self.dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query,
            excluded_methods=None
        )

        self.selected_method = method_selector_output['method_info']


    def validate_method(self, query=None):
        '''
        Do not use yet
        '''
        query = self.checkq(query)

        # TODO: Move changes from assumption_checker branch to refactoring (this branch)

        method_validator_input = MethodValidatorInput(
                method_info=self.selected_method,
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
      
        self.variables = Variables(**controls_selector_output['variables'])

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

    def execute_method(self, query=None):
        
        query = self.checkq(query)

        estimator = self.estimators[
            convert[self.selected_method['selected_method']]
        ]

        self.results = estimator(
            df=self.load_dataset(cleaned=True),
            variables=self.variables,
            query=query
        )
    
    def run_analysis(self, query, use_decision_tree: Optional[bool] = True):

        #TODO: Check out if input_text and input_parsing_tool is still needed 

        logger.info("[Causal AI Scientist Stage 1] - Dataset and Query analysis")

        self.query = query

        self.analyse_dataset(
            query=query
        )
        self.select_method(
            query=query
        )
        self.clean_dataset(
            query=query
        )
        self.execute_method(
            query=query
        )
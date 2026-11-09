"""
Core class for the CausalAgent, which orchestrates the workflow of analyzing a dataset and query,
selecting and validating methods, cleaning the dataset, executing the method, and generating explanations. 
"""

from typing import Dict, List, Any, Optional
from cais.estimator_lib import Estimators
from cais.tools.input_parser_tool import input_parser_tool
from cais.tools.dataset_analyzer_tool import dataset_analyzer_tool
from cais.tools.query_interpreter_tool import query_interpreter_tool
from cais.tools.iv_discovery_tool import iv_discovery_tool
from cais.tools.method_selector_tool import method_selector_tool
from cais.tools.controls_selector_tool import controls_selector_tool
from cais.tools.method_validator_tool import method_validator_tool
from cais.tools.method_executor_tool import method_executor_tool
from cais.tools.explanation_generator_tool import explanation_generator_tool
from cais.tools.output_formatter_tool import output_formatter_tool

from cais.methods.linear_regression.estimator import LinearRegression
from cais.methods.regression_discontinuity.estimator import RDDRegression
from cais.methods.regression_discontinuity.donut_estimator import DonutRDD
from cais.methods.difference_in_differences.estimator import DiDRegression
from cais.methods.difference_in_differences.time_window import DiDTimeWindow
from cais.methods.instrumental_variable.estimator import IVRegression
from cais.methods.propensity_score.matching import PropensityScoreMatching
from cais.methods.propensity_score.weighting import TrimmedIPW
from cais.methods.frontdoor_adjustment.estimator import FrontdoorAdjustment
from cais.models import Variables, MethodInfo
from cais.components.assumption_checks import IVTest, ObervationalTest, DiDTest, RDDTest
from cais.components.decision_tree_v2 import select_method_v2 as run_decision_tree_v2

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
DID_TIME_WINDOW = "difference_in_differences_time_window"
REGRESSION_DISCONTINUITY = "regression_discontinuity_design"
DONUT_RDD = "donut_rdd"
PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
INSTRUMENTAL_VARIABLE = "instrumental_variable"
TRIMMED_IPW = "trimmed_ipw"
FRONTDOOR = "frontdoor_adjustment"

# Temporary name conversion
convert = {
    LINEAR_REGRESSION: LinearRegression.name,
    DIFF_IN_DIFF: DiDRegression.name,
    DID_TIME_WINDOW: DiDTimeWindow.name,
    REGRESSION_DISCONTINUITY: RDDRegression.name,
    DONUT_RDD: DonutRDD.name,
    INSTRUMENTAL_VARIABLE: IVRegression.name,
    PROPENSITY_SCORE_MATCHING: PropensityScoreMatching.name,
    TRIMMED_IPW: TrimmedIPW.name,
    FRONTDOOR: FrontdoorAdjustment.name,
}

os.makedirs('./logs/', exist_ok=True)
logging.basicConfig(
    filename='./logs/agent_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CausalAgent():
    
    def __init__(
            self,
            dataset_path: Union[str, pd.DataFrame], # dataset path or dataframe directly
            dataset_description: Optional[str] = None, # Description of the dataset
            model_name: Optional[str] = None,
            provider: Optional[str] = None,
            use_iv_pipeline: bool = False,
    ):
        # Query not passed to constructor or saved so we can rerun different queries on the same dataset

        self.use_iv_pipeline = use_iv_pipeline
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
        self.tree_result: Optional[Any] = None

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
            use_iv_pipeline=self.use_iv_pipeline,
            llm=self.llm
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
        self._strip_design_columns_from_covariates()

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

    # ------------------------------------------------------------------
    # Decision tree v2 (revised tree with gates and advisories)
    # ------------------------------------------------------------------

    def _tree_properties(self) -> Dict[str, Any]:
        """Translate interpreter output into decision_tree_v2 inputs.

        Inputs the interpreter does not produce yet (mediator, placebo period,
        bounded anticipation, local manipulation, donut width) are passed as
        unavailable so the tree reports can't tell instead of guessing. The panel
        unit identifier comes from the dataset analyzer's temporal structure when
        available; the group-composition check falls back to group_variable.
        """
        variables = self.variables
        analysis = self.dataset_analysis

        covariates = list(variables.covariates or [])
        for name in (variables.confounders or []):
            if name not in covariates:
                covariates.append(name)

        has_temporal = bool(variables.time_variable)
        if analysis is not None:
            has_temporal = has_temporal or bool(getattr(analysis, "temporal_structure_detected", False))

        # Panel unit identifier, used by the group-composition check. The
        # dataset analyzer detects it while profiling temporal structure; the
        # check falls back to group_variable if this is unavailable.
        unit_variable = None
        if analysis is not None:
            temporal = getattr(analysis, "temporal_structure", None)
            if temporal is not None:
                unit_variable = getattr(temporal, "id_column", None)

        return {
            "treatment_variable": variables.treatment_variable,
            "outcome_variable": variables.outcome_variable,
            "treatment_variable_type": variables.treatment_variable_type,
            "covariates": covariates,
            "instrument_variable": variables.instrument_variable,
            "mediator_variable": getattr(variables, "mediator_variable", None),
            "running_variable": variables.running_variable,
            "cutoff_value": variables.cutoff_value,
            "time_variable": variables.time_variable,
            "group_variable": variables.group_variable,
            "unit_variable": unit_variable,
            "donut_width": None,
            "treatment_period_start": variables.treatment_time,
            "placebo_period_start": getattr(variables, "placebo_period_start", None),
            "anticipation_window_start": None,
            "is_rct": bool(variables.is_rct),
            "has_temporal_structure": has_temporal,
            "is_encouragement_design": getattr(variables, "is_encouragement_design", None),
            # None -> the tree infers it from an instrument being present
            "has_pre_treatment_variables": bool(covariates),
            "has_valid_instrument": bool(variables.instrument_variable),
            "has_candidate_mediator": None,     # tree falls back to mediator_variable
            "has_valid_backdoor_set": bool(covariates),
            "is_structure_supported": True,
            "anticipation_known_bounded": False,
            "local_manipulation": False,
            "treatment_reference_level": getattr(variables, "treatment_reference_level", None),
        }

    def select_method_v2(self, query=None, prompt_callback=None):
        """Select a method with the revised decision tree (gates + advisories).

        Unlike the legacy selector this can deliberately stop the pipeline:
        check ``tree_result.ended`` (gate failure) and
        ``tree_result.implemented`` (leaf not implemented yet) before executing.
        """
        query = self.checkq(query)

        tree_result = run_decision_tree_v2(
            self._tree_properties(),
            df=self.load_dataset(),
            description=self.dataset_description,
            llm=self.llm,
            prompt_callback=prompt_callback,
        )
        self.tree_result = tree_result
        self.selected_method = tree_result.method

        if tree_result.ended:
            justification = f"Decision tree v2 stopped: {tree_result.end_reason}"
        else:
            justification = f"Decision tree v2 selected {tree_result.method} along: {tree_result.path}"
            if tree_result.variant:
                justification += f" (variant: {tree_result.variant})"
        execution = dict(getattr(tree_result, "execution", {}) or {})
        display_name = execution.get("display_name")
        method_name = display_name or (
            tree_result.method.replace("_", " ").title() if tree_result.method else None
        )
        self.method_info = MethodInfo(
            selected_method=tree_result.method,
            method_name=method_name,
            method_justification=justification,
            method_assumptions=list(tree_result.assumptions.keys()),
            decision_tree=tree_result.to_dict(),
        )
        return tree_result

    def _tree_validation_info(self) -> Optional[Dict[str, Any]]:
        """Shape the tree result the way the explanation generator expects."""
        if self.tree_result is None:
            return None
        result = self.tree_result
        return {
            "method": result.method,
            "variant": result.variant,
            "path": result.path,
            "steps": [
                {"node": s.node, "kind": s.kind, "outcome": s.outcome, "detail": s.detail}
                for s in result.steps
            ],
            "assumptions": {name: verdict.model_dump() for name, verdict in result.assumptions.items()},
            "concerns": list(result.warnings),
            "planned_post_checks": list(result.planned_post_checks),
            "post_checks": self._post_check_advisories(),
        }

    # ------------------------------------------------------------------
    # Advisory post-modeling checks (output only, never blocking)
    # ------------------------------------------------------------------

    def _results_diagnostics(self) -> Dict[str, Any]:
        """Diagnostics from the last execution, independent of result nesting."""
        results = self.results if isinstance(self.results, dict) else {}
        nested = results.get("results")
        payload = nested if isinstance(nested, dict) else results
        diagnostics = payload.get("diagnostics") if isinstance(payload, dict) else None
        return diagnostics if isinstance(diagnostics, dict) else {}

    def _reported_post_checks(self) -> Dict[str, Any]:
        """Post-check verdicts the estimator reported in its diagnostics."""
        reported = self._results_diagnostics().get("post_checks")
        return dict(reported) if isinstance(reported, dict) else {}

    def _gps_balance_verdict(self) -> Optional[Dict[str, Any]]:
        """Normalize the GPS estimator's balance diagnostics into a verdict."""
        balance = self._results_diagnostics().get("balance_check")
        if not isinstance(balance, dict):
            return None
        summary = balance.get("summary_stats") or {}
        n_imbalanced = summary.get("num_covariates_potentially_imbalanced_p005")
        if n_imbalanced is None:
            return {
                "passed": None,
                "reasoning": "the GPS diagnostics did not report an imbalance count",
            }
        n_imbalanced = int(n_imbalanced)
        return {
            "passed": n_imbalanced == 0,
            "reasoning": (
                f"{n_imbalanced} covariate(s) potentially imbalanced across GPS strata "
                "(p < 0.05 heuristic)." if n_imbalanced else
                "no covariate was flagged as imbalanced across GPS strata (p < 0.05 heuristic)."
            ),
        }

    def _post_check_advisories(self) -> List[Dict[str, Any]]:
        """Verdicts for the advisory post-modeling checks planned by the tree.

        These are output-only: they never change the estimate or stop the run.
        A planned check the estimator did not report is surfaced as
        ``passed=None`` ("could not be evaluated") rather than silently dropped.
        """
        planned = list(getattr(self.tree_result, "planned_post_checks", []) or [])
        reported = self._reported_post_checks()
        advisories: List[Dict[str, Any]] = []
        for name in planned:
            verdict = reported.get(name)
            if verdict is None and name == "balance_after_gps":
                verdict = self._gps_balance_verdict()
            if verdict is None and name == "gps_specification":
                verdict = {
                    "passed": None,
                    "reasoning": (
                        "the GPS model-specification check (residual normality) is not "
                        "implemented in this version"
                    ),
                }
            if not isinstance(verdict, dict):
                verdict = {"passed": None, "reasoning": "the estimator did not report this check"}
            advisories.append({
                "name": name,
                "passed": verdict.get("passed"),
                "reasoning": verdict.get("reasoning") or "",
            })
        return advisories

    def _tree_stop_output(self) -> Dict[str, Any]:
        """Result for a study the tree refused to hand to an estimator."""
        result = self.tree_result
        reason = result.end_reason or f"{result.method} is not implemented yet."

        lines = [f"No causal effect was estimated. The decision tree stopped: {reason}", "", "Decision path:"]
        for step in result.steps:
            detail = f" ({step.detail})" if step.detail else ""
            lines.append(f"- {step.node}: {step.outcome}{detail}")
        if result.assumptions:
            lines.append("")
            lines.append("Checks:")
            for name, verdict in result.assumptions.items():
                label = {True: "pass", False: "fail", None: "can't tell"}[verdict.passed]
                lines.append(f"- {name}: {label} - {verdict.reasoning}")

        self.results = {
            "status": "not_estimated",
            "method": result.method,
            "end_reason": reason,
            "tree": result.to_dict(),
        }
        self.explanations = "\n".join(lines)
        return {"results": self.results, "explanation": self.explanations}

    def discover_instruments(self, query=None):
        query = self.checkq(query)

        iv_discovery_output = iv_discovery_tool.func(
            variables=self.variables,
            dataset_analysis=self.dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query
        )
        
        if hasattr(iv_discovery_output, "model_dump"):
            iv_discovery_output_dict = iv_discovery_output.model_dump()
        else:
            iv_discovery_output_dict = iv_discovery_output
            
        self.variables = Variables(**iv_discovery_output_dict["variables"])
        self._strip_design_columns_from_covariates()
        return self.variables

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

        controls_selector_output = controls_selector_tool.func(
            method_name=self.selected_method,
            variables=self.variables,
            dataset_analysis=self.dataset_analysis,
            dataset_description=self.dataset_description,
            original_query=query,
        )
      
        self.variables = Variables(**controls_selector_output['variables']) # refined controls; need to update after cleaning
        self._strip_design_columns_from_covariates()

    def _strip_design_columns_from_covariates(self) -> None:
        """Remove design-role columns from the covariate/confounder lists.

        Columns that define the identification design (treatment, outcome,
        running variable, instrument, mediator) must never appear in the
        covariate list: downstream consumers (e.g. the RDD covariate-continuity
        check) would otherwise test a column against the rule that defines it
        and read the rule itself as a violation.
        """
        variables = self.variables
        design = {
            variables.treatment_variable, variables.outcome_variable,
            variables.running_variable, variables.instrument_variable,
            getattr(variables, "mediator_variable", None),
        }
        design = {name for name in design if name}
        if not design:
            return
        removed = [c for c in (variables.covariates or []) if c in design]
        variables.covariates = [c for c in (variables.covariates or []) if c not in design]
        if removed and hasattr(variables, "confounders"):
            variables.confounders = [
                c for c in (variables.confounders or []) if c not in design
            ]
        if removed:
            logger.info(
                "Removed design columns %s from covariate list", removed
            )

    def clean_dataset(self, query=None):   

        query = self.checkq(query)

        cleaning_output = dataset_cleaner_tool.func(
            dataset_path=self.dataset_path,
            variables=self.variables.model_dump(),
            dataset_description=self.dataset_description,
            original_query=query,
            causal_method=self.selected_method
        )
        self.cleaned_dataset_path = cleaning_output.get("cleaned_dataset_path")
        
        # Check if file was actually created/returned
        if not self.cleaned_dataset_path or not os.path.exists(self.cleaned_dataset_path):
            stderr = cleaning_output.get("stderr", "No stderr available.")
            logger.error(f"Dataset cleaning failed to produce a file at {self.cleaned_dataset_path}. Stderr: {stderr}")
            raise FileNotFoundError(f"Cleaned dataset NOT found at {self.cleaned_dataset_path}. Cleaning stderr: {stderr}")
            
        return self.cleaned_dataset_path

    def execute_method(self, query=None, remove_cleaned=True):
        
        query = self.checkq(query)
        logger.info(f"Starting method execution. Trying to run {self.selected_method}")

        # The tree may hand down leaf-specific execution settings (e.g. the
        # propensity-score bounds for trimmed IPW). The leaf id stays in the
        # trace, while ``execution.method`` names the estimator to run.
        execution = dict(getattr(self.tree_result, "execution", {}) or {})
        execution_method = execution.get("method") or self.selected_method
        method_options = dict(execution.get("options") or {})
        # Leaf-level variable remap: e.g. the ITT leaf runs diff-in-means with
        # the assignment column as the grouping variable, not the received
        # treatment. Applied before data prep so NaN handling keys on the
        # right column on both primary and fallback execution paths.
        leaf_treatment = method_options.pop("treatment_variable", None)
        if leaf_treatment and self.variables is not None:
            logger.info(
                "Execution leaf overrides the treatment column: %s -> %s",
                self.variables.treatment_variable,
                leaf_treatment,
            )
            self.variables.treatment_variable = leaf_treatment
        logger.info(f"Executing '{execution_method}' with options {method_options}")
        try:
            estimator = self.estimators[
                convert[execution_method]
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
                query=query,
                **method_options
            ) | self.llm_info # append llm info
        except:
            method_executor_input = MethodExecutorInput(
                        method = execution_method,
                        variables=self.variables,
                        dataset_path=self.cleaned_dataset_path,
                        dataset_analysis=self.dataset_analysis,
                        dataset_description=self.dataset_description,
                        original_query = query,
                        method_options=method_options or None,
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
            validation_info=self._tree_validation_info(),
            dataset_description=self.dataset_description,
            original_query=query
        )['explanation']

        if self.cleaned_dataset_path and remove_cleaned:
            if isinstance(self.load_dataset(cleaned=True), pd.DataFrame):
                # os.remove(self.cleaned_dataset_path)
                self.cleaned_dataset_path=None
                logger.info("Succesfully Removed Cleaned Dataset.")

        return {
            "results" : self.results,
            "explanation": self.explanations
        }
    
    def run_analysis(self, query, llm_method_selection: Optional[bool] = True,
                     method_selection: str = "tree_v2",
                     prompt_callback: Optional[Any] = None,
                     interactive: bool = True):

        logger.info("[Causal AI Scientist Stage 1] - Dataset and Query analysis")

        self.query = query

        self.analyse_dataset(
            query=query
        )
        if method_selection == "legacy":
            self.select_method(
                query=query,
                llm_decision=llm_method_selection
            )
        else:
            tree_result = self.select_method_v2(
                query=query,
                prompt_callback=prompt_callback if interactive else None,
            )
            if tree_result.ended or not tree_result.implemented:
                logger.info("Decision tree v2 stopped: %s", tree_result.end_reason or tree_result.method)
                return self._tree_stop_output()
        if self.selected_method == INSTRUMENTAL_VARIABLE and self.use_iv_pipeline:
            logger.info("Instrumental Variable method selected. Running IV Discovery...")
            self.discover_instruments(
                query=query
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
                        use_method_validator: bool = True,
                        use_iv_pipeline: bool = False) -> Dict[str, Any]:
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

        dataset_analysis_result = dataset_analyzer_tool.func(dataset_path=input_parsing_result["dataset_path"], dataset_description=input_parsing_result["dataset_description"], original_query=input_parsing_result["original_query"], use_iv_pipeline=use_iv_pipeline).analysis_results
        
        query_info = QueryInfo(
            query_text=input_parsing_result["original_query"],
            potential_treatments=input_parsing_result["extracted_variables"].get("treatment"),
            potential_outcomes=input_parsing_result["extracted_variables"].get("outcome"),
            covariates_hints=input_parsing_result["extracted_variables"].get("covariates_mentioned"),
            instrument_hints=input_parsing_result["extracted_variables"].get("instruments_mentioned")
        )

        query_interpreter_output = query_interpreter_tool.func(dataset_analysis=dataset_analysis_result, dataset_description=input_parsing_result["dataset_description"], original_query=input_parsing_result["original_query"]).variables

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
        
        if method_name == INSTRUMENTAL_VARIABLE and use_iv_pipeline:
            logger.info("Instrumental Variable method selected. Running IV Discovery...")
            iv_discovery_output = iv_discovery_tool.func(
                variables=query_interpreter_output,
                dataset_analysis=dataset_analysis_result,
                dataset_description=input_parsing_result["dataset_description"],
                original_query=input_parsing_result["original_query"]
            )
            # update variables
            if hasattr(iv_discovery_output, "model_dump"):
                iv_discovery_output_dict = iv_discovery_output.model_dump()
            else:
                iv_discovery_output_dict = iv_discovery_output
            query_interpreter_output = Variables(**iv_discovery_output_dict["variables"])

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
        
        # include query_info in result
        result["query_info"] = {
            "query_text": input_parsing_result["original_query"],
            "potential_treatments": input_parsing_result["extracted_variables"].get("treatment"),
            "potential_outcomes": input_parsing_result["extracted_variables"].get("outcome"),
            "covariates_hints": input_parsing_result["extracted_variables"].get("covariates_mentioned"),
            "instrument_hints": input_parsing_result["extracted_variables"].get("instruments_mentioned")
        }
        
        #result['results']['results']["method_used"] = method_validator_output.get('method')
        logger.debug(result)
        logger.info("Causal analysis run finished.")
        
        # Remove the cleaned csv
        # logger.info("Removing cleaned csv.")
        # os.remove(cleaned_path)

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
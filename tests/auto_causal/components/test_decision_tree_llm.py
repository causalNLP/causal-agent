import unittest
from unittest.mock import patch, MagicMock
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from auto_causal.components.decision_tree_llm import DecisionTreeLLMEngine
from auto_causal.components.decision_tree import (
    METHOD_ASSUMPTIONS,
    CORRELATION_ANALYSIS,
    DIFF_IN_DIFF,
    INSTRUMENTAL_VARIABLE,
    LINEAR_REGRESSION,
    PROPENSITY_SCORE_MATCHING,
    REGRESSION_DISCONTINUITY,
    DIFF_IN_MEANS
)

class TestDecisionTreeLLMEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionTreeLLMEngine(verbose=False)
        self.mock_dataset_analysis = {
            "temporal_structure": {"has_temporal_structure": True, "time_variables": ["year"]},
            "potential_instruments": ["Z1"],
            "running_variable_analysis": {"is_candidate": False}
        }
        self.mock_variables = {
            "treatment_variable": "T",
            "outcome_variable": "Y",
            "covariates": ["X1", "X2"],
            "time_variable": "year",
            "instrument_variable": "Z1",
            "treatment_variable_type": "binary"
        }
        self.mock_llm = MagicMock(spec=BaseChatModel)

    def _create_mock_llm_response(self, response_dict):
        ai_message = AIMessage(content=json.dumps(response_dict))
        self.mock_llm.invoke = MagicMock(return_value=ai_message)

    def _create_mock_llm_raw_response(self, raw_content_str):
        ai_message = AIMessage(content=raw_content_str)
        self.mock_llm.invoke = MagicMock(return_value=ai_message)

    def test_select_method_rct_no_covariates_llm_selects_diff_in_means(self):
        self._create_mock_llm_response({
            "selected_method": DIFF_IN_MEANS,
            "method_justification": "LLM: RCT with no covariates, DiM is appropriate.",
            "alternative_methods": []
        })
        rct_variables = self.mock_variables.copy()
        rct_variables["covariates"] = []
        result = self.engine.select_method(
            self.mock_dataset_analysis, rct_variables, is_rct=True, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], DIFF_IN_MEANS)
        self.assertEqual(result["method_justification"], "LLM: RCT with no covariates, DiM is appropriate.")
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[DIFF_IN_MEANS])
        self.mock_llm.invoke.assert_called_once()

    def test_select_method_rct_with_covariates_llm_selects_linear_regression(self):
        self._create_mock_llm_response({
            "selected_method": LINEAR_REGRESSION,
            "method_justification": "LLM: RCT with covariates, Linear Regression for precision.",
            "alternative_methods": []
        })
        result = self.engine.select_method(
            self.mock_dataset_analysis, self.mock_variables, is_rct=True, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], LINEAR_REGRESSION)
        self.assertEqual(result["method_justification"], "LLM: RCT with covariates, Linear Regression for precision.")
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[LINEAR_REGRESSION])

    def test_select_method_observational_temporal_llm_selects_did(self):
        self._create_mock_llm_response({
            "selected_method": DIFF_IN_DIFF,
            "method_justification": "LLM: Observational with temporal data, DiD selected.",
            "alternative_methods": [INSTRUMENTAL_VARIABLE]
        })
        result = self.engine.select_method(
            self.mock_dataset_analysis, self.mock_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], DIFF_IN_DIFF)
        self.assertEqual(result["method_justification"], "LLM: Observational with temporal data, DiD selected.")
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[DIFF_IN_DIFF])
        self.assertEqual(result["alternative_methods"], [INSTRUMENTAL_VARIABLE])

    def test_select_method_observational_instrument_llm_selects_iv(self):
        # Modify dataset analysis to not strongly suggest DiD
        no_temporal_analysis = self.mock_dataset_analysis.copy()
        no_temporal_analysis["temporal_structure"] = {"has_temporal_structure": False}
        
        self._create_mock_llm_response({
            "selected_method": INSTRUMENTAL_VARIABLE,
            "method_justification": "LLM: Observational with instrument, IV selected.",
            "alternative_methods": []
        })
        result = self.engine.select_method(
            no_temporal_analysis, self.mock_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], INSTRUMENTAL_VARIABLE)
        self.assertEqual(result["method_justification"], "LLM: Observational with instrument, IV selected.")
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE])

    def test_select_method_observational_running_var_llm_selects_rdd(self):
        rdd_analysis = self.mock_dataset_analysis.copy()
        rdd_analysis["temporal_structure"] = {"has_temporal_structure": False} # Make DiD less likely
        rdd_variables = self.mock_variables.copy()
        rdd_variables["instrument_variable"] = None # Make IV less likely
        rdd_variables["running_variable"] = "age"
        rdd_variables["cutoff_value"] = 65
        
        self._create_mock_llm_response({
            "selected_method": REGRESSION_DISCONTINUITY,
            "method_justification": "LLM: Running var and cutoff, RDD selected.",
            "alternative_methods": []
        })
        result = self.engine.select_method(
            rdd_analysis, rdd_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], REGRESSION_DISCONTINUITY)
        self.assertEqual(result["method_justification"], "LLM: Running var and cutoff, RDD selected.")
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[REGRESSION_DISCONTINUITY])

    def test_select_method_observational_covariates_llm_selects_psm(self):
        psm_analysis = {"temporal_structure": {"has_temporal_structure": False}}
        psm_variables = {
            "treatment_variable": "T", "outcome_variable": "Y", "covariates": ["X1", "X2"],
            "treatment_variable_type": "binary"
        }
        self._create_mock_llm_response({
            "selected_method": PROPENSITY_SCORE_MATCHING,
            "method_justification": "LLM: Observational with covariates, PSM.",
            "alternative_methods": []
        })
        result = self.engine.select_method(
            psm_analysis, psm_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], PROPENSITY_SCORE_MATCHING)
        self.assertEqual(result["method_justification"], "LLM: Observational with covariates, PSM.")
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[PROPENSITY_SCORE_MATCHING])

    def test_select_method_no_llm_provided_defaults_to_correlation(self):
        result = self.engine.select_method(
            self.mock_dataset_analysis, self.mock_variables, is_rct=False, llm=None
        )
        self.assertEqual(result["selected_method"], CORRELATION_ANALYSIS)
        self.assertIn("LLM client not provided", result["method_justification"])
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[CORRELATION_ANALYSIS])

    def test_select_method_llm_returns_malformed_json_defaults_to_correlation(self):
        self._create_mock_llm_raw_response("This is not a valid JSON")
        result = self.engine.select_method(
            self.mock_dataset_analysis, self.mock_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], CORRELATION_ANALYSIS)
        self.assertIn("LLM response was not valid JSON", result["method_justification"])
        self.assertIn("This is not a valid JSON", result["method_justification"])
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[CORRELATION_ANALYSIS])

    def test_select_method_llm_returns_unknown_method_defaults_to_correlation(self):
        self._create_mock_llm_response({
            "selected_method": "SUPER_NOVEL_METHOD_X",
            "method_justification": "LLM thinks this is best.",
            "alternative_methods": []
        })
        result = self.engine.select_method(
            self.mock_dataset_analysis, self.mock_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], CORRELATION_ANALYSIS)
        self.assertIn("LLM output was problematic (selected: SUPER_NOVEL_METHOD_X)", result["method_justification"])
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[CORRELATION_ANALYSIS])

    def test_select_method_llm_call_raises_exception_defaults_to_correlation(self):
        self.mock_llm.invoke = MagicMock(side_effect=Exception("LLM API Error"))
        result = self.engine.select_method(
            self.mock_dataset_analysis, self.mock_variables, is_rct=False, llm=self.mock_llm
        )
        self.assertEqual(result["selected_method"], CORRELATION_ANALYSIS)
        self.assertIn("An unexpected error occurred during LLM method selection.", result["method_justification"])
        self.assertIn("LLM API Error", result["method_justification"])
        self.assertEqual(result["method_assumptions"], METHOD_ASSUMPTIONS[CORRELATION_ANALYSIS])

    def test_prompt_construction_content(self):
        actual_prompt_generated = []  # List to capture the prompt

        # Store the original method before patching
        original_construct_prompt = self.engine._construct_prompt

        def side_effect_for_construct_prompt(dataset_analysis, variables, is_rct):
            # Call the original _construct_prompt method using the stored original
            # self.engine is the instance, so it's implicitly passed if original_construct_prompt is bound
            # However, to be explicit and safe, if we treat original_construct_prompt as potentially unbound:
            prompt = original_construct_prompt(dataset_analysis, variables, is_rct)
            actual_prompt_generated.append(prompt)
            return prompt

        with patch.object(self.engine, '_construct_prompt', side_effect=side_effect_for_construct_prompt) as mock_construct_prompt:
            self._create_mock_llm_response({ # Need a mock response for the select_method to run
                "selected_method": DIFF_IN_DIFF, "method_justification": "Test", "alternative_methods": []
            })
            self.engine.select_method(self.mock_dataset_analysis, self.mock_variables, False, self.mock_llm)
            
            mock_construct_prompt.assert_called_once_with(self.mock_dataset_analysis, self.mock_variables, False)
            
            self.assertTrue(actual_prompt_generated, "Prompt was not generated or captured by side_effect")
            prompt_string = actual_prompt_generated[0]
            
            self.assertIn("You are an expert in causal inference.", prompt_string)
            self.assertIn(json.dumps(self.mock_dataset_analysis, indent=2), prompt_string)
            self.assertIn(json.dumps(self.mock_variables, indent=2), prompt_string)
            self.assertIn("Is the data from a Randomized Controlled Trial (RCT)? No", prompt_string)
            self.assertIn(f"- {DIFF_IN_DIFF}", prompt_string) # Check if method descriptions are there
            self.assertIn(f"- {INSTRUMENTAL_VARIABLE}", prompt_string)
            self.assertIn("Output your final decision as a JSON object", prompt_string)

    def test_llm_response_with_triple_backticks_json(self):
        raw_response = """
Some conversational text before the JSON.
```json
{
    "selected_method": "difference_in_differences",
    "method_justification": "LLM reasoned and selected DiD.",
    "alternative_methods": ["instrumental_variable"]
}
```
And some text after.
        """
        self._create_mock_llm_raw_response(raw_response)
        result = self.engine.select_method(self.mock_dataset_analysis, self.mock_variables, False, self.mock_llm)
        self.assertEqual(result["selected_method"], DIFF_IN_DIFF)
        self.assertEqual(result["method_justification"], "LLM reasoned and selected DiD.")

    def test_llm_response_with_triple_backticks_only(self):
        raw_response = """
```
{
    "selected_method": "difference_in_differences",
    "method_justification": "LLM reasoned and selected DiD with only triple backticks.",
    "alternative_methods": ["instrumental_variable"]
}
```
        """
        self._create_mock_llm_raw_response(raw_response)
        result = self.engine.select_method(self.mock_dataset_analysis, self.mock_variables, False, self.mock_llm)
        self.assertEqual(result["selected_method"], DIFF_IN_DIFF)
        self.assertEqual(result["method_justification"], "LLM reasoned and selected DiD with only triple backticks.")


    def test_llm_response_plain_json(self):
        raw_response = """
{
    "selected_method": "difference_in_differences",
    "method_justification": "LLM reasoned and selected DiD plain JSON.",
    "alternative_methods": ["instrumental_variable"]
}
        """
        self._create_mock_llm_raw_response(raw_response)
        result = self.engine.select_method(self.mock_dataset_analysis, self.mock_variables, False, self.mock_llm)
        self.assertEqual(result["selected_method"], DIFF_IN_DIFF)
        self.assertEqual(result["method_justification"], "LLM reasoned and selected DiD plain JSON.")


if __name__ == '__main__':
    unittest.main() 
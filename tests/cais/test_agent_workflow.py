import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

import cais.agent as cais_agent
from cais.agent import CausalAgent
from cais.methods.propensity_score.weighting import TrimmedIPW
from cais.models import AssumptionResult, Variables
from cais.components.decision_tree_v2 import TreeResult, TreeStep


def create_dummy_csv(path='dummy_e2e_test_data.csv'):
    df = pd.DataFrame({
        'treatment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'outcome': [10, 12, 11, 13, 9, 14, 10, 15, 11, 16],
        'covariate1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'covariate2': [5.5, 6.5, 5.8, 6.2, 5.1, 6.8, 5.3, 6.1, 5.9, 6.3],
    })
    df.to_csv(path, index=False)
    return path


class TestAgentWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy_data_path = create_dummy_csv()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_data_path):
            os.remove(cls.dummy_data_path)

    @patch('cais.agent.run_causal_analysis')
    def test_agent_invocation_mock(self, mock_run):
        """Smoke test: run_causal_analysis can be called and returns a result."""
        mock_run.return_value = {"explanation": "Agent invoked successfully (mocked)", "results": {}}

        result = cais_agent.run_causal_analysis(
            "What is the effect of treatment on outcome?",
            self.dummy_data_path,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)

    @patch('cais.agent.get_llm_client')
    def test_causal_agent_initialization(self, mock_get_llm):
        """CausalAgent initializes with correct attributes."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        agent = CausalAgent(
            dataset_path=self.dummy_data_path,
            dataset_description="A simple test dataset.",
        )

        # Core attributes should be set
        self.assertEqual(agent.dataset_path, self.dummy_data_path)
        self.assertEqual(agent.dataset_description, "A simple test dataset.")
        self.assertIs(agent.llm, mock_llm)
        self.assertIsNotNone(agent.estimators)

        # Pipeline states should start as None
        self.assertIsNone(agent.dataset_analysis)
        self.assertIsNone(agent.variables)
        self.assertIsNone(agent.selected_method)
        self.assertIsNone(agent.results)

    @patch('cais.agent.get_llm_client')
    def test_causal_agent_load_dataset(self, mock_get_llm):
        """CausalAgent.load_dataset() returns a DataFrame from the CSV path."""
        mock_get_llm.return_value = MagicMock()

        agent = CausalAgent(dataset_path=self.dummy_data_path)
        df = agent.load_dataset()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("treatment", df.columns)
        self.assertIn("outcome", df.columns)
        self.assertEqual(len(df), 10)

    @patch('cais.agent.get_llm_client')
    def test_causal_agent_checkq_stores_and_retrieves_query(self, mock_get_llm):
        """CausalAgent.checkq() stores and later retrieves the last used query."""
        mock_get_llm.return_value = MagicMock()

        agent = CausalAgent(dataset_path=self.dummy_data_path)

        # First call stores the query
        returned = agent.checkq("What is the effect of X on Y?")
        self.assertEqual(returned, "What is the effect of X on Y?")
        self.assertEqual(agent.last_used_query, "What is the effect of X on Y?")

        # Calling with None falls back to the stored query
        returned_again = agent.checkq(None)
        self.assertEqual(returned_again, "What is the effect of X on Y?")


class TestDecisionTreeV2Wiring(unittest.TestCase):
    """Agent wiring for the revised tree: no LLM calls, tree stubbed."""

    @classmethod
    def setUpClass(cls):
        cls.dummy_data_path = create_dummy_csv('dummy_tree_v2_test_data.csv')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_data_path):
            os.remove(cls.dummy_data_path)

    def _make_agent(self):
        with patch('cais.agent.get_llm_client', return_value=MagicMock()):
            agent = CausalAgent(
                dataset_path=self.dummy_data_path,
                dataset_description="A simple test dataset.",
            )
        agent.dataset_analysis = None
        agent.variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            treatment_variable_type="binary",
            covariates=["covariate1", "covariate2"],
            is_rct=False,
        )
        return agent

    def test_tree_properties_translation(self):
        agent = self._make_agent()
        agent.variables.confounders = ["covariate1", "extra_confounder"]

        props = agent._tree_properties()

        self.assertEqual(props["treatment_variable"], "treatment")
        self.assertEqual(props["outcome_variable"], "outcome")
        # confounders are merged into covariates without duplicates
        self.assertEqual(props["covariates"], ["covariate1", "covariate2", "extra_confounder"])
        self.assertTrue(props["has_valid_backdoor_set"])
        self.assertFalse(props["has_valid_instrument"])
        self.assertFalse(props["has_temporal_structure"])
        self.assertIsNone(props["mediator_variable"])
        self.assertIsNone(props["placebo_period_start"])

    def test_tree_properties_map_new_design_fields(self):
        agent = self._make_agent()
        agent.variables.mediator_variable = "mediator"
        agent.variables.placebo_period_start = 1986.0
        agent.variables.is_encouragement_design = True

        props = agent._tree_properties()

        self.assertEqual(props["mediator_variable"], "mediator")
        self.assertEqual(props["placebo_period_start"], 1986.0)
        self.assertIsNone(props["has_candidate_mediator"])  # tree falls back to mediator_variable
        self.assertTrue(props["is_encouragement_design"])

    @patch('cais.agent.run_decision_tree_v2')
    def test_select_method_v2_records_result(self, mock_tree):
        agent = self._make_agent()
        mock_tree.return_value = TreeResult(
            method="difference_in_differences",
            steps=[TreeStep("Gate: parallel trends", "gate", "pass")],
            assumptions={"parallel_trends": AssumptionResult(passed=True, reasoning="p=0.4")},
        )

        result = agent.select_method_v2()

        self.assertEqual(result.method, "difference_in_differences")
        self.assertEqual(agent.selected_method, "difference_in_differences")
        self.assertEqual(agent.method_info.selected_method, "difference_in_differences")
        self.assertEqual(agent.method_info.decision_tree["method"], "difference_in_differences")
        self.assertIn("Decision tree v2 selected", agent.method_info.method_justification)
        # the raw dataframe and description are handed to the tree
        kwargs = mock_tree.call_args.kwargs
        self.assertEqual(kwargs["description"], "A simple test dataset.")
        self.assertIsInstance(kwargs["df"], pd.DataFrame)

    def test_validation_info_shape_matches_explainer(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            method="propensity_score_matching",
            steps=[TreeStep("leaf", "leaf", "selected")],
            assumptions={"positivity": AssumptionResult(passed=True, reasoning="overlap ok", details={"n": 1})},
            warnings=["SUTVA could not be assessed (advisory)."],
            planned_post_checks=["balance_after_matching"],
        )

        info = agent._tree_validation_info()

        for key in ("method", "variant", "path", "steps", "assumptions", "concerns",
                    "planned_post_checks", "post_checks"):
            self.assertIn(key, info)
        self.assertEqual(info["method"], "propensity_score_matching")
        self.assertEqual(info["assumptions"]["positivity"]["passed"], True)
        self.assertIn("SUTVA could not be assessed (advisory).", info["concerns"])
        self.assertEqual(info["planned_post_checks"], ["balance_after_matching"])

    def test_validation_info_includes_post_check_advisories(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            method="propensity_score_matching",
            planned_post_checks=["balance_after_matching", "gps_specification"],
        )
        agent.results = {
            "effect_estimate": 1.0,
            "diagnostics": {
                "post_checks": {
                    "balance_after_matching": {
                        "passed": False,
                        "reasoning": "Still imbalanced: ['x1'].",
                    }
                }
            },
        }

        info = agent._tree_validation_info()

        checks = info["post_checks"]
        self.assertEqual([item["name"] for item in checks],
                         ["balance_after_matching", "gps_specification"])
        self.assertFalse(checks[0]["passed"])
        self.assertIn("Still imbalanced", checks[0]["reasoning"])
        # Planned but unevaluated checks are surfaced as "could not be evaluated".
        self.assertIsNone(checks[1]["passed"])
        self.assertTrue(checks[1]["reasoning"])

    def test_post_check_advisories_read_nested_fallback_results(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            method="propensity_score_weighting",
            planned_post_checks=["balance_after_weighting"],
        )
        agent.results = {
            "results": {
                "diagnostics": {
                    "post_checks": {
                        "balance_after_weighting": {
                            "passed": True,
                            "reasoning": "All balanced after IPW.",
                        }
                    }
                }
            }
        }

        info = agent._tree_validation_info()

        self.assertTrue(info["post_checks"][0]["passed"])
        self.assertIn("All balanced after IPW", info["post_checks"][0]["reasoning"])

    def test_tree_stop_output_for_gate_failure(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            ended=True,
            end_reason="Parallel trends violated.",
            steps=[
                TreeStep("Gate: parallel trends", "gate", "fail", "p=0.01"),
                TreeStep("END", "end", "stop", "Parallel trends violated."),
            ],
            assumptions={"parallel_trends": AssumptionResult(passed=False, reasoning="p=0.01")},
        )

        output = agent._tree_stop_output()

        self.assertEqual(output["results"]["status"], "not_estimated")
        self.assertEqual(output["results"]["end_reason"], "Parallel trends violated.")
        self.assertIn("Parallel trends violated.", output["explanation"])
        self.assertIn("Gate: parallel trends: fail (p=0.01)", output["explanation"])
        self.assertIn("parallel_trends: fail - p=0.01", output["explanation"])
        self.assertIn("Checks:", output["explanation"])

    def test_tree_stop_output_for_unimplemented_leaf(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            method="donut_rdd",
            implemented=False,
            warnings=["donut_rdd is not implemented yet; this study cannot be estimated."],
            steps=[TreeStep("donut_rdd", "leaf", "selected (not implemented)")],
        )

        output = agent._tree_stop_output()

        self.assertIn("donut_rdd is not implemented yet", output["explanation"])

    @patch('cais.agent.run_decision_tree_v2')
    def test_run_analysis_stops_before_controls_and_execution(self, mock_tree):
        agent = self._make_agent()
        agent.analyse_dataset = MagicMock()
        agent.select_controls = MagicMock()
        agent.clean_dataset = MagicMock()
        agent.execute_method = MagicMock()
        mock_tree.return_value = TreeResult(ended=True, end_reason="Group composition is not stable; pipeline stopped.")

        output = agent.run_analysis("Does treatment affect outcome?", method_selection="tree_v2")

        self.assertEqual(output["results"]["status"], "not_estimated")
        agent.select_controls.assert_not_called()
        agent.clean_dataset.assert_not_called()
        agent.execute_method.assert_not_called()

    @patch('cais.agent.run_decision_tree_v2')
    def test_interactive_flag_controls_prompt_callback(self, mock_tree):
        mock_tree.return_value = TreeResult(method="difference_in_differences", implemented=True)
        agent = self._make_agent()
        agent.analyse_dataset = MagicMock()
        agent.select_controls = MagicMock()
        agent.clean_dataset = MagicMock()
        agent.execute_method = MagicMock(return_value={"results": {}, "explanation": "ok"})
        callback = MagicMock(return_value="answer")

        agent.run_analysis("q", prompt_callback=callback)  # interactive defaults to True
        self.assertIs(mock_tree.call_args.kwargs["prompt_callback"], callback)

        agent.run_analysis("q", prompt_callback=callback, interactive=False)
        self.assertIsNone(mock_tree.call_args.kwargs["prompt_callback"])

    @patch('cais.agent.run_decision_tree_v2')
    def test_run_analysis_continues_for_implemented_leaf(self, mock_tree):
        agent = self._make_agent()
        agent.analyse_dataset = MagicMock()
        agent.select_controls = MagicMock()
        agent.clean_dataset = MagicMock()
        agent.execute_method = MagicMock(return_value={"results": {}, "explanation": "done"})
        mock_tree.return_value = TreeResult(method="difference_in_differences", implemented=True)

        output = agent.run_analysis("Does treatment affect outcome?", method_selection="tree_v2")

        agent.select_controls.assert_called_once()
        agent.clean_dataset.assert_called_once()
        agent.execute_method.assert_called_once()
        self.assertEqual(output["explanation"], "done")

    @patch('cais.agent.run_decision_tree_v2')
    def test_select_method_v2_uses_execution_display_name(self, mock_tree):
        agent = self._make_agent()
        mock_tree.return_value = TreeResult(
            method="trimmed_ipw",
            implemented=True,
            execution={
                "method": "trimmed_ipw",
                "display_name": "Trimmed Inverse Probability Weighting",
                "options": {"ps_bounds": [0.1, 0.9]},
            },
        )

        agent.select_method_v2()

        self.assertEqual(agent.method_info.method_name, "Trimmed Inverse Probability Weighting")
        self.assertEqual(agent.method_info.selected_method, "trimmed_ipw")
        self.assertEqual(
            agent.method_info.decision_tree["execution"]["options"]["ps_bounds"], [0.1, 0.9]
        )

    def test_execute_method_passes_tree_execution_options(self):
        agent = self._make_agent()
        captured = {}

        class FakeEstimator:
            def __call__(self, df=None, variables=None, query=None, **options):
                captured["df"] = df
                captured.update(options)
                return {"effect_estimate": 1.23}

        agent.estimators.estimators[TrimmedIPW.name] = FakeEstimator()
        agent.selected_method = "trimmed_ipw"
        agent.method_info = MagicMock()
        agent.tree_result = TreeResult(
            method="trimmed_ipw",
            implemented=True,
            execution={"method": "trimmed_ipw", "options": {"ps_bounds": [0.1, 0.9]}},
        )

        with patch('cais.agent.explanation_generator_tool') as mock_explain:
            mock_explain.func.return_value = {"explanation": "ok"}
            agent.execute_method()

        self.assertIsInstance(captured["df"], pd.DataFrame)
        self.assertEqual(captured["ps_bounds"], [0.1, 0.9])
        self.assertEqual(agent.results["effect_estimate"], 1.23)

    def test_execute_method_applies_leaf_treatment_override(self):
        """A leaf can remap the grouping variable for one execution (e.g. ITT
        runs diff-in-means on the assignment column, not the received
        treatment); the option is applied to the variables and not leaked into
        estimator kwargs."""
        agent = self._make_agent()
        captured = {}

        class FakeEstimator:
            def __call__(self, df=None, variables=None, query=None, **options):
                captured["variables"] = variables
                captured.update(options)
                return {"effect_estimate": 0.5}

        agent.estimators.estimators[TrimmedIPW.name] = FakeEstimator()
        agent.selected_method = "trimmed_ipw"
        agent.method_info = MagicMock()
        agent.tree_result = TreeResult(
            method="trimmed_ipw",
            implemented=True,
            execution={
                "method": "trimmed_ipw",
                "display_name": "Trial run with assignment grouping",
                # "covariate1" stands in for the assignment column, which must
                # exist in the dataset for the dropna in execute_method.
                "options": {"treatment_variable": "covariate1"},
            },
        )

        with patch('cais.agent.explanation_generator_tool') as mock_explain:
            mock_explain.func.return_value = {"explanation": "ok"}
            agent.execute_method()

        self.assertEqual(agent.variables.treatment_variable, "covariate1")
        self.assertEqual(captured["variables"].treatment_variable, "covariate1")
        self.assertNotIn("treatment_variable", captured)
        self.assertEqual(agent.results["effect_estimate"], 0.5)

    def test_trimmed_ipw_is_registered_for_execution(self):
        from cais.agent import convert
        from cais.estimator_lib import Estimators

        self.assertEqual(convert["trimmed_ipw"], TrimmedIPW.name)
        self.assertIn(TrimmedIPW.name, Estimators().estimators)

    def test_frontdoor_is_registered_for_execution(self):
        from cais.agent import convert
        from cais.estimator_lib import Estimators
        from cais.methods.frontdoor_adjustment.estimator import FrontdoorAdjustment

        self.assertEqual(convert["frontdoor_adjustment"], FrontdoorAdjustment.name)
        self.assertIn(FrontdoorAdjustment.name, Estimators().estimators)


if __name__ == '__main__':
    unittest.main()

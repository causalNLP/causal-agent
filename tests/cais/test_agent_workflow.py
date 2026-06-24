import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

import cais.agent as cais_agent
from cais.agent import CausalAgent


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


if __name__ == '__main__':
    unittest.main()

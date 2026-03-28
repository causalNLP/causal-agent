import unittest
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cais.agent import CausalAgent


class TestE2EIVNewPipeline(unittest.TestCase):
    def test_iv_llm_pipeline_app_engagement_push(self):
        """Run the full CAIS pipeline end-to-end (real API calls, no mocks)."""

        # --- Scenario ---
        query = "What is the effect of education on earnings??"
        dataset_path = "data/all_data/card_geographic.csv"
        dataset_description = (
            """The National Longitudinal Survey of Young Men (NLSYM) was conducted to collect data on demographics, education, and employment outcomes. Participants were tracked over time to study long-term patterns. The dataset used here comes from the 1976 wave of the survey. Variables include: lwage: log of wages educ: years of education exper: years of work experience black: 1 if the individual is Black, 0 otherwise south: 1 if the individual lives in a southern state, 0 otherwise married: 1 if married, 0 otherwise smsa: 1 if living in a metropolitan area, 0 otherwise nearc4: 1 if there is a four-year college in the county, 0 otherwise"""
        )

        print("--- Running E2E Test Output ---")
        agent = CausalAgent(
            dataset_path=dataset_path,
            dataset_description=dataset_description,
        )
        output = agent.run_analysis(
            query=query,
        )
        print(json.dumps(output, indent=2, default=str))
        print("-----------------------------------------------------")

        # --- Assertions ---
        self.assertIsNotNone(output, "Agent returned None output.")
        self.assertIsInstance(output, dict, "Agent output is not a dict.")
        self.assertNotIn("error", output, f"Agent returned error: {output.get('error')}")
        self.assertIn("results", output)
        self.assertIn("explanation", output)
        self.assertIn("instrument", json.dumps(output).lower())


if __name__ == "__main__":
    unittest.main()

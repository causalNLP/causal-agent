import unittest
import os
import re
from dotenv import load_dotenv

from cais.agent import run_causal_analysis


class TestE2ERDD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

        cls.query = "What is the effect of alcohol consumption on death by all causes at 21 years?"
        cls.dataset_path = "data/all_data/drinking.csv"
        cls.dataset_description = (
            "To estimate the impacts of alcohol on death, we could use the fact that legal drinking "
            "age imposes a discontinuity on nature. In the US, those just under 21 years don't drink "
            "(or drink much less) while those just older than 21 do drink. The csv file drinking.csv "
            "contains mortality data aggregated by age. Each row is the average age of a group of "
            "people and the average mortality by all causes (all), by moving vehicle accident (mva) "
            "and by suicide (suicide)."
        )

        if not os.path.exists(cls.dataset_path):
            raise unittest.SkipTest(
                f"Skipping E2E RDD test: dataset not found at {cls.dataset_path}"
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise unittest.SkipTest(
                "Skipping E2E RDD test: OPENAI_API_KEY not set or found in .env file."
            )

    def test_rdd_drinking_data(self):
        """Run the full agent workflow on the drinking age dataset for RDD."""
        print("--- Running E2E Test Output (RDD) ---")
        result = run_causal_analysis(
            query=self.query,
            dataset_path=self.dataset_path,
            dataset_description=self.dataset_description,
        )
        print(result)
        print("-------------------------------------")

        self.assertIsNotNone(result, "Agent returned None output.")
        self.assertIsInstance(result, dict, "Agent output is not a dict.")
        self.assertNotIn("error", result, f"Result contains error: {result.get('error')}")

        # Extract explanation string for text-based checks
        explanation = result.get("final_explanation_text", str(result))
        self.assertIsInstance(explanation, str)
        explanation_lower = explanation.lower()

        # Check for absence of common error messages
        self.assertNotIn("Traceback", explanation, "Output contains 'Traceback'.")

        # Check if the correct method was likely selected and mentioned
        self.assertIn(
            "Regression Discontinuity", explanation,
            "Method 'Regression Discontinuity' not mentioned in output."
        )

        # Check if key variables are mentioned
        self.assertIn("age", explanation_lower, "Running variable 'age' not mentioned.")
        self.assertIn("21", explanation_lower, "Cutoff '21' not mentioned.")

        # Check if an effect estimate exists in the results dict
        self.assertIn("results", result, "'results' key missing from output dict.")
        results_inner = result.get("results", {})
        self.assertIn(
            "effect_estimate", results_inner,
            "'effect_estimate' key missing from results."
        )


if __name__ == '__main__':
    unittest.main()

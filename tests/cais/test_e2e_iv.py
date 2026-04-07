import unittest
import os
import re
import pytest
from dotenv import load_dotenv

from cais.agent import run_causal_analysis

class TestE2EIV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

        cls.query = "Does the marketing push increase app purchases?"
        cls.dataset_path = "data/all_data/app_engagement_push.csv"
        cls.dataset_description = (
            "A study is conducted to measure the effect of a marketing push on user engagement, "
            "specifically in-app purchases. Some customers who were assigned to receive the push are "
            "not receiving it, because they probably have an older phone that doesn't support the kind "
            "of push the marketing team designed.\n"
            "The dataset app_engagement_push.csv contains records for 10,000 random customers. Each "
            "record includes whether an in-app purchase was made (in_app_purchase), if a marketing push "
            "was assigned to the user (push_assigned), and if the marketing push was successfully "
            "delivered (push_delivered)"
        )

        if not os.path.exists(cls.dataset_path):
            raise unittest.SkipTest(
                f"Skipping E2E IV test: dataset not found at {cls.dataset_path}"
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise unittest.SkipTest(
                "Skipping E2E IV test: OPENAI_API_KEY not set or found in .env file."
            )

    def test_iv_wage_education(self):
        """Run the full agent workflow on the app_engagement_push dataset for IV."""
        print("--- Running E2E Test Output (IV) ---")
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
            "Instrumental Variable", explanation,
            "Method 'Instrumental Variable' not mentioned in output."
        )

        # Check if key variables are mentioned
        self.assertIn("push", explanation_lower, "Treatment variable 'push' not mentioned.")
        self.assertIn("purchase", explanation_lower, "Outcome variable 'purchase' not mentioned.")

        # Check if an effect estimate exists in the results dict
        self.assertIn("results", result, "'results' key missing from output dict.")
        results_inner = result.get("results", {})
        self.assertIn(
            "effect_estimate", results_inner,
            "'effect_estimate' key missing from results."
        )


if __name__ == '__main__':
    unittest.main()

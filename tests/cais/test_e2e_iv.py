import unittest
import os
import sys
import re # For parsing results
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cais.agent import run_causal_analysis

class TestE2EIV(unittest.TestCase):
    
    def test_iv_wage_education(self):
        """Run the full agent workflow on the app_engagement_push dataset for IV."""
        
        query = "Does the marketing push increase app purchases?"
        # Assuming tests run from the project root directory
        dataset_path = str(PROJECT_ROOT / "data" / "all_data" / "app_engagement_push.csv")
        dataset_description = "A study is conducted to measure the effect of a marketing push on user engagement, specifically in-app purchases. Some customers who were assigned to receive the push are not receiving it, because they probably have an older phone that doesn’t support the kind of push the marketing team designed.\nThe dataset app_engagement_push.csv contains records for 10,000 random customers. Each record includes whether an in-app purchase was made (in_app_purchase), if a marketing push was assigned to the user (push_assigned), and if the marketing push was successfully delivered (push_delivered)"
        
        # --- Execute the Agent --- 
        # Note: Ensure any required API keys (e.g., OPENAI_API_KEY) are set 
        # in the environment where the test runs, as get_llm_client() likely needs it.
        print("--- Running E2E Test Output (IV) ---")
        final_output = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=dataset_description
        )
        print(final_output)
        print("-------------------------------------")
        
        # --- Assertions --- 
        self.assertIsNotNone(final_output, "Agent returned None output.")
        self.assertIsInstance(final_output, dict, "Agent output is not a dictionary.")

        # Check for absence of common error messages
        self.assertNotIn("error", final_output, "Output contains an error key.")

        # Check method selection (instrumental variable expected)
        method_name = final_output.get("method", "") or ""
        results_method = (
            final_output.get("results", {})
            .get("results", {})
            .get("method_used", "")
        )
        self.assertTrue(
            "instrument" in method_name.lower() or "instrument" in str(results_method).lower(),
            "Instrumental Variable method not indicated in output."
        )

        # Check identified variables are present
        variables = final_output.get("results", {}).get("variables", {})
        self.assertIsNotNone(variables.get("treatment_variable"), "Treatment variable not identified.")
        self.assertIsNotNone(variables.get("outcome_variable"), "Outcome variable not identified.")
        self.assertIsNotNone(variables.get("instrument_variable"), "Instrument variable not identified.")

        # Check if an effect estimate exists
        effect_estimate = final_output.get("results", {}).get("results", {}).get("effect_estimate")
        self.assertIsNotNone(effect_estimate, "Effect estimate missing.")

if __name__ == '__main__':
    unittest.main() 
import unittest
import os
import json
import re
import pandas as pd
# Import load_dotenv
from dotenv import load_dotenv

# Import the main entry point
from auto_causal.agent import run_causal_analysis

# Ensure necessary environment variables are set for LLM calls (e.g., OPENAI_API_KEY)
# Load from .env file if present
# from dotenv import load_dotenv
# load_dotenv()

class TestE2EDID(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

        # Define query details based on the user's request
        cls.query = "What was the impact of cigarette taxation rules on cigarette sales in California?"
        cls.dataset_description = "To estimate the effect of cigarette taxation on its consumption, data from cigarette sales were collected and analyzed across 39 states in the United States from the years 1970 to 2000. Proposition 99, a Tobacco Tax and Health Protection Act passed in California in 1988, imposed a 25-cent per pack state excise tax on tobacco cigarettes and implemented additional restrictions, including the ban on cigarette vending machines in public areas accessible by juveniles and a ban on the individual sale of single cigarettes. Revenue generated was allocated for environmental and health care programs along with anti-tobacco advertising. We aim to determine if the imposition of this tax and the subsequent regulations led to a reduction in cigarette sales. The data is in the CSV file smoking2.csv."
        # Expected effect from query data. Note: A positive value (tax increasing sales) is counter-intuitive for this scenario.
        # The actual DiD effect for Prop 99 is often cited as negative (reducing sales). We use the provided value for test structure.
        cls.expected_effect = 24.83
        # Tolerance might need adjustment based on the specific DiD model implemented by the agent
        cls.tolerance = 10.0 

        # Construct path relative to this test file's directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cls.dataset_path = os.path.join(base_dir, "data", "qrdata", "smoking2.csv")
        print(f"DEBUG: E2E test using dataset path: {cls.dataset_path}")

        # Check if data file exists
        if not os.path.exists(cls.dataset_path):
            raise FileNotFoundError(f"E2E test requires dataset at: {cls.dataset_path}")

        # Ensure API key is available (or skip test)
        if not os.getenv("OPENAI_API_KEY"):
             raise unittest.SkipTest("Skipping E2E test: OPENAI_API_KEY not set or found in .env file.")

    def extract_results_from_output(self, output_string: str) -> dict:
        '''Helper to parse relevant info from the agent's final output string.'''
        results = {
            'method': None,
            'effect': None
        }
        # Look for Method Used:
        method_match = re.search(r"Method Used:\s*([^\n]+)", output_string, re.IGNORECASE)
        if method_match:
            # Strip potential markdown and extra spaces
            method_name = method_match.group(1).strip().replace('*', '')
            results['method'] = method_name
        # Fallback: Look for 'Recommended Method:'
        elif (method_match := re.search(r"Recommended Method:\s*([^\n]+)", output_string, re.IGNORECASE)):
             method_name = method_match.group(1).strip().replace('*', '')
             results['method'] = method_name


        # Parse effect - Added more robust patterns
        effect_patterns = [
            r"Causal Effect:\s*([-\+]?\d*\.?\d+)",
            r"estimated causal effect is\s*([-\+]?\d*\.?\d+)",
            r"effect estimate:\s*([-\+]?\d*\.?\d+)"
        ]
        for pattern in effect_patterns:
            effect_match = re.search(pattern, output_string, re.IGNORECASE)
            if effect_match:
                try:
                    results['effect'] = float(effect_match.group(1))
                    break # Stop after first successful match
                except ValueError:
                    pass 

        return results

    def test_did_e2e(self):
        '''Run the full agent workflow on the smoking dataset.'''

        # run_causal_analysis now returns the final explanation string directly
        final_output_string = run_causal_analysis(self.query, self.dataset_path, self.dataset_description)

        print("\n--- E2E Test Output (DiD) ---")
        print(final_output_string)
        print("-----------------------------\n")

        # Parse the output string directly
        parsed_results = self.extract_results_from_output(final_output_string)

        # Assertions
        self.assertIsNotNone(parsed_results['method'], "Could not extract method from final output string.")
        # Check if the method is DiD (case-insensitive, ignoring spaces)
        method_lower_no_space = parsed_results['method'].lower().replace(' ', '').replace('-', '')
        expected_methods = ["differenceindifferences", "did", "diffindiff"]
        self.assertTrue(
            any(expected in method_lower_no_space for expected in expected_methods),
            f"Expected DiD method, but found: {parsed_results['method']}"
        )

        # Check numerical effect
        self.assertIsNotNone(parsed_results['effect'], "Could not extract effect estimate from final output string.")
        # Note: DiD estimates can vary based on model specification (covariates, fixed effects).
        # The expected value 24.83 might be based on a specific model or potentially incorrect.
        # Adjust tolerance accordingly.
        self.assertAlmostEqual(parsed_results['effect'], self.expected_effect, delta=self.tolerance,
                               msg=f"Estimated effect {parsed_results['effect']} not within {self.tolerance} of expected {self.expected_effect}")

if __name__ == '__main__':
    unittest.main() 
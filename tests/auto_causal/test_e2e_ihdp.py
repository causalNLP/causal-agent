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

class TestE2EIHDP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()
        
        # Define query details based on queries.json
        cls.query = "What is the effect of home visits from specialist doctors on the cognitive scores of premature infants?"
        
        # Construct path relative to this test file's directory
        # __file__ is the path to the current file
        # os.path.dirname gets the directory containing the file
        # os.path.abspath ensures it's an absolute path
        # Go up 2 levels (tests/auto_causal/ -> tests/ -> causalscientist/) then into data/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cls.dataset_path = os.path.join(base_dir, "data", "qrdata", "ihdp_1.csv")
        # cls.dataset_path = "data/qrdata/ihdp_1.csv" # Old relative path
        print(f"DEBUG: E2E test using dataset path: {cls.dataset_path}") # Add print statement

        cls.expected_effect = 4.05
        cls.tolerance = 0.5 # Allow some variance from the expected 4.05

        # Check if data file exists
        if not os.path.exists(cls.dataset_path):
            raise FileNotFoundError(f"E2E test requires dataset at: {cls.dataset_path}")
        
        # Ensure API key is available (or skip test)
        # This check will now happen *after* load_dotenv attempts to load it
        if not os.getenv("OPENAI_API_KEY"):
             raise unittest.SkipTest("Skipping E2E test: OPENAI_API_KEY not set or found in .env file.")

        # Add dataset description from queries.json
        cls.dataset_description = "The CSV file ihdp_1.csv contains data obtained from the Infant Health and Development Program (IHDP). The study is designed to evaluate the effect of home visit from specialist doctors on the cognitive test scores of premature infants. The confounders x (x1-x25) correspond to collected measurements of the children and their mothers, including measurements on the child (birth weight, head circumference, weeks born preterm, birth order, first born, neonatal health index, sex, twin status), as well as behaviors engaged in during the pregnancy (smoked cigarettes, drank alcohol, took drugs) and measurements on the mother at the time she gave birth (age, marital status, educational attainment, whether she worked during pregnancy, whether she received prenatal care) and the site (8 total) in which the family resided at the start of the intervention. There are 6 continuous covariates and 19 binary covariates."

    def extract_results_from_output(self, output_string: str) -> dict:
        '''Helper to parse relevant info from the agent's final output string.'''
        results = {
            'method': None,
            'effect': None
        }
        # Try simpler regex pattern: Look for Method Used:, space, capture until newline
        method_match = re.search(r"Method Used:\s*([^\n]+)", output_string, re.IGNORECASE)
            
        if method_match:
            # Strip potential markdown and extra spaces from the captured group
            method_name = method_match.group(1).strip().replace('*', '')
            results['method'] = method_name
        # Keep fallback checks if needed
        # else:
            # Fallback if the first pattern fails (e.g., different formatting)
            # method_match = re.search(r"recommended method is ([\w\s\.-]+)\.", output_string, re.IGNORECASE) 
            # if method_match:
            #      results['method'] = method_match.group(1).strip()

        # Parse effect
        effect_match = re.search(r"Causal Effect: ([\-\+]?\d*\.?\d+)", output_string, re.IGNORECASE)
        if not effect_match: # Try summary pattern
             effect_match = re.search(r"estimated causal effect is ([\-\+]?\d*\.?\d+)", output_string, re.IGNORECASE)
             
        if effect_match:
            try:
                results['effect'] = float(effect_match.group(1))
            except ValueError:
                pass 
        
        return results

    def test_ihdp_e2e(self):
        '''Run the full agent workflow on the IHDP dataset.'''
               
        # run_causal_analysis now returns the final explanation string directly
        final_output_string = run_causal_analysis(self.query, self.dataset_path, self.dataset_description)
        
        print("--- E2E Test Output ---")
        print(final_output_string)
        print("-----------------------")

        # Parse the output string directly
        parsed_results = self.extract_results_from_output(final_output_string)

        # Assertions
        self.assertIsNotNone(parsed_results['method'], "Could not extract method from final output string.")
        # Check if the method is one of the PS methods we refactored
        # Note: Method selection logic might still need debugging
        self.assertIn(parsed_results['method'].lower(), 
                      ["propensity score matching", "propensity score weighting", 
                       "ps.matching", "ps.weighting", "regression adjustment"], # Allow RA as decision tree might choose it
                      f"Unexpected method found: {parsed_results['method']}")
        
        # Check numerical effect
        self.assertIsNotNone(parsed_results['effect'], "Could not extract effect estimate from final output string.")
        self.assertAlmostEqual(parsed_results['effect'], self.expected_effect, delta=self.tolerance, 
                               msg=f"Estimated effect {parsed_results['effect']} not within {self.tolerance} of expected {self.expected_effect}")

if __name__ == '__main__':
    # Ensure the working directory allows finding the data file relative path
    # Might need adjustment depending on how tests are run
    # Example: os.chdir('../') if running from tests/ directory
    unittest.main() 
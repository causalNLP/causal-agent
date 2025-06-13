import unittest
import os
import sys
import re # For parsing results

# Ensure the main package is discoverable
# Adjust path as necessary based on your test execution context
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from auto_causal.agent import run_causal_analysis

class TestE2ERDD(unittest.TestCase):
    
    def test_rdd_drinking_data(self):
        """Run the full agent workflow on the drinking age dataset for RDD."""
        
        query = "What is the effect of alcohol consumption on death by all causes at 21 years?"
        # Assuming tests run from the project root directory
        dataset_path = "data/qrdata/drinking.csv" 
        dataset_description = "To estimate the impacts of alcohol on death, we could use the fact that legal drinking age imposes a discontinuity on nature. In the US, those just under 21 years don't drink (or drink much less) while those just older than 21 do drink. The csv file drinking.csv contains mortality data aggregated by age. Each row is the average age of a group of people and the average mortality by all causes (all), by moving vehicle accident (mva) and by suicide (suicide)."
        
        # --- Execute the Agent --- 
        # Note: Ensure any required API keys (e.g., OPENAI_API_KEY) are set 
        # in the environment where the test runs, as get_llm_client() likely needs it.
        print("--- Running E2E Test Output (RDD) ---")
        final_output_string = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=dataset_description
        )
        print(final_output_string)
        print("-------------------------------------")
        
        # --- Assertions --- 
        self.assertIsNotNone(final_output_string, "Agent returned None output.")
        self.assertIsInstance(final_output_string, str, "Agent output is not a string.")
        
        # Check for absence of common error messages
        self.assertNotIn("Error:", final_output_string, "Output string contains 'Error:'.")
        self.assertNotIn("Failed:", final_output_string, "Output string contains 'Failed:'.")
        self.assertNotIn("Traceback", final_output_string, "Output string contains 'Traceback'.")

        # Check if the correct method was likely selected and mentioned
        self.assertIn("Regression Discontinuity", final_output_string, "Method 'Regression Discontinuity' not mentioned in output.")
        
        # Check if key variables are mentioned 
        # (Use lowercase for case-insensitivity)
        output_lower = final_output_string.lower()
        self.assertIn("age", output_lower, "Running variable 'age' not mentioned.")
        self.assertIn("21", output_lower, "Cutoff '21' not mentioned.")
        # Outcome variable name is 'all' in the dataset
        self.assertIn("all", output_lower, "Outcome variable 'all' not mentioned.")
        
        # Check if an effect estimate section/value exists
        self.assertIn("Causal Effect", output_lower, "'Causal Effect' section missing.")
        # Check for a number pattern near the effect estimate 
        # This is less brittle than asserting the exact value 7.66
        self.assertTrue(re.search(r"causal effect:?\s*[-+]?\d*\.?\d+", output_lower),
                        "Numerical effect estimate pattern not found.")

if __name__ == '__main__':
    unittest.main() 
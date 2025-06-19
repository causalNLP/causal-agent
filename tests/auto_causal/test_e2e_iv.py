import unittest
import os
import sys
import re # For parsing results

from auto_causal.agent import run_causal_analysis

class TestE2EIV(unittest.TestCase):
    
    def test_iv_wage_education(self):
        """Run the full agent workflow on the app_engagement_push dataset for IV."""
        
        query = "Does the marketing push increase app purchases?"
        # Assuming tests run from the project root directory
        dataset_path = "data/qrdata/app_engagement_push.csv" 
        dataset_description = "A study is conducted to measure the effect of a marketing push on user engagement, specifically in-app purchases. Some customers who were assigned to receive the push are not receiving it, because they probably have an older phone that doesn’t support the kind of push the marketing team designed.\nThe dataset app_engagement_push.csv contains records for 10,000 random customers. Each record includes whether an in-app purchase was made (in_app_purchase), if a marketing push was assigned to the user (push_assigned), and if the marketing push was successfully delivered (push_delivered)"
        
        # --- Execute the Agent --- 
        # Note: Ensure any required API keys (e.g., OPENAI_API_KEY) are set 
        # in the environment where the test runs, as get_llm_client() likely needs it.
        print("--- Running E2E Test Output (IV) ---")
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
        self.assertIn("Instrumental Variable", final_output_string, "Method 'Instrumental Variable' not mentioned in output.")
        
        # Check if key variables are mentioned 
        output_lower = final_output_string.lower()
        self.assertIn("education", output_lower, "Treatment variable 'education' not mentioned.") # Or 'schooling'
        self.assertIn("wage", output_lower, "Outcome variable 'wage' not mentioned.") # Or 'log wage'
        self.assertIn("quarter", output_lower, "Instrument variable 'quarter' not mentioned.") # Or 'qob'
        
        # Check if an effect estimate section/value exists
        self.assertIn("Causal Effect", output_lower, "'Causal Effect' section missing.")
        # Check for a number pattern near the effect estimate 
        # Check for positive effect (0.0853)
        self.assertTrue(re.search(r"causal effect:?\s*\+?\d*\.?\d+", output_lower), 
                        "Numerical effect estimate pattern not found.")

if __name__ == '__main__':
    unittest.main() 
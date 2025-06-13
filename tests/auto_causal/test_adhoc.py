import unittest
import os
import json
import re
import pandas as pd
# Import load_dotenv
from dotenv import load_dotenv

# Import the main entry point
from auto_causal.agent import run_causal_analysis

load_dotenv()

# class TestAdhoc(unittest.TestCase):
#     def test_adhoc(self):
#         query = "Does receiving a voter turnout mailing increase the probability of voting compared to receiving no mailing?"
#         dataset_path = "data/voter_turnout_data.csv"
#         dataset_description = """The ISPS D001 dataset, titled "Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment," 
#         originates from a 2006 field experiment in Michigan. Researchers Gerber, Green, and Larimer investigated how different mail-based interventions 
#         influenced voter turnout in a primary election. The study encompassed 180,002 households (344,084 registered voters), randomly assigned to a control 
#         group or one of four treatment groups: Civic Duty, Hawthorne Effect, Self, and Neighbors. Each treatment involved a distinct mailing designed to 
#         apply social pressure or appeal to civic responsibility. The primary outcome measured was voter turnout in the 2006 local elections. 
#         Data were sourced from Michigan's Qualified Voter File (QVF), curated by Practical Political Consulting. The dataset includes individual 
#         and household-level information, treatment assignments, and voting outcomes. Comprehensive documentation and replication materials are available
#           to facilitate further research and analysis."""

#         result = run_causal_analysis(query, dataset_path, dataset_description)
#         print(result)

class TestAdhoc(unittest.TestCase):
    def test_adhoc_from_structured_input(self):
        # Define the input using the new structure
        test_input_data ={
        "paper": "	What is the effect of home visits on the cognitive test scores of children who actually received the intervention?",
        "dataset_description": """"The CSV file ihdp_4.csv contains data obtained from the Infant Health and Development Program (IHDP). The study is designed to evaluate the effect of home visit from specialist doctors on the cognitive test scores of premature infants. The confounders x (x1-x25) correspond to collected measurements of the children and their mothers, including measurements on the child (birth weight, head circumference, weeks born preterm, birth order, first born, neonatal health index, sex, twin status), as well as behaviors engaged in during the pregnancy (smoked cigarettes, drank alcohol, took drugs) and measurements on the mother at the time she gave birth (age, marital status, educational attainment, whether she worked during pregnancy, whether she received prenatal care) and the site (8 total) in which the family resided at the start of the intervention. There are 6 continuous covariates and 19 binary covariates.""",
        "query": "What is the effect of home visits on the cognitive test scores of children who actually received the intervention?",
        "answer": 0.0,
        "method": "TWFE",
        "dataset_path": "benchmark/all_data_1/ihdp_5.csv"
    }

        # Extract relevant info from the input data
        query = test_input_data["query"]
        dataset_path = test_input_data["dataset_path"]
        dataset_description = test_input_data["dataset_description"]
        expected_method = test_input_data["method"]
        expected_answer = test_input_data["answer"]

        # Ensure dataset_path is correct if it's relative to a specific root or needs joining
        # For example, if your tests run from the root of the project:
        # script_dir = os.path.dirname(__file__)
        # project_root = os.path.abspath(os.path.join(script_dir, "../../..")) # Adjust based on test_adhoc.py location
        # dataset_path = os.path.join(project_root, dataset_path)
        # For now, assuming dataset_path is directly usable or handled by run_causal_analysis

        print(f"Running adhoc test with query: {query}")
        print(f"Dataset path: {dataset_path}")

        # Call the main causal analysis function
        # We need to know what `run_causal_analysis` returns to make assertions.
        # Assuming it returns a dictionary that includes the method used and the effect estimate.
        result = run_causal_analysis(query, dataset_path, dataset_description)
        
        print("Causal analysis result:")
        #print(json.dumps(result, indent=2)) # Pretty print the result dictionary

        # Assertions (these are examples and depend on the actual structure of `result`)
        # You'll need to adapt these based on what `run_causal_analysis` returns.
        
        # Example: Assuming result is a dict and might have a top-level key for the final output summary
        # and within that, information about method used and effect estimate.
        # This is highly speculative and needs to be adjusted.
        final_summary = result # or result.get("summary"), etc.
        
       
if __name__ == "__main__":
    unittest.main()
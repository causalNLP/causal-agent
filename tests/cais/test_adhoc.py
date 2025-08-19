import unittest
import os
import json
import re
import pandas as pd
# Import load_dotenv
from dotenv import load_dotenv

# Import the main entry point
from cais.agent import run_causal_analysis

load_dotenv()


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

        
        print(f"Running adhoc test with query: {query}")
        print(f"Dataset path: {dataset_path}")

        ## Assuming it returns a dictionary that includes the method used and the effect estimate.
        result = run_causal_analysis(query, dataset_path, dataset_description)
        
        print("Causal analysis result:")
        #print(json.dumps(result, indent=2)) # Pretty print the result dictionary

        
        final_summary = result # or result.get("summary"), etc.
        
       
if __name__ == "__main__":
    unittest.main()
import unittest
import os
import pandas as pd
from dotenv import load_dotenv

from cais.agent import run_causal_analysis

load_dotenv()


class TestAdhoc(unittest.TestCase):

    def setUp(self):
        dataset_path = "benchmark/all_data_1/ihdp_5.csv"
        if not os.path.exists(dataset_path):
            self.skipTest(f"Skipping adhoc test: dataset not found at {dataset_path}")
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("Skipping adhoc test: OPENAI_API_KEY not set or found in .env file.")

    def test_adhoc_from_structured_input(self):
        test_input_data = {
            "paper": "What is the effect of home visits on the cognitive test scores of children who actually received the intervention?",
            "dataset_description": (
                "The CSV file ihdp_4.csv contains data obtained from the Infant Health and Development "
                "Program (IHDP). The study is designed to evaluate the effect of home visit from "
                "specialist doctors on the cognitive test scores of premature infants. The confounders x "
                "(x1-x25) correspond to collected measurements of the children and their mothers, "
                "including measurements on the child (birth weight, head circumference, weeks born "
                "preterm, birth order, first born, neonatal health index, sex, twin status), as well as "
                "behaviors engaged in during the pregnancy (smoked cigarettes, drank alcohol, took drugs) "
                "and measurements on the mother at the time she gave birth (age, marital status, "
                "educational attainment, whether she worked during pregnancy, whether she received "
                "prenatal care) and the site (8 total) in which the family resided at the start of the "
                "intervention. There are 6 continuous covariates and 19 binary covariates."
            ),
            "query": "What is the effect of home visits on the cognitive test scores of children who actually received the intervention?",
            "answer": 0.0,
            "method": "TWFE",
            "dataset_path": "benchmark/all_data_1/ihdp_5.csv",
        }

        query = test_input_data["query"]
        dataset_path = test_input_data["dataset_path"]
        dataset_description = test_input_data["dataset_description"]

        print(f"Running adhoc test with query: {query}")
        print(f"Dataset path: {dataset_path}")

        result = run_causal_analysis(query, dataset_path, dataset_description)

        print("Causal analysis result:")
        print(result)

        # Verify result structure
        self.assertIsNotNone(result, "run_causal_analysis returned None.")
        self.assertIsInstance(result, dict, "run_causal_analysis should return a dict.")
        self.assertNotIn(
            "error", result,
            f"run_causal_analysis returned an error: {result.get('error')}"
        )
        self.assertIn("explanation", result, "Result dict should contain an 'explanation' key.")
        self.assertIn("results", result, "Result dict should contain a 'results' key.")


if __name__ == "__main__":
    unittest.main()

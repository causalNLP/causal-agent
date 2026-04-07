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

        cls.query = "Does access to electricity lead to an increase in total household expendititure?"
        cls.dataset_path = "data/all_data/electrification_data.csv"
        cls.dataset_description = "The dataset was collected to better understand the impacts of rural electrification at the household level, particularly in regions where electricity access was expanding but remained incomplete. The data come from a household survey conducted across 686 households in 120 habitations in Uttar Pradesh, India. According to state regulations, households must be located within 40 meters of a power pole to be eligible for a legal electricity connection. Using this rule, the study sampled households situated 20–35 meters from the nearest pole, which were eligible to get electricity from the given pole, and 45–60 meters from the pole, those that were ineligible. Houses in the 35–45 meter range are excluded to minimize measurement error. The survey targeted areas with a balanced mix of electrified and non-electrified households and collected detailed information on household demographics, expenditures, appliance ownership and use, and daily activities.\xa0\n\nThe variables are\nfood_expenditure: total monthly household expenditure on food in rupees\neducation_expenditure: total monthly household expenditure on education in rupees\nkerosene_expenditure: total monthly household expenditure on kerosene in rupees\ntotal_expenditure: total monthly household expenditure in rupees\nage: age of the head of the household\xa0\nreligion: 1= Hindu, 0=otherwise\xa0\ndistance: distance of the household from the electric grid\ntreat: 1 if the household is connected to the grid, 0 if not connected to the grid\nforcing: 1 if the household is eligible to get connected to the grid (within 40 meters), 0 if the household is not eligible to get connected to the grid\nkerosene_lamps: 1 = household has kerosene lamp, 0 = no kerosene lamp\nnum_kerosene_lamps: number of kerosene wick lamps and lanterns owned by the household\nkerosene_lamp_hours: number of hours kerosene lamps are used daily\nkerosene_other: liters of kerosene used for other household purposes\nlighting_hours: total daily hours of household light usage\nchild_lighting: daily hours of lighting used by children for reading and studying\nadult_lighting: daily hours of lighting used by adults for reading and studying\nchild_activity: number of hours children spend at home in a given day\nadult_activity: number of hours adults spend at home in a given day\nappliances: number of appliances owned by the household\nappliance_use: number of daily hours using appliances by the household\nsatisfaction_reliability: satisfaction with the reliability of lighting\nsatisfaction_cost: satisfaction with the cost of lighting\nsatisfaction_safety: satisfaction with the safety of lighting\nsatisfaction_brightness: satisfaction with the brightness of lighting\nsatisfaction: overall satisfaction with lighting\nsatisfaction_chng: change in satisfaction with lighting over the past five years\nelec_value: willingness to pay for adequate electricity\nincome_increase: belief that electrification will increase household income\nbusiness_interest: interest in starting a new business due to electrification\nsatisfaction_business: belief that electrification supports business aspirations\naspirations: mean value of five questions measuring general aspirations\nknowledge: battery of questions related to knowledge of politics and popular culture"

        if not os.path.exists(cls.dataset_path):
            raise unittest.SkipTest(
                f"Skipping E2E IV test: dataset not found at {cls.dataset_path}"
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise unittest.SkipTest(
                "Skipping E2E IV test: OPENAI_API_KEY not set or found in .env file."
            )

    def test_iv_electrical_cost(self):
        """Run the full agent workflow on the electrification dataset for IV."""
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
        self.assertIn("treat", explanation_lower, "Treatment variable 'treat' not mentioned.")
        self.assertIn("total_expenditure", explanation_lower, "Outcome variable 'purchase' not mentioned.")
        self.assertIn("forcing", explanation_lower, "Instrumental variable 'forcing' not mentioned.")

        # Check if an effect estimate exists in the results dict
        self.assertIn("results", result, "'results' key missing from output dict.")
        results_inner = result.get("results", {})
        self.assertIn(
            "effect_estimate", results_inner,
            "'effect_estimate' key missing from results."
        )


if __name__ == '__main__':
    unittest.main()

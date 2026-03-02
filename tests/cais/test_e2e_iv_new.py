import unittest
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cais.agent import run_causal_analysis


class TestE2EIVNewPipeline(unittest.TestCase):
    def test_iv_llm_pipeline_app_engagement_push(self):
        """Run the full CAIS pipeline end-to-end (real API calls, no mocks)."""

        # --- Scenario ---
        query = "Does number of sibilings have an effect on the years of education?"
        dataset_path = "data/all_data/xiong_2022_sibling.csv"
        dataset_description = (
            "The dataset originates from the Chinese General Social Survey (CGSS), which was designed to systematically investigate social and demographic influences on education in China. It contains information at the individual, household, and regional levels, enabling analysis of how family structure and parental background shape children’s educational outcomes. Covering the period after China’s reform and opening-up, the survey integrates both family-level characteristics (e.g., number of siblings, birth order, parental education and occupation) and broader social context (e.g., urban–rural differences, regional disparities). Overall, the data provides a foundation for studying how internal family dynamics interact with external socio-economic conditions to influence educational attainment. Data variables: year_edu: Educational years, sib: Number of siblings, period: After the reform and opening up, han: Han nationality, gender: Gender, urban_father: Household type of father, job_father: Father works in a public institution, job_mother: Mother works in a public institution, party_father: Father is a party member, party_mother: Mother is a party member, edu_father: Education level of father, edu_mother: Education level of mother, non_father: Fatherless child (age 14), non_mother: Motherless child (age 14), level_edu: Degree level of education, total_income: Total family income, province: Province, countyid: County ID, sequence: Child’s birth order, only_child: Indicator if the child is an only child, eldest_child: Indicator if the child is the eldest, youngest_child: Indicator if the child is the youngest"
        )

        print("--- Running E2E Test Output ---")
        output = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=dataset_description,
        )
        print(json.dumps(output, indent=2, default=str))
        print("-----------------------------------------------------")

        # --- Assertions ---
        self.assertIsNotNone(output, "Agent returned None output.")
        self.assertIsInstance(output, dict, "Agent output is not a dict.")
        self.assertNotIn("error", output, f"Agent returned error: {output.get('error')}")
        self.assertIn("method", output)
        self.assertIn("results", output)
        self.assertIn("explanation", output)
        self.assertIn("instrument", json.dumps(output).lower())


if __name__ == "__main__":
    unittest.main()

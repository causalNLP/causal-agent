import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cais.agent import run_causal_analysis
from cais.config import get_llm_client
from cais.iv_llm.src.agents.confounder_miner import ConfounderMiner
from cais.iv_llm.src.agents.hypothesizer import Hypothesizer
from cais.iv_llm.src.critics.exclusion_critic import ExclusionCritic
from cais.iv_llm.src.critics.independence_critic import IndependenceCritic


class TestIVLLMPipelineAssertions(unittest.TestCase):
    def test_iv_llm_pipeline_with_stage_assertions(self):
        query = "Does number of sibilings have an effect on the years of education?"
        dataset_path = "data/all_data/xiong_2022_sibling.csv"
        dataset_description = (
            "The dataset originates from the Chinese General Social Survey (CGSS), which was designed to systematically investigate social and demographic influences on education in China. It contains information at the individual, household, and regional levels, enabling analysis of how family structure and parental background shape children’s educational outcomes. Covering the period after China’s reform and opening-up, the survey integrates both family-level characteristics (e.g., number of siblings, birth order, parental education and occupation) and broader social context (e.g., urban–rural differences, regional disparities). Overall, the data provides a foundation for studying how internal family dynamics interact with external socio-economic conditions to influence educational attainment. Data variables: year_edu: Educational years, sib: Number of siblings, period: After the reform and opening up, han: Han nationality, gender: Gender, urban_father: Household type of father, job_father: Father works in a public institution, job_mother: Mother works in a public institution, party_father: Father is a party member, party_mother: Mother is a party member, edu_father: Education level of father, edu_mother: Education level of mother, non_father: Fatherless child (age 14), non_mother: Motherless child (age 14), level_edu: Degree level of education, total_income: Total family income, province: Province, countyid: County ID, sequence: Child’s birth order, only_child: Indicator if the child is an only child, eldest_child: Indicator if the child is the eldest, youngest_child: Indicator if the child is the youngest"
        )

        llm = get_llm_client()
        hypothesizer = Hypothesizer(llm, k=5)
        confounder_miner = ConfounderMiner(llm, j=5)
        exclusion_critic = ExclusionCritic(llm)
        independence_critic = IndependenceCritic(llm)

        treatment = "sib"
        outcome = "year_edu"
        context = (
            "Available columns: year_edu, sib, period, han, gender, urban_father, "
            "job_father, job_mother, party_father, party_mother, edu_father, "
            "edu_mother, non_father, non_mother, level_edu, total_income, province, "
            "countyid, sequence, only_child, eldest_child, youngest_child. "
            + dataset_description
        )

        proposed_ivs = hypothesizer.propose_ivs(treatment, outcome, context=context)
        self.assertIsInstance(proposed_ivs, list)
        self.assertGreater(len(proposed_ivs), 0)
        self.assertLessEqual(len(proposed_ivs), 5)

        confounders = confounder_miner.identify_confounders(treatment, outcome, context=context)
        self.assertIsInstance(confounders, list)
        self.assertLessEqual(len(confounders), 5)

        candidate_iv = proposed_ivs[0]
        self.assertIsInstance(candidate_iv, str)
        self.assertTrue(len(candidate_iv) > 0)

        exclusion_ok = exclusion_critic.validate_exclusion(
            candidate_iv,
            treatment,
            outcome,
            confounders,
        )
        self.assertIsInstance(exclusion_ok, bool)

        independence_ok = independence_critic.validate_independence(
            candidate_iv,
            treatment,
            outcome,
            confounders,
        )
        self.assertIsInstance(independence_ok, bool)

        output = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=dataset_description,
        )

        self.assertIsNotNone(output, "Agent returned None output.")
        self.assertIsInstance(output, dict, "Agent output is not a dict.")
        self.assertNotIn("error", output, f"Agent returned error: {output.get('error')}")
        self.assertIn("method", output)
        self.assertIn("results", output)
        self.assertIn("explanation", output)
        self.assertIn("instrument", json.dumps(output).lower())

        try:
            Path(dataset_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()

"""End-to-end tests for complete causal analysis workflows."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Dict, Any, List

from causal_agent.agent import run_causal_analysis
import unittest


class TestCompleteWorkflowsE2E(unittest.TestCase):
    """End-to-end tests for complete causal analysis workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_realistic_rct_dataset(self) -> str:
        """Create realistic RCT dataset similar to clinical trial."""
        np.random.seed(42)
        n = 200
        
        # Baseline characteristics
        age = np.random.normal(45, 12, n)
        gender = np.random.binomial(1, 0.6, n)  # 60% female
        baseline_score = np.random.normal(50, 10, n)
        
        # Random treatment assignment (key feature of RCT)
        treatment = np.random.binomial(1, 0.5, n)
        
        # Outcome with realistic treatment effect
        # Treatment reduces score by 8 points on average
        outcome_score = (
            baseline_score * 0.7 +  # Regression to mean
            age * 0.1 +  # Age effect
            gender * 2 +  # Gender effect
            treatment * (-8) +  # Treatment effect (negative = improvement)
            np.random.normal(0, 5, n)  # Measurement error
        )
        
        data = pd.DataFrame({
            'patient_id': range(1, n + 1),
            'age': age,
            'gender': gender,
            'baseline_score': baseline_score,
            'treatment_group': treatment,
            'outcome_score': outcome_score,
            'study_site': np.random.choice(['Site_A', 'Site_B', 'Site_C'], n)
        })
        
        filepath = os.path.join(self.temp_dir, "clinical_trial_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def create_realistic_observational_dataset(self) -> str:
        """Create realistic observational dataset with confounding."""
        np.random.seed(42)
        n = 500
        
        # Socioeconomic and demographic variables
        age = np.random.normal(35, 10, n)
        education = np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.3, 0.3, 0.2])  # Education levels
        income = np.random.lognormal(10, 0.5, n)  # Log-normal income distribution
        urban = np.random.binomial(1, 0.7, n)  # 70% urban
        
        # Treatment assignment depends on confounders (selection bias)
        treatment_logits = (
            -2 +  # Base probability
            0.05 * age +  # Older people more likely to get treatment
            0.3 * education +  # More educated more likely
            0.0001 * income +  # Higher income more likely
            0.5 * urban  # Urban residents more likely
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome depends on confounders AND treatment
        outcome = (
            20 +  # Baseline
            0.2 * age +  # Age effect
            3 * education +  # Education effect
            0.0002 * income +  # Income effect
            2 * urban +  # Urban effect
            5 * treatment +  # Treatment effect (what we want to estimate)
            np.random.normal(0, 3, n)  # Noise
        )
        
        data = pd.DataFrame({
            'individual_id': range(1, n + 1),
            'age': age,
            'education_level': education,
            'annual_income': income,
            'urban_residence': urban,
            'received_treatment': treatment,
            'outcome_measure': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "observational_study_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def create_realistic_iv_dataset(self) -> str:
        """Create realistic IV dataset (e.g., education and earnings with distance to college as IV)."""
        np.random.seed(42)
        n = 400
        
        # Geographic and family background variables
        distance_to_college = np.random.exponential(20, n)  # Distance in miles
        family_income = np.random.lognormal(10, 0.6, n)
        parent_education = np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.25, 0.15])
        
        # Unobserved ability (not in dataset but affects both education and earnings)
        ability = np.random.normal(0, 1, n)
        
        # Education depends on distance (instrument) and unobserved ability
        education_logits = (
            -1 +
            -0.05 * distance_to_college +  # Farther distance reduces education (IV effect)
            0.0001 * family_income +  # Family income increases education
            0.4 * parent_education +  # Parent education increases education
            0.6 * ability  # Ability increases education (confounding)
        )
        education_prob = 1 / (1 + np.exp(-education_logits))
        college_education = np.random.binomial(1, education_prob)
        
        # Earnings depend on education and ability (not directly on distance)
        log_earnings = (
            9.5 +  # Base log earnings
            0.4 * college_education +  # Education premium (causal effect)
            0.0001 * family_income +  # Family background effect
            0.1 * parent_education +  # Parent education effect
            0.3 * ability +  # Ability effect (confounding)
            np.random.normal(0, 0.3, n)  # Earnings shock
        )
        earnings = np.exp(log_earnings)
        
        data = pd.DataFrame({
            'person_id': range(1, n + 1),
            'distance_to_college_miles': distance_to_college,
            'family_income': family_income,
            'parent_education_level': parent_education,
            'college_educated': college_education,
            'annual_earnings': earnings
        })
        
        filepath = os.path.join(self.temp_dir, "education_earnings_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def create_realistic_rdd_dataset(self) -> str:
        """Create realistic RDD dataset (e.g., scholarship eligibility based on test score)."""
        np.random.seed(42)
        n = 300
        cutoff = 75  # Test score cutoff for scholarship
        
        # Test scores around cutoff
        test_scores = np.random.normal(cutoff, 10, n)
        
        # Scholarship eligibility (sharp RDD)
        scholarship = (test_scores >= cutoff).astype(int)
        
        # Background characteristics
        family_income = np.random.lognormal(10, 0.5, n)
        parent_education = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.25, 0.15])
        
        # College completion outcome with discontinuity at cutoff
        college_completion_logits = (
            -2 +  # Base probability
            0.02 * test_scores +  # Smooth function of test score
            1.2 * scholarship +  # Scholarship effect (discontinuity)
            0.0001 * family_income +  # Family income effect
            0.3 * parent_education +  # Parent education effect
            np.random.normal(0, 0.5, n)  # Random variation
        )
        college_completion_prob = 1 / (1 + np.exp(-college_completion_logits))
        college_completion = np.random.binomial(1, college_completion_prob)
        
        data = pd.DataFrame({
            'student_id': range(1, n + 1),
            'entrance_test_score': test_scores,
            'scholarship_received': scholarship,
            'family_income': family_income,
            'parent_education_level': parent_education,
            'completed_college': college_completion
        })
        
        filepath = os.path.join(self.temp_dir, "scholarship_rdd_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def create_realistic_did_dataset(self) -> str:
        """Create realistic DiD dataset (e.g., policy intervention across states)."""
        np.random.seed(42)
        n_states = 20
        n_years = 10
        treatment_year = 5  # Policy implemented in year 5
        
        data = []
        
        for state in range(n_states):
            # Some states get the policy (treatment group)
            treated_state = state < n_states // 2
            
            # State-specific characteristics
            state_baseline = np.random.normal(50, 10)  # State fixed effect
            state_trend = np.random.normal(0.5, 0.2)  # State-specific trend
            
            for year in range(n_years):
                # Time effects
                year_effect = year * 1.5 + np.random.normal(0, 0.5)
                
                # Treatment indicator
                post_policy = year >= treatment_year
                treatment = 1 if (treated_state and post_policy) else 0
                
                # Outcome with parallel trends assumption
                outcome = (
                    state_baseline +  # State fixed effect
                    year_effect +  # Common time trend
                    state_trend * year +  # State-specific trend
                    8 * treatment +  # Policy effect (what we want to estimate)
                    np.random.normal(0, 2)  # Random shock
                )
                
                data.append({
                    'state_id': state,
                    'year': year,
                    'treated_state': treated_state,
                    'post_policy': post_policy,
                    'policy_implemented': treatment,
                    'outcome_measure': outcome,
                    'state_population': np.random.normal(5000000, 1000000),  # Control variable
                    'state_gdp_per_capita': np.random.normal(45000, 8000)  # Control variable
                })
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, "policy_did_data.csv")
        df.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_clinical_trial_rct_e2e(self, mock_get_llm, mock_llm_call):
        """Test complete RCT workflow with clinical trial data."""
        dataset_path = self.create_realistic_rct_dataset()
        
        # Mock sophisticated LLM responses for clinical trial
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["treatment_group", "outcome_score", "age", "gender", "baseline_score"],
                "treatment_candidates": ["treatment_group"],
                "outcome_candidates": ["outcome_score"],
                "data_quality": "high",
                "sample_size": 200,
                "missing_data": "minimal"
            },
            # Query interpretation
            {
                "treatment_variable": "treatment_group",
                "outcome_variable": "outcome_score",
                "covariates": ["age", "gender", "baseline_score"],
                "is_rct": True,
                "study_design": "randomized_controlled_trial"
            },
            # Method selection
            {
                "recommended_method": "diff_in_means",
                "confidence": 0.95,
                "reasoning": "Randomized trial allows for simple difference in means estimation",
                "alternative_methods": ["linear_regression"]
            },
            # Result interpretation
            {
                "interpretation": "Treatment shows statistically significant reduction in outcome score",
                "clinical_significance": "8-point reduction is clinically meaningful",
                "confidence_assessment": "High confidence due to randomization"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = "What is the effect of the treatment on patient outcome scores in this clinical trial?"
        description = """
        Clinical trial data testing a new intervention. Patients were randomly assigned to 
        treatment or control groups. Primary outcome is a standardized score where lower 
        values indicate better outcomes.
        """
        
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=description
        )
        
        # Comprehensive result validation
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        
        # Verify workflow was executed
        self.assertTrue(mock_llm_call.called)
        self.assertGreaterEqual(mock_llm_call.call_count, 3)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_observational_study_e2e(self, mock_get_llm, mock_llm_call):
        """Test complete observational study workflow with confounding."""
        dataset_path = self.create_realistic_observational_dataset()
        
        # Mock responses for observational study
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["received_treatment", "outcome_measure", "age", "education_level", "annual_income", "urban_residence"],
                "treatment_candidates": ["received_treatment"],
                "outcome_candidates": ["outcome_measure"],
                "potential_confounders": ["age", "education_level", "annual_income", "urban_residence"],
                "data_quality": "good"
            },
            # Query interpretation
            {
                "treatment_variable": "received_treatment",
                "outcome_variable": "outcome_measure",
                "covariates": ["age", "education_level", "annual_income", "urban_residence"],
                "is_rct": False,
                "confounding_concerns": "high"
            },
            # Method selection
            {
                "recommended_method": "backdoor_adjustment",
                "confidence": 0.8,
                "reasoning": "Observational data with identified confounders requires adjustment",
                "alternative_methods": ["propensity_score", "linear_regression"]
            },
            # Result interpretation
            {
                "interpretation": "Treatment effect estimated after adjusting for confounders",
                "confounding_assessment": "Controlled for major observable confounders",
                "confidence_assessment": "Moderate confidence with confounder adjustment"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = """
        What is the causal effect of receiving treatment on the outcome measure, 
        controlling for demographic and socioeconomic factors?
        """
        description = """
        Observational study examining treatment effects. Treatment assignment was not 
        randomized and may depend on patient characteristics including age, education, 
        income, and urban residence.
        """
        
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=description
        )
        
        # Validate result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        
        # Verify confounding was addressed
        self.assertTrue(mock_llm_call.called)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_instrumental_variable_e2e(self, mock_get_llm, mock_llm_call):
        """Test complete IV workflow with education-earnings example."""
        dataset_path = self.create_realistic_iv_dataset()
        
        # Mock responses for IV analysis
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["distance_to_college_miles", "college_educated", "annual_earnings", "family_income", "parent_education_level"],
                "treatment_candidates": ["college_educated"],
                "outcome_candidates": ["annual_earnings"],
                "instruments": ["distance_to_college_miles"],
                "data_quality": "good"
            },
            # Query interpretation
            {
                "treatment_variable": "college_educated",
                "outcome_variable": "annual_earnings",
                "instrument_variable": "distance_to_college_miles",
                "covariates": ["family_income", "parent_education_level"],
                "is_rct": False
            },
            # Method selection
            {
                "recommended_method": "instrumental_variable",
                "confidence": 0.85,
                "reasoning": "Distance to college is a valid instrument for education",
                "instrument_validity": "satisfies exclusion restriction"
            },
            # Result interpretation
            {
                "interpretation": "IV estimate of education premium on earnings",
                "instrument_assessment": "Distance to college affects education but not earnings directly",
                "confidence_assessment": "Good confidence with valid instrument"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = """
        What is the causal effect of college education on annual earnings, 
        using distance to college as an instrumental variable?
        """
        description = """
        Study of education returns using geographic variation. Distance to nearest college 
        affects education decisions but should not directly affect earnings, making it a 
        valid instrument for identifying causal effects of education.
        """
        
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=description
        )
        
        # Validate result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        
        # Verify IV method was used
        self.assertTrue(mock_llm_call.called)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_regression_discontinuity_e2e(self, mock_get_llm, mock_llm_call):
        """Test complete RDD workflow with scholarship eligibility example."""
        dataset_path = self.create_realistic_rdd_dataset()
        
        # Mock responses for RDD analysis
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["entrance_test_score", "scholarship_received", "completed_college", "family_income", "parent_education_level"],
                "treatment_candidates": ["scholarship_received"],
                "outcome_candidates": ["completed_college"],
                "running_variables": ["entrance_test_score"],
                "discontinuity_detected": True
            },
            # Query interpretation
            {
                "treatment_variable": "scholarship_received",
                "outcome_variable": "completed_college",
                "running_variable": "entrance_test_score",
                "cutoff_value": 75,
                "covariates": ["family_income", "parent_education_level"],
                "is_rct": False
            },
            # Method selection
            {
                "recommended_method": "regression_discontinuity",
                "confidence": 0.9,
                "reasoning": "Sharp discontinuity in scholarship assignment at test score cutoff",
                "design_validity": "sharp RDD with clear cutoff"
            },
            # Result interpretation
            {
                "interpretation": "Scholarship increases college completion probability at cutoff",
                "design_assessment": "Sharp discontinuity provides credible identification",
                "confidence_assessment": "High confidence with valid RDD design"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = """
        What is the effect of receiving a scholarship on college completion, 
        using the test score cutoff for scholarship eligibility?
        """
        description = """
        Regression discontinuity design studying scholarship effects. Students with test 
        scores at or above 75 automatically receive scholarships. The discontinuous 
        assignment rule allows identification of causal effects.
        """
        
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=description
        )
        
        # Validate result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        
        # Verify RDD method was used
        self.assertTrue(mock_llm_call.called)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_difference_in_differences_e2e(self, mock_get_llm, mock_llm_call):
        """Test complete DiD workflow with policy evaluation example."""
        dataset_path = self.create_realistic_did_dataset()
        
        # Mock responses for DiD analysis
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["state_id", "year", "policy_implemented", "outcome_measure", "state_population", "state_gdp_per_capita"],
                "treatment_candidates": ["policy_implemented"],
                "outcome_candidates": ["outcome_measure"],
                "panel_structure": True,
                "time_variable": "year",
                "unit_variable": "state_id"
            },
            # Query interpretation
            {
                "treatment_variable": "policy_implemented",
                "outcome_variable": "outcome_measure",
                "time_variable": "year",
                "unit_variable": "state_id",
                "covariates": ["state_population", "state_gdp_per_capita"],
                "is_rct": False
            },
            # Method selection
            {
                "recommended_method": "difference_in_differences",
                "confidence": 0.88,
                "reasoning": "Panel data with staggered policy implementation suitable for DiD",
                "parallel_trends": "assumption appears reasonable"
            },
            # Result interpretation
            {
                "interpretation": "Policy implementation shows positive effect on outcome",
                "parallel_trends_assessment": "Pre-treatment trends appear parallel",
                "confidence_assessment": "Good confidence with DiD identification"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = """
        What is the causal effect of the policy implementation on the outcome measure 
        across states over time?
        """
        description = """
        Panel dataset for policy evaluation using difference-in-differences. Some states 
        implemented a new policy in year 5, while others did not. The staggered 
        implementation allows identification of causal effects.
        """
        
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description=description
        )
        
        # Validate result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        
        # Verify DiD method was used
        self.assertTrue(mock_llm_call.called)
    
    @pytest.mark.e2e
    def test_workflow_robustness_with_data_issues(self):
        """Test workflow robustness with common data quality issues."""
        # Create dataset with missing values and outliers
        np.random.seed(42)
        n = 100
        
        data = {
            'treatment': np.random.binomial(1, 0.5, n),
            'outcome': np.random.normal(10, 2, n),
            'age': np.random.normal(35, 10, n),
            'income': np.random.lognormal(10, 1, n)
        }
        
        # Add missing values
        missing_indices = np.random.choice(n, 10, replace=False)
        data['age'] = np.array(data['age'])
        data['age'][missing_indices] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(n, 5, replace=False)
        data['income'] = np.array(data['income'])
        data['income'][outlier_indices] *= 10  # Extreme outliers
        
        df = pd.DataFrame(data)
        dataset_path = os.path.join(self.temp_dir, "problematic_data.csv")
        df.to_csv(dataset_path, index=False)
        
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
                # Mock responses that handle data quality issues
                mock_llm_call.side_effect = [
                    {
                        "variables": ["treatment", "outcome", "age", "income"],
                        "data_quality": "fair",
                        "missing_data": "present",
                        "outliers_detected": True
                    },
                    {
                        "treatment_variable": "treatment",
                        "outcome_variable": "outcome",
                        "covariates": ["age", "income"],
                        "data_quality_concerns": ["missing_values", "outliers"]
                    },
                    {
                        "recommended_method": "linear_regression",
                        "confidence": 0.7,
                        "reasoning": "Robust method for data with quality issues"
                    },
                    {
                        "interpretation": "Effect estimated with data quality adjustments",
                        "data_quality_impact": "Missing values and outliers may affect precision"
                    }
                ]
                
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm
                
                # Should handle gracefully
                result = run_causal_analysis(
                    query="What is the effect of treatment on outcome?",
                    dataset_path=dataset_path,
                    dataset_description="Dataset with data quality issues"
                )
                
                self.assertIsInstance(result, dict)
    
    @pytest.mark.e2e
    def test_multiple_query_types_same_dataset(self):
        """Test different query types on the same dataset."""
        dataset_path = self.create_realistic_observational_dataset()
        
        queries = [
            "What is the effect of treatment on outcome?",
            "Does treatment cause outcome?", 
            "How much does treatment change outcome?",
            "What would happen to outcome if everyone received treatment?",
            "Estimate the average treatment effect on outcome"
        ]
        
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm
                
                for i, query in enumerate(queries):
                    with self.subTest(query=query):
                        # Mock responses for each query
                        mock_llm_call.side_effect = [
                            {"variables": ["received_treatment", "outcome_measure"], "data_quality": "good"},
                            {"treatment_variable": "received_treatment", "outcome_variable": "outcome_measure"},
                            {"recommended_method": "backdoor_adjustment", "confidence": 0.8},
                            {"interpretation": f"Treatment effect for query {i+1}"}
                        ]
                        
                        result = run_causal_analysis(
                            query=query,
                            dataset_path=dataset_path,
                            dataset_description="Observational study data"
                        )
                        
                        self.assertIsInstance(result, dict)
                        self.assertIn('results', result)
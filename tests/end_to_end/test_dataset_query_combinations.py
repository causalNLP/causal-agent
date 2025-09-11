"""End-to-end tests for different dataset and query combinations."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Dict, Any, List, Tuple

from causal_agent.agent import run_causal_analysis
import unittest


class TestDatasetQueryCombinations(unittest.TestCase):
    """Test various combinations of datasets and query types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_dataset_variants(self) -> Dict[str, str]:
        """Create different variants of datasets for testing."""
        datasets = {}
        
        # 1. Simple binary treatment, continuous outcome
        np.random.seed(42)
        n = 150
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 10 + 3 * treatment + np.random.normal(0, 2, n)
        age = np.random.normal(40, 10, n)
        
        df1 = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome,
            'age': age
        })
        datasets['binary_continuous'] = os.path.join(self.temp_dir, "binary_continuous.csv")
        df1.to_csv(datasets['binary_continuous'], index=False)
        
        # 2. Continuous treatment, continuous outcome
        np.random.seed(42)
        treatment_cont = np.random.normal(5, 2, n)
        outcome_cont = 20 + 1.5 * treatment_cont + np.random.normal(0, 3, n)
        
        df2 = pd.DataFrame({
            'dosage': treatment_cont,
            'response': outcome_cont,
            'baseline': np.random.normal(50, 10, n)
        })
        datasets['continuous_continuous'] = os.path.join(self.temp_dir, "continuous_continuous.csv")
        df2.to_csv(datasets['continuous_continuous'], index=False)
        
        # 3. Binary treatment, binary outcome
        np.random.seed(42)
        treatment_bin = np.random.binomial(1, 0.4, n)
        outcome_logits = -1 + 2 * treatment_bin + np.random.normal(0, 1, n)
        outcome_bin = np.random.binomial(1, 1 / (1 + np.exp(-outcome_logits)))
        
        df3 = pd.DataFrame({
            'intervention': treatment_bin,
            'success': outcome_bin,
            'risk_score': np.random.normal(0, 1, n)
        })
        datasets['binary_binary'] = os.path.join(self.temp_dir, "binary_binary.csv")
        df3.to_csv(datasets['binary_binary'], index=False)
        
        # 4. Multi-level categorical treatment
        np.random.seed(42)
        treatment_cat = np.random.choice(['control', 'low_dose', 'high_dose'], n, p=[0.4, 0.3, 0.3])
        treatment_effect = np.where(treatment_cat == 'control', 0,
                                  np.where(treatment_cat == 'low_dose', 2, 5))
        outcome_cat = 15 + treatment_effect + np.random.normal(0, 2, n)
        
        df4 = pd.DataFrame({
            'treatment_group': treatment_cat,
            'outcome_score': outcome_cat,
            'covariate1': np.random.normal(0, 1, n),
            'covariate2': np.random.binomial(1, 0.6, n)
        })
        datasets['categorical_continuous'] = os.path.join(self.temp_dir, "categorical_continuous.csv")
        df4.to_csv(datasets['categorical_continuous'], index=False)
        
        # 5. Time series / panel data
        np.random.seed(42)
        panel_data = []
        for unit in range(20):
            for time in range(10):
                treated = unit < 10 and time >= 5  # Treatment starts at time 5 for first 10 units
                outcome_panel = (
                    unit * 0.5 +  # Unit fixed effect
                    time * 0.3 +  # Time trend
                    (3 if treated else 0) +  # Treatment effect
                    np.random.normal(0, 1)
                )
                panel_data.append({
                    'unit_id': unit,
                    'time_period': time,
                    'treated': int(treated),
                    'outcome': outcome_panel
                })
        
        df5 = pd.DataFrame(panel_data)
        datasets['panel_data'] = os.path.join(self.temp_dir, "panel_data.csv")
        df5.to_csv(datasets['panel_data'], index=False)
        
        # 6. High-dimensional data
        np.random.seed(42)
        n_features = 15
        X = np.random.normal(0, 1, (n, n_features))
        treatment_hd = np.random.binomial(1, 1 / (1 + np.exp(-X[:, 0] - 0.5 * X[:, 1])))
        outcome_hd = (
            2 * treatment_hd +
            0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2] +
            np.random.normal(0, 1, n)
        )
        
        feature_cols = {f'feature_{i}': X[:, i] for i in range(n_features)}
        df6 = pd.DataFrame({
            'treatment': treatment_hd,
            'outcome': outcome_hd,
            **feature_cols
        })
        datasets['high_dimensional'] = os.path.join(self.temp_dir, "high_dimensional.csv")
        df6.to_csv(datasets['high_dimensional'], index=False)
        
        return datasets
    
    def get_query_variants(self) -> List[Tuple[str, str, str]]:
        """Get different query formulations with expected method and description."""
        return [
            # (query, expected_method_type, description)
            ("What is the effect of treatment on outcome?", "basic_causal", "Simple causal effect query"),
            ("Does treatment cause outcome?", "basic_causal", "Causal relationship query"),
            ("How much does treatment increase outcome?", "quantitative", "Quantitative effect query"),
            ("What would happen to outcome if we changed treatment?", "counterfactual", "Counterfactual query"),
            ("Estimate the average treatment effect", "ate", "Average treatment effect query"),
            ("What is the causal impact controlling for confounders?", "adjusted", "Confounder-adjusted query"),
            ("Compare treatment groups", "comparison", "Group comparison query"),
            ("Analyze the intervention effect", "intervention", "Intervention analysis query")
        ]
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_binary_treatment_continuous_outcome_combinations(self, mock_get_llm, mock_llm_call):
        """Test various queries on binary treatment, continuous outcome data."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['binary_continuous']
        queries = self.get_query_variants()
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        for query, expected_type, description in queries:
            with self.subTest(query=query, expected_type=expected_type):
                # Mock responses tailored to binary treatment, continuous outcome
                mock_llm_call.side_effect = [
                    {
                        "variables": ["treatment", "outcome", "age"],
                        "treatment_candidates": ["treatment"],
                        "outcome_candidates": ["outcome"],
                        "data_type": "binary_treatment_continuous_outcome"
                    },
                    {
                        "treatment_variable": "treatment",
                        "outcome_variable": "outcome",
                        "covariates": ["age"],
                        "is_rct": True,
                        "query_type": expected_type
                    },
                    {
                        "recommended_method": "diff_in_means" if expected_type == "basic_causal" else "linear_regression",
                        "confidence": 0.9,
                        "reasoning": f"Appropriate method for {expected_type} query"
                    },
                    {
                        "interpretation": f"Analysis completed for {expected_type} query",
                        "effect_size": "moderate",
                        "confidence_assessment": "high"
                    }
                ]
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description=f"Binary treatment, continuous outcome: {description}"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_continuous_treatment_combinations(self, mock_get_llm, mock_llm_call):
        """Test queries on continuous treatment data."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['continuous_continuous']
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Continuous treatment specific queries
        continuous_queries = [
            "What is the dose-response relationship between dosage and response?",
            "How does increasing dosage by 1 unit affect response?",
            "What is the optimal dosage level?",
            "Estimate the linear effect of dosage on response"
        ]
        
        for query in continuous_queries:
            with self.subTest(query=query):
                mock_llm_call.side_effect = [
                    {
                        "variables": ["dosage", "response", "baseline"],
                        "treatment_candidates": ["dosage"],
                        "outcome_candidates": ["response"],
                        "treatment_type": "continuous"
                    },
                    {
                        "treatment_variable": "dosage",
                        "outcome_variable": "response",
                        "covariates": ["baseline"],
                        "is_rct": False,
                        "treatment_type": "continuous"
                    },
                    {
                        "recommended_method": "linear_regression",
                        "confidence": 0.85,
                        "reasoning": "Linear regression appropriate for continuous treatment"
                    },
                    {
                        "interpretation": "Dose-response relationship estimated",
                        "linearity_assessment": "linear relationship assumed",
                        "confidence_assessment": "good"
                    }
                ]
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description="Continuous dosage study with response outcome"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_binary_outcome_combinations(self, mock_get_llm, mock_llm_call):
        """Test queries on binary outcome data."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['binary_binary']
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Binary outcome specific queries
        binary_outcome_queries = [
            "What is the effect of intervention on success probability?",
            "Does intervention increase the odds of success?",
            "How much does intervention change the risk of success?",
            "What is the risk ratio for intervention vs control?"
        ]
        
        for query in binary_outcome_queries:
            with self.subTest(query=query):
                mock_llm_call.side_effect = [
                    {
                        "variables": ["intervention", "success", "risk_score"],
                        "treatment_candidates": ["intervention"],
                        "outcome_candidates": ["success"],
                        "outcome_type": "binary"
                    },
                    {
                        "treatment_variable": "intervention",
                        "outcome_variable": "success",
                        "covariates": ["risk_score"],
                        "is_rct": True,
                        "outcome_type": "binary"
                    },
                    {
                        "recommended_method": "diff_in_means",  # For binary outcomes, can use proportion difference
                        "confidence": 0.88,
                        "reasoning": "Difference in proportions for binary outcome"
                    },
                    {
                        "interpretation": "Effect on success probability estimated",
                        "effect_measure": "risk_difference",
                        "confidence_assessment": "good"
                    }
                ]
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description="Intervention study with binary success outcome"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_categorical_treatment_combinations(self, mock_get_llm, mock_llm_call):
        """Test queries on categorical treatment data."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['categorical_continuous']
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Categorical treatment specific queries
        categorical_queries = [
            "Compare the effects of different treatment groups",
            "What is the effect of high_dose vs control?",
            "Which treatment group has the best outcome?",
            "Estimate pairwise treatment comparisons"
        ]
        
        for query in categorical_queries:
            with self.subTest(query=query):
                mock_llm_call.side_effect = [
                    {
                        "variables": ["treatment_group", "outcome_score", "covariate1", "covariate2"],
                        "treatment_candidates": ["treatment_group"],
                        "outcome_candidates": ["outcome_score"],
                        "treatment_type": "categorical",
                        "treatment_levels": ["control", "low_dose", "high_dose"]
                    },
                    {
                        "treatment_variable": "treatment_group",
                        "outcome_variable": "outcome_score",
                        "covariates": ["covariate1", "covariate2"],
                        "is_rct": True,
                        "treatment_type": "categorical"
                    },
                    {
                        "recommended_method": "linear_regression",  # Can handle categorical treatments
                        "confidence": 0.87,
                        "reasoning": "Linear regression with categorical treatment indicators"
                    },
                    {
                        "interpretation": "Multi-group treatment effects estimated",
                        "comparison_type": "pairwise",
                        "confidence_assessment": "good"
                    }
                ]
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description="Multi-arm trial with control, low dose, and high dose groups"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_panel_data_combinations(self, mock_get_llm, mock_llm_call):
        """Test queries on panel/longitudinal data."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['panel_data']
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Panel data specific queries
        panel_queries = [
            "What is the effect of treatment over time?",
            "Estimate the treatment effect using difference-in-differences",
            "How does treatment affect outcome across units and time?",
            "What is the dynamic treatment effect?"
        ]
        
        for query in panel_queries:
            with self.subTest(query=query):
                mock_llm_call.side_effect = [
                    {
                        "variables": ["unit_id", "time_period", "treated", "outcome"],
                        "treatment_candidates": ["treated"],
                        "outcome_candidates": ["outcome"],
                        "data_structure": "panel",
                        "time_variable": "time_period",
                        "unit_variable": "unit_id"
                    },
                    {
                        "treatment_variable": "treated",
                        "outcome_variable": "outcome",
                        "time_variable": "time_period",
                        "unit_variable": "unit_id",
                        "is_rct": False,
                        "data_structure": "panel"
                    },
                    {
                        "recommended_method": "difference_in_differences",
                        "confidence": 0.85,
                        "reasoning": "Panel data with treatment variation over time suitable for DiD"
                    },
                    {
                        "interpretation": "Treatment effect estimated using panel variation",
                        "identification_strategy": "difference_in_differences",
                        "confidence_assessment": "good"
                    }
                ]
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description="Panel data with units observed over multiple time periods"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)
    
    @pytest.mark.e2e
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_high_dimensional_combinations(self, mock_get_llm, mock_llm_call):
        """Test queries on high-dimensional data."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['high_dimensional']
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # High-dimensional data specific queries
        hd_queries = [
            "What is the treatment effect controlling for all features?",
            "Estimate treatment effect with high-dimensional confounders",
            "Use regularization to estimate causal effects",
            "What is the treatment effect after feature selection?"
        ]
        
        for query in hd_queries:
            with self.subTest(query=query):
                feature_vars = [f'feature_{i}' for i in range(15)]
                mock_llm_call.side_effect = [
                    {
                        "variables": ["treatment", "outcome"] + feature_vars,
                        "treatment_candidates": ["treatment"],
                        "outcome_candidates": ["outcome"],
                        "data_characteristics": "high_dimensional",
                        "n_features": 15
                    },
                    {
                        "treatment_variable": "treatment",
                        "outcome_variable": "outcome",
                        "covariates": feature_vars,
                        "is_rct": False,
                        "dimensionality": "high"
                    },
                    {
                        "recommended_method": "backdoor_adjustment",  # Or could be regularized regression
                        "confidence": 0.75,
                        "reasoning": "High-dimensional confounders require careful adjustment"
                    },
                    {
                        "interpretation": "Treatment effect estimated with high-dimensional adjustment",
                        "regularization": "may_be_needed",
                        "confidence_assessment": "moderate"
                    }
                ]
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description="High-dimensional observational data with many potential confounders"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)
    
    @pytest.mark.e2e
    def test_query_robustness_across_datasets(self):
        """Test how the same query performs across different dataset types."""
        datasets = self.create_dataset_variants()
        standard_query = "What is the effect of treatment on outcome?"
        
        # Map dataset types to their treatment/outcome variable names
        variable_mappings = {
            'binary_continuous': ('treatment', 'outcome'),
            'continuous_continuous': ('dosage', 'response'),
            'binary_binary': ('intervention', 'success'),
            'categorical_continuous': ('treatment_group', 'outcome_score'),
            'panel_data': ('treated', 'outcome'),
            'high_dimensional': ('treatment', 'outcome')
        }
        
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm
                
                for dataset_type, dataset_path in datasets.items():
                    with self.subTest(dataset_type=dataset_type):
                        treatment_var, outcome_var = variable_mappings[dataset_type]
                        
                        # Mock responses adapted to each dataset type
                        mock_llm_call.side_effect = [
                            {
                                "variables": [treatment_var, outcome_var],
                                "treatment_candidates": [treatment_var],
                                "outcome_candidates": [outcome_var],
                                "dataset_type": dataset_type
                            },
                            {
                                "treatment_variable": treatment_var,
                                "outcome_variable": outcome_var,
                                "dataset_type": dataset_type
                            },
                            {
                                "recommended_method": "linear_regression",  # Generic method
                                "confidence": 0.8,
                                "reasoning": f"Appropriate for {dataset_type} data"
                            },
                            {
                                "interpretation": f"Effect estimated for {dataset_type}",
                                "dataset_adaptation": "successful"
                            }
                        ]
                        
                        result = run_causal_analysis(
                            query=standard_query,
                            dataset_path=dataset_path,
                            dataset_description=f"Testing standard query on {dataset_type} data"
                        )
                        
                        self.assertIsInstance(result, dict)
                        self.assertIn('results', result)
    
    @pytest.mark.e2e
    def test_complex_query_parsing(self):
        """Test parsing of complex, multi-part queries."""
        datasets = self.create_dataset_variants()
        dataset_path = datasets['binary_continuous']
        
        complex_queries = [
            """
            I want to understand the causal effect of treatment on outcome. 
            Please control for age as a potential confounder. 
            Also, I'm interested in whether the effect varies by age group.
            """,
            """
            Estimate the average treatment effect of treatment on outcome.
            Use appropriate methods for causal inference.
            Report confidence intervals and statistical significance.
            """,
            """
            What is the treatment effect? I think age might be a confounder.
            The treatment was randomly assigned, so this should be like an RCT.
            Please use the most appropriate causal inference method.
            """,
            """
            Compare treated vs untreated groups on outcome.
            Account for baseline differences in age.
            I need both the effect size and its uncertainty.
            """
        ]
        
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm
                
                for i, query in enumerate(complex_queries):
                    with self.subTest(query_index=i):
                        mock_llm_call.side_effect = [
                            {
                                "variables": ["treatment", "outcome", "age"],
                                "treatment_candidates": ["treatment"],
                                "outcome_candidates": ["outcome"],
                                "query_complexity": "high"
                            },
                            {
                                "treatment_variable": "treatment",
                                "outcome_variable": "outcome",
                                "covariates": ["age"],
                                "query_components": ["main_effect", "confounder_control", "uncertainty"]
                            },
                            {
                                "recommended_method": "linear_regression",
                                "confidence": 0.85,
                                "reasoning": "Handles multiple query components"
                            },
                            {
                                "interpretation": "Complex query analysis completed",
                                "components_addressed": ["effect_estimation", "confounder_adjustment", "uncertainty_quantification"]
                            }
                        ]
                        
                        result = run_causal_analysis(
                            query=query,
                            dataset_path=dataset_path,
                            dataset_description="Dataset for complex query testing"
                        )
                        
                        self.assertIsInstance(result, dict)
                        self.assertIn('results', result)
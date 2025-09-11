"""Integration tests for complete causal analysis workflows."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from causal_agent.agent import run_causal_analysis
from causal_agent.models import QueryInfo, MethodInfo, MethodValidatorInput, MethodExecutorInput
import unittest


class TestAgentWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete agent workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self, dataset_type: str = "rct", n_samples: int = 100) -> str:
        """Create a test dataset file."""
        if dataset_type == "rct":
            # Simple RCT data
            np.random.seed(42)
            treatment = np.random.binomial(1, 0.5, n_samples)
            outcome = np.random.normal(10, 2, n_samples)
            age = np.random.normal(35, 10, n_samples)
            gender = np.random.binomial(1, 0.5, n_samples)
            
            # Add treatment effect
            outcome = outcome + treatment * 2.5
            
            data = {
                'treatment': treatment,
                'outcome': outcome,
                'age': age,
                'gender': gender
            }
            
        elif dataset_type == "observational":
            # Observational data with confounding
            np.random.seed(42)
            age = np.random.normal(35, 10, n_samples)
            # Treatment depends on age (confounding)
            treatment_prob = 1 / (1 + np.exp(-(age - 35) / 10))
            treatment = np.random.binomial(1, treatment_prob)
            # Outcome depends on both age and treatment
            outcome = 5 + 0.1 * age + 3.0 * treatment + np.random.normal(0, 1, n_samples)
            
            data = {
                'treatment': treatment,
                'outcome': outcome,
                'age': age,
                'gender': np.random.binomial(1, 0.5, n_samples)
            }
            
        elif dataset_type == "iv":
            # Instrumental variable data
            np.random.seed(42)
            instrument = np.random.binomial(1, 0.5, n_samples)
            unobserved = np.random.normal(0, 1, n_samples)
            
            # Treatment depends on instrument and unobserved confounder
            treatment_prob = 1 / (1 + np.exp(-2 * instrument - unobserved))
            treatment = np.random.binomial(1, treatment_prob)
            
            # Outcome depends on treatment and unobserved confounder
            outcome = 10 + 2.0 * treatment + unobserved + np.random.normal(0, 0.5, n_samples)
            
            data = {
                'instrument': instrument,
                'treatment': treatment,
                'outcome': outcome,
                'age': np.random.normal(35, 10, n_samples)
            }
            
        elif dataset_type == "rdd":
            # Regression discontinuity data
            np.random.seed(42)
            running_var = np.random.uniform(-3, 3, n_samples)
            treatment = (running_var >= 0).astype(int)
            
            # Outcome with discontinuity at cutoff
            outcome = 10 + 0.5 * running_var + 2.0 * treatment + np.random.normal(0, 0.5, n_samples)
            
            data = {
                'running_var': running_var,
                'treatment': treatment,
                'outcome': outcome,
                'age': np.random.normal(35, 10, n_samples)
            }
            
        df = pd.DataFrame(data)
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        filepath = os.path.join(self.temp_dir, f"test_{dataset_type}_data.csv")
        df.to_csv(filepath, index=False)
        
        # Verify file was created
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Failed to create test dataset at {filepath}")
            
        return filepath
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_rct_workflow_integration(self, mock_get_llm, mock_llm_call):
        """Test complete workflow with RCT data."""
        # Create test dataset
        dataset_path = self.create_test_dataset("rct")
        
        # Mock LLM responses for different stages
        mock_llm_call.side_effect = [
            # Dataset analysis response
            {
                "variables": ["treatment", "outcome", "age", "gender"],
                "treatment_candidates": ["treatment"],
                "outcome_candidates": ["outcome"],
                "data_quality": "good"
            },
            # Query interpretation response
            {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "covariates": ["age", "gender"],
                "is_rct": True
            },
            # Method selection response
            {
                "recommended_method": "diff_in_means",
                "confidence": 0.9,
                "reasoning": "RCT data suitable for simple difference in means"
            },
            # Result interpretation response
            {
                "interpretation": "Treatment has positive effect",
                "confidence_assessment": "High confidence due to randomization"
            }
        ]
        
        # Mock LLM client
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = "What is the effect of treatment on outcome?"
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description="RCT testing treatment effect"
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        
        # Verify workflow components were called
        self.assertTrue(mock_llm_call.called)
        self.assertGreater(mock_llm_call.call_count, 0)
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_observational_workflow_integration(self, mock_get_llm, mock_llm_call):
        """Test complete workflow with observational data."""
        # Create test dataset
        dataset_path = self.create_test_dataset("observational")
        
        # Mock LLM responses
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["treatment", "outcome", "age", "gender"],
                "treatment_candidates": ["treatment"],
                "outcome_candidates": ["outcome"],
                "confounders": ["age"]
            },
            # Query interpretation
            {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "covariates": ["age", "gender"],
                "is_rct": False
            },
            # Method selection
            {
                "recommended_method": "backdoor_adjustment",
                "confidence": 0.8,
                "reasoning": "Observational data with identified confounders"
            },
            # Result interpretation
            {
                "interpretation": "Adjusted treatment effect accounting for confounders",
                "confidence_assessment": "Moderate confidence with confounder adjustment"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = "What is the causal effect of treatment on outcome, controlling for age?"
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description="Observational study with potential confounding"
        )
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_iv_workflow_integration(self, mock_get_llm, mock_llm_call):
        """Test complete workflow with instrumental variable data."""
        # Create test dataset
        dataset_path = self.create_test_dataset("iv")
        
        # Mock LLM responses
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["instrument", "treatment", "outcome", "age"],
                "treatment_candidates": ["treatment"],
                "outcome_candidates": ["outcome"],
                "instruments": ["instrument"]
            },
            # Query interpretation
            {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "instrument_variable": "instrument",
                "covariates": ["age"],
                "is_rct": False
            },
            # Method selection
            {
                "recommended_method": "instrumental_variable",
                "confidence": 0.85,
                "reasoning": "Instrumental variable available for causal identification"
            },
            # Result interpretation
            {
                "interpretation": "IV estimate of causal effect",
                "confidence_assessment": "Good confidence with valid instrument"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = "What is the causal effect of treatment on outcome using instrument as IV?"
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description="Study with instrumental variable"
        )
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_rdd_workflow_integration(self, mock_get_llm, mock_llm_call):
        """Test complete workflow with regression discontinuity data."""
        # Create test dataset
        dataset_path = self.create_test_dataset("rdd")
        
        # Mock LLM responses
        mock_llm_call.side_effect = [
            # Dataset analysis
            {
                "variables": ["running_var", "treatment", "outcome", "age"],
                "treatment_candidates": ["treatment"],
                "outcome_candidates": ["outcome"],
                "running_variables": ["running_var"]
            },
            # Query interpretation
            {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "running_variable": "running_var",
                "cutoff_value": 0.0,
                "covariates": ["age"],
                "is_rct": False
            },
            # Method selection
            {
                "recommended_method": "regression_discontinuity",
                "confidence": 0.9,
                "reasoning": "Clear discontinuity design with running variable"
            },
            # Result interpretation
            {
                "interpretation": "RDD estimate at discontinuity",
                "confidence_assessment": "High confidence with sharp discontinuity"
            }
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Run analysis
        query = "What is the effect of treatment at the cutoff using running_var?"
        result = run_causal_analysis(
            query=query,
            dataset_path=dataset_path,
            dataset_description="Regression discontinuity design"
        )
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
    
    def test_workflow_error_handling(self):
        """Test workflow error handling with invalid inputs."""
        # Test with non-existent file
        with self.assertRaises(Exception):
            run_causal_analysis(
                query="Test query",
                dataset_path="/nonexistent/file.csv",
                dataset_description="Test"
            )
        
        # Test with empty query
        dataset_path = self.create_test_dataset("rct")
        with patch('causal_agent.config.get_llm_client') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            
            result = run_causal_analysis(
                query="",
                dataset_path=dataset_path,
                dataset_description="Test"
            )
            
            # Should handle gracefully and return error info
            self.assertIsInstance(result, dict)
    
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_workflow_with_different_query_types(self, mock_get_llm, mock_llm_call):
        """Test workflow with different types of causal queries."""
        dataset_path = self.create_test_dataset("observational")
        
        # Mock basic responses
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome", "age"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome", "covariates": ["age"]},
            {"recommended_method": "backdoor_adjustment", "confidence": 0.8},
            {"interpretation": "Treatment effect estimated"}
        ] * 10  # Repeat for multiple queries
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Test different query formulations
        queries = [
            "What is the effect of treatment on outcome?",
            "Does treatment cause changes in outcome?",
            "How much does treatment increase outcome?",
            "What would happen to outcome if we changed treatment?",
            "Estimate the causal impact of treatment on outcome controlling for age"
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description="Test dataset"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('results', result)


class TestAgentComponentIntegration(unittest.TestCase):
    """Test integration between different agent components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('causal_agent.config.get_llm_client')
    def test_component_data_flow_integration(self, mock_get_llm):
        """Test data flow between components."""
        from causal_agent.tools.input_parser_tool import input_parser_tool
        from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
        from causal_agent.tools.query_interpreter_tool import query_interpreter_tool
        
        # Create test data
        data = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0],
            'outcome': [10, 12, 11, 13, 9],
            'age': [25, 30, 35, 40, 45]
        })
        dataset_path = os.path.join(self.temp_dir, "test_data.csv")
        data.to_csv(dataset_path, index=False)
        
        # Mock LLM
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Test component chain
        query = "What is the effect of treatment on outcome?"
        
        # Step 1: Input parsing
        input_result = input_parser_tool(
            f"Query: {query}\nDataset: {dataset_path}\nDescription: Test data"
        )
        
        self.assertIn('original_query', input_result)
        self.assertIn('dataset_path', input_result)
        
        # Step 2: Dataset analysis (using the parsed dataset path)
        with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
            mock_llm_call.return_value = {
                "variables": ["treatment", "outcome", "age"],
                "treatment_candidates": ["treatment"],
                "outcome_candidates": ["outcome"]
            }
            
            analysis_result = dataset_analyzer_tool.func(
                dataset_path=input_result['dataset_path'],
                dataset_description=input_result['dataset_description'],
                original_query=input_result['original_query']
            )
            
            self.assertIn('analysis_results', analysis_result)
        
        # Step 3: Query interpretation (using analysis results)
        with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
            mock_llm_call.return_value = {
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "covariates": ["age"],
                "is_rct": True
            }
            
            query_info = QueryInfo(
                query_text=input_result['original_query'],
                potential_treatments=input_result['extracted_variables'].get('treatment'),
                potential_outcomes=input_result['extracted_variables'].get('outcome'),
                covariates_hints=input_result['extracted_variables'].get('covariates_mentioned'),
                instrument_hints=input_result['extracted_variables'].get('instruments_mentioned')
            )
            
            interpretation_result = query_interpreter_tool.func(
                query_info=query_info,
                dataset_analysis=analysis_result.analysis_results,
                dataset_description=input_result['dataset_description'],
                original_query=input_result['original_query']
            )
            
            self.assertIn('variables', interpretation_result)
            
        # Verify data flows correctly between components
        self.assertEqual(
            interpretation_result.variables['treatment_variable'],
            'treatment'
        )
        self.assertEqual(
            interpretation_result.variables['outcome_variable'],
            'outcome'
        )
    
    def test_method_selection_validation_integration(self):
        """Test integration between method selection and validation."""
        from causal_agent.tools.method_selector_tool import method_selector_tool
        from causal_agent.tools.method_validator_tool import method_validator_tool
        
        # Mock variables and dataset analysis
        variables = {
            'treatment_variable': 'treatment',
            'outcome_variable': 'outcome',
            'covariates': ['age', 'gender'],
            'is_rct': False
        }
        
        dataset_analysis = {
            'n_samples': 500,
            'n_features': 4,
            'data_quality_score': 0.8
        }
        
        with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
            # Method selection response
            mock_llm_call.return_value = {
                "recommended_method": "backdoor_adjustment",
                "confidence": 0.8,
                "reasoning": "Observational data with confounders"
            }
            
            # Test method selection
            selection_result = method_selector_tool.func(
                variables=variables,
                dataset_analysis=dataset_analysis,
                dataset_description="Test observational data",
                original_query="Test query",
                excluded_methods=None
            )
            
            self.assertIn('method_info', selection_result)
            
            # Test method validation using selection result
            method_info = MethodInfo(**selection_result['method_info'])
            
            validation_input = MethodValidatorInput(
                method_info=method_info,
                variables=variables,
                dataset_analysis=dataset_analysis,
                dataset_description="Test observational data",
                original_query="Test query"
            )
            
            # Mock validation response
            mock_llm_call.return_value = {
                "validation_status": "valid",
                "confidence": 0.85,
                "warnings": [],
                "recommendations": []
            }
            
            validation_result = method_validator_tool.func(validation_input)
            
            self.assertIn('method', validation_result)
            self.assertEqual(validation_result['method'], 'backdoor_adjustment')


class TestWorkflowPerformanceIntegration(unittest.TestCase):
    """Test workflow performance with different dataset sizes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_performance_dataset(self, n_samples: int) -> str:
        """Create dataset for performance testing."""
        np.random.seed(42)
        data = {
            'treatment': np.random.binomial(1, 0.5, n_samples),
            'outcome': np.random.normal(10, 2, n_samples),
            'age': np.random.normal(35, 10, n_samples),
            'gender': np.random.binomial(1, 0.5, n_samples),
            'income': np.random.lognormal(10, 1, n_samples)
        }
        # Add treatment effect
        data['outcome'] += data['treatment'] * 2.0
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, f"perf_data_{n_samples}.csv")
        df.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.performance
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_workflow_performance_small_dataset(self, mock_get_llm, mock_llm_call):
        """Test workflow performance with small dataset (100 samples)."""
        import time
        
        dataset_path = self.create_performance_dataset(100)
        
        # Mock responses
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "diff_in_means", "confidence": 0.9},
            {"interpretation": "Treatment effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        start_time = time.time()
        result = run_causal_analysis(
            query="What is the effect of treatment on outcome?",
            dataset_path=dataset_path,
            dataset_description="Small performance test dataset"
        )
        execution_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 30.0, "Small dataset should process quickly")
        self.assertIsInstance(result, dict)
    
    @pytest.mark.performance
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_workflow_performance_medium_dataset(self, mock_get_llm, mock_llm_call):
        """Test workflow performance with medium dataset (1000 samples)."""
        import time
        
        dataset_path = self.create_performance_dataset(1000)
        
        # Mock responses
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "backdoor_adjustment", "confidence": 0.8},
            {"interpretation": "Treatment effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        start_time = time.time()
        result = run_causal_analysis(
            query="What is the effect of treatment on outcome?",
            dataset_path=dataset_path,
            dataset_description="Medium performance test dataset"
        )
        execution_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 60.0, "Medium dataset should process within reasonable time")
        self.assertIsInstance(result, dict)
    
    @pytest.mark.performance
    @pytest.mark.slow
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    @patch('causal_agent.config.get_llm_client')
    def test_workflow_performance_large_dataset(self, mock_get_llm, mock_llm_call):
        """Test workflow performance with large dataset (5000 samples)."""
        import time
        
        dataset_path = self.create_performance_dataset(5000)
        
        # Mock responses
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "propensity_score", "confidence": 0.85},
            {"interpretation": "Treatment effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        start_time = time.time()
        result = run_causal_analysis(
            query="What is the effect of treatment on outcome?",
            dataset_path=dataset_path,
            dataset_description="Large performance test dataset"
        )
        execution_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 120.0, "Large dataset should process within 2 minutes")
        self.assertIsInstance(result, dict)
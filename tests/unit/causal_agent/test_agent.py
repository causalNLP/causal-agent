"""
Tests for the agent module.
"""

import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock, call
import json

from causal_agent.agent import run_causal_analysis


class TestAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create test dataset
        test_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'outcome': [10, 12, 11, 13, 9, 14],
            'covariate': [1, 2, 1, 2, 1, 2]
        })
        test_data.to_csv(self.test_dataset_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('causal_agent.agent.explanation_generator_tool')
    @patch('causal_agent.agent.method_executor_tool')
    @patch('causal_agent.agent.method_validator_tool')
    @patch('causal_agent.agent.method_selector_tool')
    @patch('causal_agent.agent.query_interpreter_tool')
    @patch('causal_agent.agent.dataset_analyzer_tool')
    @patch('causal_agent.agent.input_parser_tool')
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_basic(self, mock_get_llm, mock_input_parser, 
                                       mock_dataset_analyzer, mock_query_interpreter,
                                       mock_method_selector, mock_method_validator,
                                       mock_method_executor, mock_explanation_generator):
        """Test basic run_causal_analysis functionality."""
        
        # Mock LLM client
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock input parser
        mock_input_parser.return_value = {
            "dataset_path": self.test_dataset_path,
            "dataset_description": "Test dataset",
            "original_query": "What is the effect of treatment on outcome?",
            "extracted_variables": {
                "treatment": ["treatment"],
                "outcome": ["outcome"],
                "covariates_mentioned": ["covariate"],
                "instruments_mentioned": []
            }
        }
        
        # Mock dataset analyzer
        mock_dataset_analyzer_result = MagicMock()
        mock_dataset_analyzer_result.analysis_results = {
            "dataset_info": {
                "num_rows": 6, 
                "num_columns": 3,
                "file_path": self.test_dataset_path,
                "file_name": "test_data.csv"
            },
            "columns": ["treatment", "outcome", "covariate"],
            "potential_treatments": ["treatment"],
            "potential_outcomes": ["outcome"],
            "temporal_structure_detected": False,
            "panel_data_detected": False,
            "potential_instruments_detected": False,
            "discontinuities_detected": False,
            "temporal_structure": {
                "has_temporal_structure": False,
                "temporal_columns": [],
                "is_panel_data": False,
                "id_column": None
            },
            "sample_size": 6,
            "num_covariates_estimate": 1
        }
        mock_dataset_analyzer.func.return_value = mock_dataset_analyzer_result
        
        # Mock query interpreter
        mock_query_interpreter_result = MagicMock()
        mock_query_interpreter_result.variables = {
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "covariates": ["covariate"],
            "treatment_variable_type": "binary"
        }
        mock_query_interpreter.func.return_value = mock_query_interpreter_result
        
        # Mock method selector
        mock_method_selector.func.return_value = {
            "method_info": {
                "method": "linear_regression",
                "method_justification": "Linear regression selected",
                "method_assumptions": ["linearity", "no_confounding"]
            }
        }
        
        # Mock method validator
        mock_method_validator.func.return_value = {
            "method": "linear_regression",
            "validation_passed": True,
            "validation_details": {"assumptions_met": True}
        }
        
        # Mock method executor
        mock_method_executor.func.return_value = {
            "effect_estimate": 2.5,
            "standard_error": 0.5,
            "confidence_interval": [1.5, 3.5],
            "p_value": 0.001
        }
        
        # Mock explanation generator
        mock_explanation_generator.func.return_value = {
            "results": {
                "results": {
                    "effect_estimate": 2.5,
                    "standard_error": 0.5,
                    "confidence_interval": [1.5, 3.5],
                    "p_value": 0.001,
                    "method_used": "linear_regression"
                },
                "variables": {
                    "treatment_variable": "treatment",
                    "outcome_variable": "outcome",
                    "covariates": ["covariate"]
                }
            },
            "explanation": "The treatment has a positive effect on the outcome."
        }
        
        # Run the analysis
        result = run_causal_analysis(
            query="What is the effect of treatment on outcome?",
            dataset_path=self.test_dataset_path,
            dataset_description="Test dataset"
        )
        
        # Verify all tools were called
        mock_input_parser.assert_called_once()
        mock_dataset_analyzer.func.assert_called_once()
        mock_query_interpreter.func.assert_called_once()
        mock_method_selector.func.assert_called_once()
        mock_method_validator.func.assert_called_once()
        mock_method_executor.func.assert_called_once()
        mock_explanation_generator.func.assert_called_once()
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)
        self.assertIn("results", result["results"])
        self.assertEqual(result["results"]["results"]["effect_estimate"], 2.5)
        self.assertEqual(result["results"]["results"]["method_used"], "linear_regression")
    
    @patch('causal_agent.agent.explanation_generator_tool')
    @patch('causal_agent.agent.method_executor_tool')
    @patch('causal_agent.agent.method_validator_tool')
    @patch('causal_agent.agent.method_selector_tool')
    @patch('causal_agent.agent.query_interpreter_tool')
    @patch('causal_agent.agent.dataset_analyzer_tool')
    @patch('causal_agent.agent.input_parser_tool')
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_without_description(self, mock_get_llm, mock_input_parser, 
                                                     mock_dataset_analyzer, mock_query_interpreter,
                                                     mock_method_selector, mock_method_validator,
                                                     mock_method_executor, mock_explanation_generator):
        """Test run_causal_analysis without dataset description."""
        
        # Setup mocks similar to basic test but without description
        mock_get_llm.return_value = MagicMock()
        
        mock_input_parser.return_value = {
            "dataset_path": self.test_dataset_path,
            "dataset_description": None,
            "original_query": "What is the effect of treatment on outcome?",
            "extracted_variables": {
                "treatment": ["treatment"],
                "outcome": ["outcome"],
                "covariates_mentioned": [],
                "instruments_mentioned": []
            }
        }
        
        mock_dataset_analyzer_result = MagicMock()
        mock_dataset_analyzer_result.analysis_results = {}
        mock_dataset_analyzer.func.return_value = mock_dataset_analyzer_result
        
        mock_query_interpreter_result = MagicMock()
        mock_query_interpreter_result.variables = {}
        mock_query_interpreter.func.return_value = mock_query_interpreter_result
        
        mock_method_selector.func.return_value = {"method_info": {"method": "correlation_analysis"}}
        mock_method_validator.func.return_value = {"method": "correlation_analysis"}
        mock_method_executor.func.return_value = {"correlation": 0.8}
        mock_explanation_generator.func.return_value = {
            "results": {"results": {"correlation": 0.8}, "variables": {}},
            "explanation": "Correlation analysis performed."
        }
        
        # Run without description
        result = run_causal_analysis(
            query="What is the effect of treatment on outcome?",
            dataset_path=self.test_dataset_path
        )
        
        # Verify input parser was called with None description
        input_call_args = mock_input_parser.call_args[0][0]
        self.assertIn("What is the effect of treatment on outcome?", input_call_args)
        self.assertIn(self.test_dataset_path, input_call_args)
        
        self.assertIsInstance(result, dict)
    
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_with_llm_model_env(self, mock_get_llm):
        """Test that LLM model is configured from environment variables."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        with patch.dict(os.environ, {'LLM_MODEL': 'gpt-3.5-turbo'}):
            with patch('causal_agent.agent.input_parser_tool') as mock_input_parser:
                mock_input_parser.side_effect = Exception("Stop early")
                
                try:
                    run_causal_analysis("test query", self.test_dataset_path)
                except:
                    pass  # We expect this to fail, we just want to check LLM setup
                
                # Verify get_llm_client was called with temperature=0 for non-o3 models
                mock_get_llm.assert_called_with(temperature=0)
    
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_with_o3_model(self, mock_get_llm):
        """Test that o3 models use different LLM configuration."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        with patch.dict(os.environ, {'LLM_MODEL': 'o3'}):
            with patch('causal_agent.agent.input_parser_tool') as mock_input_parser:
                mock_input_parser.side_effect = Exception("Stop early")
                
                try:
                    run_causal_analysis("test query", self.test_dataset_path)
                except:
                    pass  # We expect this to fail, we just want to check LLM setup
                
                # Verify get_llm_client was called without temperature for o3 models
                mock_get_llm.assert_called_with()
    
    @patch('causal_agent.agent.explanation_generator_tool')
    @patch('causal_agent.agent.method_executor_tool')
    @patch('causal_agent.agent.method_validator_tool')
    @patch('causal_agent.agent.method_selector_tool')
    @patch('causal_agent.agent.query_interpreter_tool')
    @patch('causal_agent.agent.dataset_analyzer_tool')
    @patch('causal_agent.agent.input_parser_tool')
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_with_complex_variables(self, mock_get_llm, mock_input_parser, 
                                                        mock_dataset_analyzer, mock_query_interpreter,
                                                        mock_method_selector, mock_method_validator,
                                                        mock_method_executor, mock_explanation_generator):
        """Test run_causal_analysis with complex variable extraction."""
        
        mock_get_llm.return_value = MagicMock()
        
        # Mock complex variable extraction
        mock_input_parser.return_value = {
            "dataset_path": self.test_dataset_path,
            "dataset_description": "Complex dataset",
            "original_query": "What is the effect of education on income controlling for age and gender?",
            "extracted_variables": {
                "treatment": ["education"],
                "outcome": ["income"],
                "covariates_mentioned": ["age", "gender"],
                "instruments_mentioned": ["distance_to_school"]
            }
        }
        
        mock_dataset_analyzer_result = MagicMock()
        mock_dataset_analyzer_result.analysis_results = {
            "dataset_info": {"num_rows": 6, "num_columns": 4},
            "potential_instruments_detected": True,
            "temporal_structure_detected": False,
            "temporal_structure": {"has_temporal_structure": False},
            "sample_size": 6,
            "num_covariates_estimate": 2
        }
        mock_dataset_analyzer.func.return_value = mock_dataset_analyzer_result
        
        mock_query_interpreter_result = MagicMock()
        mock_query_interpreter_result.variables = {
            "treatment_variable": "education",
            "outcome_variable": "income",
            "covariates": ["age", "gender"],
            "instrument_variable": "distance_to_school"
        }
        mock_query_interpreter.func.return_value = mock_query_interpreter_result
        
        mock_method_selector.func.return_value = {
            "method_info": {"method": "instrumental_variable"}
        }
        mock_method_validator.func.return_value = {"method": "instrumental_variable"}
        mock_method_executor.func.return_value = {"effect_estimate": 1.5}
        mock_explanation_generator.func.return_value = {
            "results": {"results": {"effect_estimate": 1.5}, "variables": {}},
            "explanation": "IV analysis performed."
        }
        
        result = run_causal_analysis(
            query="What is the effect of education on income controlling for age and gender?",
            dataset_path=self.test_dataset_path,
            dataset_description="Complex dataset"
        )
        
        # Verify QueryInfo was constructed with complex variables
        query_interpreter_call = mock_query_interpreter.func.call_args
        query_info = query_interpreter_call[1]['query_info']
        self.assertEqual(query_info.potential_treatments, ["education"])
        self.assertEqual(query_info.potential_outcomes, ["income"])
        self.assertEqual(query_info.covariates_hints, ["age", "gender"])
        self.assertEqual(query_info.instrument_hints, ["distance_to_school"])
        
        self.assertIsInstance(result, dict)
    
    @patch('causal_agent.agent.input_parser_tool')
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_input_parser_error(self, mock_get_llm, mock_input_parser):
        """Test error handling when input parser fails."""
        mock_get_llm.return_value = MagicMock()
        mock_input_parser.side_effect = Exception("Input parsing failed")
        
        result = run_causal_analysis("test query", self.test_dataset_path)
        
        # Should return error dict instead of raising exception
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("Input parsing failed", result["error"])
    
    @patch('causal_agent.agent.explanation_generator_tool')
    @patch('causal_agent.agent.method_executor_tool')
    @patch('causal_agent.agent.method_validator_tool')
    @patch('causal_agent.agent.method_selector_tool')
    @patch('causal_agent.agent.query_interpreter_tool')
    @patch('causal_agent.agent.dataset_analyzer_tool')
    @patch('causal_agent.agent.input_parser_tool')
    @patch('causal_agent.agent.get_llm_client')
    def test_run_causal_analysis_deprecated_api_key_parameter(self, mock_get_llm, mock_input_parser, 
                                                              mock_dataset_analyzer, mock_query_interpreter,
                                                              mock_method_selector, mock_method_validator,
                                                              mock_method_executor, mock_explanation_generator):
        """Test that deprecated api_key parameter is ignored."""
        
        # Setup minimal mocks
        mock_get_llm.return_value = MagicMock()
        mock_input_parser.return_value = {
            "dataset_path": self.test_dataset_path,
            "dataset_description": None,
            "original_query": "test",
            "extracted_variables": {"treatment": [], "outcome": [], "covariates_mentioned": [], "instruments_mentioned": []}
        }
        
        mock_dataset_analyzer_result = MagicMock()
        mock_dataset_analyzer_result.analysis_results = {}
        mock_dataset_analyzer.func.return_value = mock_dataset_analyzer_result
        
        mock_query_interpreter_result = MagicMock()
        mock_query_interpreter_result.variables = {}
        mock_query_interpreter.func.return_value = mock_query_interpreter_result
        
        mock_method_selector.func.return_value = {"method_info": {"method": "correlation_analysis"}}
        mock_method_validator.func.return_value = {"method": "correlation_analysis"}
        mock_method_executor.func.return_value = {}
        mock_explanation_generator.func.return_value = {"results": {"results": {}, "variables": {}}}
        
        # Call with deprecated api_key parameter
        result = run_causal_analysis(
            query="test query",
            dataset_path=self.test_dataset_path,
            api_key="deprecated_key"  # This should be ignored
        )
        
        # Should still work and return a result
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
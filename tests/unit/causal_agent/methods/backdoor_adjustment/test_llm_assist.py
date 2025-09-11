"""Unit tests for backdoor adjustment LLM assistance."""

import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from unittest.mock import MagicMock, patch

from causal_agent.methods.backdoor_adjustment.llm_assist import (
    identify_backdoor_set, 
    interpret_backdoor_results
)
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestBackdoorAdjustmentLLMAssist(MethodTestCase):
    """Test cases for backdoor adjustment LLM assistance functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=200, treatment_effect=3.0, random_seed=789)
        )
        self.confounded_data = self.generator.generate_observational_data()
        
        # Create sample columns and fitted results for testing
        self.df_cols = ['treatment', 'outcome', 'confounder_0', 'confounder_1', 'irrelevant_var']
        
        # Create fitted model results
        df_analysis = self.confounded_data.dropna()
        treatment = 'treatment'
        covariates = ['confounder_0', 'confounder_1']
        X = df_analysis[[treatment] + covariates]
        X = sm.add_constant(X)
        y = df_analysis['outcome']
        model = sm.OLS(y, X)
        self.results = model.fit()
        self.covariates = covariates
    
    def test_identify_backdoor_set_no_llm(self):
        """Test backdoor set identification without LLM."""
        existing_covariates = ['confounder_0', 'confounder_1']
        
        result = identify_backdoor_set(
            df_cols=self.df_cols,
            treatment='treatment',
            outcome='outcome',
            existing_covariates=existing_covariates,
            llm=None
        )
        
        # Should return existing covariates when no LLM provided
        self.assertEqual(result, existing_covariates)
    
    def test_identify_backdoor_set_empty_existing(self):
        """Test backdoor set identification with empty existing covariates."""
        result = identify_backdoor_set(
            df_cols=self.df_cols,
            treatment='treatment',
            outcome='outcome',
            existing_covariates=None,
            llm=None
        )
        
        # Should return empty list when no LLM and no existing covariates
        self.assertEqual(result, [])
    
    @patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output')
    def test_identify_backdoor_set_with_llm_success(self, mock_llm_call):
        """Test successful backdoor set identification with LLM."""
        # Mock LLM response
        mock_llm_call.return_value = {
            "suggested_backdoor_set": ["confounder_0", "confounder_1"]
        }
        
        mock_llm = MagicMock()
        existing_covariates = ['confounder_0']
        
        result = identify_backdoor_set(
            df_cols=self.df_cols,
            treatment='treatment',
            outcome='outcome',
            query="What causes both treatment and outcome?",
            existing_covariates=existing_covariates,
            llm=mock_llm
        )
        
        # Should combine existing and LLM suggestions, removing duplicates
        expected = ['confounder_0', 'confounder_1']
        self.assertEqual(result, expected)
        
        # Verify LLM was called with correct parameters
        mock_llm_call.assert_called_once_with(mock_llm, pytest.approx(str, rel=1e-3))
    
    @patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output')
    def test_identify_backdoor_set_llm_invalid_response(self, mock_llm_call):
        """Test backdoor set identification with invalid LLM response."""
        # Mock invalid LLM response
        mock_llm_call.return_value = {"invalid_key": "invalid_value"}
        
        mock_llm = MagicMock()
        existing_covariates = ['confounder_0']
        
        result = identify_backdoor_set(
            df_cols=self.df_cols,
            treatment='treatment',
            outcome='outcome',
            existing_covariates=existing_covariates,
            llm=mock_llm
        )
        
        # Should fall back to existing covariates
        self.assertEqual(result, existing_covariates)
    
    @patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output')
    def test_identify_backdoor_set_llm_non_string_items(self, mock_llm_call):
        """Test backdoor set identification with non-string items in LLM response."""
        # Mock LLM response with mixed types
        mock_llm_call.return_value = {
            "suggested_backdoor_set": ["confounder_0", 123, "confounder_1", None]
        }
        
        mock_llm = MagicMock()
        
        result = identify_backdoor_set(
            df_cols=self.df_cols,
            treatment='treatment',
            outcome='outcome',
            llm=mock_llm
        )
        
        # Should filter out non-string items
        expected = ['confounder_0', 'confounder_1']
        self.assertEqual(result, expected)
    
    def test_identify_backdoor_set_no_potential_confounders(self):
        """Test backdoor set identification when no potential confounders exist."""
        # Only treatment and outcome columns
        minimal_cols = ['treatment', 'outcome']
        
        result = identify_backdoor_set(
            df_cols=minimal_cols,
            treatment='treatment',
            outcome='outcome',
            llm=MagicMock()
        )
        
        # Should return empty list when no potential confounders
        self.assertEqual(result, [])
    
    def test_interpret_backdoor_results_no_llm(self):
        """Test result interpretation without LLM."""
        diagnostics = {"status": "Success", "details": {"r_squared": 0.75}}
        
        result = interpret_backdoor_results(
            results=self.results,
            diagnostics=diagnostics,
            treatment_var='treatment',
            covariates=self.covariates,
            llm=None
        )
        
        # Should return default message when no LLM
        self.assertEqual(result, "LLM interpretation not available for Backdoor Adjustment.")
    
    @patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output')
    def test_interpret_backdoor_results_with_llm_success(self, mock_llm_call):
        """Test successful result interpretation with LLM."""
        # Mock LLM response
        mock_interpretation = "The treatment effect of 3.0 is statistically significant and represents a substantial positive impact on the outcome."
        mock_llm_call.return_value = {
            "interpretation": mock_interpretation
        }
        
        mock_llm = MagicMock()
        diagnostics = {
            "status": "Success", 
            "details": {
                "r_squared": 0.75,
                "residuals_normality_status": "Normal",
                "homoscedasticity_status": "Homoscedastic",
                "multicollinearity_status": "Low"
            }
        }
        
        result = interpret_backdoor_results(
            results=self.results,
            diagnostics=diagnostics,
            treatment_var='treatment',
            covariates=self.covariates,
            llm=mock_llm
        )
        
        self.assertEqual(result, mock_interpretation)
        mock_llm_call.assert_called_once()
    
    @patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output')
    def test_interpret_backdoor_results_llm_invalid_response(self, mock_llm_call):
        """Test result interpretation with invalid LLM response."""
        # Mock invalid LLM response
        mock_llm_call.return_value = {"invalid_key": "invalid_value"}
        
        mock_llm = MagicMock()
        diagnostics = {"status": "Success", "details": {}}
        
        result = interpret_backdoor_results(
            results=self.results,
            diagnostics=diagnostics,
            treatment_var='treatment',
            covariates=self.covariates,
            llm=mock_llm
        )
        
        # Should return default message for invalid response
        self.assertEqual(result, "LLM interpretation not available for Backdoor Adjustment.")
    
    @patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output')
    def test_interpret_backdoor_results_llm_exception(self, mock_llm_call):
        """Test result interpretation when LLM call raises exception."""
        # Mock LLM call to raise exception
        mock_llm_call.side_effect = Exception("LLM service unavailable")
        
        mock_llm = MagicMock()
        diagnostics = {"status": "Success", "details": {}}
        
        result = interpret_backdoor_results(
            results=self.results,
            diagnostics=diagnostics,
            treatment_var='treatment',
            covariates=self.covariates,
            llm=mock_llm
        )
        
        # Should return error message
        self.assertIn("Error generating interpretation", result)
        self.assertIn("LLM service unavailable", result)
    
    def test_interpret_backdoor_results_missing_treatment_var(self):
        """Test result interpretation when treatment variable is missing from results."""
        # Create results without the expected treatment variable
        mock_results = MagicMock()
        mock_results.params = pd.Series({'const': 1.0, 'other_var': 0.5})
        mock_results.pvalues = pd.Series({'const': 0.1, 'other_var': 0.05})
        mock_results.conf_int.return_value = pd.DataFrame({
            0: [0.8, 0.3], 
            1: [1.2, 0.7]
        }, index=['const', 'other_var'])
        
        diagnostics = {"status": "Success", "details": {}}
        
        with patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output') as mock_llm_call:
            mock_llm_call.return_value = {"interpretation": "Test interpretation"}
            
            result = interpret_backdoor_results(
                results=mock_results,
                diagnostics=diagnostics,
                treatment_var='missing_treatment',
                covariates=['covariate1'],
                llm=MagicMock()
            )
            
            # Should handle missing treatment variable gracefully
            self.assertIsInstance(result, str)
    
    @pytest.mark.parametrize("diagnostic_status", ["Success", "Failed", "Unknown"])
    def test_interpret_backdoor_results_different_diagnostic_status(self, diagnostic_status):
        """Test result interpretation with different diagnostic statuses."""
        diagnostics = {
            "status": diagnostic_status,
            "details": {"r_squared": 0.5} if diagnostic_status == "Success" else {}
        }
        
        with patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output') as mock_llm_call:
            mock_llm_call.return_value = {"interpretation": f"Interpretation for {diagnostic_status} diagnostics"}
            
            result = interpret_backdoor_results(
                results=self.results,
                diagnostics=diagnostics,
                treatment_var='treatment',
                covariates=self.covariates,
                llm=MagicMock()
            )
            
            # Should handle different diagnostic statuses
            self.assertIsInstance(result, str)
            if diagnostic_status == "Success":
                self.assertIn("Interpretation for Success diagnostics", result)
    
    def test_identify_backdoor_set_prompt_construction(self):
        """Test that the prompt for backdoor set identification is properly constructed."""
        with patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output') as mock_llm_call:
            mock_llm_call.return_value = {"suggested_backdoor_set": []}
            
            identify_backdoor_set(
                df_cols=['treatment', 'outcome', 'confounder_0', 'confounder_1'],
                treatment='treatment',
                outcome='outcome',
                query="Test query for causal effect",
                existing_covariates=['confounder_0'],
                llm=MagicMock()
            )
            
            # Verify the prompt contains expected elements
            call_args = mock_llm_call.call_args[0]
            prompt = call_args[1]
            
            self.assertIn('treatment', prompt)
            self.assertIn('outcome', prompt)
            self.assertIn('Test query for causal effect', prompt)
            self.assertIn('confounder_0', prompt)
            self.assertIn('confounder_1', prompt)
            self.assertIn('backdoor adjustment', prompt.lower())
    
    def test_interpret_results_prompt_construction(self):
        """Test that the prompt for result interpretation is properly constructed."""
        diagnostics = {
            "status": "Success",
            "details": {
                "r_squared": 0.75,
                "residuals_normality_status": "Normal",
                "homoscedasticity_status": "Homoscedastic",
                "multicollinearity_status": "Low"
            }
        }
        
        with patch('causal_agent.methods.backdoor_adjustment.llm_assist.call_llm_with_json_output') as mock_llm_call:
            mock_llm_call.return_value = {"interpretation": "Test interpretation"}
            
            interpret_backdoor_results(
                results=self.results,
                diagnostics=diagnostics,
                treatment_var='treatment',
                covariates=self.covariates,
                llm=MagicMock()
            )
            
            # Verify the prompt contains expected elements
            call_args = mock_llm_call.call_args[0]
            prompt = call_args[1]
            
            self.assertIn('treatment', prompt)
            self.assertIn('Backdoor Adjustment', prompt)
            self.assertIn('0.75', prompt)  # R-squared value
            self.assertIn('Normal', prompt)  # Diagnostic status
            self.assertIn('assumption', prompt.lower())


if __name__ == '__main__':
    pytest.main([__file__])
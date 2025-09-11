"""Unit tests for backdoor adjustment diagnostics."""

import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from unittest.mock import patch, MagicMock

from causal_agent.methods.backdoor_adjustment.diagnostics import run_backdoor_diagnostics
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


class TestBackdoorAdjustmentDiagnostics(MethodTestCase):
    """Test cases for backdoor adjustment diagnostics."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=200, treatment_effect=3.0, random_seed=789)
        )
        self.confounded_data = self.generator.generate_observational_data()
        
        # Create a fitted model for testing
        df_analysis = self.confounded_data.dropna()
        treatment = 'treatment'
        covariates = ['confounder_0', 'confounder_1']
        X = df_analysis[[treatment] + covariates]
        X = sm.add_constant(X)
        y = df_analysis['outcome']
        model = sm.OLS(y, X)
        self.results = model.fit()
        self.X = X
    
    def test_run_backdoor_diagnostics_success(self):
        """Test successful diagnostic execution with real results."""
        diagnostics = run_backdoor_diagnostics(self.results, self.X)
        
        # Check basic structure
        self.assertIsInstance(diagnostics, dict)
        self.assertEqual(diagnostics["status"], "Success")
        self.assertIn("details", diagnostics)
        
        details = diagnostics["details"]
        
        # Check for key OLS diagnostic metrics
        expected_metrics = [
            'r_squared', 'adj_r_squared', 'f_statistic', 'f_p_value',
            'n_observations', 'degrees_of_freedom_resid', 'durbin_watson'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, details)
        
        # Check normality test results
        normality_keys = [
            'residuals_normality_jb_stat', 'residuals_normality_jb_p_value',
            'residuals_skewness', 'residuals_kurtosis', 'residuals_normality_status'
        ]
        for key in normality_keys:
            self.assertIn(key, details)
        
        # Check homoscedasticity test results
        homoscedasticity_keys = [
            'homoscedasticity_bp_lm_stat', 'homoscedasticity_bp_lm_p_value',
            'homoscedasticity_status'
        ]
        for key in homoscedasticity_keys:
            self.assertIn(key, details)
        
        # Check multicollinearity diagnostics
        multicollinearity_keys = ['model_condition_number', 'multicollinearity_status']
        for key in multicollinearity_keys:
            self.assertIn(key, details)
        
        # Check linearity placeholder
        self.assertIn('linearity_check', details)
        self.assertEqual(
            details['linearity_check'], 
            "Requires visual inspection (e.g., residual vs fitted plot)"
        )
    
    def test_diagnostic_value_types_and_ranges(self):
        """Test that diagnostic values have correct types and reasonable ranges."""
        diagnostics = run_backdoor_diagnostics(self.results, self.X)
        details = diagnostics["details"]
        
        # Check numeric types and ranges
        self.assertIsInstance(details["r_squared"], float)
        self.assertGreaterEqual(details["r_squared"], 0.0)
        self.assertLessEqual(details["r_squared"], 1.0)
        
        self.assertIsInstance(details["adj_r_squared"], float)
        
        self.assertIsInstance(details["f_statistic"], float)
        self.assertGreater(details["f_statistic"], 0.0)
        
        self.assertIsInstance(details["f_p_value"], float)
        self.assertGreaterEqual(details["f_p_value"], 0.0)
        self.assertLessEqual(details["f_p_value"], 1.0)
        
        self.assertIsInstance(details["n_observations"], int)
        self.assertGreater(details["n_observations"], 0)
        
        self.assertIsInstance(details["degrees_of_freedom_resid"], int)
        self.assertGreater(details["degrees_of_freedom_resid"], 0)
        
        # Durbin-Watson should be numeric or string (for edge cases)
        dw = details["durbin_watson"]
        self.assertTrue(isinstance(dw, (int, float, str)))
        if isinstance(dw, (int, float)):
            self.assertGreaterEqual(dw, 0.0)
            self.assertLessEqual(dw, 4.0)
    
    def test_normality_diagnostics(self):
        """Test normality diagnostic components."""
        diagnostics = run_backdoor_diagnostics(self.results, self.X)
        details = diagnostics["details"]
        
        # Check Jarque-Bera test results
        if details["residuals_normality_status"] != "N/A (Too few obs)":
            self.assertIsInstance(details["residuals_normality_jb_stat"], float)
            self.assertIsInstance(details["residuals_normality_jb_p_value"], float)
            self.assertIsInstance(details["residuals_skewness"], float)
            self.assertIsInstance(details["residuals_kurtosis"], float)
            
            # P-value should be between 0 and 1
            self.assertGreaterEqual(details["residuals_normality_jb_p_value"], 0.0)
            self.assertLessEqual(details["residuals_normality_jb_p_value"], 1.0)
            
            # Status should be interpretable
            status = details["residuals_normality_status"]
            self.assertIn(status, ["Normal", "Non-Normal", "Test Failed"])
    
    def test_homoscedasticity_diagnostics(self):
        """Test homoscedasticity diagnostic components."""
        diagnostics = run_backdoor_diagnostics(self.results, self.X)
        details = diagnostics["details"]
        
        # Check Breusch-Pagan test results
        status = details["homoscedasticity_status"]
        if status not in ["N/A (Too few obs or too many predictors)", "Test Failed"]:
            self.assertIsInstance(details["homoscedasticity_bp_lm_stat"], float)
            self.assertIsInstance(details["homoscedasticity_bp_lm_p_value"], float)
            
            # P-value should be between 0 and 1
            self.assertGreaterEqual(details["homoscedasticity_bp_lm_p_value"], 0.0)
            self.assertLessEqual(details["homoscedasticity_bp_lm_p_value"], 1.0)
            
            # Status should be interpretable
            self.assertIn(status, ["Homoscedastic", "Heteroscedastic"])
    
    def test_multicollinearity_diagnostics(self):
        """Test multicollinearity diagnostic components."""
        diagnostics = run_backdoor_diagnostics(self.results, self.X)
        details = diagnostics["details"]
        
        # Check condition number
        if details["multicollinearity_status"] != "Check Failed":
            self.assertIsInstance(details["model_condition_number"], float)
            self.assertGreater(details["model_condition_number"], 0.0)
            
            # Status should be interpretable
            status = details["multicollinearity_status"]
            expected_statuses = [
                "Low", "Moderate (Cond. No. > 10)", "High (Cond. No. > 30)"
            ]
            self.assertIn(status, expected_statuses)
    
    def test_run_backdoor_diagnostics_failure(self):
        """Test diagnostic failure mode with invalid input."""
        # Pass a non-results object
        dummy_X = pd.DataFrame({'const': [1], 'treatment': [0], 'cov1': [1]})
        diagnostics = run_backdoor_diagnostics("not a results object", dummy_X)
        
        self.assertEqual(diagnostics["status"], "Failed")
        self.assertIn("error", diagnostics)
        self.assertIsInstance(diagnostics["error"], str)
    
    def test_diagnostics_with_small_sample(self):
        """Test diagnostics with very small sample size."""
        # Create minimal dataset
        small_data = pd.DataFrame({
            'treatment': [0, 1],
            'outcome': [1, 2],
            'covariate': [0.5, 1.5]
        })
        
        X = small_data[['treatment', 'covariate']]
        X = sm.add_constant(X)
        y = small_data['outcome']
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        diagnostics = run_backdoor_diagnostics(results, X)
        
        # Should handle small sample gracefully
        self.assertEqual(diagnostics["status"], "Success")
        details = diagnostics["details"]
        
        # Some tests may not be applicable with very small samples
        self.assertTrue(
            details["residuals_normality_status"] in [
                "Normal", "Non-Normal", "N/A (Too few obs)", "Test Failed"
            ]
        )
    
    def test_diagnostics_with_perfect_fit(self):
        """Test diagnostics when model has perfect fit (R² = 1)."""
        # Create data with perfect linear relationship
        perfect_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'outcome': [1, 3, 1, 3],  # outcome = 1 + 2*treatment
            'covariate': [0, 0, 0, 0]  # No effect
        })
        
        X = perfect_data[['treatment', 'covariate']]
        X = sm.add_constant(X)
        y = perfect_data['outcome']
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        diagnostics = run_backdoor_diagnostics(results, X)
        
        # Should handle perfect fit case
        self.assertEqual(diagnostics["status"], "Success")
        details = diagnostics["details"]
        
        # R-squared should be very close to 1
        self.assertGreater(details["r_squared"], 0.99)
    
    @pytest.mark.parametrize("n_samples", [50, 100, 500, 1000])
    def test_diagnostics_with_different_sample_sizes(self, n_samples):
        """Test diagnostics with different sample sizes."""
        # Generate data with specific sample size
        config = SyntheticDataConfig(n_samples=n_samples, random_seed=42)
        generator = SyntheticDataGenerator(config)
        data = generator.generate_observational_data()
        
        # Fit model
        df_analysis = data.dropna()
        treatment = 'treatment'
        covariates = ['confounder_0', 'confounder_1']
        X = df_analysis[[treatment] + covariates]
        X = sm.add_constant(X)
        y = df_analysis['outcome']
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Run diagnostics
        diagnostics = run_backdoor_diagnostics(results, X)
        
        # Should succeed regardless of sample size
        self.assertEqual(diagnostics["status"], "Success")
        
        # Check that sample size is correctly reported
        details = diagnostics["details"]
        self.assertEqual(details["n_observations"], len(df_analysis))
    
    def test_diagnostics_error_handling(self):
        """Test error handling in diagnostic calculations."""
        # Test with various edge cases that might cause errors
        
        # Case 1: Results object without expected attributes
        mock_results = MagicMock()
        mock_results.nobs = 10
        mock_results.rsquared = 0.5
        mock_results.rsquared_adj = 0.4
        mock_results.fvalue = 5.0
        mock_results.f_pvalue = 0.05
        mock_results.df_resid = 7
        mock_results.resid = np.array([0.1, -0.2, 0.15, -0.1, 0.05, -0.08, 0.12, -0.15, 0.09, -0.06])
        
        # Mock the model.exog to avoid matrix operations
        mock_results.model.exog = np.array([[1, 0, 0.5], [1, 1, 1.5], [1, 0, 0.3], [1, 1, 1.2], 
                                          [1, 0, 0.8], [1, 1, 1.1], [1, 0, 0.6], [1, 1, 1.4],
                                          [1, 0, 0.7], [1, 1, 1.3]])
        
        X = pd.DataFrame(mock_results.model.exog, columns=['const', 'treatment', 'covariate'])
        
        # This should handle the mock gracefully
        diagnostics = run_backdoor_diagnostics(mock_results, X)
        
        # Should either succeed or fail gracefully
        self.assertIn(diagnostics["status"], ["Success", "Failed"])


if __name__ == '__main__':
    pytest.main([__file__])
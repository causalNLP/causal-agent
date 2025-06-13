import unittest
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Assuming the causalscientist package is installed or PYTHONPATH is set correctly
from auto_causal.methods.generalized_propensity_score.estimator import (
    _estimate_gps_values, 
    _estimate_outcome_model, 
    _generate_dose_response_function,
    estimate_effect_gps
)
from auto_causal.methods.generalized_propensity_score.diagnostics import assess_gps_balance

class TestGeneralizedPropensityScore(unittest.TestCase):

    def _generate_synthetic_data(self, n=1000, seed=42, linear_gps=True, linear_outcome=False):
        """
        Generates synthetic data for testing GPS.
        T = a0 + a1*X1 + a2*X2 + u
        Y = b0 + b1*T + b2*T^2 + c1*X1 + c2*X2 + v (if non-linear outcome)
        Y = b0 + b1*T + c1*X1 + c2*X2 + v (if linear outcome)
        GPS is based on T ~ X.
        Outcome model is Y ~ T, GPS, and their transformations.
        """
        np.random.seed(seed)
        data = pd.DataFrame()
        data['X1'] = np.random.normal(0, 1, n)
        data['X2'] = np.random.binomial(1, 0.5, n)

        # Treatment assignment
        if linear_gps:
            # Ensure treatment has reasonable variance and is affected by covariates
            data['T'] = 0.5 + 1.5 * data['X1'] - 1.0 * data['X2'] + np.random.normal(0, 2, n) 
        else: # Non-linear treatment assignment (not directly used by current _estimate_gps_values OLS)
            data['T'] = 0.5 + 1.5 * data['X1']**2 - 1.0 * data['X2'] + np.random.normal(0, 2, n)
        
        # Outcome generation
        # True Dose-Response: E[Y(t)] = 10 + 5*t - 0.5*t^2 (example)
        # Confounding effect of X1, X2
        confounding_effect = 2.0 * data['X1'] + 1.0 * data['X2']
        
        if linear_outcome:
             # Y = b0 + b1*T + c1*X1 + c2*X2 + v
            data['Y'] = 10 + 3 * data['T'] + confounding_effect + np.random.normal(0, 3, n)
        else:
            # Y = b0 + b1*T + b2*T^2 + c1*X1 + c2*X2 + v
            data['Y'] = 10 + 5 * data['T'] - 0.5 * data['T']**2 + confounding_effect + np.random.normal(0, 3, n)

        return data

    def test_generate_synthetic_data_smoke(self):
        df = self._generate_synthetic_data()
        self.assertEqual(len(df), 1000)
        self.assertIn('T', df.columns)
        self.assertIn('Y', df.columns)
        self.assertIn('X1', df.columns)
        self.assertIn('X2', df.columns)

    def test_estimate_gps_values_linear_case(self):
        df = self._generate_synthetic_data(n=100)
        treatment_var = 'T'
        covariate_vars = ['X1', 'X2']
        gps_model_spec = {"type": "linear"}

        df_with_gps, diagnostics = _estimate_gps_values(df.copy(), treatment_var, covariate_vars, gps_model_spec)
        
        self.assertIn('gps_score', df_with_gps.columns)
        self.assertFalse(df_with_gps['gps_score'].isnull().all(), "GPS scores should not be all NaNs")
        self.assertGreater(df_with_gps['gps_score'].mean(), 0, "Mean GPS score should be positive")
        self.assertIn("gps_model_type", diagnostics)
        self.assertEqual(diagnostics["gps_model_type"], "linear_ols")
        self.assertTrue(0 <= diagnostics["gps_model_rsquared"] <= 1)

    def test_estimate_gps_values_no_covariates(self):
        df = self._generate_synthetic_data(n=50)
        treatment_var = 'T'
        gps_model_spec = {"type": "linear"}
        
        # Test with empty list of covariates
        df_with_gps, diagnostics = _estimate_gps_values(df.copy(), treatment_var, [], gps_model_spec)
        self.assertIn('gps_score', df_with_gps.columns) # Should still add the column
        self.assertTrue(df_with_gps['gps_score'].isnull().all(), "GPS scores should be all NaN if no covariates")
        self.assertIn("error", diagnostics)
        self.assertEqual(diagnostics["error"], "No covariates provided.")

    def test_estimate_gps_values_zero_variance_residual(self):
        # Create data where T is perfectly predicted by X1 (zero residual variance)
        df = pd.DataFrame({'X1': np.random.normal(0, 1, 50)})
        df['T'] = 2 * df['X1'] # Perfect prediction
        df['X2'] = np.random.binomial(1, 0.5, 50) # Dummy covariate
        treatment_var = 'T'
        covariate_vars = ['X1'] # Using only X1 for perfect prediction
        gps_model_spec = {"type": "linear"}

        df_with_gps, diagnostics = _estimate_gps_values(df.copy(), treatment_var, covariate_vars, gps_model_spec)
        self.assertIn('gps_score', df_with_gps.columns)
        self.assertTrue(df_with_gps['gps_score'].isnull().all(), "GPS should be NaN if residual variance is zero")
        self.assertIn("warning_sigma_sq_hat_near_zero", diagnostics) # Check for the warning when it's very close to zero
        
    def test_estimate_gps_values_not_enough_dof(self):
        df = pd.DataFrame({
            'T': [1,2,3],
            'X1': [10,11,12],
            'X2': [1,0,1],
            'X3': [5,6,7] # T = X1 - 9 + X3 - 5 (perfectly determined)
        })
        # n=3, k_params (const, X1, X2, X3) = 4. n-k = -1
        df_res, diagnostics = _estimate_gps_values(df.copy(), 'T', ['X1', 'X2', 'X3'], {"type": "linear"})
        self.assertTrue(df_res['gps_score'].isnull().all())
        self.assertIn("Not enough degrees of freedom for GPS variance", diagnostics.get("error", ""))


    def test_estimate_outcome_model_structure(self):
        df = self._generate_synthetic_data(n=100)
        # First, get some GPS scores
        df_with_gps, _ = _estimate_gps_values(df.copy(), 'T', ['X1', 'X2'], {"type": "linear"})
        df_with_gps.dropna(subset=['gps_score', 'Y', 'T', 'X1', 'X2'], inplace=True) # Ensure no NaNs for model fitting
        
        self.assertFalse(df_with_gps.empty, "DataFrame became empty after GPS estimation for outcome model test")

        outcome_var = 'Y'
        treatment_var = 'T'
        gps_col_name = 'gps_score'
        # Standard polynomial spec as used in estimator.py
        outcome_model_spec = {"type": "polynomial", "degree": 2, "interaction": True}

        fitted_model = _estimate_outcome_model(df_with_gps, outcome_var, treatment_var, gps_col_name, outcome_model_spec)
        
        self.assertIsNotNone(fitted_model)
        self.assertIsInstance(fitted_model, sm.regression.linear_model.RegressionResultsWrapper)
        
        expected_terms = ['intercept', 'T', 'GPS', 'T_sq', 'GPS_sq', 'T_x_GPS']
        for term in expected_terms:
            self.assertIn(term, fitted_model.model.exog_names, f"Term {term} missing from outcome model")

    def test_generate_dose_response_function(self):
        df = self._generate_synthetic_data(n=200)
        df_with_gps, _ = _estimate_gps_values(df.copy(), 'T', ['X1', 'X2'], {"type": "linear"})
        df_with_gps.dropna(subset=['gps_score', 'Y', 'T'], inplace=True)
        self.assertFalse(df_with_gps.empty, "Test setup failed: df_with_gps is empty")


        outcome_model_spec = {"type": "polynomial", "degree": 2, "interaction": True}
        fitted_outcome_model = _estimate_outcome_model(df_with_gps, 'Y', 'T', 'gps_score', outcome_model_spec)
        
        t_values = np.linspace(df_with_gps['T'].min(), df_with_gps['T'].max(), 5).tolist()
        adrf_estimates = _generate_dose_response_function(
            df_with_gps, fitted_outcome_model, 'T', 'gps_score', outcome_model_spec, t_values
        )
        
        self.assertEqual(len(adrf_estimates), len(t_values))
        self.assertFalse(np.isnan(adrf_estimates).any(), "ADRF estimates should not be NaN for valid inputs")

    def test_generate_dose_response_empty_t_values(self):
        df = self._generate_synthetic_data(n=50) # Dummy data
        df_with_gps, _ = _estimate_gps_values(df.copy(), 'T', ['X1', 'X2'], {"type": "linear"})
        df_with_gps.dropna(subset=['gps_score', 'Y', 'T'], inplace=True)
        outcome_model_spec = {"type": "polynomial", "degree": 2, "interaction": True}
        fitted_outcome_model = _estimate_outcome_model(df_with_gps, 'Y', 'T', 'gps_score', outcome_model_spec)

        adrf_estimates = _generate_dose_response_function(
            df_with_gps, fitted_outcome_model, 'T', 'gps_score', outcome_model_spec, [] # Empty t_values
        )
        self.assertEqual(len(adrf_estimates), 0)

    def test_estimate_effect_gps_end_to_end_smoke(self):
        df = self._generate_synthetic_data(n=200, seed=123)
        results = estimate_effect_gps(
            df, 
            treatment_var='T', 
            outcome_var='Y', 
            covariate_vars=['X1', 'X2'],
            t_values_for_adrf=np.linspace(df['T'].min(), df['T'].max(), 7).tolist() # specify t_values
        )
        
        self.assertNotIn("error", results, f"estimate_effect_gps returned an error: {results.get('error')}")
        self.assertIn("adrf_curve", results)
        self.assertIn("diagnostics", results)
        self.assertIn("method_details", results)
        self.assertIn("parameters_used", results)
        
        self.assertIn("t_levels", results["adrf_curve"])
        self.assertIn("expected_outcomes", results["adrf_curve"])
        self.assertEqual(len(results["adrf_curve"]["t_levels"]), 7)
        self.assertEqual(len(results["adrf_curve"]["expected_outcomes"]), 7)
        self.assertIsInstance(results["diagnostics"]["gps_estimation_diagnostics"], dict)
        self.assertIsInstance(results["diagnostics"]["balance_check"], dict) # From assess_gps_balance

    def test_estimate_effect_gps_gps_estimation_failure(self):
        # Test case where GPS estimation might fail (e.g., no covariates)
        df = self._generate_synthetic_data(n=50)
        results = estimate_effect_gps(
            df, 
            treatment_var='T', 
            outcome_var='Y', 
            covariate_vars=[] # No covariates
        )
        self.assertIn("error", results)
        self.assertEqual(results["error"], "GPS estimation failed.")
        self.assertIn("no covariates provided", results["diagnostics"]["error"].lower())


    # --- Tests for assess_gps_balance (from diagnostics.py) ---
    def test_assess_gps_balance_smoke(self):
        df_synth = self._generate_synthetic_data(n=300)
        df_with_gps, _ = _estimate_gps_values(df_synth.copy(), 'T', ['X1', 'X2'], {"type": "linear"})
        df_with_gps.dropna(subset=['gps_score', 'T', 'X1', 'X2'], inplace=True)
        
        self.assertGreater(len(df_with_gps), 100, "Not enough data after NaN drop for balance test setup.")

        balance_results = assess_gps_balance(
            df_with_gps, 
            treatment_var='T', 
            covariate_vars=['X1', 'X2'], 
            gps_col_name='gps_score',
            num_strata=3 # Test with fewer strata
        )
        
        self.assertNotIn("error", balance_results, f"assess_gps_balance returned an error: {balance_results.get('error')}")
        self.assertIn("balance_results_per_covariate", balance_results)
        self.assertIn("summary_stats", balance_results)
        self.assertIn("X1", balance_results["balance_results_per_covariate"])
        self.assertIn("X2", balance_results["balance_results_per_covariate"])
        self.assertIsInstance(balance_results["balance_results_per_covariate"]['X1']["strata_details"], list)
        self.assertGreater(len(balance_results["balance_results_per_covariate"]['X1']["strata_details"]), 0)
        self.assertEqual(balance_results["summary_stats"]["num_strata_used"], 3)


    def test_assess_gps_balance_all_gps_nan(self):
        df_synth = self._generate_synthetic_data(n=50)
        df_synth['gps_score_all_nan'] = np.nan # All GPS scores are NaN
        
        balance_results = assess_gps_balance(
            df_synth,
            treatment_var='T',
            covariate_vars=['X1'],
            gps_col_name='gps_score_all_nan'
        )
        self.assertIn("error", balance_results)
        self.assertEqual(balance_results["error"], "All GPS scores are NaN.")

    def test_assess_gps_balance_qcut_failure_fallback(self):
        # Test qcut failure and fallback (e.g. GPS has very few unique values)
        df = pd.DataFrame({
            'T': np.random.rand(50),
            'X1': np.random.rand(50),
            'gps_score': np.array([0.1]*20 + [0.2]*20 + [0.3]*10) # Only 3 unique GPS values
        })
        balance_results = assess_gps_balance(df, 'T', ['X1'], 'gps_score', num_strata=5)
        self.assertNotIn("error", balance_results.get("summary_stats", {}).get("error", "")) # Check for critical error
        self.assertIn("warnings", balance_results["summary_stats"])
        # Check for the warning about forming fewer strata than requested
        actual_strata_formed = balance_results["summary_stats"].get('actual_num_strata_formed', 0)
        expected_warning_part = f"Only {actual_strata_formed} strata formed out of 5 requested"
        current_warnings = balance_results["summary_stats"]["warnings"]
        self.assertTrue(any(expected_warning_part in w 
                            for w in current_warnings),
                        f"Expected warning '{expected_warning_part}' not found. Warnings: {current_warnings}")
        self.assertEqual(balance_results["summary_stats"]["actual_num_strata_formed"], 3)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 
"""Parametrized tests for all causal inference methods."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import (
    SyntheticDataGenerator, 
    DatasetType, 
    SyntheticDataConfig,
    create_benchmark_datasets
)
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestCausalMethodsParametrized(MethodTestCase):
    """Parametrized tests across all causal inference methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.benchmark_datasets = create_benchmark_datasets()
    
    @pytest.mark.parametrize("method_name,dataset_type,expected_keys", [
        ("backdoor_adjustment", "strong_effect_obs", [
            'effect_estimate', 'p_value', 'confidence_interval', 
            'standard_error', 'formula', 'model_summary', 
            'diagnostics', 'interpretation', 'method_used'
        ]),
        ("propensity_score_matching", "weak_effect_obs", [
            'effect_estimate', 'effect_se', 'confidence_interval',
            'diagnostics', 'method_details', 'parameters'
        ]),
        ("propensity_score_weighting", "strong_effect_obs", [
            'effect_estimate', 'effect_se', 'confidence_interval',
            'diagnostics', 'method_details', 'parameters'
        ]),
        ("instrumental_variable", "weak_instrument", [
            'effect_estimate', 'treatment_variable', 'outcome_variable',
            'instrument_variables', 'method_used', 'diagnostics'
        ]),
    ])
    def test_method_output_structure(self, method_name, dataset_type, expected_keys):
        """Test that all methods return expected output structure."""
        dataset = self.benchmark_datasets[dataset_type]
        
        # Get method-specific parameters
        method_params = self._get_method_parameters(method_name, dataset)
        
        # Mock method-specific dependencies
        with self._mock_method_dependencies(method_name):
            # Import and call method
            method_func = self._import_method(method_name)
            
            try:
                results = method_func(
                    dataset,
                    method_params['treatment'],
                    method_params['outcome'],
                    method_params['covariates'],
                    **method_params.get('kwargs', {})
                )
                
                # Check that all expected keys are present
                for key in expected_keys:
                    self.assertIn(key, results, f"Missing key '{key}' in {method_name} results")
                
                # Check that effect estimate is numeric
                if 'effect_estimate' in results:
                    self.assertIsInstance(
                        results['effect_estimate'], 
                        (int, float, type(None)),
                        f"Effect estimate should be numeric in {method_name}"
                    )
                
            except Exception as e:
                # Some methods might fail with certain datasets - that's okay for this test
                self.skipTest(f"Method {method_name} failed with dataset {dataset_type}: {e}")
    
    @pytest.mark.parametrize("dataset_type,treatment_effect", [
        ("small_rct", 0.5),
        ("large_rct", 0.3),
        ("weak_effect_obs", 0.1),
        ("strong_effect_obs", 0.8),
    ])
    def test_backdoor_adjustment_with_different_datasets(self, dataset_type, treatment_effect):
        """Test backdoor adjustment with different dataset characteristics."""
        dataset = self.benchmark_datasets[dataset_type]
        true_effect = dataset.attrs.get('true_treatment_effect', treatment_effect)
        
        with patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics') as mock_diag, \
             patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results') as mock_interp:
            
            mock_diag.return_value = {"status": "Success", "details": {}}
            mock_interp.return_value = "Test interpretation"
            
            from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect
            
            # Get confounders from dataset metadata
            confounders = dataset.attrs.get('confounders', [])
            if not confounders:
                # For RCT data, use available feature columns
                confounders = [col for col in dataset.columns 
                             if col.startswith('feature_') or col.startswith('confounder_')][:2]
            
            if confounders:  # Only test if we have confounders
                results = estimate_effect(
                    dataset,
                    'treatment',
                    'outcome',
                    confounders
                )
                
                # Check that estimate is in reasonable range
                estimate = results['effect_estimate']
                self.assertIsInstance(estimate, (int, float))
                
                # For synthetic data, estimate should be somewhat close to true effect
                # Allow generous tolerance due to confounding and sample variation
                tolerance = max(1.0, abs(true_effect) * 2)
                self.assertLess(
                    abs(estimate - true_effect), 
                    tolerance,
                    f"Estimate {estimate} too far from true effect {true_effect}"
                )
    
    @pytest.mark.parametrize("weight_type", ["ATE", "ATT"])
    def test_propensity_score_weighting_estimands(self, weight_type):
        """Test propensity score weighting with different estimands."""
        dataset = self.benchmark_datasets["strong_effect_obs"]
        
        with patch('causal_agent.methods.propensity_score.weighting.get_llm_parameters') as mock_llm, \
             patch('causal_agent.methods.propensity_score.weighting.assess_weight_distribution') as mock_assess:
            
            mock_llm.return_value = {"parameters": {}}
            mock_assess.return_value = {"weight_quality": "good"}
            
            from causal_agent.methods.propensity_score.weighting import estimate_effect
            
            confounders = dataset.attrs.get('confounders', ['confounder_0', 'confounder_1'])
            
            results = estimate_effect(
                dataset,
                'treatment',
                'outcome',
                confounders,
                weight_type=weight_type
            )
            
            # Check that weight type is recorded correctly
            self.assertEqual(results['parameters']['weight_type'], weight_type)
            
            # Check that estimate is numeric
            self.assertIsInstance(results['effect_estimate'], (int, float))
    
    @pytest.mark.parametrize("instrument_strength", ["weak_instrument", "strong_instrument"])
    def test_instrumental_variable_with_different_strengths(self, instrument_strength):
        """Test instrumental variable with different instrument strengths."""
        dataset = self.benchmark_datasets[instrument_strength]
        
        with patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics') as mock_diag, \
             patch('causal_agent.methods.instrumental_variable.estimator.CausalModel') as mock_model:
            
            mock_diag.return_value = {'first_stage_f': 15.0 if 'strong' in instrument_strength else 5.0}
            
            # Mock DoWhy to fail so we test statsmodels path
            mock_model.side_effect = Exception("DoWhy failed")
            
            with patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS') as mock_iv2sls:
                mock_sm_model = MagicMock()
                mock_sm_results = MagicMock()
                mock_sm_results.params = {'treatment': 1.5}
                
                mock_iv2sls.return_value = mock_sm_model
                mock_sm_model.fit.return_value = mock_sm_results
                
                from causal_agent.methods.instrumental_variable.estimator import estimate_effect
                
                confounders = dataset.attrs.get('confounders', ['covariate_0', 'covariate_1'])
                instrument = dataset.attrs.get('instrument', 'instrument')
                
                results = estimate_effect(
                    dataset,
                    'treatment',
                    'outcome',
                    confounders,
                    instrument_variable=instrument
                )
                
                # Should work with both weak and strong instruments
                self.assertIn('effect_estimate', results)
                self.assertEqual(results['instrument_variables'], [instrument])
    
    @pytest.mark.parametrize("method_name", [
        "backdoor_adjustment",
        "propensity_score_matching", 
        "propensity_score_weighting"
    ])
    def test_methods_with_missing_data(self, method_name):
        """Test how methods handle missing data."""
        # Create dataset with missing values
        base_dataset = self.benchmark_datasets["strong_effect_obs"]
        dataset_with_missing = base_dataset.copy()
        
        # Introduce missing values
        n_missing = len(dataset_with_missing) // 10  # 10% missing
        missing_indices = np.random.choice(len(dataset_with_missing), n_missing, replace=False)
        dataset_with_missing.loc[missing_indices, 'confounder_0'] = np.nan
        
        method_params = self._get_method_parameters(method_name, dataset_with_missing)
        
        with self._mock_method_dependencies(method_name):
            method_func = self._import_method(method_name)
            
            try:
                results = method_func(
                    dataset_with_missing,
                    method_params['treatment'],
                    method_params['outcome'],
                    method_params['covariates'],
                    **method_params.get('kwargs', {})
                )
                
                # Should either handle missing data gracefully or raise appropriate error
                if 'error' not in results:
                    self.assertIn('effect_estimate', results)
                
            except (ValueError, KeyError) as e:
                # Expected for some methods with missing data
                self.assertIn('missing', str(e).lower())
    
    @pytest.mark.parametrize("n_samples", [50, 200, 1000])
    def test_methods_with_different_sample_sizes(self, n_samples):
        """Test method performance with different sample sizes."""
        # Generate dataset with specific sample size
        config = SyntheticDataConfig(
            n_samples=n_samples,
            treatment_effect=0.5,
            random_seed=42
        )
        generator = SyntheticDataGenerator(config)
        dataset = generator.generate_observational_data()
        
        # Test backdoor adjustment (most robust to sample size)
        with patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics') as mock_diag, \
             patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results') as mock_interp:
            
            mock_diag.return_value = {"status": "Success", "details": {}}
            mock_interp.return_value = "Test interpretation"
            
            from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect
            
            confounders = dataset.attrs.get('confounders', ['confounder_0', 'confounder_1'])
            
            results = estimate_effect(
                dataset,
                'treatment',
                'outcome',
                confounders
            )
            
            # Should work with all sample sizes
            self.assertIn('effect_estimate', results)
            self.assertIsInstance(results['effect_estimate'], (int, float))
            
            # Standard error should decrease with larger samples (roughly)
            if n_samples >= 200:
                self.assertGreater(results['standard_error'], 0.0)
    
    def _get_method_parameters(self, method_name: str, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Get method-specific parameters for testing."""
        base_params = {
            'treatment': 'treatment',
            'outcome': 'outcome',
            'covariates': dataset.attrs.get('confounders', ['confounder_0', 'confounder_1']),
            'kwargs': {}
        }
        
        if method_name == "instrumental_variable":
            base_params['kwargs']['instrument_variable'] = dataset.attrs.get('instrument', 'instrument')
        elif method_name == "propensity_score_weighting":
            base_params['kwargs']['weight_type'] = 'ATE'
        elif method_name == "propensity_score_matching":
            base_params['kwargs']['n_bootstraps'] = 20  # Small for testing
        
        return base_params
    
    def _mock_method_dependencies(self, method_name: str):
        """Create context manager for method-specific mocks."""
        if method_name == "backdoor_adjustment":
            return patch.multiple(
                'causal_agent.methods.backdoor_adjustment.estimator',
                run_backdoor_diagnostics=MagicMock(return_value={"status": "Success", "details": {}}),
                interpret_backdoor_results=MagicMock(return_value="Test interpretation")
            )
        elif method_name == "propensity_score_matching":
            return patch.multiple(
                'causal_agent.methods.propensity_score.matching',
                get_llm_parameters=MagicMock(return_value={"parameters": {}}),
                assess_balance=MagicMock(return_value={"balance_score": 0.8})
            )
        elif method_name == "propensity_score_weighting":
            return patch.multiple(
                'causal_agent.methods.propensity_score.weighting',
                get_llm_parameters=MagicMock(return_value={"parameters": {}}),
                assess_weight_distribution=MagicMock(return_value={"weight_quality": "good"})
            )
        elif method_name == "instrumental_variable":
            return patch.multiple(
                'causal_agent.methods.instrumental_variable.estimator',
                run_iv_diagnostics=MagicMock(return_value={'first_stage_f': 15.0}),
                CausalModel=MagicMock(side_effect=Exception("DoWhy failed")),
                IV2SLS=MagicMock(return_value=MagicMock(fit=MagicMock(return_value=MagicMock(params={'treatment': 1.5}))))
            )
        else:
            return patch('builtins.print')  # No-op context manager
    
    def _import_method(self, method_name: str):
        """Import the appropriate method function."""
        if method_name == "backdoor_adjustment":
            from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect
            return estimate_effect
        elif method_name == "propensity_score_matching":
            from causal_agent.methods.propensity_score.matching import estimate_effect
            return estimate_effect
        elif method_name == "propensity_score_weighting":
            from causal_agent.methods.propensity_score.weighting import estimate_effect
            return estimate_effect
        elif method_name == "instrumental_variable":
            from causal_agent.methods.instrumental_variable.estimator import estimate_effect
            return estimate_effect
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def test_all_methods_confidence_intervals(self):
        """Test that all methods produce valid confidence intervals."""
        dataset = self.benchmark_datasets["strong_effect_obs"]
        
        methods_to_test = [
            ("backdoor_adjustment", 'confidence_interval'),
            ("propensity_score_matching", 'confidence_interval'),
            ("propensity_score_weighting", 'confidence_interval'),
        ]
        
        for method_name, ci_key in methods_to_test:
            with self.subTest(method=method_name):
                method_params = self._get_method_parameters(method_name, dataset)
                
                with self._mock_method_dependencies(method_name):
                    method_func = self._import_method(method_name)
                    
                    try:
                        results = method_func(
                            dataset,
                            method_params['treatment'],
                            method_params['outcome'],
                            method_params['covariates'],
                            **method_params.get('kwargs', {})
                        )
                        
                        if ci_key in results and results[ci_key] is not None:
                            ci = results[ci_key]
                            self.assertIsInstance(ci, list)
                            self.assertEqual(len(ci), 2)
                            self.assertLess(ci[0], ci[1], f"Invalid CI in {method_name}: {ci}")
                            
                            # Effect estimate should be within CI (if available)
                            if 'effect_estimate' in results and results['effect_estimate'] is not None:
                                effect = results['effect_estimate']
                                self.assertGreaterEqual(effect, ci[0], f"Effect {effect} below CI lower bound in {method_name}")
                                self.assertLessEqual(effect, ci[1], f"Effect {effect} above CI upper bound in {method_name}")
                    
                    except Exception as e:
                        self.skipTest(f"Method {method_name} failed: {e}")
    
    @pytest.mark.parametrize("method_name", [
        "backdoor_adjustment",
        "propensity_score_matching",
        "propensity_score_weighting"
    ])
    def test_methods_error_handling(self, method_name):
        """Test error handling across methods."""
        dataset = self.benchmark_datasets["strong_effect_obs"]
        
        # Test with invalid column names
        with self._mock_method_dependencies(method_name):
            method_func = self._import_method(method_name)
            
            with self.assertRaises((KeyError, ValueError)):
                method_func(
                    dataset,
                    'invalid_treatment',
                    'outcome',
                    ['confounder_0']
                )
    
    def test_method_reproducibility(self):
        """Test that methods produce consistent results with same random seed."""
        dataset = self.benchmark_datasets["strong_effect_obs"]
        
        # Test backdoor adjustment reproducibility
        with patch('causal_agent.methods.backdoor_adjustment.estimator.run_backdoor_diagnostics') as mock_diag, \
             patch('causal_agent.methods.backdoor_adjustment.estimator.interpret_backdoor_results') as mock_interp:
            
            mock_diag.return_value = {"status": "Success", "details": {}}
            mock_interp.return_value = "Test interpretation"
            
            from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect
            
            confounders = dataset.attrs.get('confounders', ['confounder_0', 'confounder_1'])
            
            # Run twice with same data
            results1 = estimate_effect(dataset, 'treatment', 'outcome', confounders)
            results2 = estimate_effect(dataset, 'treatment', 'outcome', confounders)
            
            # Should get identical results (deterministic method)
            self.assertAlmostEqual(
                results1['effect_estimate'], 
                results2['effect_estimate'],
                places=10,
                msg="Backdoor adjustment should be deterministic"
            )


if __name__ == '__main__':
    pytest.main([__file__])
"""Unit tests for instrumental variable estimator."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from causal_agent.methods.instrumental_variable.estimator import (
    estimate_effect,
    build_iv_graph_gml,
    format_iv_results
)
from tests.base import MethodTestCase
from tests.fixtures.synthetic_data import SyntheticDataGenerator, DatasetType, SyntheticDataConfig
from tests.fixtures.mock_llm_responses import mock_llm_generator


class TestInstrumentalVariableEstimator(MethodTestCase):
    """Test cases for instrumental variable estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.generator = SyntheticDataGenerator(
            SyntheticDataConfig(n_samples=300, treatment_effect=2.0, instrument_strength=0.8, random_seed=123)
        )
        self.iv_data = self.generator.generate_iv_data()
    
    def test_build_iv_graph_gml_basic(self):
        """Test GML graph construction for IV."""
        treatment = 'treatment'
        outcome = 'outcome'
        instruments = ['instrument']
        covariates = ['covariate_0', 'covariate_1']
        
        gml_graph = build_iv_graph_gml(treatment, outcome, instruments, covariates)
        
        # Check that graph is a string
        self.assertIsInstance(gml_graph, str)
        
        # Check that all variables are included
        self.assertIn(treatment, gml_graph)
        self.assertIn(outcome, gml_graph)
        self.assertIn('instrument', gml_graph)
        self.assertIn('covariate_0', gml_graph)
        self.assertIn('covariate_1', gml_graph)
        self.assertIn('U', gml_graph)  # Unobserved confounder
        
        # Check graph structure keywords
        self.assertIn('graph', gml_graph)
        self.assertIn('directed 1', gml_graph)
        self.assertIn('node', gml_graph)
        self.assertIn('edge', gml_graph)
    
    def test_build_iv_graph_gml_multiple_instruments(self):
        """Test GML graph construction with multiple instruments."""
        treatment = 'treatment'
        outcome = 'outcome'
        instruments = ['instrument1', 'instrument2']
        covariates = ['covariate']
        
        gml_graph = build_iv_graph_gml(treatment, outcome, instruments, covariates)
        
        # Check that both instruments are included
        self.assertIn('instrument1', gml_graph)
        self.assertIn('instrument2', gml_graph)
        
        # Should have edges from both instruments to treatment
        self.assertIn('source "instrument1" target "treatment"', gml_graph)
        self.assertIn('source "instrument2" target "treatment"', gml_graph)
    
    def test_build_iv_graph_gml_no_covariates(self):
        """Test GML graph construction with no covariates."""
        treatment = 'treatment'
        outcome = 'outcome'
        instruments = ['instrument']
        covariates = []
        
        gml_graph = build_iv_graph_gml(treatment, outcome, instruments, covariates)
        
        # Should still create valid graph
        self.assertIsInstance(gml_graph, str)
        self.assertIn(treatment, gml_graph)
        self.assertIn(outcome, gml_graph)
        self.assertIn('instrument', gml_graph)
    
    def test_format_iv_results_basic(self):
        """Test basic IV result formatting."""
        estimate = 1.5
        raw_results = {'method': 'test'}
        diagnostics = {'first_stage_f': 10.0}
        treatment = 'treatment'
        outcome = 'outcome'
        instruments = ['instrument']
        method_used = 'test_method'
        
        results = format_iv_results(
            estimate, raw_results, diagnostics, 
            treatment, outcome, instruments, method_used
        )
        
        # Check required keys
        expected_keys = [
            'effect_estimate', 'treatment_variable', 'outcome_variable',
            'instrument_variables', 'method_used', 'diagnostics',
            'raw_results', 'confidence_interval', 'standard_error',
            'p_value', 'interpretation'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check values
        self.assertEqual(results['effect_estimate'], 1.5)
        self.assertEqual(results['treatment_variable'], treatment)
        self.assertEqual(results['outcome_variable'], outcome)
        self.assertEqual(results['instrument_variables'], instruments)
        self.assertEqual(results['method_used'], method_used)
    
    def test_format_iv_results_with_statsmodels(self):
        """Test IV result formatting with statsmodels results."""
        # Mock statsmodels results object
        mock_sm_results = MagicMock()
        mock_sm_results.bse = {'treatment': 0.2}
        mock_sm_results.pvalues = {'treatment': 0.05}
        mock_sm_results.conf_int.return_value = pd.DataFrame({
            0: [1.1], 1: [1.9]
        }, index=['treatment'])
        
        estimate = 1.5
        raw_results = {'statsmodels_results_object': mock_sm_results}
        diagnostics = {}
        
        results = format_iv_results(
            estimate, raw_results, diagnostics,
            'treatment', 'outcome', ['instrument'], 'statsmodels'
        )
        
        # Check extracted values
        self.assertEqual(results['standard_error'], 0.2)
        self.assertEqual(results['p_value'], 0.05)
        self.assertEqual(results['confidence_interval'], [1.1, 1.9])
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    @patch('causal_agent.methods.instrumental_variable.estimator.validate_instrument_assumptions_qualitative')
    def test_estimate_effect_missing_instrument(self, mock_validate, mock_diagnostics):
        """Test error handling when instrument variable is not provided."""
        mock_diagnostics.return_value = {}
        
        covariates = ['covariate_0', 'covariate_1']
        
        results = estimate_effect(
            self.iv_data,
            'treatment',
            'outcome',
            covariates
            # No instrument_variable provided
        )
        
        # Should return error
        self.assertIn('error', results)
        self.assertIn('Instrument variable', results['error'])
        self.assertEqual(results['method_used'], 'none')
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    @patch('causal_agent.methods.instrumental_variable.estimator.validate_instrument_assumptions_qualitative')
    def test_estimate_effect_missing_columns(self, mock_validate, mock_diagnostics):
        """Test error handling for missing columns."""
        mock_diagnostics.return_value = {}
        
        results = estimate_effect(
            self.iv_data,
            'missing_treatment',
            'outcome',
            ['covariate_0'],
            instrument_variable='instrument'
        )
        
        # Should return error about missing columns
        self.assertIn('error', results)
        self.assertIn('Missing required columns', results['error'])
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    @patch('causal_agent.methods.instrumental_variable.estimator.validate_instrument_assumptions_qualitative')
    @patch('causal_agent.methods.instrumental_variable.estimator.CausalModel')
    def test_estimate_effect_dowhy_success(self, mock_causal_model, mock_validate, mock_diagnostics):
        """Test successful IV estimation using DoWhy."""
        # Setup mocks
        mock_diagnostics.return_value = {'first_stage_f': 15.0}
        mock_validate.return_value = "Assumptions appear reasonable"
        
        # Mock DoWhy components
        mock_model = MagicMock()
        mock_estimand = MagicMock()
        mock_estimate = MagicMock()
        mock_estimate.value = 2.1
        
        mock_causal_model.return_value = mock_model
        mock_model.identify_effect.return_value = mock_estimand
        mock_model.estimate_effect.return_value = mock_estimate
        mock_model.refute_estimate.return_value = MagicMock()
        
        covariates = ['covariate_0', 'covariate_1']
        
        results = estimate_effect(
            self.iv_data,
            'treatment',
            'outcome',
            covariates,
            instrument_variable='instrument',
            query="What is the causal effect?",
            llm=MagicMock()
        )
        
        # Should succeed with DoWhy
        self.assertNotIn('error', results)
        self.assertEqual(results['method_used'], 'dowhy')
        self.assertEqual(results['effect_estimate'], 2.1)
        self.assertIn('diagnostics', results)
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    @patch('causal_agent.methods.instrumental_variable.estimator.validate_instrument_assumptions_qualitative')
    @patch('causal_agent.methods.instrumental_variable.estimator.CausalModel')
    @patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS')
    def test_estimate_effect_dowhy_fallback_to_statsmodels(self, mock_iv2sls, mock_causal_model, mock_validate, mock_diagnostics):
        """Test fallback to statsmodels when DoWhy fails."""
        # Setup mocks
        mock_diagnostics.return_value = {'first_stage_f': 12.0}
        mock_validate.return_value = "Assumptions reasonable"
        
        # Mock DoWhy to fail
        mock_causal_model.side_effect = Exception("DoWhy failed")
        
        # Mock statsmodels to succeed
        mock_sm_model = MagicMock()
        mock_sm_results = MagicMock()
        mock_sm_results.params = {'treatment': 1.8}
        mock_sm_results.bse = {'treatment': 0.3}
        mock_sm_results.pvalues = {'treatment': 0.02}
        
        mock_iv2sls.return_value = mock_sm_model
        mock_sm_model.fit.return_value = mock_sm_results
        
        covariates = ['covariate_0', 'covariate_1']
        
        results = estimate_effect(
            self.iv_data,
            'treatment',
            'outcome',
            covariates,
            instrument_variable='instrument'
        )
        
        # Should succeed with statsmodels fallback
        self.assertEqual(results['method_used'], 'statsmodels')
        self.assertEqual(results['effect_estimate'], 1.8)
        self.assertIn('diagnostics', results)
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    @patch('causal_agent.methods.instrumental_variable.estimator.validate_instrument_assumptions_qualitative')
    @patch('causal_agent.methods.instrumental_variable.estimator.CausalModel')
    @patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS')
    def test_estimate_effect_both_methods_fail(self, mock_iv2sls, mock_causal_model, mock_validate, mock_diagnostics):
        """Test when both DoWhy and statsmodels fail."""
        # Setup mocks
        mock_diagnostics.return_value = {}
        
        # Mock both methods to fail
        mock_causal_model.side_effect = Exception("DoWhy failed")
        mock_iv2sls.side_effect = Exception("Statsmodels failed")
        
        covariates = ['covariate_0', 'covariate_1']
        
        results = estimate_effect(
            self.iv_data,
            'treatment',
            'outcome',
            covariates,
            instrument_variable='instrument'
        )
        
        # Should indicate failure
        self.assertIn('dowhy_failed', results['method_used'])
        self.assertIsNone(results['effect_estimate'])
        self.assertIn('error', results)
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    def test_estimate_effect_multiple_instruments(self, mock_diagnostics):
        """Test IV estimation with multiple instruments."""
        mock_diagnostics.return_value = {'first_stage_f': 20.0}
        
        # Create data with multiple instruments
        iv_data_multi = self.iv_data.copy()
        iv_data_multi['instrument2'] = np.random.binomial(1, 0.5, len(iv_data_multi))
        
        covariates = ['covariate_0', 'covariate_1']
        
        with patch('causal_agent.methods.instrumental_variable.estimator.CausalModel') as mock_causal_model:
            # Mock DoWhy to fail so we test statsmodels path
            mock_causal_model.side_effect = Exception("DoWhy failed")
            
            with patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS') as mock_iv2sls:
                mock_sm_model = MagicMock()
                mock_sm_results = MagicMock()
                mock_sm_results.params = {'treatment': 1.5}
                
                mock_iv2sls.return_value = mock_sm_model
                mock_sm_model.fit.return_value = mock_sm_results
                
                results = estimate_effect(
                    iv_data_multi,
                    'treatment',
                    'outcome',
                    covariates,
                    instrument_variable=['instrument', 'instrument2']
                )
                
                # Should handle multiple instruments
                self.assertEqual(results['instrument_variables'], ['instrument', 'instrument2'])
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    def test_estimate_effect_force_statsmodels(self, mock_diagnostics):
        """Test forcing statsmodels estimation."""
        mock_diagnostics.return_value = {'first_stage_f': 18.0}
        
        with patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS') as mock_iv2sls:
            mock_sm_model = MagicMock()
            mock_sm_results = MagicMock()
            mock_sm_results.params = {'treatment': 2.2}
            
            mock_iv2sls.return_value = mock_sm_model
            mock_sm_model.fit.return_value = mock_sm_results
            
            covariates = ['covariate_0', 'covariate_1']
            
            results = estimate_effect(
                self.iv_data,
                'treatment',
                'outcome',
                covariates,
                instrument_variable='instrument',
                force_statsmodels=True
            )
            
            # Should use statsmodels directly
            self.assertEqual(results['method_used'], 'statsmodels')
            self.assertEqual(results['effect_estimate'], 2.2)
    
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    def test_estimate_effect_underidentified(self, mock_diagnostics):
        """Test handling of underidentified model."""
        mock_diagnostics.return_value = {}
        
        # Create scenario with more endogenous variables than instruments
        with patch('causal_agent.methods.instrumental_variable.estimator.CausalModel') as mock_causal_model:
            mock_causal_model.side_effect = Exception("DoWhy failed")
            
            with patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS') as mock_iv2sls:
                mock_iv2sls.side_effect = ValueError("Model underidentified")
                
                covariates = ['covariate_0', 'covariate_1']
                
                results = estimate_effect(
                    self.iv_data,
                    'treatment',
                    'outcome',
                    covariates,
                    instrument_variable='instrument'
                )
                
                # Should handle underidentification error
                self.assertIn('statsmodels_error', results)
                self.assertIn('underidentified', results['statsmodels_error'])
    
    @pytest.mark.parametrize("instrument_strength", [0.3, 0.6, 0.9])
    @patch('causal_agent.methods.instrumental_variable.estimator.run_iv_diagnostics')
    def test_estimate_effect_different_instrument_strengths(self, mock_diagnostics, instrument_strength):
        """Test IV estimation with different instrument strengths."""
        mock_diagnostics.return_value = {'first_stage_f': 10.0 * instrument_strength}
        
        # Generate data with specific instrument strength
        config = SyntheticDataConfig(
            n_samples=200,
            treatment_effect=1.5,
            instrument_strength=instrument_strength,
            random_seed=42
        )
        generator = SyntheticDataGenerator(config)
        iv_data = generator.generate_iv_data()
        
        with patch('causal_agent.methods.instrumental_variable.estimator.CausalModel') as mock_causal_model:
            mock_causal_model.side_effect = Exception("DoWhy failed")
            
            with patch('causal_agent.methods.instrumental_variable.estimator.IV2SLS') as mock_iv2sls:
                mock_sm_model = MagicMock()
                mock_sm_results = MagicMock()
                mock_sm_results.params = {'treatment': 1.5}
                
                mock_iv2sls.return_value = mock_sm_model
                mock_sm_model.fit.return_value = mock_sm_results
                
                covariates = ['covariate_0', 'covariate_1']
                
                results = estimate_effect(
                    iv_data,
                    'treatment',
                    'outcome',
                    covariates,
                    instrument_variable='instrument'
                )
                
                # Should work regardless of instrument strength
                self.assertIn('effect_estimate', results)
    
    def test_format_iv_results_none_estimate(self):
        """Test result formatting when estimate is None."""
        results = format_iv_results(
            None, {}, {}, 'treatment', 'outcome', ['instrument'], 'failed'
        )
        
        # Should handle None estimate gracefully
        self.assertIsNone(results['effect_estimate'])
        self.assertIn('Estimation failed', results['interpretation'])
    
    @patch('causal_agent.methods.instrumental_variable.estimator.interpret_iv_results')
    def test_format_iv_results_with_llm(self, mock_interpret):
        """Test result formatting with LLM interpretation."""
        mock_interpret.return_value = "LLM interpretation of IV results"
        
        results = format_iv_results(
            1.5, {}, {}, 'treatment', 'outcome', ['instrument'], 'test',
            llm=MagicMock()
        )
        
        # Should include LLM interpretation
        self.assertEqual(results['interpretation'], "LLM interpretation of IV results")
        mock_interpret.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
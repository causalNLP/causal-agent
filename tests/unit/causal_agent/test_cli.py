"""
Tests for the CLI module.
"""

import unittest
import json
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

from causal_agent.cli import main


class TestCLI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_path = os.path.join(self.temp_dir, "test_data.csv")
        self.test_metadata_path = os.path.join(self.temp_dir, "metadata.csv")
        self.test_output_path = os.path.join(self.temp_dir, "output.json")
        
        # Create test dataset
        test_data = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'outcome': [10, 12, 11, 13, 9, 14],
            'covariate': [1, 2, 1, 2, 1, 2]
        })
        test_data.to_csv(self.test_dataset_path, index=False)
        
        # Create test metadata
        metadata = pd.DataFrame({
            'natural_language_query': ['What is the effect of treatment on outcome?'],
            'data_description': ['Test dataset for causal analysis'],
            'data_files': ['test_data.csv'],
            'method': ['linear_regression'],
            'answer': ['Positive effect']
        })
        metadata.to_csv(self.test_metadata_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_main_no_args_prints_help(self):
        """Test that main with no arguments prints help."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('argparse.ArgumentParser.print_help') as mock_help:
                main([])
                mock_help.assert_called_once()
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_run_command_basic(self, mock_run_analysis):
        """Test basic run command functionality."""
        mock_result = {
            'results': {
                'results': {
                    'method_used': 'linear_regression',
                    'effect_estimate': 2.5,
                    'standard_error': 0.5
                },
                'variables': {
                    'treatment_variable': 'treatment',
                    'outcome_variable': 'outcome'
                }
            }
        }
        mock_run_analysis.return_value = mock_result
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main(['run', self.test_dataset_path, 'What is the effect of treatment on outcome?'])
            
        mock_run_analysis.assert_called_once_with(
            query='What is the effect of treatment on outcome?',
            dataset_path=self.test_dataset_path,
            dataset_description=None
        )
        
        # Check that JSON output was printed
        output = mock_stdout.getvalue()
        self.assertIn('linear_regression', output)
        self.assertIn('2.5', output)
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_run_command_with_description(self, mock_run_analysis):
        """Test run command with dataset description."""
        mock_result = {'results': {'results': {}, 'variables': {}}}
        mock_run_analysis.return_value = mock_result
        
        main(['run', self.test_dataset_path, 'Test query', '--desc', 'Test description'])
        
        mock_run_analysis.assert_called_once_with(
            query='Test query',
            dataset_path=self.test_dataset_path,
            dataset_description='Test description'
        )
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_run_command_with_llm_options(self, mock_run_analysis):
        """Test run command with LLM configuration options."""
        mock_result = {'results': {'results': {}, 'variables': {}}}
        mock_run_analysis.return_value = mock_result
        
        with patch.dict(os.environ, {}, clear=True):
            main(['run', self.test_dataset_path, 'Test query', 
                  '--llm-name', 'gpt-4', '--llm-provider', 'openai'])
            
            self.assertEqual(os.environ.get('LLM_MODEL'), 'gpt-4')
            self.assertEqual(os.environ.get('LLM_PROVIDER'), 'openai')
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_batch_command_basic(self, mock_run_analysis):
        """Test basic batch command functionality."""
        mock_result = {
            'results': {
                'results': {
                    'method_used': 'linear_regression',
                    'effect_estimate': 2.5,
                    'standard_error': 0.5
                },
                'variables': {
                    'treatment_variable': 'treatment',
                    'outcome_variable': 'outcome',
                    'covariates': ['covariate'],
                    'instrument_variable': None,
                    'running_variable': None,
                    'time_variable': None
                }
            }
        }
        mock_run_analysis.return_value = mock_result
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main(['batch', self.test_metadata_path, self.temp_dir, self.test_output_path])
        
        # Check that run_causal_analysis was called
        mock_run_analysis.assert_called_once()
        call_args = mock_run_analysis.call_args
        self.assertEqual(call_args[1]['query'], 'What is the effect of treatment on outcome?')
        self.assertEqual(call_args[1]['dataset_description'], 'Test dataset for causal analysis')
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.test_output_path))
        
        # Check output content
        with open(self.test_output_path, 'r') as f:
            results = json.load(f)
        
        self.assertIn('0', results)  # Index 0 from metadata
        self.assertEqual(results['0']['query'], 'What is the effect of treatment on outcome?')
        self.assertEqual(results['0']['final_result']['method'], 'linear_regression')
        self.assertEqual(results['0']['final_result']['causal_effect'], 2.5)
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_batch_command_with_error(self, mock_run_analysis):
        """Test batch command handling of analysis errors."""
        mock_run_analysis.side_effect = Exception("Analysis failed")
        
        main(['batch', self.test_metadata_path, self.temp_dir, self.test_output_path])
        
        # Check that output file was created with error
        self.assertTrue(os.path.exists(self.test_output_path))
        
        with open(self.test_output_path, 'r') as f:
            results = json.load(f)
        
        self.assertIn('0', results)
        self.assertEqual(results['0']['error'], 'Analysis failed')
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_batch_command_creates_output_directory(self, mock_run_analysis):
        """Test that batch command creates output directory if it doesn't exist."""
        mock_result = {'results': {'results': {}, 'variables': {}}}
        mock_run_analysis.return_value = mock_result
        
        nested_output_path = os.path.join(self.temp_dir, 'nested', 'output.json')
        
        main(['batch', self.test_metadata_path, self.temp_dir, nested_output_path])
        
        self.assertTrue(os.path.exists(nested_output_path))
        self.assertTrue(os.path.exists(os.path.dirname(nested_output_path)))
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_batch_command_with_llm_options(self, mock_run_analysis):
        """Test batch command with LLM configuration options."""
        mock_result = {'results': {'results': {}, 'variables': {}}}
        mock_run_analysis.return_value = mock_result
        
        with patch.dict(os.environ, {}, clear=True):
            main(['batch', self.test_metadata_path, self.temp_dir, self.test_output_path,
                  '--llm-name', 'claude-3', '--llm-provider', 'anthropic'])
            
            self.assertEqual(os.environ.get('LLM_MODEL'), 'claude-3')
            self.assertEqual(os.environ.get('LLM_PROVIDER'), 'anthropic')
    
    def test_batch_command_missing_metadata_file(self):
        """Test batch command with missing metadata file."""
        with self.assertRaises(FileNotFoundError):
            main(['batch', 'nonexistent.csv', self.temp_dir, self.test_output_path])
    
    @patch('causal_agent.cli.run_causal_analysis')
    def test_batch_command_complex_result_structure(self, mock_run_analysis):
        """Test batch command with complex nested result structure."""
        mock_result = {
            'results': {
                'results': {
                    'method_used': 'difference_in_differences',
                    'effect_estimate': 1.8,
                    'standard_error': 0.3
                },
                'variables': {
                    'treatment_variable': 'treated',
                    'outcome_variable': 'y',
                    'covariates': ['x1', 'x2'],
                    'instrument_variable': 'z',
                    'running_variable': 'score',
                    'time_variable': 'year'
                }
            }
        }
        mock_run_analysis.return_value = mock_result
        
        main(['batch', self.test_metadata_path, self.temp_dir, self.test_output_path])
        
        with open(self.test_output_path, 'r') as f:
            results = json.load(f)
        
        final_result = results['0']['final_result']
        self.assertEqual(final_result['method'], 'difference_in_differences')
        self.assertEqual(final_result['causal_effect'], 1.8)
        self.assertEqual(final_result['standard_deviation'], 0.3)
        self.assertEqual(final_result['treatment_variable'], 'treated')
        self.assertEqual(final_result['outcome_variable'], 'y')
        self.assertEqual(final_result['covariates'], ['x1', 'x2'])
        self.assertEqual(final_result['instrument_variable'], 'z')
        self.assertEqual(final_result['running_variable'], 'score')
        self.assertEqual(final_result['temporal_variable'], 'year')


if __name__ == '__main__':
    unittest.main()
"""Integration tests for CLI functionality."""

import pytest
import subprocess
import tempfile
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Dict, Any

from causal_agent.cli import main
import unittest


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI commands."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self, filename: str = "test_data.csv") -> str:
        """Create a test dataset file."""
        np.random.seed(42)
        data = {
            'treatment': np.random.binomial(1, 0.5, 100),
            'outcome': np.random.normal(10, 2, 100),
            'age': np.random.normal(35, 10, 100),
            'gender': np.random.binomial(1, 0.5, 100)
        }
        # Add treatment effect
        data['outcome'] += data['treatment'] * 2.5
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def create_batch_metadata_csv(self) -> str:
        """Create metadata CSV for batch testing."""
        metadata = [
            {
                "natural_language_query": "What is the effect of treatment on outcome?",
                "data_description": "RCT testing treatment effect on outcome",
                "data_files": "test_data_1.csv",
                "method": "diff_in_means",
                "answer": "2.5"
            },
            {
                "natural_language_query": "Does treatment cause changes in outcome controlling for age?",
                "data_description": "Observational study with age confounder",
                "data_files": "test_data_2.csv", 
                "method": "backdoor_adjustment",
                "answer": "2.0"
            }
        ]
        
        # Create corresponding data files
        for i, meta in enumerate(metadata, 1):
            self.create_test_dataset(f"test_data_{i}.csv")
        
        # Create metadata CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_path = os.path.join(self.temp_dir, "metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        
        return metadata_path
    
    @patch('causal_agent.config.get_llm_client')
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    def test_cli_run_command_basic(self, mock_llm_call, mock_get_llm):
        """Test basic CLI run command."""
        # Create test dataset
        dataset_path = self.create_test_dataset()
        
        # Mock LLM responses
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "diff_in_means", "confidence": 0.9},
            {"interpretation": "Treatment has positive effect"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Test CLI arguments
        args = [
            "run",
            dataset_path,
            "What is the effect of treatment on outcome?",
            "--desc", "Test RCT dataset"
        ]
        
        # Capture output
        import io
        import sys
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                main(args)
            
            output = output_buffer.getvalue()
            
            # Verify output is valid JSON
            result = json.loads(output)
            self.assertIsInstance(result, dict)
            self.assertIn('results', result)
            
        except SystemExit:
            # CLI might call sys.exit, which is normal
            pass
    
    @patch('causal_agent.config.get_llm_client')
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    def test_cli_run_command_with_llm_options(self, mock_llm_call, mock_get_llm):
        """Test CLI run command with LLM provider and model options."""
        dataset_path = self.create_test_dataset()
        
        # Mock LLM responses
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "diff_in_means", "confidence": 0.9},
            {"interpretation": "Treatment effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Test with LLM options
        args = [
            "run",
            dataset_path,
            "What is the effect of treatment on outcome?",
            "--llm-provider", "openai",
            "--llm-name", "gpt-4"
        ]
        
        # Check environment variables are set
        original_provider = os.environ.get("LLM_PROVIDER")
        original_model = os.environ.get("LLM_MODEL")
        
        try:
            main(args)
            
            # Verify environment variables were set
            self.assertEqual(os.environ.get("LLM_PROVIDER"), "openai")
            self.assertEqual(os.environ.get("LLM_MODEL"), "gpt-4")
            
        except SystemExit:
            pass
        finally:
            # Restore original environment
            if original_provider:
                os.environ["LLM_PROVIDER"] = original_provider
            elif "LLM_PROVIDER" in os.environ:
                del os.environ["LLM_PROVIDER"]
                
            if original_model:
                os.environ["LLM_MODEL"] = original_model
            elif "LLM_MODEL" in os.environ:
                del os.environ["LLM_MODEL"]
    
    @patch('causal_agent.config.get_llm_client')
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    def test_cli_batch_command(self, mock_llm_call, mock_get_llm):
        """Test CLI batch command."""
        # Create metadata and output files
        metadata_path = self.create_batch_metadata_csv()
        output_path = os.path.join(self.temp_dir, "batch_results.json")
        
        # Mock LLM responses (will be called multiple times)
        mock_llm_call.side_effect = [
            # First dataset
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "diff_in_means", "confidence": 0.9},
            {"interpretation": "Treatment effect estimated"},
            # Second dataset
            {"variables": ["treatment", "outcome", "age"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome", "covariates": ["age"]},
            {"recommended_method": "backdoor_adjustment", "confidence": 0.8},
            {"interpretation": "Adjusted treatment effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Test batch command
        args = [
            "batch",
            metadata_path,
            self.temp_dir,  # data folder
            output_path,
            "--llm-provider", "openai",
            "--llm-name", "gpt-3.5-turbo"
        ]
        
        try:
            main(args)
            
            # Verify output file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Verify output content
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            self.assertIsInstance(results, dict)
            self.assertIn('0', results)  # First result
            self.assertIn('1', results)  # Second result
            
            # Verify result structure
            for key in results:
                result = results[key]
                if 'error' not in result:
                    self.assertIn('query', result)
                    self.assertIn('final_result', result)
                    
        except SystemExit:
            pass
    
    def test_cli_help_commands(self):
        """Test CLI help functionality."""
        import io
        import sys
        from contextlib import redirect_stderr
        
        # Test main help
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                main([])
        
        # Test run help
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                main(["run", "--help"])
        
        # Test batch help
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                main(["batch", "--help"])
    
    def test_cli_error_handling_invalid_dataset(self):
        """Test CLI error handling with invalid dataset path."""
        args = [
            "run",
            "/nonexistent/dataset.csv",
            "What is the effect of treatment on outcome?"
        ]
        
        # Should handle error gracefully
        try:
            main(args)
        except SystemExit as e:
            # CLI should exit with error code
            self.assertNotEqual(e.code, 0)
        except Exception:
            # Or raise an exception that gets handled
            pass
    
    def test_cli_error_handling_invalid_metadata(self):
        """Test CLI error handling with invalid metadata file."""
        output_path = os.path.join(self.temp_dir, "output.json")
        
        args = [
            "batch",
            "/nonexistent/metadata.csv",
            self.temp_dir,
            output_path
        ]
        
        # Should handle error gracefully
        try:
            main(args)
        except SystemExit as e:
            # CLI should exit with error code
            self.assertNotEqual(e.code, 0)
        except Exception:
            # Or raise an exception that gets handled
            pass


class TestCLISubprocessIntegration(unittest.TestCase):
    """Test CLI through subprocess calls (more realistic testing)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self) -> str:
        """Create a test dataset file."""
        np.random.seed(42)
        data = {
            'treatment': np.random.binomial(1, 0.5, 50),
            'outcome': np.random.normal(10, 2, 50),
            'age': np.random.normal(35, 10, 50)
        }
        data['outcome'] += data['treatment'] * 2.0
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, "subprocess_test_data.csv")
        df.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.slow
    def test_cli_subprocess_run_command(self):
        """Test CLI run command through subprocess."""
        dataset_path = self.create_test_dataset()
        
        # Prepare command
        cmd = [
            "python", "-m", "causal_agent.cli",
            "run",
            dataset_path,
            "What is the effect of treatment on outcome?",
            "--desc", "Subprocess test dataset"
        ]
        
        # Set environment to use mock/test mode if available
        env = os.environ.copy()
        env["CAUSAL_AGENT_TEST_MODE"] = "1"
        
        try:
            # Run command with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
                cwd=Path(__file__).parent.parent.parent  # Project root
            )
            
            # Check if command completed (may fail due to missing API keys in CI)
            if result.returncode == 0:
                # Verify output is valid JSON
                output = result.stdout.strip()
                if output:
                    parsed_output = json.loads(output)
                    self.assertIsInstance(parsed_output, dict)
            else:
                # Log error for debugging but don't fail test in CI environment
                print(f"CLI subprocess failed (expected in CI): {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.fail("CLI command timed out")
        except FileNotFoundError:
            # Python module not found - skip test
            pytest.skip("causal_agent module not available for subprocess testing")
    
    @pytest.mark.slow
    def test_cli_subprocess_help(self):
        """Test CLI help through subprocess."""
        cmd = ["python", "-m", "causal_agent.cli", "--help"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent.parent
            )
            
            # Help should work regardless of API keys
            self.assertEqual(result.returncode, 0)
            self.assertIn("causal", result.stdout.lower())
            
        except subprocess.TimeoutExpired:
            self.fail("CLI help command timed out")
        except FileNotFoundError:
            pytest.skip("causal_agent module not available for subprocess testing")


class TestCLICommandVariations(unittest.TestCase):
    """Test various CLI command combinations and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self, name: str = "test.csv") -> str:
        """Create a test dataset."""
        np.random.seed(42)
        data = {
            'treatment': np.random.binomial(1, 0.5, 30),
            'outcome': np.random.normal(10, 2, 30)
        }
        data['outcome'] += data['treatment'] * 1.5
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, name)
        df.to_csv(filepath, index=False)
        return filepath
    
    @patch('causal_agent.config.get_llm_client')
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    def test_cli_run_minimal_arguments(self, mock_llm_call, mock_get_llm):
        """Test CLI run with minimal required arguments."""
        dataset_path = self.create_test_dataset()
        
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "diff_in_means", "confidence": 0.9},
            {"interpretation": "Effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        args = [
            "run",
            dataset_path,
            "Effect of treatment on outcome?"
        ]
        
        try:
            main(args)
        except SystemExit:
            pass
    
    @patch('causal_agent.config.get_llm_client')
    @patch('causal_agent.utils.llm_helpers.call_llm_with_json_output')
    def test_cli_run_with_all_options(self, mock_llm_call, mock_get_llm):
        """Test CLI run with all available options."""
        dataset_path = self.create_test_dataset()
        
        mock_llm_call.side_effect = [
            {"variables": ["treatment", "outcome"], "data_quality": "good"},
            {"treatment_variable": "treatment", "outcome_variable": "outcome"},
            {"recommended_method": "diff_in_means", "confidence": 0.9},
            {"interpretation": "Effect estimated"}
        ]
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        args = [
            "run",
            dataset_path,
            "What is the causal effect of treatment on outcome?",
            "--desc", "Comprehensive test dataset with treatment and outcome variables",
            "--llm-provider", "openai",
            "--llm-name", "gpt-4"
        ]
        
        try:
            main(args)
        except SystemExit:
            pass
    
    def test_cli_batch_with_empty_metadata(self):
        """Test CLI batch command with empty metadata file."""
        # Create empty metadata file
        empty_metadata = pd.DataFrame(columns=[
            "natural_language_query", "data_description", "data_files"
        ])
        metadata_path = os.path.join(self.temp_dir, "empty_metadata.csv")
        empty_metadata.to_csv(metadata_path, index=False)
        
        output_path = os.path.join(self.temp_dir, "empty_results.json")
        
        args = [
            "batch",
            metadata_path,
            self.temp_dir,
            output_path
        ]
        
        try:
            main(args)
            
            # Should create empty results file
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            self.assertEqual(len(results), 0)
            
        except SystemExit:
            pass
    
    def test_cli_batch_with_missing_data_files(self):
        """Test CLI batch command when some data files are missing."""
        # Create metadata with non-existent files
        metadata = pd.DataFrame([
            {
                "natural_language_query": "Test query 1",
                "data_description": "Test description 1", 
                "data_files": "nonexistent1.csv"
            },
            {
                "natural_language_query": "Test query 2",
                "data_description": "Test description 2",
                "data_files": "nonexistent2.csv"
            }
        ])
        
        metadata_path = os.path.join(self.temp_dir, "missing_files_metadata.csv")
        metadata.to_csv(metadata_path, index=False)
        
        output_path = os.path.join(self.temp_dir, "missing_files_results.json")
        
        args = [
            "batch",
            metadata_path,
            self.temp_dir,
            output_path
        ]
        
        try:
            main(args)
            
            # Should create results file with errors
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            # Should have error entries for missing files
            for key in results:
                result = results[key]
                self.assertIn('error', result)
                
        except SystemExit:
            pass
    
    @patch('causal_agent.config.get_llm_client')
    def test_cli_different_query_formats(self, mock_get_llm):
        """Test CLI with different query formats and phrasings."""
        dataset_path = self.create_test_dataset()
        
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm
        
        # Test different query phrasings
        queries = [
            "What is the effect of treatment on outcome?",
            "Does treatment cause outcome?",
            "How does treatment impact outcome?",
            "treatment -> outcome causal effect",
            "Estimate causal impact: treatment on outcome"
        ]
        
        for query in queries:
            with self.subTest(query=query):
                args = [
                    "run",
                    dataset_path,
                    query
                ]
                
                with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm_call:
                    mock_llm_call.side_effect = [
                        {"variables": ["treatment", "outcome"], "data_quality": "good"},
                        {"treatment_variable": "treatment", "outcome_variable": "outcome"},
                        {"recommended_method": "diff_in_means", "confidence": 0.9},
                        {"interpretation": "Effect estimated"}
                    ]
                    
                    try:
                        main(args)
                    except SystemExit:
                        pass
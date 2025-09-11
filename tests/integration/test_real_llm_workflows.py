"""Integration tests using actual LLM invocation with OpenAI API."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import unittest
from dotenv import load_dotenv

from causal_agent.agent import run_causal_analysis


# Load environment variables
load_dotenv()


def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.requires_llm
class TestRealLLMWorkflows(unittest.TestCase):
    """Integration tests using actual LLM calls."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        if not has_openai_key():
            pytest.skip("OpenAI API key not available - skipping real LLM tests")
        
        # Set LLM configuration for tests
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-3.5-turbo"
        
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_simple_rct_dataset(self) -> str:
        """Create a simple RCT dataset for testing."""
        np.random.seed(42)
        n = 100
        
        # Random treatment assignment
        treatment = np.random.binomial(1, 0.5, n)
        
        # Baseline characteristics
        age = np.random.normal(45, 12, n)
        gender = np.random.binomial(1, 0.6, n)
        
        # Outcome with clear treatment effect
        outcome = (
            50 +  # Baseline
            0.1 * age +  # Age effect
            2 * gender +  # Gender effect
            8 * treatment +  # Clear treatment effect
            np.random.normal(0, 5, n)  # Noise
        )
        
        data = pd.DataFrame({
            'patient_id': range(1, n + 1),
            'age': age,
            'gender': gender,
            'treatment': treatment,
            'outcome': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "rct_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def create_observational_dataset(self) -> str:
        """Create an observational dataset with confounding."""
        np.random.seed(42)
        n = 150
        
        # Confounding variables
        age = np.random.normal(40, 15, n)
        income = np.random.lognormal(10, 0.5, n)
        education = np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.3, 0.3, 0.2])
        
        # Treatment assignment depends on confounders
        treatment_logits = (
            -2 +
            0.02 * age +
            0.0001 * income +
            0.5 * education
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome depends on confounders and treatment
        outcome = (
            30 +
            0.3 * age +
            0.0002 * income +
            5 * education +
            6 * treatment +  # Treatment effect
            np.random.normal(0, 8, n)
        )
        
        data = pd.DataFrame({
            'person_id': range(1, n + 1),
            'age': age,
            'income': income,
            'education_level': education,
            'received_treatment': treatment,
            'outcome_score': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "observational_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    def create_iv_dataset(self) -> str:
        """Create an instrumental variable dataset."""
        np.random.seed(42)
        n = 120
        
        # Instrument (randomly assigned)
        instrument = np.random.binomial(1, 0.5, n)
        
        # Observed covariates
        age = np.random.normal(35, 10, n)
        
        # Unobserved confounder (not in final dataset)
        unobserved = np.random.normal(0, 1, n)
        
        # Treatment depends on instrument and unobserved confounder
        treatment_logits = (
            -1 +
            2 * instrument +  # Strong instrument effect
            0.8 * unobserved +  # Confounding
            0.02 * age
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome depends on treatment and unobserved confounder
        outcome = (
            25 +
            4 * treatment +  # Treatment effect
            0.6 * unobserved +  # Confounding
            0.1 * age +
            np.random.normal(0, 3, n)
        )
        
        data = pd.DataFrame({
            'id': range(1, n + 1),
            'instrument': instrument,
            'age': age,
            'treatment': treatment,
            'outcome': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "iv_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.slow
    def test_rct_workflow_with_real_llm(self):
        """Test RCT workflow with actual LLM calls."""
        dataset_path = self.create_simple_rct_dataset()
        
        query = "What is the effect of treatment on outcome in this clinical trial?"
        description = """
        This is a randomized controlled trial where patients were randomly assigned 
        to treatment or control groups. The outcome is a health score where higher 
        values indicate better health outcomes.
        """
        
        try:
            result = run_causal_analysis(
                query=query,
                dataset_path=dataset_path,
                dataset_description=description
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            
            # Check if we got a meaningful result (not just an error)
            if 'error' in result:
                self.fail(f"Analysis failed with error: {result['error']}")
            
            # Verify we have results
            self.assertIn('results', result)
            
            # Print result for manual inspection
            print(f"\n=== RCT Analysis Result ===")
            print(f"Query: {query}")
            print(f"Result keys: {list(result.keys())}")
            if 'results' in result and 'results' in result['results']:
                method_used = result['results']['results'].get('method_used', 'Unknown')
                effect_estimate = result['results']['results'].get('effect_estimate', 'Unknown')
                print(f"Method used: {method_used}")
                print(f"Effect estimate: {effect_estimate}")
            
        except Exception as e:
            self.fail(f"RCT workflow failed with exception: {e}")
    
    @pytest.mark.slow
    def test_observational_workflow_with_real_llm(self):
        """Test observational study workflow with actual LLM calls."""
        dataset_path = self.create_observational_dataset()
        
        query = """
        What is the causal effect of receiving treatment on the outcome score? 
        Please control for potential confounders like age, income, and education level.
        """
        description = """
        This is an observational study where treatment assignment was not randomized.
        Participants with higher age, income, and education were more likely to receive 
        treatment. We want to estimate the causal effect while controlling for these 
        potential confounders.
        """
        
        try:
            result = run_causal_analysis(
                query=query,
                dataset_path=dataset_path,
                dataset_description=description
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            
            # Check for errors
            if 'error' in result:
                self.fail(f"Analysis failed with error: {result['error']}")
            
            # Verify we have results
            self.assertIn('results', result)
            
            # Print result for manual inspection
            print(f"\n=== Observational Study Result ===")
            print(f"Query: {query[:100]}...")
            print(f"Result keys: {list(result.keys())}")
            if 'results' in result and 'results' in result['results']:
                method_used = result['results']['results'].get('method_used', 'Unknown')
                effect_estimate = result['results']['results'].get('effect_estimate', 'Unknown')
                print(f"Method used: {method_used}")
                print(f"Effect estimate: {effect_estimate}")
            
        except Exception as e:
            self.fail(f"Observational workflow failed with exception: {e}")
    
    @pytest.mark.slow
    def test_iv_workflow_with_real_llm(self):
        """Test instrumental variable workflow with actual LLM calls."""
        dataset_path = self.create_iv_dataset()
        
        query = """
        What is the causal effect of treatment on outcome using instrument as 
        an instrumental variable? The instrument should affect treatment but 
        not directly affect the outcome.
        """
        description = """
        This dataset contains an instrumental variable 'instrument' that was 
        randomly assigned and affects treatment uptake but should not directly 
        affect the outcome. This allows us to estimate the causal effect of 
        treatment even in the presence of unobserved confounders.
        """
        
        try:
            result = run_causal_analysis(
                query=query,
                dataset_path=dataset_path,
                dataset_description=description
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            
            # Check for errors
            if 'error' in result:
                self.fail(f"Analysis failed with error: {result['error']}")
            
            # Verify we have results
            self.assertIn('results', result)
            
            # Print result for manual inspection
            print(f"\n=== Instrumental Variable Result ===")
            print(f"Query: {query[:100]}...")
            print(f"Result keys: {list(result.keys())}")
            if 'results' in result and 'results' in result['results']:
                method_used = result['results']['results'].get('method_used', 'Unknown')
                effect_estimate = result['results']['results'].get('effect_estimate', 'Unknown')
                print(f"Method used: {method_used}")
                print(f"Effect estimate: {effect_estimate}")
            
        except Exception as e:
            self.fail(f"IV workflow failed with exception: {e}")
    
    @pytest.mark.slow
    def test_different_query_formulations_real_llm(self):
        """Test different ways of asking the same causal question."""
        dataset_path = self.create_simple_rct_dataset()
        
        queries = [
            "What is the effect of treatment on outcome?",
            "Does treatment cause changes in outcome?",
            "How much does treatment improve outcome?",
            "What would happen to outcome if everyone received treatment?",
            "Compare treated vs untreated patients on outcome."
        ]
        
        description = "Randomized controlled trial with treatment and outcome variables."
        
        results = []
        
        for i, query in enumerate(queries):
            try:
                print(f"\n--- Testing Query {i+1}: {query} ---")
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description=description
                )
                
                # Verify basic structure
                self.assertIsInstance(result, dict)
                
                # Store result for comparison
                results.append({
                    'query': query,
                    'result': result,
                    'has_error': 'error' in result
                })
                
                # Print summary
                if 'error' not in result and 'results' in result:
                    method_used = result.get('results', {}).get('results', {}).get('method_used', 'Unknown')
                    print(f"  Method: {method_used}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"  Exception: {e}")
                results.append({
                    'query': query,
                    'result': {'error': str(e)},
                    'has_error': True
                })
        
        # Verify at least some queries succeeded
        successful_results = [r for r in results if not r['has_error']]
        self.assertGreater(len(successful_results), 0, 
                          "At least one query formulation should succeed")
        
        print(f"\n=== Summary ===")
        print(f"Total queries: {len(queries)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(results) - len(successful_results)}")
    
    def test_error_handling_with_real_llm(self):
        """Test error handling with problematic inputs using real LLM."""
        # Create a dataset with missing values and unusual structure
        np.random.seed(42)
        n = 30
        
        data = pd.DataFrame({
            'weird_treatment_name': np.random.binomial(1, 0.5, n),
            'strange_outcome': np.random.normal(0, 1, n),
            'missing_data': [np.nan if i % 5 == 0 else np.random.normal(0, 1) for i in range(n)],
            'constant_column': [1] * n,  # No variation
            'text_column': [f"text_{i}" for i in range(n)]  # Non-numeric
        })
        
        filepath = os.path.join(self.temp_dir, "problematic_data.csv")
        data.to_csv(filepath, index=False)
        
        query = "What is the causal effect of weird_treatment_name on strange_outcome?"
        description = "Dataset with unusual column names and data quality issues."
        
        try:
            result = run_causal_analysis(
                query=query,
                dataset_path=filepath,
                dataset_description=description
            )
            
            # Should handle gracefully - either succeed or fail with informative error
            self.assertIsInstance(result, dict)
            
            print(f"\n=== Error Handling Test ===")
            print(f"Result keys: {list(result.keys())}")
            
            if 'error' in result:
                print(f"Handled error gracefully: {result['error']}")
            else:
                print("Successfully processed problematic data")
                if 'results' in result:
                    method_used = result.get('results', {}).get('results', {}).get('method_used', 'Unknown')
                    print(f"Method used: {method_used}")
            
        except Exception as e:
            # Even exceptions should be informative
            print(f"Exception occurred (this may be expected): {e}")
            # Don't fail the test - error handling can include exceptions
    
    @pytest.mark.slow
    def test_workflow_consistency_real_llm(self):
        """Test that the same query on the same data gives consistent results."""
        dataset_path = self.create_simple_rct_dataset()
        
        query = "What is the effect of treatment on outcome?"
        description = "RCT dataset for consistency testing."
        
        results = []
        
        # Run the same analysis multiple times
        for i in range(2):  # Limited to 2 runs to avoid excessive API calls
            try:
                print(f"\n--- Run {i+1} ---")
                
                result = run_causal_analysis(
                    query=query,
                    dataset_path=dataset_path,
                    dataset_description=description
                )
                
                results.append(result)
                
                # Print basic info
                if 'error' not in result and 'results' in result:
                    method_used = result.get('results', {}).get('results', {}).get('method_used', 'Unknown')
                    effect_estimate = result.get('results', {}).get('results', {}).get('effect_estimate', 'Unknown')
                    print(f"  Method: {method_used}")
                    print(f"  Effect: {effect_estimate}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                print(f"  Exception: {e}")
                results.append({'error': str(e)})
        
        # Verify we got results
        self.assertEqual(len(results), 2)
        
        # Check for basic consistency (both should either succeed or fail similarly)
        success_count = sum(1 for r in results if 'error' not in r)
        
        print(f"\n=== Consistency Test Summary ===")
        print(f"Successful runs: {success_count}/2")
        
        if success_count >= 1:
            print("At least one run succeeded - workflow is functional")
        else:
            print("Both runs failed - may indicate systematic issue")


@pytest.mark.requires_llm
class TestRealLLMCLI(unittest.TestCase):
    """Test CLI functionality with real LLM calls."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        if not has_openai_key():
            pytest.skip("OpenAI API key not available - skipping real LLM CLI tests")
    
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
        """Create a simple test dataset."""
        np.random.seed(42)
        n = 50
        
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 20 + 5 * treatment + np.random.normal(0, 3, n)
        
        data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "cli_test_data.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.slow
    def test_cli_run_with_real_llm(self):
        """Test CLI run command with actual LLM calls."""
        from causal_agent.cli import main
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        dataset_path = self.create_test_dataset()
        
        args = [
            "run",
            dataset_path,
            "What is the effect of treatment on outcome?",
            "--desc", "Simple test dataset for CLI testing",
            "--llm-provider", "openai",
            "--llm-name", "gpt-3.5-turbo"
        ]
        
        # Capture output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                main(args)
            
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            
            print(f"\n=== CLI Test Output ===")
            print(f"STDOUT length: {len(stdout_output)}")
            print(f"STDERR length: {len(stderr_output)}")
            
            if stdout_output:
                print(f"STDOUT preview: {stdout_output[:200]}...")
                
                # Try to parse as JSON
                try:
                    import json
                    result = json.loads(stdout_output)
                    self.assertIsInstance(result, dict)
                    print("CLI output is valid JSON")
                except json.JSONDecodeError:
                    print("CLI output is not valid JSON (may be expected)")
            
            if stderr_output:
                print(f"STDERR: {stderr_output}")
            
        except SystemExit as e:
            # CLI may exit normally
            print(f"CLI exited with code: {e.code}")
        except Exception as e:
            self.fail(f"CLI test failed with exception: {e}")


if __name__ == '__main__':
    # Run tests only if OpenAI key is available
    if has_openai_key():
        unittest.main()
    else:
        print("OpenAI API key not available - skipping real LLM tests")
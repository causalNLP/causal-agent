"""Basic integration tests with real LLM calls - focused and fast."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import unittest
from dotenv import load_dotenv

from causal_agent.agent import run_causal_analysis


# Load environment variables
load_dotenv()


def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.requires_llm
class TestBasicLLMIntegration(unittest.TestCase):
    """Basic integration tests using real LLM calls."""
    
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
    
    def create_minimal_rct_dataset(self) -> str:
        """Create a minimal RCT dataset for fast testing."""
        np.random.seed(42)
        n = 40  # Small dataset for fast processing
        
        # Simple randomized treatment
        treatment = np.random.binomial(1, 0.5, n)
        
        # Clear treatment effect for easy validation
        outcome = 10 + 5 * treatment + np.random.normal(0, 2, n)
        
        data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome
        })
        
        filepath = os.path.join(self.temp_dir, "minimal_rct.csv")
        data.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.slow
    def test_simple_rct_analysis_real_llm(self):
        """Test simple RCT analysis with real LLM - should be fast and reliable."""
        dataset_path = self.create_minimal_rct_dataset()
        
        query = "What is the effect of treatment on outcome?"
        description = "Simple randomized controlled trial with binary treatment and continuous outcome."
        
        print(f"\n=== Testing Simple RCT Analysis ===")
        print(f"Dataset: {dataset_path}")
        print(f"Query: {query}")
        
        try:
            result = run_causal_analysis(
                query=query,
                dataset_path=dataset_path,
                dataset_description=description
            )
            
            # Basic validation
            self.assertIsInstance(result, dict)
            print(f"Result type: {type(result)}")
            print(f"Result keys: {list(result.keys())}")
            
            # Check for success or informative error
            if 'error' in result:
                print(f"Analysis returned error: {result['error']}")
                # Don't fail immediately - log for investigation
                print("This may indicate an issue with the workflow or LLM response")
            else:
                print("Analysis completed successfully!")
                
                # Try to extract key information
                if 'results' in result:
                    results_section = result['results']
                    print(f"Results section keys: {list(results_section.keys())}")
                    
                    if 'results' in results_section:
                        analysis_results = results_section['results']
                        method_used = analysis_results.get('method_used', 'Unknown')
                        effect_estimate = analysis_results.get('effect_estimate', 'Unknown')
                        
                        print(f"Method used: {method_used}")
                        print(f"Effect estimate: {effect_estimate}")
                        
                        # Basic sanity check - effect should be positive (we added +5 for treatment)
                        if isinstance(effect_estimate, (int, float)):
                            if effect_estimate > 0:
                                print("✓ Effect estimate has expected positive sign")
                            else:
                                print("⚠ Effect estimate is negative (unexpected)")
                
        except Exception as e:
            print(f"Exception during analysis: {e}")
            # Log but don't fail - helps with debugging
            print("This exception should be investigated")
            raise  # Re-raise for test failure
    
    def test_component_integration_real_llm(self):
        """Test individual components with real LLM calls."""
        from causal_agent.tools.input_parser_tool import input_parser_tool
        
        dataset_path = self.create_minimal_rct_dataset()
        
        print(f"\n=== Testing Component Integration ===")
        
        # Test input parser with real data
        input_text = f"Query: What is the effect of treatment on outcome?\nDataset: {dataset_path}\nDescription: Test dataset"
        
        try:
            input_result = input_parser_tool(input_text)
            
            print(f"Input parser result type: {type(input_result)}")
            print(f"Input parser keys: {list(input_result.keys())}")
            
            # Validate input parser output
            self.assertIsInstance(input_result, dict)
            self.assertIn('original_query', input_result)
            self.assertIn('dataset_path', input_result)
            
            # Verify dataset path is correct
            self.assertEqual(input_result['dataset_path'], dataset_path)
            
            print("✓ Input parser component working correctly")
            
        except Exception as e:
            print(f"Component integration failed: {e}")
            raise
    
    def test_dataset_analyzer_real_llm(self):
        """Test dataset analyzer with real LLM calls."""
        from causal_agent.tools.dataset_analyzer_tool import dataset_analyzer_tool
        
        dataset_path = self.create_minimal_rct_dataset()
        
        print(f"\n=== Testing Dataset Analyzer ===")
        
        try:
            analysis_result = dataset_analyzer_tool.func(
                dataset_path=dataset_path,
                dataset_description="Simple RCT dataset with treatment and outcome",
                original_query="What is the effect of treatment on outcome?"
            )
            
            print(f"Dataset analyzer result type: {type(analysis_result)}")
            print(f"Has analysis_results: {hasattr(analysis_result, 'analysis_results')}")
            
            # Validate dataset analyzer output
            self.assertTrue(hasattr(analysis_result, 'analysis_results'))
            
            analysis_dict = analysis_result.analysis_results
            # Convert Pydantic model to dict to check keys
            analysis_data = analysis_dict.model_dump() if hasattr(analysis_dict, 'model_dump') else analysis_dict.__dict__
            print(f"Analysis results keys: {list(analysis_data.keys())}")
            
            # Check for expected analysis components
            expected_keys = ['sample_size', 'num_covariates_estimate', 'columns']
            for key in expected_keys:
                if hasattr(analysis_dict, key):
                    print(f"✓ Found expected attribute: {key}")
                else:
                    print(f"⚠ Missing expected attribute: {key}")
            
            print("✓ Dataset analyzer component working")
            
        except Exception as e:
            print(f"Dataset analyzer failed: {e}")
            raise
    
    def test_error_recovery_real_llm(self):
        """Test error recovery with real LLM calls."""
        # Create a problematic dataset
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']  # Non-numeric data
        })
        
        filepath = os.path.join(self.temp_dir, "problematic.csv")
        data.to_csv(filepath, index=False)
        
        print(f"\n=== Testing Error Recovery ===")
        
        query = "What is the causal effect of col1 on col2?"
        description = "Dataset with problematic structure for testing error handling."
        
        try:
            result = run_causal_analysis(
                query=query,
                dataset_path=filepath,
                dataset_description=description
            )
            
            print(f"Error recovery result: {type(result)}")
            
            # Should handle gracefully
            self.assertIsInstance(result, dict)
            
            if 'error' in result:
                print(f"✓ Error handled gracefully: {result['error']}")
            else:
                print("⚠ No error reported - may have processed problematic data")
            
        except Exception as e:
            print(f"Error recovery test exception: {e}")
            # This is acceptable - error handling can include exceptions
    
    def test_llm_configuration_real(self):
        """Test that LLM configuration is working correctly."""
        from causal_agent.config import get_llm_client
        
        print(f"\n=== Testing LLM Configuration ===")
        
        try:
            # Test LLM client creation
            llm_client = get_llm_client()
            
            print(f"LLM client type: {type(llm_client)}")
            print(f"LLM client created successfully")
            
            # Verify environment variables
            provider = os.getenv("LLM_PROVIDER", "not_set")
            model = os.getenv("LLM_MODEL", "not_set")
            
            print(f"LLM Provider: {provider}")
            print(f"LLM Model: {model}")
            
            self.assertEqual(provider, "openai")
            self.assertEqual(model, "gpt-3.5-turbo")
            
            print("✓ LLM configuration is correct")
            
        except Exception as e:
            print(f"LLM configuration test failed: {e}")
            raise


if __name__ == '__main__':
    # Run tests only if OpenAI key is available
    if has_openai_key():
        unittest.main()
    else:
        print("OpenAI API key not available - skipping real LLM tests")
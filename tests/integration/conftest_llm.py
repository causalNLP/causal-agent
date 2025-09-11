"""Configuration for real LLM integration tests."""

import pytest
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def pytest_configure(config):
    """Configure pytest for LLM tests."""
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring actual LLM API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle LLM requirements."""
    skip_llm = pytest.mark.skip(reason="OpenAI API key not available")
    
    for item in items:
        if "requires_llm" in item.keywords:
            if not os.getenv("OPENAI_API_KEY"):
                item.add_marker(skip_llm)


@pytest.fixture(scope="session")
def openai_api_key():
    """Provide OpenAI API key for tests."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not available")
    return api_key


@pytest.fixture(scope="session")
def llm_config():
    """Provide LLM configuration for tests."""
    return {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,  # Low temperature for more consistent results
        "max_tokens": 2000
    }


@pytest.fixture(autouse=True, scope="session")
def setup_llm_environment(llm_config):
    """Set up LLM environment variables for tests."""
    original_provider = os.environ.get("LLM_PROVIDER")
    original_model = os.environ.get("LLM_MODEL")
    
    # Set test configuration
    os.environ["LLM_PROVIDER"] = llm_config["provider"]
    os.environ["LLM_MODEL"] = llm_config["model"]
    
    yield
    
    # Restore original configuration
    if original_provider:
        os.environ["LLM_PROVIDER"] = original_provider
    elif "LLM_PROVIDER" in os.environ:
        del os.environ["LLM_PROVIDER"]
        
    if original_model:
        os.environ["LLM_MODEL"] = original_model
    elif "LLM_MODEL" in os.environ:
        del os.environ["LLM_MODEL"]


@pytest.fixture
def rate_limit_delay():
    """Provide rate limiting for API calls."""
    import time
    
    def delay(seconds=1):
        """Add delay between API calls to respect rate limits."""
        time.sleep(seconds)
    
    return delay


class LLMTestHelper:
    """Helper class for LLM integration tests."""
    
    @staticmethod
    def validate_analysis_result(result):
        """Validate the structure of an analysis result."""
        assert isinstance(result, dict), "Result should be a dictionary"
        
        if 'error' in result:
            return False, f"Analysis failed with error: {result['error']}"
        
        if 'results' not in result:
            return False, "Result missing 'results' key"
        
        return True, "Result structure is valid"
    
    @staticmethod
    def extract_effect_estimate(result):
        """Extract effect estimate from analysis result."""
        try:
            return result['results']['results']['effect_estimate']
        except (KeyError, TypeError):
            return None
    
    @staticmethod
    def extract_method_used(result):
        """Extract method used from analysis result."""
        try:
            return result['results']['results']['method_used']
        except (KeyError, TypeError):
            return None
    
    @staticmethod
    def print_result_summary(result, query=""):
        """Print a summary of the analysis result."""
        print(f"\n=== Analysis Result Summary ===")
        if query:
            print(f"Query: {query}")
        
        is_valid, message = LLMTestHelper.validate_analysis_result(result)
        print(f"Valid result: {is_valid}")
        
        if is_valid:
            method = LLMTestHelper.extract_method_used(result)
            effect = LLMTestHelper.extract_effect_estimate(result)
            print(f"Method used: {method}")
            print(f"Effect estimate: {effect}")
        else:
            print(f"Issue: {message}")


@pytest.fixture
def llm_test_helper():
    """Provide LLM test helper."""
    return LLMTestHelper()
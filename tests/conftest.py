"""Pytest configuration and shared fixtures for causal_agent tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional
import os
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and register data management plugin
pytest_plugins = ["tests.fixtures.pytest_data_plugin"]


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="causal_agent_test_")
    yield Path(temp_dir)
    # Cleanup after all tests
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration settings."""
    from tests.fixtures import get_test_config
    return get_test_config()


@pytest.fixture(scope="session")
def config_manager():
    """Provide test configuration manager."""
    from tests.fixtures import get_config_manager
    return get_config_manager()


# ============================================================================
# Function-level Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    from tests.fixtures import mock_llm_generator
    
    with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock_llm:
        # Use sophisticated mock responses from fixtures
        def mock_response(*args, **kwargs):
            # Determine response type based on call context
            if "method" in str(args) or "select" in str(args):
                return mock_llm_generator.get_method_selection_response()
            elif "analyze" in str(args) or "dataset" in str(args):
                return mock_llm_generator.get_dataset_analysis_response()
            else:
                # Default response
                return mock_llm_generator.get_method_selection_response()
        
        mock_llm.side_effect = mock_response
        yield mock_llm


@pytest.fixture
def synthetic_data_generator():
    """Provide synthetic data generator for tests."""
    from tests.fixtures import get_synthetic_data_generator
    return get_synthetic_data_generator()


@pytest.fixture
def sample_rct_data(synthetic_data_generator):
    """Generate sample RCT dataset for testing."""
    return synthetic_data_generator.generate_rct_data()


@pytest.fixture
def sample_observational_data(synthetic_data_generator):
    """Generate sample observational dataset for testing."""
    return synthetic_data_generator.generate_observational_data()


@pytest.fixture
def sample_iv_data(synthetic_data_generator):
    """Generate sample instrumental variable dataset for testing."""
    return synthetic_data_generator.generate_iv_data()


@pytest.fixture
def sample_rdd_data(synthetic_data_generator):
    """Generate sample regression discontinuity dataset for testing."""
    return synthetic_data_generator.generate_rdd_data()


@pytest.fixture
def sample_did_data(synthetic_data_generator):
    """Generate sample difference-in-differences dataset for testing."""
    return synthetic_data_generator.generate_did_data()


@pytest.fixture
def benchmark_datasets():
    """Provide benchmark datasets for comprehensive testing."""
    from tests.fixtures import get_benchmark_datasets
    return get_benchmark_datasets()


@pytest.fixture
def standard_datasets():
    """Provide standard validation datasets."""
    from tests.fixtures import get_standard_datasets
    return get_standard_datasets()


@pytest.fixture
def performance_datasets():
    """Provide datasets for performance testing."""
    from tests.fixtures import get_performance_datasets
    return get_performance_datasets()


@pytest.fixture
def mock_dataset_analyzer():
    """Mock dataset analyzer component."""
    mock_analyzer = Mock()
    mock_analyzer.analyze_dataset.return_value = {
        "n_samples": 500,
        "n_features": 5,
        "treatment_variable": "treatment",
        "outcome_variable": "outcome",
        "potential_confounders": ["feature_0", "feature_1"],
        "data_quality_score": 0.85
    }
    return mock_analyzer


@pytest.fixture
def mock_method_selector():
    """Mock method selector component."""
    mock_selector = Mock()
    mock_selector.select_method.return_value = {
        "recommended_method": "backdoor_adjustment",
        "confidence": 0.8,
        "alternative_methods": ["propensity_score", "linear_regression"],
        "reasoning": "Dataset appears suitable for backdoor adjustment"
    }
    return mock_selector


@pytest.fixture
def test_query_simple():
    """Simple test query for causal analysis."""
    return {
        "query": "What is the effect of treatment on outcome?",
        "treatment_variable": "treatment",
        "outcome_variable": "outcome"
    }


@pytest.fixture
def test_query_complex():
    """Complex test query with additional specifications."""
    return {
        "query": "Estimate the causal effect of education on income, controlling for background variables",
        "treatment_variable": "education_years",
        "outcome_variable": "annual_income", 
        "confounders": ["age", "gender", "family_background"],
        "method_preference": "propensity_score",
        "confidence_level": 0.95
    }


# ============================================================================
# Parametrized Fixtures
# ============================================================================

@pytest.fixture(params=["backdoor_adjustment", "propensity_score", "linear_regression"])
def causal_method_name(request):
    """Parametrized fixture for different causal methods."""
    return request.param


@pytest.fixture(params=[100, 500, 1000])
def dataset_size(request):
    """Parametrized fixture for different dataset sizes."""
    return request.param


@pytest.fixture(params=[0.1, 0.3, 0.5, 0.8])
def treatment_effect_size(request):
    """Parametrized fixture for different treatment effect sizes."""
    return request.param


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def assert_helpers():
    """Provide assertion helper functions."""
    class AssertHelpers:
        @staticmethod
        def assert_causal_result_structure(result: Dict[str, Any]):
            """Assert that a causal result has the expected structure."""
            required_keys = ["effect_estimate", "confidence_interval", "method_used"]
            for key in required_keys:
                assert key in result, f"Required key '{key}' not found in result"
            
            # Check types
            assert isinstance(result["effect_estimate"], (int, float, np.number))
            
            if result["confidence_interval"]:
                ci = result["confidence_interval"]
                assert isinstance(ci, (list, tuple))
                assert len(ci) == 2
                assert ci[0] <= ci[1]
        
        @staticmethod
        def assert_dataframe_structure(df: pd.DataFrame, 
                                     expected_columns: List[str],
                                     min_rows: int = 1):
            """Assert DataFrame has expected structure."""
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= min_rows
            
            for col in expected_columns:
                assert col in df.columns, f"Column '{col}' not found"
        
        @staticmethod
        def assert_performance_within_limits(execution_time: float,
                                           memory_usage: float,
                                           max_time: float = 10.0,
                                           max_memory_mb: float = 100.0):
            """Assert performance metrics are within acceptable limits."""
            assert execution_time <= max_time, f"Execution time {execution_time}s exceeds limit {max_time}s"
            assert memory_usage <= max_memory_mb, f"Memory usage {memory_usage}MB exceeds limit {max_memory_mb}MB"
    
    return AssertHelpers()


@pytest.fixture
def data_generator():
    """Provide data generation utilities."""
    class DataGenerator:
        @staticmethod
        def create_synthetic_dataset(n_samples: int = 100,
                                   n_features: int = 3,
                                   treatment_effect: float = 0.5,
                                   noise_level: float = 0.1,
                                   seed: int = 42) -> pd.DataFrame:
            """Create synthetic dataset with known causal structure."""
            np.random.seed(seed)
            
            # Generate features
            features = np.random.normal(0, 1, (n_samples, n_features))
            
            # Generate treatment (can depend on features for observational data)
            treatment_prob = 1 / (1 + np.exp(-features[:, 0]))  # Logistic
            treatment = np.random.binomial(1, treatment_prob)
            
            # Generate outcome with known treatment effect
            outcome = (features.sum(axis=1) * 0.1 + 
                      treatment_effect * treatment + 
                      np.random.normal(0, noise_level, n_samples))
            
            # Create DataFrame
            data = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
            data['treatment'] = treatment
            data['outcome'] = outcome
            
            return data
        
        @staticmethod
        def create_time_series_data(n_periods: int = 50,
                                  n_units: int = 20,
                                  treatment_start: int = 25,
                                  treatment_effect: float = 0.3) -> pd.DataFrame:
            """Create panel data for difference-in-differences testing."""
            np.random.seed(42)
            
            data = []
            for unit in range(n_units):
                for period in range(n_periods):
                    # Some units get treated after treatment_start
                    treated_unit = unit < n_units // 2
                    post_treatment = period >= treatment_start
                    treatment = 1 if (treated_unit and post_treatment) else 0
                    
                    # Outcome with unit and time fixed effects
                    outcome = (unit * 0.1 + period * 0.05 + 
                              treatment_effect * treatment + 
                              np.random.normal(0, 0.2))
                    
                    data.append({
                        'unit': unit,
                        'period': period,
                        'treatment': treatment,
                        'outcome': outcome,
                        'treated_unit': treated_unit,
                        'post_treatment': post_treatment
                    })
            
            return pd.DataFrame(data)
    
    return DataGenerator()


# ============================================================================
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Enhanced cleanup using data manager
    from tests.fixtures.data_manager import get_test_data_manager
    
    try:
        manager = get_test_data_manager()
        # Clean up old temp resources (older than 5 minutes)
        manager.temp_manager.cleanup_old_resources(max_age_seconds=300)
    except Exception:
        # Fallback to basic cleanup
        temp_patterns = ["test_*.csv", "temp_*.json", "*.tmp"]
        current_dir = Path.cwd()
        
        for pattern in temp_patterns:
            for temp_file in current_dir.glob(pattern):
                try:
                    temp_file.unlink()
                except (OSError, PermissionError):
                    pass  # Ignore cleanup errors
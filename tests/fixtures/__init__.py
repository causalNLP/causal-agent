"""Shared test fixtures and data for causal_agent tests."""

# Import all fixture components for easy access
from .synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    DatasetType,
    create_benchmark_datasets
)

from .mock_llm_responses import (
    MockLLMResponseGenerator,
    MockLLMResponse,
    LLMResponseType,
    mock_method_selection,
    mock_dataset_analysis,
    mock_result_interpretation,
    mock_llm_generator
)

from .shared_datasets import (
    SharedDatasetManager,
    DatasetMetadata,
    shared_dataset_manager,
    get_standard_datasets,
    get_benchmark_datasets,
    get_performance_datasets,
    load_real_world_datasets,
    create_method_specific_datasets
)

from .test_config import (
    CausalAgentTestConfig,
    TestConfigManager,
    TestEnvironment,
    LogLevel,
    LLMConfig,
    DataConfig,
    PerformanceConfig,
    CoverageConfig,
    CIConfig,
    get_test_config,
    get_config_manager,
    setup_test_config,
    create_test_config_for_method
)

from .data_manager import (
    TestDataManager,
    DataCache,
    TempResourceManager,
    TestIsolationManager,
    CacheEntry,
    TempResource,
    get_test_data_manager,
    setup_test_data_manager,
    get_cached_dataset,
    create_temp_dataset_file,
    isolated_test,
    cleanup_test_data
)

# Convenience functions for common fixture usage
def get_synthetic_data_generator(seed: int = 42) -> SyntheticDataGenerator:
    """Get a configured synthetic data generator."""
    config = SyntheticDataConfig(random_seed=seed)
    return SyntheticDataGenerator(config)


def get_mock_llm_generator() -> MockLLMResponseGenerator:
    """Get the mock LLM response generator."""
    return mock_llm_generator


def get_dataset_manager() -> SharedDatasetManager:
    """Get the shared dataset manager."""
    return shared_dataset_manager


def get_data_manager() -> TestDataManager:
    """Get the test data manager."""
    return get_test_data_manager()


def setup_test_environment(environment: str = "local", **config_overrides):
    """Setup test environment with appropriate configuration."""
    env_configs = {
        "local": {
            "debug_mode": True,
            "verbose_output": True,
            "llm": {"mock_llm": True},
            "data": {"cleanup_temp_data": False}
        },
        "ci": {
            "debug_mode": False,
            "verbose_output": False,
            "llm": {"mock_llm": True},
            "data": {"cleanup_temp_data": True},
            "performance": {"enable_profiling": False}
        },
        "performance": {
            "performance": {"enable_profiling": True},
            "llm": {"mock_llm": True},
            "data": {"cache_datasets": True}
        }
    }
    
    base_config = env_configs.get(environment, env_configs["local"])
    base_config.update(config_overrides)
    
    return setup_test_config(**base_config)


# Export all public components
__all__ = [
    # Synthetic data
    "SyntheticDataGenerator",
    "SyntheticDataConfig", 
    "DatasetType",
    "create_benchmark_datasets",
    
    # Mock LLM responses
    "MockLLMResponseGenerator",
    "MockLLMResponse",
    "LLMResponseType",
    "mock_method_selection",
    "mock_dataset_analysis", 
    "mock_result_interpretation",
    "mock_llm_generator",
    
    # Shared datasets
    "SharedDatasetManager",
    "DatasetMetadata",
    "shared_dataset_manager",
    "get_standard_datasets",
    "get_benchmark_datasets",
    "get_performance_datasets",
    "load_real_world_datasets",
    "create_method_specific_datasets",
    
    # Test configuration
    "CausalAgentTestConfig",
    "TestConfigManager",
    "TestEnvironment",
    "LogLevel",
    "LLMConfig",
    "DataConfig", 
    "PerformanceConfig",
    "CoverageConfig",
    "CIConfig",
    "get_test_config",
    "get_config_manager",
    "setup_test_config",
    "create_test_config_for_method",
    
    # Data management
    "TestDataManager",
    "DataCache",
    "TempResourceManager", 
    "TestIsolationManager",
    "CacheEntry",
    "TempResource",
    "get_test_data_manager",
    "setup_test_data_manager",
    "get_cached_dataset",
    "create_temp_dataset_file",
    "isolated_test",
    "cleanup_test_data",
    
    # Convenience functions
    "get_synthetic_data_generator",
    "get_mock_llm_generator",
    "get_dataset_manager",
    "get_data_manager",
    "setup_test_environment"
]
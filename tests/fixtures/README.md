# Test Fixtures and Data Management

This directory contains comprehensive test fixtures and data management utilities for the causal_agent testing infrastructure. The fixtures provide synthetic data generation, mock LLM responses, shared dataset management, and test configuration management.

## Overview

The test fixtures system consists of four main components:

1. **Synthetic Data Generation** (`synthetic_data.py`) - Generate various types of causal inference datasets
2. **Mock LLM Responses** (`mock_llm_responses.py`) - Provide deterministic LLM responses for testing
3. **Shared Dataset Management** (`shared_datasets.py`) - Manage and cache datasets across tests
4. **Test Configuration** (`test_config.py`) - Centralized configuration management for tests

## Quick Start

```python
from tests.fixtures import (
    get_synthetic_data_generator,
    get_mock_llm_generator,
    get_dataset_manager,
    get_test_config
)

# Generate synthetic data
generator = get_synthetic_data_generator()
rct_data = generator.generate_rct_data()
obs_data = generator.generate_observational_data()

# Get mock LLM responses
llm_generator = get_mock_llm_generator()
method_response = llm_generator.get_method_selection_response("propensity_score")

# Access shared datasets
datasets = get_dataset_manager().get_validation_datasets()

# Get test configuration
config = get_test_config()
```

## Synthetic Data Generation

### Supported Dataset Types

- **RCT (Randomized Controlled Trial)** - Random treatment assignment
- **Observational** - Treatment assignment with confounding
- **Instrumental Variable** - Datasets with instruments for identification
- **Regression Discontinuity** - Sharp cutoff-based treatment assignment
- **Difference-in-Differences** - Panel data with treatment timing variation
- **Front Door** - Datasets suitable for front-door criterion

### Basic Usage

```python
from tests.fixtures import SyntheticDataGenerator, SyntheticDataConfig, DatasetType

# Create generator with custom configuration
config = SyntheticDataConfig(
    n_samples=1000,
    treatment_effect=0.5,
    noise_level=0.1,
    random_seed=42
)
generator = SyntheticDataGenerator(config)

# Generate specific dataset types
rct_data = generator.generate_rct_data()
obs_data = generator.generate_observational_data()
iv_data = generator.generate_iv_data()

# Generate multiple datasets
datasets = generator.generate_multiple_datasets([
    DatasetType.RCT,
    DatasetType.OBSERVATIONAL,
    DatasetType.INSTRUMENTAL_VARIABLE
])
```

### Dataset Metadata

All generated datasets include metadata attributes:

```python
data = generator.generate_rct_data()
print(data.attrs['dataset_type'])           # 'randomized_controlled_trial'
print(data.attrs['true_treatment_effect'])  # 0.5
print(data.attrs['confounders'])           # [] (no confounders in RCT)
```

## Mock LLM Responses

### Response Types

- **Method Selection** - Causal method recommendations
- **Dataset Analysis** - Dataset quality and structure analysis
- **Result Interpretation** - Causal effect interpretation
- **Assumption Validation** - Assumption checking results
- **Diagnostic Analysis** - Method diagnostic results
- **Query Parsing** - User query interpretation
- **Error Explanation** - Error analysis and suggestions

### Basic Usage

```python
from tests.fixtures import MockLLMResponseGenerator, LLMResponseType

generator = MockLLMResponseGenerator()

# Get structured responses
method_response = generator.get_response(LLMResponseType.METHOD_SELECTION)
dataset_response = generator.get_response(LLMResponseType.DATASET_ANALYSIS)

# Get quick responses for common scenarios
method_dict = generator.get_method_selection_response("backdoor_adjustment", confidence=0.8)
dataset_dict = generator.get_dataset_analysis_response(n_obs=500, data_quality=0.9)

# Create custom responses
custom_response = generator.create_custom_response(
    LLMResponseType.RESULT_INTERPRETATION,
    {"effect_estimate": 0.45, "confidence_interval": [0.2, 0.7]},
    confidence=0.85
)
```

### Integration with Tests

```python
import pytest
from unittest.mock import patch
from tests.fixtures import mock_llm_generator

@pytest.fixture
def mock_llm():
    with patch('causal_agent.utils.llm_helpers.call_llm_with_json_output') as mock:
        mock.return_value = mock_llm_generator.get_method_selection_response()
        yield mock

def test_method_selection(mock_llm):
    # Your test code here
    pass
```

## Shared Dataset Management

### Dataset Registration and Loading

```python
from tests.fixtures import SharedDatasetManager, DatasetMetadata

manager = SharedDatasetManager()

# Register a dataset
metadata = DatasetMetadata(
    name="my_test_dataset",
    dataset_type="synthetic",
    n_samples=500,
    n_features=5,
    true_treatment_effect=0.5,
    description="Custom test dataset",
    tags=["test", "rct"]
)

dataset_path = manager.register_dataset("my_test_dataset", data, metadata)

# Load dataset
loaded_data, loaded_metadata = manager.load_dataset("my_test_dataset")

# List datasets
all_datasets = manager.list_datasets()
rct_datasets = manager.list_datasets(tags=["rct"])
```

### Benchmark Datasets

```python
# Create comprehensive benchmark suite
benchmark_datasets = manager.create_benchmark_suite()

# Get validation datasets
validation_datasets = manager.get_validation_datasets()

# Get performance testing datasets
performance_datasets = manager.get_performance_datasets()
```

### Method-Specific Datasets

```python
from tests.fixtures import create_method_specific_datasets

method_datasets = create_method_specific_datasets()

# Access datasets optimized for specific methods
backdoor_datasets = method_datasets["backdoor_adjustment"]
ps_datasets = method_datasets["propensity_score"]
iv_datasets = method_datasets["instrumental_variable"]
```

## Test Configuration Management

### Configuration Structure

```python
from tests.fixtures import CausalAgentTestConfig, TestEnvironment, LLMConfig

config = CausalAgentTestConfig(
    environment=TestEnvironment.LOCAL,
    debug_mode=True,
    llm=LLMConfig(mock_llm=True, temperature=0.0),
    random_seed=42
)
```

### Environment-Specific Configuration

```python
from tests.fixtures import setup_test_environment

# Setup for different environments
local_config = setup_test_environment("local", debug_mode=True)
ci_config = setup_test_environment("ci", verbose_output=False)
perf_config = setup_test_environment("performance", enable_profiling=True)
```

### Method-Specific Configuration

```python
from tests.fixtures import create_test_config_for_method

# Get optimized configuration for specific methods
ps_config = create_test_config_for_method("propensity_score")
iv_config = create_test_config_for_method("instrumental_variable")

# Access method-specific settings
ps_settings = ps_config.method_configs["propensity_score"]
max_time = ps_settings["max_execution_time"]
```

## Pytest Integration

### Using Fixtures in Tests

```python
import pytest

def test_with_synthetic_data(sample_rct_data):
    """Test using synthetic RCT data fixture."""
    assert len(sample_rct_data) > 0
    assert 'treatment' in sample_rct_data.columns

def test_with_mock_llm(mock_llm_client):
    """Test using mock LLM client fixture."""
    # Your causal agent code that calls LLM
    pass

def test_with_config(test_config):
    """Test using test configuration fixture."""
    assert test_config.random_seed == 42
    assert test_config.llm.mock_llm is True

@pytest.mark.parametrize("method_name", ["backdoor_adjustment", "propensity_score"])
def test_multiple_methods(method_name, synthetic_data_generator):
    """Parametrized test across multiple methods."""
    data = synthetic_data_generator.generate_observational_data()
    # Test method with data
```

### Custom Fixtures

```python
@pytest.fixture
def custom_dataset():
    """Create custom dataset for specific test needs."""
    from tests.fixtures import get_synthetic_data_generator, SyntheticDataConfig
    
    config = SyntheticDataConfig(
        n_samples=200,
        treatment_effect=0.8,
        confounding_strength=0.5
    )
    generator = get_synthetic_data_generator()
    generator.config = config
    return generator.generate_observational_data()

@pytest.fixture
def custom_llm_response():
    """Create custom LLM response for specific test."""
    from tests.fixtures import get_mock_llm_generator
    
    generator = get_mock_llm_generator()
    return generator.get_method_selection_response(
        method="instrumental_variable",
        confidence=0.9
    )
```

## Performance Testing

### Dataset Scaling

```python
from tests.fixtures import get_performance_datasets

# Get datasets of different sizes for performance testing
perf_datasets = get_performance_datasets()

for name, data in perf_datasets.items():
    # Run performance tests with different dataset sizes
    start_time = time.time()
    result = run_causal_method(data)
    execution_time = time.time() - start_time
    
    assert execution_time < config.performance.max_execution_time_seconds
```

### Memory Usage Testing

```python
import psutil
import os

def test_memory_usage(performance_datasets, test_config):
    """Test memory usage stays within limits."""
    process = psutil.Process(os.getpid())
    
    for name, data in performance_datasets.items():
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run your causal method
        result = run_method(data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < test_config.performance.max_memory_usage_mb
```

## Configuration Files

### YAML Configuration

```yaml
# test_config.yaml
environment: "local"
debug_mode: true
verbose_output: true

llm:
  mock_llm: true
  temperature: 0.0
  timeout_seconds: 30

data:
  use_synthetic_data: true
  synthetic_data_seed: 42
  cache_datasets: true

performance:
  max_execution_time_seconds: 10.0
  max_memory_usage_mb: 500.0
  enable_profiling: false

coverage:
  minimum_coverage_percentage: 80.0
  coverage_report_format: ["html", "xml"]
```

### Environment Variables

```bash
# Override configuration with environment variables
export CAUSAL_AGENT_TEST_DEBUG=true
export CAUSAL_AGENT_TEST_MOCK_LLM=true
export CAUSAL_AGENT_TEST_SEED=42
export CAUSAL_AGENT_TEST_MIN_COVERAGE=85.0
```

## Best Practices

### 1. Use Appropriate Dataset Types

```python
# For unit tests - small, fast datasets
@pytest.fixture
def unit_test_data():
    config = SyntheticDataConfig(n_samples=100, noise_level=0.1)
    generator = SyntheticDataGenerator(config)
    return generator.generate_rct_data()

# For integration tests - realistic datasets
@pytest.fixture  
def integration_test_data():
    return get_standard_datasets()["obs_weak_confounding"]

# For performance tests - large datasets
@pytest.fixture
def performance_test_data():
    return get_performance_datasets()["performance_n10000"]
```

### 2. Mock LLM Responses Appropriately

```python
# For deterministic tests
def test_method_selection_deterministic(mock_llm_client):
    mock_llm_client.return_value = mock_method_selection("backdoor_adjustment", 0.8)
    # Test code

# For testing error handling
def test_llm_error_handling(mock_llm_client):
    mock_llm_client.side_effect = Exception("API Error")
    # Test error handling code
```

### 3. Use Configuration Effectively

```python
# Method-specific configuration
@pytest.fixture
def propensity_score_config():
    return create_test_config_for_method("propensity_score", 
                                       overlap_threshold=0.1,
                                       caliper=0.2)

# Environment-specific setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    if os.getenv('CI'):
        setup_test_config(environment="ci")
    else:
        setup_test_config(environment="local")
```

### 4. Dataset Caching and Cleanup

```python
# Cache expensive datasets
@pytest.fixture(scope="session")
def expensive_dataset():
    manager = get_dataset_manager()
    try:
        data, _ = manager.load_dataset("expensive_computation")
    except ValueError:
        # Generate if not cached
        data = generate_expensive_dataset()
        metadata = DatasetMetadata(...)
        manager.register_dataset("expensive_computation", data, metadata)
    return data

# Cleanup temporary data
@pytest.fixture(autouse=True)
def cleanup_temp_data():
    yield
    # Cleanup code here
```

## Examples

See `examples/fixture_usage_examples.py` for comprehensive examples of using all fixture components together.

## Directory Structure

```
tests/fixtures/
├── __init__.py                 # Main exports and convenience functions
├── synthetic_data.py          # Synthetic data generation
├── mock_llm_responses.py      # Mock LLM response fixtures
├── shared_datasets.py         # Dataset management and caching
├── test_config.py            # Test configuration management
├── data/                     # Cached datasets
│   ├── synthetic/           # Synthetic datasets
│   ├── real/               # Real-world datasets
│   └── benchmark/          # Benchmark datasets
├── examples/               # Usage examples
│   └── fixture_usage_examples.py
└── README.md              # This documentation
```
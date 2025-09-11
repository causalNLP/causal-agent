# Test Data Management System

This document describes the comprehensive test data management and cleanup system implemented for the causal_agent project.

## Overview

The test data management system provides:

1. **Efficient Data Loading and Caching** - Intelligent caching of datasets to speed up test execution
2. **Temporary Resource Management** - Automatic cleanup of temporary files and directories
3. **Test Isolation** - Prevents test interference through isolated execution contexts
4. **Performance Monitoring** - Tracks resource usage and execution times
5. **Automatic Cleanup** - Ensures no test artifacts are left behind

## Components

### 1. DataCache

Thread-safe LRU cache for test datasets with configurable size limits.

```python
from tests.fixtures.data_manager import DataCache

cache = DataCache(max_size_mb=100, max_entries=50)
cache.put("dataset_key", dataframe)
data = cache.get("dataset_key")
```

**Features:**
- LRU eviction policy
- Size-based limits (memory and entry count)
- Thread-safe operations
- Performance statistics

### 2. TempResourceManager

Manages temporary files and directories with automatic cleanup.

```python
from tests.fixtures.data_manager import TempResourceManager

manager = TempResourceManager()
temp_file = manager.create_temp_file(suffix=".csv", content="data")
temp_dir = manager.create_temp_dir(prefix="test_")
dataset_file = manager.save_temp_dataset(dataframe, "dataset_name")
```

**Features:**
- Automatic cleanup on exit
- Age-based cleanup policies
- Resource tracking and statistics
- Custom cleanup functions

### 3. TestIsolationManager

Provides isolated execution contexts to prevent test interference.

```python
from tests.fixtures.data_manager import TestIsolationManager

manager = TestIsolationManager()
with manager.isolated_test("test_name") as test_id:
    # Test runs in isolation
    # Random state is isolated
    # Resources are tracked for cleanup
    pass
```

**Features:**
- Random state isolation
- Working directory isolation
- Resource tracking per test
- Automatic cleanup of abandoned tests

### 4. TestDataManager

Main interface combining all data management functionality.

```python
from tests.fixtures.data_manager import get_test_data_manager

manager = get_test_data_manager()
data = manager.get_dataset("dataset_key", cache=True)
temp_file = manager.create_temp_dataset(data, "temp_name")
workspace = manager.create_temp_workspace("test_name")
```

**Features:**
- Unified interface for all data operations
- Configurable caching and cleanup policies
- Performance monitoring
- Background maintenance tasks

## Usage Examples

### Basic Data Caching

```python
from tests.fixtures.data_manager import get_test_data_manager

def test_with_cached_data():
    manager = get_test_data_manager()
    
    # First call generates data
    data1 = manager.get_dataset("rct_standard", cache=True)
    
    # Second call retrieves from cache (much faster)
    data2 = manager.get_dataset("rct_standard", cache=True)
    
    assert data1.equals(data2)
```

### Custom Data Generation

```python
def custom_generator(n_samples=500, effect_size=0.5):
    # Your custom data generation logic
    return pd.DataFrame({...})

def test_with_custom_data():
    manager = get_test_data_manager()
    
    data = manager.get_dataset(
        "custom_key",
        generator_func=custom_generator,
        n_samples=1000,
        effect_size=0.8,
        cache=True
    )
```

### Test Isolation

```python
from tests.fixtures.data_manager import isolated_test

def test_with_isolation():
    with isolated_test("my_test") as context:
        workspace = context["workspace"]
        data_manager = context["data_manager"]
        
        # Create test files in isolated workspace
        (workspace / "data.csv").write_text("test,data")
        
        # Generate test data
        data = data_manager.get_dataset("test_data")
        
        # All resources cleaned up automatically
```

### Pytest Integration

The system integrates seamlessly with pytest through fixtures:

```python
def test_with_fixtures(test_data_manager, isolated_test_data):
    # test_data_manager provides the main manager
    data = test_data_manager.get_dataset("test_key")
    
    # isolated_test_data provides isolated context
    workspace = isolated_test_data["workspace"]
```

## Configuration

Configure the system through `tests/fixtures/data_management_config.yaml`:

```yaml
data:
  max_cache_size_mb: 100
  cleanup_temp_data: true
  preload_datasets:
    - "rct_standard"
    - "observational_standard"

performance:
  max_execution_time_seconds: 10.0
  max_memory_usage_mb: 500.0
  enable_profiling: false

isolation:
  enable_test_isolation: true
  isolate_random_state: true
  cleanup_abandoned_tests: true
```

## Pytest Plugin

The system includes a pytest plugin that automatically:

- Initializes data management at session start
- Preloads common datasets
- Monitors test performance
- Cleans up resources at session end
- Generates performance reports

Enable by adding to `conftest.py`:

```python
pytest_plugins = ["tests.fixtures.pytest_data_plugin"]
```

## Performance Benefits

The data management system provides significant performance improvements:

1. **Caching**: 10-100x speedup for repeated dataset access
2. **Preloading**: Common datasets loaded once at session start
3. **Efficient Cleanup**: Background cleanup prevents resource accumulation
4. **Memory Management**: LRU eviction prevents memory exhaustion

## Best Practices

### 1. Use Descriptive Cache Keys

```python
# Good
data = manager.get_dataset("rct_n500_effect0.5_seed42", cache=True)

# Bad
data = manager.get_dataset("data1", cache=True)
```

### 2. Cache Expensive Operations

```python
def expensive_data_generation():
    # Complex data generation logic
    return complex_dataset

# Cache the result
data = manager.get_dataset(
    "expensive_dataset_key",
    generator_func=expensive_data_generation,
    cache=True
)
```

### 3. Use Isolation for Independent Tests

```python
def test_independent_analysis():
    with isolated_test("analysis_test") as context:
        # Test runs in complete isolation
        # No interference from other tests
        pass
```

### 4. Monitor Resource Usage

```python
def test_with_monitoring():
    manager = get_test_data_manager()
    
    # Check stats before
    initial_stats = manager.get_stats()
    
    # Run test operations
    data = manager.get_dataset("large_dataset")
    
    # Check stats after
    final_stats = manager.get_stats()
    
    # Verify resource usage is reasonable
    memory_used = final_stats["cache"]["total_size_mb"]
    assert memory_used < 100  # Less than 100MB
```

## Troubleshooting

### High Memory Usage

If tests are using too much memory:

1. Reduce cache size: `max_cache_size_mb: 50`
2. Enable aggressive cleanup: `cleanup_after_each_test: true`
3. Check for memory leaks in test code

### Slow Test Execution

If tests are running slowly:

1. Enable preloading: Add datasets to `preload_datasets`
2. Increase cache size: `max_cache_size_mb: 200`
3. Use caching for expensive operations

### Resource Cleanup Issues

If temporary files are not being cleaned up:

1. Check cleanup configuration: `cleanup_temp_data: true`
2. Verify pytest plugin is loaded
3. Call `cleanup_test_data()` manually if needed

### Test Interference

If tests are interfering with each other:

1. Use test isolation: `with isolated_test("test_name"):`
2. Enable random state isolation: `isolate_random_state: true`
3. Check for global state modifications

## API Reference

### TestDataManager

- `get_dataset(key, generator_func=None, cache=True, **kwargs)` - Get or generate dataset
- `create_temp_dataset(data, name)` - Save dataset to temporary file
- `create_temp_workspace(test_name)` - Create temporary workspace directory
- `isolated_test_data(test_name, cleanup_after=True)` - Context manager for isolated testing
- `preload_common_datasets(dataset_keys)` - Preload datasets into cache
- `get_stats()` - Get comprehensive statistics
- `cleanup_all()` - Clean up all managed resources

### DataCache

- `get(key)` - Retrieve dataset from cache
- `put(key, data, metadata=None)` - Store dataset in cache
- `clear()` - Clear all cached data
- `get_stats()` - Get cache statistics

### TempResourceManager

- `create_temp_file(suffix, prefix, content=None)` - Create temporary file
- `create_temp_dir(prefix)` - Create temporary directory
- `save_temp_dataset(data, name)` - Save dataset to temporary file
- `cleanup_resource(path)` - Clean up specific resource
- `cleanup_old_resources(max_age_seconds)` - Clean up old resources
- `cleanup_all()` - Clean up all resources

### TestIsolationManager

- `isolated_test(test_name, cleanup_after=True)` - Context manager for test isolation
- `register_test_resource(test_id, resource_path)` - Register resource for cleanup
- `get_active_tests()` - Get list of active test IDs
- `cleanup_abandoned_tests(max_age_seconds)` - Clean up abandoned test contexts

## Integration with Existing Tests

The data management system is designed to integrate seamlessly with existing tests:

1. **Backward Compatibility**: Existing tests continue to work without changes
2. **Opt-in Features**: New features are opt-in through fixtures or explicit usage
3. **Minimal Overhead**: System has minimal performance impact when not actively used
4. **Gradual Adoption**: Tests can be migrated to use new features incrementally

## Future Enhancements

Planned improvements include:

1. **Distributed Caching**: Share cache across test processes
2. **Persistent Cache**: Cache datasets across test sessions
3. **Smart Preloading**: Automatically identify datasets to preload
4. **Resource Quotas**: Per-test resource limits
5. **Advanced Monitoring**: Detailed performance profiling
6. **Cloud Storage**: Integration with cloud storage for large datasets
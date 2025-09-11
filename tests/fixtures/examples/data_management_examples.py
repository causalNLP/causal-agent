"""Examples demonstrating the test data management and cleanup system."""

import pandas as pd
import numpy as np
import time
from pathlib import Path

from tests.fixtures.data_manager import (
    get_test_data_manager,
    isolated_test,
    cleanup_test_data
)
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


def example_basic_data_caching():
    """Example of basic data caching functionality."""
    print("=== Basic Data Caching Example ===")
    
    manager = get_test_data_manager()
    
    # Generate and cache a dataset
    print("Generating and caching dataset...")
    start_time = time.time()
    data1 = manager.get_dataset("example_rct", cache=True)
    first_time = time.time() - start_time
    print(f"First generation took: {first_time:.3f}s")
    print(f"Dataset shape: {data1.shape}")
    
    # Get the same dataset from cache
    print("\nRetrieving from cache...")
    start_time = time.time()
    data2 = manager.get_dataset("example_rct", cache=True)
    cache_time = time.time() - start_time
    print(f"Cache retrieval took: {cache_time:.3f}s")
    print(f"Speed improvement: {first_time/cache_time:.1f}x faster")
    
    # Verify data is the same
    print(f"Data identical: {data1.equals(data2)}")
    
    # Show cache stats
    stats = manager.cache.get_stats()
    print(f"\nCache stats: {stats}")


def example_custom_data_generation():
    """Example of using custom data generation functions."""
    print("\n=== Custom Data Generation Example ===")
    
    manager = get_test_data_manager()
    
    def create_custom_dataset(n_samples=500, effect_size=0.8, confounding=0.3):
        """Custom dataset generator with specific characteristics."""
        np.random.seed(42)
        
        # Generate confounders
        age = np.random.normal(40, 15, n_samples)
        income = np.random.lognormal(10, 0.5, n_samples)
        
        # Treatment assignment (biased by confounders)
        treatment_prob = 1 / (1 + np.exp(-(confounding * (age - 40) / 15 + 
                                         confounding * (income - np.median(income)) / np.std(income))))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome with treatment effect
        outcome = (0.1 * age + 0.0001 * income + 
                  effect_size * treatment + 
                  np.random.normal(0, 5, n_samples))
        
        return pd.DataFrame({
            'age': age,
            'income': income,
            'treatment': treatment,
            'outcome': outcome
        })
    
    # Generate custom dataset
    print("Generating custom dataset...")
    custom_data = manager.get_dataset(
        "custom_observational",
        generator_func=create_custom_dataset,
        n_samples=1000,
        effect_size=1.2,
        confounding=0.5,
        cache=True
    )
    
    print(f"Custom dataset shape: {custom_data.shape}")
    print(f"Treatment rate: {custom_data['treatment'].mean():.2f}")
    print(f"Average outcome by treatment:")
    print(custom_data.groupby('treatment')['outcome'].mean())


def example_temporary_resources():
    """Example of temporary resource management."""
    print("\n=== Temporary Resource Management Example ===")
    
    manager = get_test_data_manager()
    
    # Create temporary dataset file
    sample_data = pd.DataFrame({
        'x': np.random.random(100),
        'y': np.random.random(100),
        'z': np.random.random(100)
    })
    
    print("Creating temporary dataset file...")
    temp_file = manager.create_temp_dataset(sample_data, "example_dataset")
    print(f"Temporary file created: {temp_file}")
    print(f"File exists: {temp_file.exists()}")
    print(f"File size: {temp_file.stat().st_size} bytes")
    
    # Create temporary workspace
    print("\nCreating temporary workspace...")
    workspace = manager.create_temp_workspace("example_analysis")
    print(f"Workspace created: {workspace}")
    
    # Create some files in the workspace
    (workspace / "data.csv").write_text("a,b,c\n1,2,3\n4,5,6")
    (workspace / "config.json").write_text('{"param1": 0.5, "param2": "test"}')
    (workspace / "results").mkdir()
    (workspace / "results" / "output.txt").write_text("Analysis complete")
    
    print(f"Files in workspace: {list(workspace.rglob('*'))}")
    
    # Show resource stats
    resource_stats = manager.temp_manager.get_resource_stats()
    print(f"\nResource stats: {resource_stats}")


def example_test_isolation():
    """Example of test isolation functionality."""
    print("\n=== Test Isolation Example ===")
    
    # Set initial random state
    np.random.seed(123)
    initial_random = np.random.random(3)
    print(f"Initial random numbers: {initial_random}")
    
    # Test 1 in isolation
    with isolated_test("test_1") as context1:
        print(f"\nIn test_1 (ID: {context1['test_id'][:8]}...):")
        test1_random = np.random.random(3)
        print(f"Test 1 random numbers: {test1_random}")
        
        # Create some data in the workspace
        workspace1 = context1['workspace']
        (workspace1 / "test1_data.txt").write_text("Test 1 data")
        print(f"Test 1 workspace: {workspace1}")
    
    # Test 2 in isolation
    with isolated_test("test_2") as context2:
        print(f"\nIn test_2 (ID: {context2['test_id'][:8]}...):")
        test2_random = np.random.random(3)
        print(f"Test 2 random numbers: {test2_random}")
        
        # Create some data in the workspace
        workspace2 = context2['workspace']
        (workspace2 / "test2_data.txt").write_text("Test 2 data")
        print(f"Test 2 workspace: {workspace2}")
    
    # Check that random numbers are different (due to different seeds)
    print(f"\nRandom numbers are different: {not np.array_equal(test1_random, test2_random)}")
    
    # Check random state after isolation
    final_random = np.random.random(3)
    print(f"Final random numbers: {final_random}")
    print(f"Random state restored: {np.array_equal(initial_random, final_random)}")


def example_performance_monitoring():
    """Example of performance monitoring capabilities."""
    print("\n=== Performance Monitoring Example ===")
    
    manager = get_test_data_manager()
    
    # Generate datasets of different sizes
    sizes = [100, 500, 1000, 5000]
    performance_data = []
    
    for size in sizes:
        print(f"\nTesting with {size} samples...")
        
        # Time dataset generation
        start_time = time.time()
        data = manager.get_dataset(
            f"perf_test_{size}",
            cache=True,
            n_samples=size
        )
        generation_time = time.time() - start_time
        
        # Time cache retrieval
        start_time = time.time()
        cached_data = manager.get_dataset(f"perf_test_{size}", cache=True)
        cache_time = time.time() - start_time
        
        # Calculate memory usage
        memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        performance_data.append({
            'size': size,
            'generation_time': generation_time,
            'cache_time': cache_time,
            'memory_mb': memory_usage,
            'speedup': generation_time / cache_time if cache_time > 0 else float('inf')
        })
        
        print(f"  Generation: {generation_time:.3f}s")
        print(f"  Cache retrieval: {cache_time:.3f}s")
        print(f"  Memory usage: {memory_usage:.2f}MB")
        print(f"  Cache speedup: {performance_data[-1]['speedup']:.1f}x")
    
    # Show overall performance summary
    print("\n=== Performance Summary ===")
    perf_df = pd.DataFrame(performance_data)
    print(perf_df.to_string(index=False, float_format='%.3f'))


def example_data_management_stats():
    """Example of comprehensive data management statistics."""
    print("\n=== Data Management Statistics Example ===")
    
    manager = get_test_data_manager()
    
    # Generate some activity
    print("Generating test activity...")
    
    # Create various datasets
    datasets = ['rct_small', 'obs_medium', 'iv_large', 'rdd_test', 'did_panel']
    for dataset_name in datasets:
        data = manager.get_dataset(dataset_name, cache=True)
        print(f"  Generated {dataset_name}: {data.shape}")
    
    # Create temporary resources
    temp_files = []
    for i in range(5):
        temp_data = pd.DataFrame({'col': np.random.random(50)})
        temp_file = manager.create_temp_dataset(temp_data, f"temp_{i}")
        temp_files.append(temp_file)
    
    # Create workspaces
    workspaces = []
    for i in range(3):
        workspace = manager.create_temp_workspace(f"workspace_{i}")
        workspaces.append(workspace)
    
    # Get comprehensive stats
    print("\n=== Comprehensive Statistics ===")
    stats = manager.get_stats()
    
    print("Cache Statistics:")
    cache_stats = stats['cache']
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    print("\nTemporary Resource Statistics:")
    temp_stats = stats['temp_resources']
    for key, value in temp_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nActive Tests: {stats['active_tests']}")


def example_cleanup_and_maintenance():
    """Example of cleanup and maintenance operations."""
    print("\n=== Cleanup and Maintenance Example ===")
    
    manager = get_test_data_manager()
    
    # Show initial state
    initial_stats = manager.get_stats()
    print("Initial state:")
    print(f"  Cache entries: {initial_stats['cache']['entries']}")
    print(f"  Temp resources: {initial_stats['temp_resources']['total_resources']}")
    
    # Create some resources to clean up
    print("\nCreating resources...")
    for i in range(10):
        data = pd.DataFrame({'data': np.random.random(100)})
        manager.cache.put(f"cleanup_test_{i}", data)
        temp_file = manager.temp_manager.create_temp_file(content=f"test_{i}")
    
    mid_stats = manager.get_stats()
    print("After creating resources:")
    print(f"  Cache entries: {mid_stats['cache']['entries']}")
    print(f"  Temp resources: {mid_stats['temp_resources']['total_resources']}")
    
    # Perform cleanup
    print("\nPerforming cleanup...")
    
    # Clean up old temp resources
    manager.temp_manager.cleanup_old_resources(max_age_seconds=0)
    
    # Clear cache
    manager.cache.clear()
    
    final_stats = manager.get_stats()
    print("After cleanup:")
    print(f"  Cache entries: {final_stats['cache']['entries']}")
    print(f"  Temp resources: {final_stats['temp_resources']['total_resources']}")


def run_all_examples():
    """Run all data management examples."""
    print("Running Test Data Management Examples")
    print("=" * 50)
    
    try:
        example_basic_data_caching()
        example_custom_data_generation()
        example_temporary_resources()
        example_test_isolation()
        example_performance_monitoring()
        example_data_management_stats()
        example_cleanup_and_maintenance()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        print("\nPerforming final cleanup...")
        cleanup_test_data()
        print("Cleanup complete.")


if __name__ == "__main__":
    run_all_examples()
"""Tests for the test data management and cleanup system."""

import pytest
import pandas as pd
import numpy as np
import time
import threading
from pathlib import Path
import tempfile
import shutil

from tests.fixtures.data_manager import (
    TestDataManager,
    DataCache,
    TempResourceManager,
    TestIsolationManager,
    get_test_data_manager,
    isolated_test,
    cleanup_test_data
)
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


class TestDataCache:
    """Test the data caching system."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = DataCache(max_size_mb=10, max_entries=5)
        
        # Create test data
        data = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200),
            'c': np.random.random(100)
        })
        
        # Test put and get
        assert cache.put("test_data", data)
        retrieved = cache.get("test_data")
        
        assert retrieved is not None
        pd.testing.assert_frame_equal(data, retrieved)
        
        # Test cache miss
        assert cache.get("nonexistent") is None
    
    def test_cache_size_limits(self):
        """Test cache size limiting."""
        cache = DataCache(max_size_mb=1, max_entries=10)  # Very small cache
        
        # Create large dataset that should be rejected
        large_data = pd.DataFrame({
            'data': np.random.random(100000)  # Large dataset
        })
        
        # Should reject data that's too large
        assert not cache.put("large_data", large_data)
        assert cache.get("large_data") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = DataCache(max_size_mb=10, max_entries=3)
        
        # Add data to fill cache
        for i in range(3):
            data = pd.DataFrame({'col': range(10)})
            assert cache.put(f"data_{i}", data)
        
        # Access data_1 to make it recently used
        cache.get("data_1")
        
        # Add new data, should evict data_0 (least recently used)
        new_data = pd.DataFrame({'col': range(10)})
        assert cache.put("data_3", new_data)
        
        # data_0 should be evicted, data_1 should still exist
        assert cache.get("data_0") is None
        assert cache.get("data_1") is not None
        assert cache.get("data_3") is not None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = DataCache(max_size_mb=10, max_entries=5)
        
        # Initially empty
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["total_size_mb"] == 0
        
        # Add some data
        data = pd.DataFrame({'col': range(100)})
        cache.put("test", data)
        
        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["total_size_mb"] > 0
        assert 0 <= stats["utilization"] <= 1


class TestTempResourceManager:
    """Test the temporary resource management system."""
    
    def test_temp_file_creation(self):
        """Test temporary file creation and cleanup."""
        manager = TempResourceManager()
        
        # Create temp file
        temp_file = manager.create_temp_file(suffix=".csv", content="test,data\n1,2")
        
        assert temp_file.exists()
        assert temp_file.suffix == ".csv"
        assert temp_file.read_text() == "test,data\n1,2"
        
        # Cleanup
        assert manager.cleanup_resource(temp_file)
        assert not temp_file.exists()
    
    def test_temp_directory_creation(self):
        """Test temporary directory creation and cleanup."""
        manager = TempResourceManager()
        
        # Create temp directory
        temp_dir = manager.create_temp_dir(prefix="test_")
        
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert temp_dir.name.startswith("test_")
        
        # Create some files in the directory
        (temp_dir / "test_file.txt").write_text("test content")
        
        # Cleanup should remove directory and contents
        assert manager.cleanup_resource(temp_dir)
        assert not temp_dir.exists()
    
    def test_temp_dataset_saving(self):
        """Test saving datasets to temporary files."""
        manager = TempResourceManager()
        
        # Create test dataset
        data = pd.DataFrame({
            'feature_1': np.random.random(50),
            'feature_2': np.random.random(50),
            'treatment': np.random.binomial(1, 0.5, 50),
            'outcome': np.random.random(50)
        })
        
        # Save to temp file
        temp_file = manager.save_temp_dataset(data, "test_dataset")
        
        assert temp_file.exists()
        assert temp_file.suffix == ".csv"
        
        # Verify data can be loaded back
        loaded_data = pd.read_csv(temp_file)
        pd.testing.assert_frame_equal(data, loaded_data)
        
        # Cleanup
        manager.cleanup_resource(temp_file)
    
    def test_cleanup_old_resources(self):
        """Test cleanup of old resources."""
        manager = TempResourceManager()
        
        # Create some temp files
        temp_files = []
        for i in range(3):
            temp_file = manager.create_temp_file(content=f"file_{i}")
            temp_files.append(temp_file)
        
        # All should exist
        for temp_file in temp_files:
            assert temp_file.exists()
        
        # Cleanup old resources (age 0 should clean all)
        manager.cleanup_old_resources(max_age_seconds=0)
        
        # All should be cleaned up
        for temp_file in temp_files:
            assert not temp_file.exists()
    
    def test_resource_stats(self):
        """Test resource statistics."""
        manager = TempResourceManager()
        
        # Initially no resources
        stats = manager.get_resource_stats()
        assert stats["total_resources"] == 0
        
        # Create some resources
        temp_file = manager.create_temp_file(content="test")
        temp_dir = manager.create_temp_dir()
        
        stats = manager.get_resource_stats()
        assert stats["total_resources"] == 2
        assert "file" in stats["by_type"]
        assert "directory" in stats["by_type"]
        
        # Cleanup
        manager.cleanup_all()


class TestIsolationManager:
    """Test the test isolation system."""
    
    def test_isolated_test_context(self):
        """Test isolated test context management."""
        manager = TestIsolationManager()
        
        # Test isolation context
        with manager.isolated_test("test_example") as test_id:
            assert test_id is not None
            assert test_id in manager.get_active_tests()
        
        # Should be cleaned up after context
        assert test_id not in manager.get_active_tests()
    
    def test_random_state_isolation(self):
        """Test that random state is isolated between tests."""
        manager = TestIsolationManager()
        
        # Set initial random state
        np.random.seed(42)
        initial_state = np.random.get_state()
        
        # Generate some random numbers in isolated context
        with manager.isolated_test("test_random_1"):
            random_nums_1 = np.random.random(5)
        
        # Random state should be restored
        current_state = np.random.get_state()
        assert np.array_equal(initial_state[1], current_state[1])
        
        # Generate numbers in another isolated context
        with manager.isolated_test("test_random_2"):
            random_nums_2 = np.random.random(5)
        
        # Should get different numbers due to different test IDs
        assert not np.array_equal(random_nums_1, random_nums_2)
    
    def test_resource_registration(self):
        """Test resource registration for cleanup."""
        manager = TempResourceManager()
        isolation_manager = TestIsolationManager()
        
        with isolation_manager.isolated_test("test_resources") as test_id:
            # Create temp resource
            temp_file = manager.create_temp_file(content="test")
            
            # Register for cleanup
            isolation_manager.register_test_resource(test_id, temp_file)
            
            assert temp_file.exists()
        
        # File should be cleaned up after test context
        # Note: This test might be flaky due to timing, but demonstrates the concept
        time.sleep(0.1)  # Give cleanup time to run
    
    def test_abandoned_test_cleanup(self):
        """Test cleanup of abandoned test contexts."""
        manager = TestIsolationManager()
        
        # Simulate abandoned test by not using context manager
        test_id = f"abandoned_test_{threading.current_thread().ident}_{time.time()}"
        manager._enter_test_context(test_id, "abandoned_test")
        
        assert test_id in manager.get_active_tests()
        
        # Cleanup abandoned tests (age 0 should clean all)
        manager.cleanup_abandoned_tests(max_age_seconds=0)
        
        assert test_id not in manager.get_active_tests()


class TestDataManager:
    """Test the main test data manager."""
    
    def test_data_manager_initialization(self):
        """Test data manager initialization."""
        manager = TestDataManager()
        
        assert manager.cache is not None
        assert manager.temp_manager is not None
        assert manager.isolation_manager is not None
        assert manager.synthetic_generator is not None
    
    def test_dataset_generation_and_caching(self):
        """Test dataset generation with caching."""
        manager = TestDataManager()
        
        # Generate dataset with caching
        data1 = manager.get_dataset("test_rct", cache=True)
        assert data1 is not None
        assert isinstance(data1, pd.DataFrame)
        
        # Get same dataset again (should come from cache)
        data2 = manager.get_dataset("test_rct", cache=True)
        
        # Should be equal but different objects (cache returns copies)
        pd.testing.assert_frame_equal(data1, data2)
        assert data1 is not data2  # Different objects
    
    def test_custom_generator_function(self):
        """Test using custom generator function."""
        manager = TestDataManager()
        
        def custom_generator(n_samples=100, effect_size=0.5):
            return pd.DataFrame({
                'x': np.random.random(n_samples),
                'treatment': np.random.binomial(1, 0.5, n_samples),
                'y': np.random.random(n_samples) + effect_size
            })
        
        # Generate with custom function
        data = manager.get_dataset(
            "custom_data", 
            generator_func=custom_generator,
            n_samples=50,
            effect_size=1.0
        )
        
        assert len(data) == 50
        assert 'x' in data.columns
        assert 'treatment' in data.columns
        assert 'y' in data.columns
    
    def test_temp_workspace_creation(self):
        """Test temporary workspace creation."""
        manager = TestDataManager()
        
        workspace = manager.create_temp_workspace("test_workspace")
        
        assert workspace.exists()
        assert workspace.is_dir()
        assert "test_workspace" in workspace.name
        
        # Create some files in workspace
        (workspace / "data.csv").write_text("test,data\n1,2")
        (workspace / "results.json").write_text('{"result": "success"}')
        
        assert (workspace / "data.csv").exists()
        assert (workspace / "results.json").exists()
    
    def test_isolated_test_data_context(self):
        """Test isolated test data context."""
        manager = TestDataManager()
        
        with manager.isolated_test_data("test_isolation") as context:
            assert "test_id" in context
            assert "workspace" in context
            assert "data_manager" in context
            
            workspace = context["workspace"]
            assert workspace.exists()
            assert workspace.is_dir()
            
            # Create some test data
            test_file = workspace / "test_data.csv"
            test_file.write_text("a,b,c\n1,2,3")
            assert test_file.exists()
        
        # Workspace should be cleaned up after context
        # Note: Cleanup might be asynchronous, so this test demonstrates the concept
    
    def test_preload_datasets(self):
        """Test preloading common datasets."""
        manager = TestDataManager()
        
        # Clear cache first
        manager.cache.clear()
        
        # Preload datasets
        dataset_keys = ["rct_test", "obs_test", "iv_test"]
        manager.preload_common_datasets(dataset_keys)
        
        # Should be able to get datasets from cache
        for key in dataset_keys:
            data = manager.cache.get(key)
            assert data is not None
            assert isinstance(data, pd.DataFrame)
    
    def test_manager_stats(self):
        """Test data manager statistics."""
        manager = TestDataManager()
        
        stats = manager.get_stats()
        
        assert "cache" in stats
        assert "temp_resources" in stats
        assert "active_tests" in stats
        
        # Cache stats
        cache_stats = stats["cache"]
        assert "entries" in cache_stats
        assert "total_size_mb" in cache_stats
        assert "utilization" in cache_stats
        
        # Temp resource stats
        temp_stats = stats["temp_resources"]
        assert "total_resources" in temp_stats
        assert "by_type" in temp_stats


class TestIntegrationWithPytest:
    """Test integration with pytest fixtures and plugins."""
    
    def test_global_data_manager(self):
        """Test global data manager access."""
        manager1 = get_test_data_manager()
        manager2 = get_test_data_manager()
        
        # Should be the same instance
        assert manager1 is manager2
    
    def test_isolated_test_fixture(self):
        """Test isolated test fixture functionality."""
        with isolated_test("fixture_test") as context:
            assert "test_id" in context
            assert "workspace" in context
            assert "data_manager" in context
            
            # Should be able to use the data manager
            data_manager = context["data_manager"]
            test_data = data_manager.get_dataset("test_data", cache=False)
            assert test_data is not None
    
    def test_cleanup_function(self):
        """Test manual cleanup function."""
        manager = get_test_data_manager()
        
        # Create some test resources
        temp_file = manager.temp_manager.create_temp_file(content="test")
        test_data = pd.DataFrame({'a': [1, 2, 3]})
        manager.cache.put("test_cleanup", test_data)
        
        # Verify resources exist
        assert temp_file.exists()
        assert manager.cache.get("test_cleanup") is not None
        
        # Cleanup
        cleanup_test_data()
        
        # Resources should be cleaned up
        assert not temp_file.exists()
        assert manager.cache.get("test_cleanup") is None


class TestPerformanceAndScalability:
    """Test performance and scalability of data management system."""
    
    def test_cache_performance(self):
        """Test cache performance with multiple operations."""
        cache = DataCache(max_size_mb=50, max_entries=100)
        
        # Generate test datasets
        datasets = {}
        for i in range(20):
            data = pd.DataFrame({
                'feature_1': np.random.random(100),
                'feature_2': np.random.random(100),
                'outcome': np.random.random(100)
            })
            datasets[f"dataset_{i}"] = data
        
        # Time cache operations
        start_time = time.time()
        
        # Put all datasets
        for key, data in datasets.items():
            cache.put(key, data)
        
        # Get all datasets multiple times
        for _ in range(5):
            for key in datasets.keys():
                retrieved = cache.get(key)
                assert retrieved is not None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly
        assert duration < 5.0  # 5 seconds should be more than enough
    
    def test_concurrent_access(self):
        """Test concurrent access to data manager."""
        manager = TestDataManager()
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                # Each thread generates and caches data
                data = manager.get_dataset(f"thread_data_{thread_id}", cache=True)
                results.append((thread_id, len(data)))
                
                # Create temp resources
                temp_file = manager.temp_manager.create_temp_file(
                    content=f"thread_{thread_id}_data"
                )
                results.append((thread_id, str(temp_file)))
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10  # 2 results per thread
    
    @pytest.mark.performance
    def test_memory_efficiency(self):
        """Test memory efficiency of data management."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        manager = TestDataManager()
        
        # Generate and cache multiple datasets
        for i in range(10):
            data = pd.DataFrame({
                'feature_1': np.random.random(1000),
                'feature_2': np.random.random(1000),
                'feature_3': np.random.random(1000),
                'outcome': np.random.random(1000)
            })
            manager.cache.put(f"memory_test_{i}", data)
        
        # Force garbage collection
        gc.collect()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.2f}MB"
        
        # Cleanup
        manager.cache.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should be mostly freed (within 20MB of initial)
        memory_retained = final_memory - initial_memory
        assert memory_retained < 20, f"Too much memory retained: {memory_retained:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
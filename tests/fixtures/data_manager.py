"""Advanced test data management and cleanup system for causal_agent tests."""

import os
import shutil
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import pickle
import hashlib
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from .test_config import get_test_config
from .synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached dataset entry."""
    key: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self.data.memory_usage(deep=True).sum()


@dataclass
class TempResource:
    """Represents a temporary resource that needs cleanup."""
    path: Path
    resource_type: str  # 'file', 'directory', 'dataset'
    created_at: float
    cleanup_func: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataCache:
    """Thread-safe LRU cache for test datasets with size limits."""
    
    def __init__(self, max_size_mb: int = 100, max_entries: int = 50):
        """Initialize the data cache."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_size = 0
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get dataset from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry.data.copy()  # Return copy to prevent modification
            return None
    
    def put(self, key: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put dataset in cache. Returns True if cached, False if rejected."""
        if metadata is None:
            metadata = {}
        
        # Calculate size
        data_size = data.memory_usage(deep=True).sum()
        
        # Check if data is too large for cache
        if data_size > self.max_size_bytes * 0.5:  # Don't cache if > 50% of total cache
            return False
        
        with self._lock:
            current_time = time.time()
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size -= old_entry.size_bytes
                del self._cache[key]
            
            # Make room if necessary
            while (len(self._cache) >= self.max_entries or 
                   self._total_size + data_size > self.max_size_bytes):
                if not self._evict_lru():
                    return False  # Can't make room
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                data=data.copy(),
                metadata=metadata,
                created_at=current_time,
                last_accessed=current_time,
                size_bytes=data_size
            )
            
            self._cache[key] = entry
            self._total_size += data_size
            return True
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        
        # Find LRU entry
        lru_key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].last_accessed)
        
        entry = self._cache[lru_key]
        self._total_size -= entry.size_bytes
        del self._cache[lru_key]
        return True
    
    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "total_size_mb": self._total_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self._total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
                "hit_rate": self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        if total_accesses == 0:
            return 0.0
        # This is a simplified hit rate calculation
        return min(1.0, total_accesses / (len(self._cache) * 2))


class TempResourceManager:
    """Manager for temporary resources with automatic cleanup."""
    
    def __init__(self):
        """Initialize the temp resource manager."""
        self._resources: Dict[str, TempResource] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown = False
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_all)
    
    def create_temp_file(self, suffix: str = ".csv", prefix: str = "test_", 
                        content: Optional[str] = None) -> Path:
        """Create a temporary file."""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, delete=False
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        if content:
            temp_path.write_text(content)
        
        resource = TempResource(
            path=temp_path,
            resource_type="file",
            created_at=time.time()
        )
        
        with self._lock:
            self._resources[str(temp_path)] = resource
        
        return temp_path
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        
        resource = TempResource(
            path=temp_dir,
            resource_type="directory", 
            created_at=time.time()
        )
        
        with self._lock:
            self._resources[str(temp_dir)] = resource
        
        return temp_dir
    
    def save_temp_dataset(self, data: pd.DataFrame, name: str = "dataset") -> Path:
        """Save dataset to temporary file."""
        temp_file = self.create_temp_file(suffix=".csv", prefix=f"{name}_")
        data.to_csv(temp_file, index=False)
        
        # Update resource metadata
        with self._lock:
            if str(temp_file) in self._resources:
                self._resources[str(temp_file)].resource_type = "dataset"
                self._resources[str(temp_file)].metadata = {
                    "dataset_name": name,
                    "shape": data.shape,
                    "columns": list(data.columns)
                }
        
        return temp_file
    
    def register_cleanup(self, path: Union[str, Path], 
                        cleanup_func: Optional[Callable] = None):
        """Register a path for cleanup."""
        path = Path(path)
        
        resource = TempResource(
            path=path,
            resource_type="custom",
            created_at=time.time(),
            cleanup_func=cleanup_func
        )
        
        with self._lock:
            self._resources[str(path)] = resource
    
    def cleanup_resource(self, path: Union[str, Path]) -> bool:
        """Clean up a specific resource."""
        path_str = str(path)
        
        with self._lock:
            if path_str not in self._resources:
                return False
            
            resource = self._resources[path_str]
            
            try:
                if resource.cleanup_func:
                    resource.cleanup_func()
                elif resource.path.exists():
                    if resource.path.is_file():
                        resource.path.unlink()
                    elif resource.path.is_dir():
                        shutil.rmtree(resource.path)
                
                del self._resources[path_str]
                return True
                
            except Exception as e:
                logger.warning(f"Failed to cleanup resource {path}: {e}")
                return False
    
    def cleanup_old_resources(self, max_age_seconds: float = 3600):
        """Clean up resources older than specified age."""
        current_time = time.time()
        to_cleanup = []
        
        with self._lock:
            for path_str, resource in self._resources.items():
                if current_time - resource.created_at > max_age_seconds:
                    to_cleanup.append(path_str)
        
        for path_str in to_cleanup:
            self.cleanup_resource(path_str)
    
    def cleanup_all(self):
        """Clean up all managed resources."""
        with self._lock:
            paths_to_cleanup = list(self._resources.keys())
        
        for path_str in paths_to_cleanup:
            self.cleanup_resource(path_str)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about managed resources."""
        with self._lock:
            stats = {
                "total_resources": len(self._resources),
                "by_type": {},
                "total_size_mb": 0
            }
            
            for resource in self._resources.values():
                resource_type = resource.resource_type
                stats["by_type"][resource_type] = stats["by_type"].get(resource_type, 0) + 1
                
                # Calculate size if possible
                try:
                    if resource.path.exists():
                        if resource.path.is_file():
                            stats["total_size_mb"] += resource.path.stat().st_size / (1024 * 1024)
                        elif resource.path.is_dir():
                            for file_path in resource.path.rglob("*"):
                                if file_path.is_file():
                                    stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                except Exception:
                    pass  # Ignore errors in size calculation
            
            return stats


class TestIsolationManager:
    """Manager for test isolation to prevent test interference."""
    
    def __init__(self):
        """Initialize the test isolation manager."""
        self._test_contexts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._active_tests: Set[str] = set()
    
    @contextmanager
    def isolated_test(self, test_name: str, cleanup_after: bool = True):
        """Context manager for isolated test execution."""
        test_id = f"{test_name}_{threading.current_thread().ident}_{time.time()}"
        
        try:
            self._enter_test_context(test_id, test_name)
            yield test_id
        finally:
            if cleanup_after:
                self._exit_test_context(test_id)
    
    def _enter_test_context(self, test_id: str, test_name: str):
        """Enter test context with isolation setup."""
        with self._lock:
            self._active_tests.add(test_id)
            self._test_contexts[test_id] = {
                "test_name": test_name,
                "start_time": time.time(),
                "temp_resources": [],
                "cached_data": [],
                "random_state": np.random.get_state(),
                "original_cwd": os.getcwd()
            }
        
        # Set deterministic random seed for test
        np.random.seed(hash(test_id) % (2**32))
    
    def _exit_test_context(self, test_id: str):
        """Exit test context with cleanup."""
        with self._lock:
            if test_id not in self._test_contexts:
                return
            
            context = self._test_contexts[test_id]
            
            # Restore random state
            try:
                np.random.set_state(context["random_state"])
            except Exception:
                pass  # Ignore if state restoration fails
            
            # Restore working directory
            try:
                os.chdir(context["original_cwd"])
            except Exception:
                pass
            
            # Clean up test-specific resources
            for resource_path in context.get("temp_resources", []):
                try:
                    if Path(resource_path).exists():
                        if Path(resource_path).is_file():
                            Path(resource_path).unlink()
                        else:
                            shutil.rmtree(resource_path)
                except Exception:
                    pass
            
            # Remove from active tests
            self._active_tests.discard(test_id)
            del self._test_contexts[test_id]
    
    def register_test_resource(self, test_id: str, resource_path: Union[str, Path]):
        """Register a resource for cleanup when test exits."""
        with self._lock:
            if test_id in self._test_contexts:
                self._test_contexts[test_id]["temp_resources"].append(str(resource_path))
    
    def get_active_tests(self) -> List[str]:
        """Get list of currently active test IDs."""
        with self._lock:
            return list(self._active_tests)
    
    def cleanup_abandoned_tests(self, max_age_seconds: float = 1800):
        """Clean up contexts from tests that didn't exit properly."""
        current_time = time.time()
        abandoned_tests = []
        
        with self._lock:
            for test_id, context in self._test_contexts.items():
                if current_time - context["start_time"] > max_age_seconds:
                    abandoned_tests.append(test_id)
        
        for test_id in abandoned_tests:
            logger.warning(f"Cleaning up abandoned test context: {test_id}")
            self._exit_test_context(test_id)


class TestDataManager:
    """Main test data management system combining caching, cleanup, and isolation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test data manager."""
        self.config = config or get_test_config().data
        
        # Initialize components
        cache_size_mb = getattr(self.config, 'max_cache_size_mb', 100)
        self.cache = DataCache(max_size_mb=cache_size_mb)
        self.temp_manager = TempResourceManager()
        self.isolation_manager = TestIsolationManager()
        
        # Data generators
        self.synthetic_generator = SyntheticDataGenerator()
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self.temp_manager.cleanup_old_resources(max_age_seconds=3600)
                    self.isolation_manager.cleanup_abandoned_tests(max_age_seconds=1800)
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get_dataset(self, dataset_key: str, generator_func: Optional[Callable] = None,
                   cache: bool = True, **generator_kwargs) -> pd.DataFrame:
        """Get dataset with caching and efficient loading."""
        # Try cache first
        if cache:
            cached_data = self.cache.get(dataset_key)
            if cached_data is not None:
                return cached_data
        
        # Generate or load data
        if generator_func:
            data = generator_func(**generator_kwargs)
        else:
            # Try to generate from key
            data = self._generate_from_key(dataset_key, **generator_kwargs)
        
        # Cache if requested
        if cache and data is not None:
            self.cache.put(dataset_key, data, metadata=generator_kwargs)
        
        return data
    
    def _generate_from_key(self, dataset_key: str, **kwargs) -> pd.DataFrame:
        """Generate dataset based on key pattern."""
        if "rct" in dataset_key.lower():
            return self.synthetic_generator.generate_rct_data()
        elif "observational" in dataset_key.lower() or "obs" in dataset_key.lower():
            return self.synthetic_generator.generate_observational_data()
        elif "iv" in dataset_key.lower() or "instrumental" in dataset_key.lower():
            return self.synthetic_generator.generate_iv_data()
        elif "rdd" in dataset_key.lower() or "discontinuity" in dataset_key.lower():
            return self.synthetic_generator.generate_rdd_data()
        elif "did" in dataset_key.lower() or "difference" in dataset_key.lower():
            return self.synthetic_generator.generate_did_data()
        else:
            # Default to observational
            return self.synthetic_generator.generate_observational_data()
    
    def create_temp_dataset(self, data: pd.DataFrame, name: str = "temp_dataset") -> Path:
        """Create temporary dataset file."""
        return self.temp_manager.save_temp_dataset(data, name)
    
    def create_temp_workspace(self, test_name: str) -> Path:
        """Create temporary workspace for test."""
        workspace = self.temp_manager.create_temp_dir(prefix=f"{test_name}_workspace_")
        return workspace
    
    @contextmanager
    def isolated_test_data(self, test_name: str, cleanup_after: bool = True):
        """Context manager for isolated test data management."""
        with self.isolation_manager.isolated_test(test_name, cleanup_after) as test_id:
            # Create test-specific workspace
            workspace = self.create_temp_workspace(test_name)
            self.isolation_manager.register_test_resource(test_id, workspace)
            
            yield {
                "test_id": test_id,
                "workspace": workspace,
                "data_manager": self
            }
    
    def preload_common_datasets(self, dataset_keys: List[str]):
        """Preload commonly used datasets into cache."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for key in dataset_keys:
                future = executor.submit(self.get_dataset, key, cache=True)
                futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    logger.warning(f"Failed to preload dataset: {e}")
    
    def cleanup_all(self):
        """Clean up all managed resources."""
        self.cache.clear()
        self.temp_manager.cleanup_all()
        self.isolation_manager.cleanup_abandoned_tests(max_age_seconds=0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about data management."""
        return {
            "cache": self.cache.get_stats(),
            "temp_resources": self.temp_manager.get_resource_stats(),
            "active_tests": len(self.isolation_manager.get_active_tests())
        }


# Global test data manager instance
_test_data_manager = None


def get_test_data_manager() -> TestDataManager:
    """Get the global test data manager instance."""
    global _test_data_manager
    if _test_data_manager is None:
        _test_data_manager = TestDataManager()
    return _test_data_manager


def setup_test_data_manager(config: Optional[Dict[str, Any]] = None) -> TestDataManager:
    """Setup test data manager with custom configuration."""
    global _test_data_manager
    _test_data_manager = TestDataManager(config)
    return _test_data_manager


# Convenience functions for common operations
def get_cached_dataset(key: str, generator_func: Optional[Callable] = None, **kwargs) -> pd.DataFrame:
    """Get dataset with caching."""
    return get_test_data_manager().get_dataset(key, generator_func, cache=True, **kwargs)


def create_temp_dataset_file(data: pd.DataFrame, name: str = "dataset") -> Path:
    """Create temporary dataset file."""
    return get_test_data_manager().create_temp_dataset(data, name)


@contextmanager
def isolated_test(test_name: str, cleanup_after: bool = True):
    """Context manager for isolated test execution."""
    with get_test_data_manager().isolated_test_data(test_name, cleanup_after) as context:
        yield context


def cleanup_test_data():
    """Clean up all test data."""
    manager = get_test_data_manager()
    manager.cleanup_all()
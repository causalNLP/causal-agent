"""Pytest plugin for test data management and cleanup integration."""

import pytest
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from .data_manager import (
    get_test_data_manager,
    TestDataManager,
    isolated_test,
    cleanup_test_data
)
from .test_config import get_test_config


logger = logging.getLogger(__name__)


class TestDataPlugin:
    """Pytest plugin for managing test data lifecycle."""
    
    def __init__(self):
        """Initialize the plugin."""
        self.data_manager = None
        self.session_start_time = None
        self.test_stats = {}
        self.failed_tests = []
    
    def pytest_configure(self, config):
        """Configure the plugin when pytest starts."""
        # Initialize data manager
        self.data_manager = get_test_data_manager()
        
        # Preload common datasets if configured
        test_config = get_test_config()
        if hasattr(test_config.data, 'preload_datasets'):
            common_datasets = getattr(test_config.data, 'preload_datasets', [])
            if common_datasets:
                logger.info(f"Preloading {len(common_datasets)} common datasets...")
                self.data_manager.preload_common_datasets(common_datasets)
    
    def pytest_sessionstart(self, session):
        """Called when test session starts."""
        self.session_start_time = time.time()
        logger.info("Test data management session started")
        
        # Log initial stats
        stats = self.data_manager.get_stats()
        logger.debug(f"Initial data manager stats: {stats}")
    
    def pytest_sessionfinish(self, session, exitstatus):
        """Called when test session finishes."""
        session_duration = time.time() - self.session_start_time
        
        # Log final stats
        final_stats = self.data_manager.get_stats()
        logger.info(f"Test session completed in {session_duration:.2f}s")
        logger.info(f"Final data manager stats: {final_stats}")
        
        # Generate summary report
        self._generate_session_report(session, exitstatus, session_duration, final_stats)
        
        # Cleanup if configured
        test_config = get_test_config()
        if getattr(test_config.data, 'cleanup_temp_data', True):
            logger.info("Cleaning up test data...")
            self.data_manager.cleanup_all()
    
    def pytest_runtest_setup(self, item):
        """Called before each test runs."""
        test_name = item.nodeid
        
        # Initialize test stats
        self.test_stats[test_name] = {
            "start_time": time.time(),
            "data_operations": 0,
            "cache_hits": 0,
            "temp_files_created": 0
        }
    
    def pytest_runtest_teardown(self, item, nextitem):
        """Called after each test completes."""
        test_name = item.nodeid
        
        if test_name in self.test_stats:
            self.test_stats[test_name]["end_time"] = time.time()
            self.test_stats[test_name]["duration"] = (
                self.test_stats[test_name]["end_time"] - 
                self.test_stats[test_name]["start_time"]
            )
        
        # Clean up test-specific resources if configured
        test_config = get_test_config()
        if getattr(test_config.data, 'cleanup_after_each_test', False):
            # Clean up old temp resources (but not too aggressively)
            self.data_manager.temp_manager.cleanup_old_resources(max_age_seconds=60)
    
    def pytest_runtest_makereport(self, item, call):
        """Called when test report is created."""
        if call.when == "call":  # Only for the actual test call, not setup/teardown
            test_name = item.nodeid
            
            if call.excinfo is not None:  # Test failed
                self.failed_tests.append({
                    "test_name": test_name,
                    "exception": str(call.excinfo.value),
                    "stats": self.test_stats.get(test_name, {})
                })
    
    def _generate_session_report(self, session, exitstatus, duration, final_stats):
        """Generate a summary report of the test session."""
        report = {
            "session_duration": duration,
            "exit_status": exitstatus,
            "data_manager_stats": final_stats,
            "test_count": len(self.test_stats),
            "failed_tests": len(self.failed_tests),
            "average_test_duration": self._calculate_average_test_duration(),
            "data_efficiency": self._calculate_data_efficiency()
        }
        
        # Save report if output directory is configured
        test_config = get_test_config()
        if hasattr(test_config, 'output_dir') and test_config.output_dir:
            report_path = Path(test_config.output_dir) / "test_data_report.json"
            try:
                import json
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Test data report saved to {report_path}")
            except Exception as e:
                logger.warning(f"Failed to save test data report: {e}")
    
    def _calculate_average_test_duration(self) -> float:
        """Calculate average test duration."""
        durations = [
            stats.get("duration", 0) 
            for stats in self.test_stats.values() 
            if "duration" in stats
        ]
        return sum(durations) / len(durations) if durations else 0.0
    
    def _calculate_data_efficiency(self) -> Dict[str, Any]:
        """Calculate data management efficiency metrics."""
        cache_stats = self.data_manager.cache.get_stats()
        
        return {
            "cache_utilization": cache_stats.get("utilization", 0),
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "memory_efficiency": cache_stats.get("total_size_mb", 0) / cache_stats.get("max_size_mb", 1)
        }


# Plugin instance
_plugin_instance = None


def pytest_configure(config):
    """Configure pytest with data management plugin."""
    global _plugin_instance
    _plugin_instance = TestDataPlugin()
    _plugin_instance.pytest_configure(config)
    
    # Register plugin hooks
    config.pluginmanager.register(_plugin_instance, "test_data_plugin")


def pytest_sessionstart(session):
    """Session start hook."""
    if _plugin_instance:
        _plugin_instance.pytest_sessionstart(session)


def pytest_sessionfinish(session, exitstatus):
    """Session finish hook."""
    if _plugin_instance:
        _plugin_instance.pytest_sessionfinish(session, exitstatus)


def pytest_runtest_setup(item):
    """Test setup hook."""
    if _plugin_instance:
        _plugin_instance.pytest_runtest_setup(item)


def pytest_runtest_teardown(item, nextitem):
    """Test teardown hook."""
    if _plugin_instance:
        _plugin_instance.pytest_runtest_teardown(item, nextitem)


def pytest_runtest_makereport(item, call):
    """Test report hook."""
    if _plugin_instance:
        _plugin_instance.pytest_runtest_makereport(item, call)


# Fixtures for test data management
@pytest.fixture
def test_data_manager():
    """Provide test data manager instance."""
    return get_test_data_manager()


@pytest.fixture
def isolated_test_data(request):
    """Provide isolated test data context."""
    test_name = request.node.name
    with isolated_test(test_name, cleanup_after=True) as context:
        yield context


@pytest.fixture
def temp_dataset_file():
    """Create temporary dataset file."""
    import pandas as pd
    import numpy as np
    
    # Create sample dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(0, 1, 100),
        'treatment': np.random.binomial(1, 0.5, 100),
        'outcome': np.random.normal(0, 1, 100)
    })
    
    manager = get_test_data_manager()
    temp_file = manager.create_temp_dataset(data, "fixture_dataset")
    
    yield temp_file
    
    # Cleanup is handled by the data manager


@pytest.fixture
def cached_synthetic_data():
    """Provide cached synthetic datasets."""
    manager = get_test_data_manager()
    
    datasets = {}
    dataset_configs = [
        ("small_rct", "rct_small"),
        ("medium_obs", "observational_medium"),
        ("iv_data", "iv_standard"),
        ("rdd_data", "rdd_standard")
    ]
    
    for key, dataset_type in dataset_configs:
        datasets[key] = manager.get_dataset(dataset_type, cache=True)
    
    return datasets


@pytest.fixture(scope="session")
def preloaded_datasets():
    """Session-scoped fixture for preloaded datasets."""
    manager = get_test_data_manager()
    
    # Preload common datasets at session start
    common_datasets = [
        "rct_standard",
        "observational_standard", 
        "iv_standard",
        "rdd_standard",
        "did_standard"
    ]
    
    manager.preload_common_datasets(common_datasets)
    
    return {key: manager.get_dataset(key, cache=True) for key in common_datasets}


@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage."""
    import psutil
    import time
    
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance metrics
    logger.info(f"Test performance: {duration:.3f}s, memory delta: {memory_delta:.2f}MB")
    
    # Check against thresholds
    test_config = get_test_config()
    max_duration = getattr(test_config.performance, 'max_execution_time_seconds', 10.0)
    max_memory = getattr(test_config.performance, 'max_memory_usage_mb', 100.0)
    
    if duration > max_duration:
        logger.warning(f"Test exceeded time limit: {duration:.3f}s > {max_duration}s")
    
    if memory_delta > max_memory:
        logger.warning(f"Test exceeded memory limit: {memory_delta:.2f}MB > {max_memory}MB")


# Markers for data management
def pytest_configure(config):
    """Add custom markers for data management."""
    config.addinivalue_line(
        "markers", "data_intensive: mark test as data intensive (may need special handling)"
    )
    config.addinivalue_line(
        "markers", "no_cleanup: mark test to skip automatic cleanup"
    )
    config.addinivalue_line(
        "markers", "preload_data: mark test to preload specific datasets"
    )
    config.addinivalue_line(
        "markers", "isolated: mark test to run in complete isolation"
    )


# Utility functions for test data management
def get_test_workspace(test_name: str) -> Path:
    """Get or create test-specific workspace."""
    manager = get_test_data_manager()
    return manager.create_temp_workspace(test_name)


def cache_dataset(key: str, data, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Cache a dataset for reuse."""
    manager = get_test_data_manager()
    return manager.cache.put(key, data, metadata)


def get_cached_dataset(key: str):
    """Get dataset from cache."""
    manager = get_test_data_manager()
    return manager.cache.get(key)


def cleanup_test_resources():
    """Manual cleanup of test resources."""
    cleanup_test_data()
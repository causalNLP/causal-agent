"""Base classes and utilities for performance testing of causal inference methods."""

import time
import psutil
import gc
import tracemalloc
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pytest

from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig, DatasetType


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    execution_time: float
    peak_memory_mb: float
    memory_growth_mb: float
    cpu_percent: float
    method_name: str
    dataset_size: int
    dataset_type: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different dataset sizes."""
    max_execution_time_seconds: Dict[int, float] = field(default_factory=lambda: {
        100: 5.0,      # Small datasets: 5 seconds
        500: 15.0,     # Medium datasets: 15 seconds
        1000: 30.0,    # Large datasets: 30 seconds
        5000: 120.0,   # Very large datasets: 2 minutes
        10000: 300.0   # Huge datasets: 5 minutes
    })
    max_memory_mb: Dict[int, float] = field(default_factory=lambda: {
        100: 50.0,     # Small datasets: 50 MB
        500: 100.0,    # Medium datasets: 100 MB
        1000: 200.0,   # Large datasets: 200 MB
        5000: 500.0,   # Very large datasets: 500 MB
        10000: 1000.0  # Huge datasets: 1 GB
    })
    max_cpu_percent: float = 95.0  # Maximum CPU usage percentage


class PerformanceProfiler:
    """Utility class for profiling performance of causal inference methods."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.process = psutil.Process()
        self.start_memory = None
        self.start_time = None
        
    def start_profiling(self):
        """Start performance profiling."""
        # Force garbage collection before starting
        gc.collect()
        
        # Start memory tracing
        tracemalloc.start()
        
        # Record initial state
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
        
    def stop_profiling(self, method_name: str, dataset_size: int, dataset_type: str) -> PerformanceMetrics:
        """Stop profiling and return metrics."""
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        # Get memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - self.start_memory
        
        # Get peak memory from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024  # MB
        tracemalloc.stop()
        
        # Get CPU usage (average over last second)
        cpu_percent = self.process.cpu_percent()
        
        return PerformanceMetrics(
            execution_time=execution_time,
            peak_memory_mb=peak_memory_mb,
            memory_growth_mb=memory_growth,
            cpu_percent=cpu_percent,
            method_name=method_name,
            dataset_size=dataset_size,
            dataset_type=dataset_type
        )


class PerformanceTestBase(ABC):
    """Base class for performance tests of causal inference methods."""
    
    def __init__(self):
        """Initialize the performance test."""
        self.profiler = PerformanceProfiler()
        self.thresholds = PerformanceThresholds()
        self.data_generator = SyntheticDataGenerator()
        
    @abstractmethod
    def get_method_instance(self):
        """Get an instance of the causal method to test."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of the method being tested."""
        pass
    
    def generate_test_datasets(self, sizes: List[int]) -> Dict[str, List[pd.DataFrame]]:
        """Generate test datasets of various sizes for performance testing."""
        datasets = {}
        
        # Generate datasets for each supported type
        dataset_types = [
            DatasetType.RCT,
            DatasetType.OBSERVATIONAL,
            DatasetType.INSTRUMENTAL_VARIABLE,
            DatasetType.REGRESSION_DISCONTINUITY,
            DatasetType.DIFFERENCE_IN_DIFFERENCES
        ]
        
        for dataset_type in dataset_types:
            datasets[dataset_type.value] = []
            
            for size in sizes:
                config = SyntheticDataConfig(
                    n_samples=size,
                    random_seed=42 + size  # Different seed for each size
                )
                self.data_generator.config = config
                
                dataset = self.data_generator.generate_dataset(dataset_type)
                datasets[dataset_type.value].append(dataset)
                
        return datasets
    
    def run_performance_test(self, dataset: pd.DataFrame, 
                           treatment: str = 'treatment',
                           outcome: str = 'outcome',
                           covariates: Optional[List[str]] = None) -> PerformanceMetrics:
        """Run performance test on a single dataset."""
        if covariates is None:
            # Auto-detect covariates (all columns except treatment and outcome)
            covariates = [col for col in dataset.columns 
                         if col not in [treatment, outcome]]
        
        method = self.get_method_instance()
        dataset_type = dataset.attrs.get('dataset_type', 'unknown')
        
        # Start profiling
        self.profiler.start_profiling()
        
        try:
            # Run the method
            result = method.estimate_effect(dataset, treatment, outcome, covariates)
            
            # Stop profiling and get metrics
            metrics = self.profiler.stop_profiling(
                self.get_method_name(),
                len(dataset),
                dataset_type
            )
            
            # Add method-specific metrics
            if isinstance(result, dict):
                metrics.additional_metrics.update({
                    'effect_estimate': result.get('effect_estimate'),
                    'confidence_interval': result.get('confidence_interval'),
                    'p_value': result.get('p_value')
                })
            
            return metrics
            
        except Exception as e:
            # Still stop profiling even if method fails
            metrics = self.profiler.stop_profiling(
                self.get_method_name(),
                len(dataset),
                dataset_type
            )
            metrics.additional_metrics['error'] = str(e)
            return metrics
    
    def validate_performance(self, metrics: PerformanceMetrics) -> Dict[str, bool]:
        """Validate performance metrics against thresholds."""
        validation_results = {}
        
        # Find appropriate threshold based on dataset size
        size_thresholds = sorted(self.thresholds.max_execution_time_seconds.keys())
        threshold_size = next((s for s in size_thresholds if s >= metrics.dataset_size), 
                             size_thresholds[-1])
        
        # Validate execution time
        max_time = self.thresholds.max_execution_time_seconds[threshold_size]
        validation_results['execution_time_ok'] = metrics.execution_time <= max_time
        
        # Validate memory usage
        max_memory = self.thresholds.max_memory_mb[threshold_size]
        validation_results['memory_usage_ok'] = metrics.peak_memory_mb <= max_memory
        
        # Validate CPU usage
        validation_results['cpu_usage_ok'] = metrics.cpu_percent <= self.thresholds.max_cpu_percent
        
        return validation_results
    
    def run_scalability_test(self, sizes: List[int] = None) -> List[PerformanceMetrics]:
        """Run scalability test across different dataset sizes."""
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        datasets = self.generate_test_datasets(sizes)
        all_metrics = []
        
        # Test each dataset type and size combination
        for dataset_type, dataset_list in datasets.items():
            for i, dataset in enumerate(dataset_list):
                try:
                    metrics = self.run_performance_test(dataset)
                    all_metrics.append(metrics)
                except Exception as e:
                    # Create error metrics
                    error_metrics = PerformanceMetrics(
                        execution_time=-1,
                        peak_memory_mb=-1,
                        memory_growth_mb=-1,
                        cpu_percent=-1,
                        method_name=self.get_method_name(),
                        dataset_size=sizes[i],
                        dataset_type=dataset_type,
                        additional_metrics={'error': str(e)}
                    )
                    all_metrics.append(error_metrics)
        
        return all_metrics
    
    def run_memory_stress_test(self, base_size: int = 1000, 
                              iterations: int = 10) -> List[PerformanceMetrics]:
        """Run memory stress test with repeated executions."""
        config = SyntheticDataConfig(n_samples=base_size, random_seed=42)
        self.data_generator.config = config
        
        # Use observational data for stress testing
        dataset = self.data_generator.generate_observational_data()
        
        metrics_list = []
        
        for i in range(iterations):
            metrics = self.run_performance_test(dataset)
            metrics.additional_metrics['iteration'] = i
            metrics_list.append(metrics)
            
            # Force garbage collection between iterations
            gc.collect()
        
        return metrics_list


class MethodWrapper:
    """Wrapper to make functions work like method classes."""
    
    def __init__(self, method_function):
        """Initialize with a method function."""
        self.method_function = method_function
    
    def estimate_effect(self, dataset: pd.DataFrame, treatment: str, 
                       outcome: str, covariates: List[str]) -> Dict[str, Any]:
        """Call the wrapped function with appropriate parameters."""
        return self.method_function(dataset, treatment, outcome, covariates)


def benchmark_method_performance(method_class_or_function, method_name: str, 
                               sizes: List[int] = None) -> Dict[str, Any]:
    """Benchmark performance of a causal inference method."""
    
    class MethodPerformanceTest(PerformanceTestBase):
        def get_method_instance(self):
            # Handle both classes and functions
            if hasattr(method_class_or_function, '__call__') and hasattr(method_class_or_function, '__name__'):
                # It's a function, wrap it
                return MethodWrapper(method_class_or_function)
            else:
                # It's a class, instantiate it
                return method_class_or_function()
        
        def get_method_name(self) -> str:
            return method_name
    
    test = MethodPerformanceTest()
    
    # Run scalability test
    scalability_metrics = test.run_scalability_test(sizes)
    
    # Run memory stress test
    memory_metrics = test.run_memory_stress_test()
    
    # Analyze results
    results = {
        'method_name': method_name,
        'scalability_metrics': scalability_metrics,
        'memory_stress_metrics': memory_metrics,
        'summary': _analyze_performance_results(scalability_metrics, memory_metrics)
    }
    
    return results


def _analyze_performance_results(scalability_metrics: List[PerformanceMetrics],
                               memory_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
    """Analyze performance results and generate summary statistics."""
    
    # Filter out error metrics
    valid_scalability = [m for m in scalability_metrics if m.execution_time > 0]
    valid_memory = [m for m in memory_metrics if m.execution_time > 0]
    
    if not valid_scalability:
        return {'error': 'No valid scalability metrics'}
    
    summary = {
        'scalability_analysis': {
            'min_execution_time': min(m.execution_time for m in valid_scalability),
            'max_execution_time': max(m.execution_time for m in valid_scalability),
            'avg_execution_time': np.mean([m.execution_time for m in valid_scalability]),
            'min_memory_usage': min(m.peak_memory_mb for m in valid_scalability),
            'max_memory_usage': max(m.peak_memory_mb for m in valid_scalability),
            'avg_memory_usage': np.mean([m.peak_memory_mb for m in valid_scalability]),
        }
    }
    
    if valid_memory:
        # Check for memory leaks (increasing memory usage over iterations)
        memory_usage_trend = [m.peak_memory_mb for m in valid_memory]
        memory_growth = memory_usage_trend[-1] - memory_usage_trend[0] if len(memory_usage_trend) > 1 else 0
        
        summary['memory_analysis'] = {
            'memory_growth_over_iterations': memory_growth,
            'potential_memory_leak': memory_growth > 10.0,  # More than 10MB growth
            'avg_memory_per_iteration': np.mean(memory_usage_trend),
            'memory_stability': np.std(memory_usage_trend) < 5.0  # Low variance indicates stability
        }
    
    return summary
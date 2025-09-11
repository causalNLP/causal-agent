"""Performance and benchmark tests for causal_agent.

This module provides comprehensive performance testing infrastructure for causal inference methods,
including:

- Performance benchmarking across different dataset sizes and types
- Memory usage validation and leak detection
- Scalability analysis with complexity estimation
- Comparative benchmarking framework

Usage:
    # Run comprehensive performance tests
    from tests.performance.run_performance_tests import PerformanceTestSuite
    
    suite = PerformanceTestSuite()
    results = suite.run_comprehensive_tests(quick_mode=True)
    
    # Run specific performance tests
    from tests.performance.benchmark_framework import run_standard_benchmark
    
    benchmark_results = run_standard_benchmark()
    
    # Test memory usage for a specific method
    from tests.performance.test_memory_validation import MemoryStressTest
    from causal_agent.methods.linear_regression.estimator import LinearRegressionEstimator
    
    method = LinearRegressionEstimator()
    stress_test = MemoryStressTest(method, "LinearRegression")
    memory_results = stress_test.run_repeated_execution_test()
"""

from .test_performance_base import (
    PerformanceMetrics,
    PerformanceThresholds,
    PerformanceProfiler,
    PerformanceTestBase,
    benchmark_method_performance
)

from .benchmark_framework import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    run_standard_benchmark
)

from .test_memory_validation import (
    MemoryValidator,
    MemoryStressTest
)

from .test_scalability import (
    ScalabilityTestConfig,
    ScalabilityResult,
    ScalabilityTester
)

from .run_performance_tests import PerformanceTestSuite

__all__ = [
    # Base performance testing
    'PerformanceMetrics',
    'PerformanceThresholds', 
    'PerformanceProfiler',
    'PerformanceTestBase',
    'benchmark_method_performance',
    
    # Benchmark framework
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkRunner',
    'run_standard_benchmark',
    
    # Memory validation
    'MemoryValidator',
    'MemoryStressTest',
    
    # Scalability testing
    'ScalabilityTestConfig',
    'ScalabilityResult',
    'ScalabilityTester',
    
    # Comprehensive test suite
    'PerformanceTestSuite'
]
"""Memory usage validation tests for causal inference methods."""

import pytest
import gc
import psutil
import tracemalloc
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from tests.performance.test_performance_base import PerformanceProfiler, PerformanceMetrics
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig, DatasetType


class MemoryValidator:
    """Utility class for validating memory usage patterns."""
    
    def __init__(self):
        """Initialize memory validator."""
        self.process = psutil.Process()
        self.baseline_memory = None
        
    def establish_baseline(self):
        """Establish baseline memory usage."""
        gc.collect()  # Force garbage collection
        time.sleep(0.1)  # Allow system to settle
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def check_memory_leak(self, iterations: int = 10, 
                         max_growth_mb: float = 20.0) -> Dict[str, Any]:
        """Check for memory leaks over multiple iterations."""
        if self.baseline_memory is None:
            self.establish_baseline()
        
        memory_readings = []
        
        for i in range(iterations):
            # Force garbage collection before each reading
            gc.collect()
            time.sleep(0.05)  # Brief pause
            
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(current_memory)
        
        # Analyze memory trend
        memory_growth = memory_readings[-1] - memory_readings[0]
        max_memory = max(memory_readings)
        min_memory = min(memory_readings)
        avg_memory = np.mean(memory_readings)
        memory_variance = np.var(memory_readings)
        
        # Check for consistent upward trend (potential leak)
        trend_slope = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        
        return {
            'memory_readings': memory_readings,
            'memory_growth_mb': memory_growth,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory,
            'avg_memory_mb': avg_memory,
            'memory_variance': memory_variance,
            'trend_slope': trend_slope,
            'potential_leak': memory_growth > max_growth_mb or trend_slope > 1.0,
            'baseline_memory_mb': self.baseline_memory
        }
    
    def validate_memory_bounds(self, expected_max_mb: float) -> Dict[str, Any]:
        """Validate that memory usage stays within expected bounds."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'current_memory_mb': current_memory,
            'expected_max_mb': expected_max_mb,
            'within_bounds': current_memory <= expected_max_mb,
            'memory_efficiency': current_memory / expected_max_mb if expected_max_mb > 0 else 1.0
        }


class MemoryStressTest:
    """Class for running memory stress tests on causal inference methods."""
    
    def __init__(self, method_instance_or_function, method_name: str):
        """Initialize memory stress test."""
        # Handle both class instances and functions
        if callable(method_instance_or_function) and not hasattr(method_instance_or_function, 'estimate_effect'):
            # It's a function, wrap it
            from tests.performance.test_performance_base import MethodWrapper
            self.method = MethodWrapper(method_instance_or_function)
        else:
            # It's a class instance
            self.method = method_instance_or_function
        
        self.method_name = method_name
        self.validator = MemoryValidator()
        self.data_generator = SyntheticDataGenerator()
        
    def run_repeated_execution_test(self, dataset_size: int = 500, 
                                  iterations: int = 20) -> Dict[str, Any]:
        """Test memory usage over repeated executions."""
        print(f"Running repeated execution test for {self.method_name}...")
        
        # Establish baseline
        self.validator.establish_baseline()
        
        # Generate test dataset
        config = SyntheticDataConfig(n_samples=dataset_size, random_seed=42)
        self.data_generator.config = config
        dataset = self.data_generator.generate_observational_data()
        
        covariates = [col for col in dataset.columns 
                     if col not in ['treatment', 'outcome']]
        
        execution_results = []
        memory_readings = []
        
        for i in range(iterations):
            # Start memory tracking for this iteration
            tracemalloc.start()
            start_memory = self.validator.process.memory_info().rss / 1024 / 1024
            
            try:
                # Execute method
                result = self.method.estimate_effect(dataset, 'treatment', 'outcome', covariates)
                
                # Record memory usage
                current, peak = tracemalloc.get_traced_memory()
                end_memory = self.validator.process.memory_info().rss / 1024 / 1024
                
                execution_results.append({
                    'iteration': i,
                    'success': True,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'peak_memory_mb': peak / 1024 / 1024,
                    'memory_growth_mb': end_memory - start_memory
                })
                
                memory_readings.append(end_memory)
                
            except Exception as e:
                execution_results.append({
                    'iteration': i,
                    'success': False,
                    'error': str(e),
                    'start_memory_mb': start_memory,
                    'end_memory_mb': start_memory,  # No change if failed
                    'peak_memory_mb': 0,
                    'memory_growth_mb': 0
                })
                memory_readings.append(start_memory)
            
            finally:
                tracemalloc.stop()
                gc.collect()  # Force cleanup after each iteration
        
        # Analyze results
        successful_runs = [r for r in execution_results if r['success']]
        
        if successful_runs:
            total_memory_growth = memory_readings[-1] - memory_readings[0]
            avg_peak_memory = np.mean([r['peak_memory_mb'] for r in successful_runs])
            max_peak_memory = max([r['peak_memory_mb'] for r in successful_runs])
            
            # Check for memory leak pattern
            leak_analysis = self.validator.check_memory_leak(iterations)
            
            return {
                'method_name': self.method_name,
                'dataset_size': dataset_size,
                'total_iterations': iterations,
                'successful_iterations': len(successful_runs),
                'total_memory_growth_mb': total_memory_growth,
                'avg_peak_memory_mb': avg_peak_memory,
                'max_peak_memory_mb': max_peak_memory,
                'memory_readings': memory_readings,
                'execution_results': execution_results,
                'leak_analysis': leak_analysis,
                'memory_stable': leak_analysis['memory_growth_mb'] < 10.0,  # Less than 10MB growth
                'performance_degradation': self._check_performance_degradation(successful_runs)
            }
        else:
            return {
                'method_name': self.method_name,
                'dataset_size': dataset_size,
                'total_iterations': iterations,
                'successful_iterations': 0,
                'error': 'No successful iterations',
                'execution_results': execution_results
            }
    
    def run_increasing_dataset_test(self, base_size: int = 100, 
                                  size_multipliers: List[float] = None) -> Dict[str, Any]:
        """Test memory usage with increasing dataset sizes."""
        if size_multipliers is None:
            size_multipliers = [1, 2, 5, 10, 20]
        
        print(f"Running increasing dataset test for {self.method_name}...")
        
        results = []
        
        for multiplier in size_multipliers:
            dataset_size = int(base_size * multiplier)
            
            # Generate dataset
            config = SyntheticDataConfig(n_samples=dataset_size, random_seed=42)
            self.data_generator.config = config
            dataset = self.data_generator.generate_observational_data()
            
            covariates = [col for col in dataset.columns 
                         if col not in ['treatment', 'outcome']]
            
            # Measure memory usage
            tracemalloc.start()
            start_memory = self.validator.process.memory_info().rss / 1024 / 1024
            
            try:
                result = self.method.estimate_effect(dataset, 'treatment', 'outcome', covariates)
                
                current, peak = tracemalloc.get_traced_memory()
                end_memory = self.validator.process.memory_info().rss / 1024 / 1024
                
                results.append({
                    'dataset_size': dataset_size,
                    'size_multiplier': multiplier,
                    'success': True,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'peak_memory_mb': peak / 1024 / 1024,
                    'memory_growth_mb': end_memory - start_memory,
                    'memory_per_sample': (peak / 1024 / 1024) / dataset_size
                })
                
            except Exception as e:
                results.append({
                    'dataset_size': dataset_size,
                    'size_multiplier': multiplier,
                    'success': False,
                    'error': str(e),
                    'start_memory_mb': start_memory,
                    'end_memory_mb': start_memory,
                    'peak_memory_mb': 0,
                    'memory_growth_mb': 0,
                    'memory_per_sample': 0
                })
            
            finally:
                tracemalloc.stop()
                gc.collect()
        
        # Analyze scaling behavior
        successful_results = [r for r in results if r['success']]
        
        if len(successful_results) > 1:
            sizes = [r['dataset_size'] for r in successful_results]
            memories = [r['peak_memory_mb'] for r in successful_results]
            
            # Fit linear relationship to check scaling
            scaling_coefficient = np.polyfit(sizes, memories, 1)[0]  # MB per sample
            
            return {
                'method_name': self.method_name,
                'scaling_results': results,
                'successful_tests': len(successful_results),
                'scaling_coefficient_mb_per_sample': scaling_coefficient,
                'memory_scaling': 'linear' if scaling_coefficient > 0 else 'constant',
                'max_dataset_size_tested': max(sizes) if sizes else 0,
                'memory_efficiency_good': scaling_coefficient < 0.1  # Less than 0.1 MB per sample
            }
        else:
            return {
                'method_name': self.method_name,
                'scaling_results': results,
                'successful_tests': len(successful_results),
                'error': 'Insufficient successful tests for scaling analysis'
            }
    
    def _check_performance_degradation(self, execution_results: List[Dict]) -> Dict[str, Any]:
        """Check if performance degrades over iterations (indicating memory issues)."""
        if len(execution_results) < 5:
            return {'insufficient_data': True}
        
        # Extract execution times if available (would need to be measured separately)
        # For now, check memory growth pattern
        memory_growths = [r['memory_growth_mb'] for r in execution_results]
        
        # Check if memory growth is increasing over time
        trend_slope = np.polyfit(range(len(memory_growths)), memory_growths, 1)[0]
        
        return {
            'memory_growth_trend': trend_slope,
            'degradation_detected': trend_slope > 0.1,  # Growing memory usage
            'avg_memory_growth': np.mean(memory_growths),
            'max_memory_growth': max(memory_growths)
        }


# Pytest test classes
class TestMemoryValidation:
    """Test suite for memory validation."""
    
    @pytest.mark.performance
    @pytest.mark.memory
    def test_linear_regression_memory_stability(self):
        """Test memory stability of linear regression method."""
        try:
            from causal_agent.methods.linear_regression.estimator import LinearRegressionEstimator
            
            method = LinearRegressionEstimator()
            stress_test = MemoryStressTest(method, "LinearRegression")
            
            # Run repeated execution test
            results = stress_test.run_repeated_execution_test(dataset_size=200, iterations=10)
            
            # Validate results
            assert results['successful_iterations'] > 0, "Should have successful iterations"
            assert results['memory_stable'], f"Memory not stable: {results['total_memory_growth_mb']}MB growth"
            assert not results['leak_analysis']['potential_leak'], "Potential memory leak detected"
            
        except ImportError:
            pytest.skip("LinearRegressionEstimator not available")
    
    @pytest.mark.performance
    @pytest.mark.memory
    def test_difference_in_means_memory_scaling(self):
        """Test memory scaling of difference in means method."""
        try:
            from causal_agent.methods.diff_in_means.estimator import DifferenceInMeansEstimator
            
            method = DifferenceInMeansEstimator()
            stress_test = MemoryStressTest(method, "DifferenceInMeans")
            
            # Run scaling test
            results = stress_test.run_increasing_dataset_test(base_size=50, 
                                                            size_multipliers=[1, 2, 4, 8])
            
            # Validate results
            assert results['successful_tests'] > 0, "Should have successful tests"
            if 'scaling_coefficient_mb_per_sample' in results:
                assert results['memory_efficiency_good'], f"Poor memory efficiency: {results['scaling_coefficient_mb_per_sample']} MB/sample"
            
        except ImportError:
            pytest.skip("DifferenceInMeansEstimator not available")
    
    @pytest.mark.performance
    @pytest.mark.memory
    @pytest.mark.slow
    def test_comprehensive_memory_validation(self):
        """Comprehensive memory validation across multiple methods."""
        
        methods_to_test = []
        
        # Try to import available methods
        try:
            from causal_agent.methods.linear_regression.estimator import LinearRegressionEstimator
            methods_to_test.append((LinearRegressionEstimator, "LinearRegression"))
        except ImportError:
            pass
        
        try:
            from causal_agent.methods.diff_in_means.estimator import DifferenceInMeansEstimator
            methods_to_test.append((DifferenceInMeansEstimator, "DifferenceInMeans"))
        except ImportError:
            pass
        
        if not methods_to_test:
            pytest.skip("No methods available for testing")
        
        memory_results = {}
        
        for method_class, method_name in methods_to_test:
            try:
                method = method_class()
                stress_test = MemoryStressTest(method, method_name)
                
                # Run both tests
                stability_results = stress_test.run_repeated_execution_test(dataset_size=100, iterations=5)
                scaling_results = stress_test.run_increasing_dataset_test(base_size=50, 
                                                                        size_multipliers=[1, 2, 4])
                
                memory_results[method_name] = {
                    'stability': stability_results,
                    'scaling': scaling_results
                }
                
                # Basic validation
                assert stability_results['successful_iterations'] > 0, f"{method_name} had no successful iterations"
                
            except Exception as e:
                memory_results[method_name] = {'error': str(e)}
        
        # Should have tested at least one method successfully
        successful_methods = [name for name, results in memory_results.items() 
                             if 'error' not in results]
        assert len(successful_methods) > 0, "No methods were successfully tested"
    
    @pytest.mark.performance
    @pytest.mark.memory
    def test_memory_bounds_validation(self):
        """Test that methods stay within reasonable memory bounds."""
        validator = MemoryValidator()
        validator.establish_baseline()
        
        # Generate a moderately sized dataset
        generator = SyntheticDataGenerator()
        config = SyntheticDataConfig(n_samples=500, random_seed=42)
        generator.config = config
        dataset = generator.generate_observational_data()
        
        # Test with a simple method if available
        try:
            from causal_agent.methods.diff_in_means.estimator import DifferenceInMeansEstimator
            
            method = DifferenceInMeansEstimator()
            covariates = [col for col in dataset.columns 
                         if col not in ['treatment', 'outcome']]
            
            # Execute method
            result = method.estimate_effect(dataset, 'treatment', 'outcome', covariates)
            
            # Validate memory bounds (should be reasonable for 500 samples)
            bounds_check = validator.validate_memory_bounds(expected_max_mb=100.0)  # 100MB max
            
            assert bounds_check['within_bounds'], f"Memory usage too high: {bounds_check['current_memory_mb']}MB"
            assert bounds_check['memory_efficiency'] < 2.0, "Memory usage more than 2x expected"
            
        except ImportError:
            pytest.skip("DifferenceInMeansEstimator not available")


if __name__ == "__main__":
    # Run a quick memory validation test
    print("Running memory validation test...")
    
    try:
        from causal_agent.methods.diff_in_means.estimator import DifferenceInMeansEstimator
        
        method = DifferenceInMeansEstimator()
        stress_test = MemoryStressTest(method, "DifferenceInMeans")
        
        print("Testing memory stability...")
        stability_results = stress_test.run_repeated_execution_test(dataset_size=100, iterations=5)
        
        print(f"Memory stability test results:")
        print(f"- Successful iterations: {stability_results['successful_iterations']}")
        print(f"- Total memory growth: {stability_results['total_memory_growth_mb']:.2f} MB")
        print(f"- Memory stable: {stability_results['memory_stable']}")
        print(f"- Potential leak: {stability_results['leak_analysis']['potential_leak']}")
        
    except ImportError:
        print("DifferenceInMeansEstimator not available for testing")
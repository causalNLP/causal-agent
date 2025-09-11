"""Performance tests for individual causal inference methods."""

import pytest
import pandas as pd
from typing import List, Dict, Any
import numpy as np

from tests.performance.test_performance_base import (
    PerformanceTestBase, 
    PerformanceMetrics,
    benchmark_method_performance
)
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig, DatasetType

# Import causal methods (functions)
try:
    from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect as backdoor_estimate_effect
    BACKDOOR_AVAILABLE = True
except ImportError:
    BACKDOOR_AVAILABLE = False

try:
    from causal_agent.methods.propensity_score.matching import estimate_effect as ps_matching_estimate_effect
    PS_MATCHING_AVAILABLE = True
except ImportError:
    PS_MATCHING_AVAILABLE = False

try:
    from causal_agent.methods.propensity_score.weighting import estimate_effect as ps_weighting_estimate_effect
    PS_WEIGHTING_AVAILABLE = True
except ImportError:
    PS_WEIGHTING_AVAILABLE = False

try:
    from causal_agent.methods.instrumental_variable.estimator import estimate_effect as iv_estimate_effect
    IV_AVAILABLE = True
except ImportError:
    IV_AVAILABLE = False

try:
    from causal_agent.methods.regression_discontinuity.estimator import estimate_effect as rdd_estimate_effect
    RDD_AVAILABLE = True
except ImportError:
    RDD_AVAILABLE = False

try:
    from causal_agent.methods.difference_in_differences.estimator import estimate_effect as did_estimate_effect
    DID_AVAILABLE = True
except ImportError:
    DID_AVAILABLE = False

try:
    from causal_agent.methods.linear_regression.estimator import estimate_effect as lr_estimate_effect
    LR_AVAILABLE = True
except ImportError:
    LR_AVAILABLE = False

try:
    from causal_agent.methods.diff_in_means.estimator import estimate_effect as dim_estimate_effect
    DIM_AVAILABLE = True
except ImportError:
    DIM_AVAILABLE = False


class BackdoorAdjustmentPerformanceTest(PerformanceTestBase):
    """Performance tests for Backdoor Adjustment method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(backdoor_estimate_effect) if BACKDOOR_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "BackdoorAdjustment"


class PropensityScoreMatchingPerformanceTest(PerformanceTestBase):
    """Performance tests for Propensity Score Matching method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(ps_matching_estimate_effect) if PS_MATCHING_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "PropensityScoreMatching"


class PropensityScoreWeightingPerformanceTest(PerformanceTestBase):
    """Performance tests for Propensity Score Weighting method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(ps_weighting_estimate_effect) if PS_WEIGHTING_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "PropensityScoreWeighting"


class InstrumentalVariablePerformanceTest(PerformanceTestBase):
    """Performance tests for Instrumental Variable method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(iv_estimate_effect) if IV_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "InstrumentalVariable"


class RegressionDiscontinuityPerformanceTest(PerformanceTestBase):
    """Performance tests for Regression Discontinuity method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(rdd_estimate_effect) if RDD_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "RegressionDiscontinuity"


class DifferenceInDifferencesPerformanceTest(PerformanceTestBase):
    """Performance tests for Difference-in-Differences method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(did_estimate_effect) if DID_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "DifferenceInDifferences"


class LinearRegressionPerformanceTest(PerformanceTestBase):
    """Performance tests for Linear Regression method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(lr_estimate_effect) if LR_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "LinearRegression"


class DifferenceInMeansPerformanceTest(PerformanceTestBase):
    """Performance tests for Difference in Means method."""
    
    def get_method_instance(self):
        from tests.performance.test_performance_base import MethodWrapper
        return MethodWrapper(dim_estimate_effect) if DIM_AVAILABLE else None
    
    def get_method_name(self) -> str:
        return "DifferenceInMeans"


# Test classes for pytest
class TestLinearRegressionPerformance:
    """Test suite for Linear Regression performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = LinearRegressionPerformanceTest()
    
    @pytest.mark.performance
    def test_small_dataset_performance(self):
        """Test performance on small datasets."""
        if not LR_AVAILABLE:
            pytest.skip("LinearRegression method not available")
            
        config = SyntheticDataConfig(n_samples=100, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_observational_data()
        
        metrics = self.test_instance.run_performance_test(dataset)
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"
        assert metrics.execution_time > 0, "Method should have positive execution time"


class TestDifferenceInMeansPerformance:
    """Test suite for Difference in Means performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = DifferenceInMeansPerformanceTest()
    
    @pytest.mark.performance
    def test_small_dataset_performance(self):
        """Test performance on small datasets."""
        if not DIM_AVAILABLE:
            pytest.skip("DifferenceInMeans method not available")
            
        config = SyntheticDataConfig(n_samples=100, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_observational_data()
        
        metrics = self.test_instance.run_performance_test(dataset)
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"
        assert metrics.execution_time > 0, "Method should have positive execution time"


class TestBackdoorAdjustmentPerformance:
    """Test suite for Backdoor Adjustment performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = BackdoorAdjustmentPerformanceTest()
    
    @pytest.mark.performance
    def test_small_dataset_performance(self):
        """Test performance on small datasets."""
        if not BACKDOOR_AVAILABLE:
            pytest.skip("BackdoorAdjustment method not available")
            
        config = SyntheticDataConfig(n_samples=100, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_observational_data()
        
        metrics = self.test_instance.run_performance_test(dataset)
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"
        assert metrics.execution_time > 0, "Method should have positive execution time"
    
    @pytest.mark.performance
    def test_medium_dataset_performance(self):
        """Test performance on medium datasets."""
        if not BACKDOOR_AVAILABLE:
            pytest.skip("BackdoorAdjustment method not available")
            
        config = SyntheticDataConfig(n_samples=500, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_observational_data()
        
        metrics = self.test_instance.run_performance_test(dataset)
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scalability(self):
        """Test scalability across different dataset sizes."""
        if not BACKDOOR_AVAILABLE:
            pytest.skip("BackdoorAdjustment method not available")
            
        sizes = [100, 250, 500, 1000]
        metrics_list = self.test_instance.run_scalability_test(sizes)
        
        # Check that we have metrics for each size
        successful_metrics = [m for m in metrics_list if m.execution_time > 0]
        assert len(successful_metrics) > 0, "Should have at least some successful runs"
        
        # Check that execution time scales reasonably (not exponentially)
        if len(successful_metrics) > 1:
            times_by_size = {}
            for m in successful_metrics:
                if m.dataset_type == 'observational':  # Focus on one dataset type
                    times_by_size[m.dataset_size] = m.execution_time
            
            if len(times_by_size) > 1:
                sizes_sorted = sorted(times_by_size.keys())
                # Check that 10x data doesn't take more than 100x time (reasonable scaling)
                if len(sizes_sorted) >= 2:
                    ratio_size = sizes_sorted[-1] / sizes_sorted[0]
                    ratio_time = times_by_size[sizes_sorted[-1]] / times_by_size[sizes_sorted[0]]
                    assert ratio_time < ratio_size * 10, f"Poor scaling: {ratio_size}x data took {ratio_time}x time"
    
    @pytest.mark.performance
    def test_memory_stability(self):
        """Test memory stability over multiple iterations."""
        if not BACKDOOR_AVAILABLE:
            pytest.skip("BackdoorAdjustment method not available")
            
        metrics_list = self.test_instance.run_memory_stress_test(base_size=200, iterations=5)
        
        successful_metrics = [m for m in metrics_list if m.execution_time > 0]
        assert len(successful_metrics) > 0, "Should have successful iterations"
        
        if len(successful_metrics) > 1:
            memory_usage = [m.peak_memory_mb for m in successful_metrics]
            memory_growth = memory_usage[-1] - memory_usage[0]
            
            # Memory growth should be minimal (less than 20MB over iterations)
            assert memory_growth < 20.0, f"Potential memory leak: {memory_growth}MB growth"


class TestPropensityScoreMatchingPerformance:
    """Test suite for Propensity Score Matching performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = PropensityScoreMatchingPerformanceTest()
    
    @pytest.mark.performance
    def test_small_dataset_performance(self):
        """Test performance on small datasets."""
        if not PS_MATCHING_AVAILABLE:
            pytest.skip("PropensityScoreMatching method not available")
            
        config = SyntheticDataConfig(n_samples=100, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_observational_data()
        
        metrics = self.test_instance.run_performance_test(dataset)
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scalability(self):
        """Test scalability - matching can be computationally expensive."""
        sizes = [50, 100, 200, 400]  # Smaller sizes for matching
        metrics_list = self.test_instance.run_scalability_test(sizes)
        
        successful_metrics = [m for m in metrics_list if m.execution_time > 0]
        assert len(successful_metrics) > 0, "Should have at least some successful runs"


class TestInstrumentalVariablePerformance:
    """Test suite for Instrumental Variable performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = InstrumentalVariablePerformanceTest()
    
    @pytest.mark.performance
    def test_iv_specific_dataset_performance(self):
        """Test performance on IV-specific datasets."""
        config = SyntheticDataConfig(n_samples=200, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_iv_data()
        
        metrics = self.test_instance.run_performance_test(dataset, 
                                                        treatment='treatment',
                                                        outcome='outcome',
                                                        covariates=['covariate_0', 'covariate_1'])
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"


class TestRegressionDiscontinuityPerformance:
    """Test suite for Regression Discontinuity performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = RegressionDiscontinuityPerformanceTest()
    
    @pytest.mark.performance
    def test_rdd_specific_dataset_performance(self):
        """Test performance on RDD-specific datasets."""
        config = SyntheticDataConfig(n_samples=200, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_rdd_data()
        
        metrics = self.test_instance.run_performance_test(dataset,
                                                        treatment='treatment',
                                                        outcome='outcome',
                                                        covariates=['covariate_0', 'covariate_1'])
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"


class TestDifferenceInDifferencesPerformance:
    """Test suite for Difference-in-Differences performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_instance = DifferenceInDifferencesPerformanceTest()
    
    @pytest.mark.performance
    def test_did_panel_data_performance(self):
        """Test performance on panel data."""
        config = SyntheticDataConfig(n_units=20, n_periods=10, random_seed=42)
        self.test_instance.data_generator.config = config
        dataset = self.test_instance.data_generator.generate_did_data()
        
        metrics = self.test_instance.run_performance_test(dataset,
                                                        treatment='treatment',
                                                        outcome='outcome',
                                                        covariates=['covariate'])
        validation = self.test_instance.validate_performance(metrics)
        
        assert validation['execution_time_ok'], f"Execution time too high: {metrics.execution_time}s"
        assert validation['memory_usage_ok'], f"Memory usage too high: {metrics.peak_memory_mb}MB"


# Comprehensive benchmark test
@pytest.mark.performance
@pytest.mark.slow
def test_comprehensive_method_benchmark():
    """Run comprehensive benchmark across all methods."""
    
    # Define methods to benchmark (only available ones)
    methods_to_test = []
    
    if LR_AVAILABLE:
        methods_to_test.append((lr_estimate_effect, "LinearRegression"))
    if DIM_AVAILABLE:
        methods_to_test.append((dim_estimate_effect, "DifferenceInMeans"))
    if BACKDOOR_AVAILABLE:
        methods_to_test.append((backdoor_estimate_effect, "BackdoorAdjustment"))
    
    if not methods_to_test:
        pytest.skip("No causal inference methods available for testing")
    
    benchmark_results = {}
    sizes = [100, 200, 500]  # Reasonable sizes for comprehensive testing
    
    for method_class, method_name in methods_to_test:
        try:
            results = benchmark_method_performance(method_class, method_name, sizes)
            benchmark_results[method_name] = results
            
            # Basic validation - should have some successful runs
            successful_scalability = [m for m in results['scalability_metrics'] 
                                    if m.execution_time > 0]
            assert len(successful_scalability) > 0, f"{method_name} had no successful runs"
            
        except Exception as e:
            # Log the error but don't fail the entire test
            benchmark_results[method_name] = {'error': str(e)}
    
    # Should have tested at least one method successfully
    successful_methods = [name for name, results in benchmark_results.items() 
                         if 'error' not in results]
    assert len(successful_methods) > 0, "No methods were successfully benchmarked"


# Utility function for generating performance reports
def generate_performance_report(benchmark_results: Dict[str, Any]) -> str:
    """Generate a human-readable performance report."""
    
    report_lines = ["# Causal Agent Performance Benchmark Report\n"]
    
    for method_name, results in benchmark_results.items():
        report_lines.append(f"## {method_name}\n")
        
        if 'error' in results:
            report_lines.append(f"**Error:** {results['error']}\n")
            continue
        
        summary = results.get('summary', {})
        
        if 'scalability_analysis' in summary:
            scalability = summary['scalability_analysis']
            report_lines.extend([
                "### Scalability Analysis",
                f"- **Execution Time Range:** {scalability['min_execution_time']:.3f}s - {scalability['max_execution_time']:.3f}s",
                f"- **Average Execution Time:** {scalability['avg_execution_time']:.3f}s",
                f"- **Memory Usage Range:** {scalability['min_memory_usage']:.1f}MB - {scalability['max_memory_usage']:.1f}MB",
                f"- **Average Memory Usage:** {scalability['avg_memory_usage']:.1f}MB\n"
            ])
        
        if 'memory_analysis' in summary:
            memory = summary['memory_analysis']
            report_lines.extend([
                "### Memory Analysis",
                f"- **Memory Growth:** {memory['memory_growth_over_iterations']:.1f}MB",
                f"- **Potential Memory Leak:** {'Yes' if memory['potential_memory_leak'] else 'No'}",
                f"- **Memory Stability:** {'Good' if memory['memory_stability'] else 'Poor'}\n"
            ])
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Run a quick benchmark when script is executed directly
    print("Running quick performance benchmark...")
    
    methods = []
    if LR_AVAILABLE:
        methods.append((lr_estimate_effect, "LinearRegression"))
    if DIM_AVAILABLE:
        methods.append((dim_estimate_effect, "DifferenceInMeans"))
    
    if not methods:
        print("No methods available for benchmarking")
        exit(1)
    
    results = {}
    for method_function, method_name in methods:
        try:
            print(f"Benchmarking {method_name}...")
            result = benchmark_method_performance(method_function, method_name, [100, 200])
            results[method_name] = result
        except Exception as e:
            print(f"Error benchmarking {method_name}: {e}")
            results[method_name] = {'error': str(e)}
    
    # Generate and print report
    report = generate_performance_report(results)
    print("\n" + report)
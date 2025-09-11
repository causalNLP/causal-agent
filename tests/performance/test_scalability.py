"""Scalability tests for causal inference methods with varying dataset sizes."""

import pytest
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

from tests.performance.test_performance_base import PerformanceProfiler, PerformanceMetrics
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig, DatasetType


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability tests."""
    base_size: int = 100
    size_multipliers: List[float] = None
    dataset_types: List[DatasetType] = None
    iterations_per_size: int = 3
    max_execution_time_seconds: float = 300.0  # 5 minutes max
    max_memory_mb: float = 1000.0  # 1GB max
    
    def __post_init__(self):
        if self.size_multipliers is None:
            self.size_multipliers = [1, 2, 5, 10, 20, 50]
        if self.dataset_types is None:
            self.dataset_types = [DatasetType.OBSERVATIONAL, DatasetType.RCT]


@dataclass
class ScalabilityResult:
    """Result of a scalability test."""
    method_name: str
    dataset_size: int
    dataset_type: str
    execution_time: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    iteration: int = 0


class ScalabilityTester:
    """Class for running scalability tests on causal inference methods."""
    
    def __init__(self, method_instance_or_function, method_name: str, config: ScalabilityTestConfig = None):
        """Initialize scalability tester."""
        # Handle both class instances and functions
        if callable(method_instance_or_function) and not hasattr(method_instance_or_function, 'estimate_effect'):
            # It's a function, wrap it
            from tests.performance.test_performance_base import MethodWrapper
            self.method = MethodWrapper(method_instance_or_function)
        else:
            # It's a class instance
            self.method = method_instance_or_function
        
        self.method_name = method_name
        self.config = config or ScalabilityTestConfig()
        self.profiler = PerformanceProfiler()
        self.data_generator = SyntheticDataGenerator()
        self.results: List[ScalabilityResult] = []
    
    def run_scalability_test(self) -> List[ScalabilityResult]:
        """Run comprehensive scalability test."""
        print(f"Running scalability test for {self.method_name}...")
        
        self.results = []
        
        for multiplier in self.config.size_multipliers:
            dataset_size = int(self.config.base_size * multiplier)
            
            print(f"  Testing dataset size: {dataset_size}")
            
            for dataset_type in self.config.dataset_types:
                for iteration in range(self.config.iterations_per_size):
                    result = self._test_single_configuration(
                        dataset_size, dataset_type, iteration
                    )
                    self.results.append(result)
                    
                    # Stop if we hit resource limits
                    if not result.success and "timeout" in (result.error_message or "").lower():
                        print(f"    Stopping due to timeout at size {dataset_size}")
                        return self.results
                    
                    if result.memory_usage_mb > self.config.max_memory_mb:
                        print(f"    Stopping due to memory limit at size {dataset_size}")
                        return self.results
        
        return self.results
    
    def _test_single_configuration(self, dataset_size: int, dataset_type: DatasetType, 
                                 iteration: int) -> ScalabilityResult:
        """Test a single configuration (size, type, iteration)."""
        
        try:
            # Generate dataset
            config = SyntheticDataConfig(
                n_samples=dataset_size, 
                random_seed=42 + iteration
            )
            self.data_generator.config = config
            dataset = self._generate_dataset(dataset_type)
            
            # Determine columns
            treatment_col = 'treatment'
            outcome_col = 'outcome'
            covariate_cols = [col for col in dataset.columns 
                            if col not in [treatment_col, outcome_col]]
            
            # Special handling for different dataset types
            if dataset_type == DatasetType.INSTRUMENTAL_VARIABLE:
                covariate_cols = [col for col in covariate_cols if col != 'instrument']
            elif dataset_type == DatasetType.REGRESSION_DISCONTINUITY:
                covariate_cols = [col for col in covariate_cols if col != 'running_var']
            elif dataset_type == DatasetType.DIFFERENCE_IN_DIFFERENCES:
                covariate_cols = [col for col in covariate_cols 
                                if col not in ['unit', 'period', 'treated_unit', 'post_treatment']]
            
            # Start profiling
            self.profiler.start_profiling()
            start_time = time.time()
            
            # Execute method with timeout
            result = self._execute_with_timeout(
                dataset, treatment_col, outcome_col, covariate_cols
            )
            
            execution_time = time.time() - start_time
            
            # Stop profiling
            metrics = self.profiler.stop_profiling(
                self.method_name, dataset_size, dataset_type.value
            )
            
            return ScalabilityResult(
                method_name=self.method_name,
                dataset_size=dataset_size,
                dataset_type=dataset_type.value,
                execution_time=execution_time,
                memory_usage_mb=metrics.peak_memory_mb,
                success=True,
                iteration=iteration
            )
            
        except TimeoutError:
            return ScalabilityResult(
                method_name=self.method_name,
                dataset_size=dataset_size,
                dataset_type=dataset_type.value,
                execution_time=self.config.max_execution_time_seconds,
                memory_usage_mb=0,
                success=False,
                error_message="Execution timeout",
                iteration=iteration
            )
        except Exception as e:
            return ScalabilityResult(
                method_name=self.method_name,
                dataset_size=dataset_size,
                dataset_type=dataset_type.value,
                execution_time=0,
                memory_usage_mb=0,
                success=False,
                error_message=str(e),
                iteration=iteration
            )
    
    def _generate_dataset(self, dataset_type: DatasetType) -> pd.DataFrame:
        """Generate dataset of specified type."""
        return self.data_generator.generate_dataset(dataset_type)
    
    def _execute_with_timeout(self, dataset: pd.DataFrame, treatment: str, 
                            outcome: str, covariates: List[str]):
        """Execute method with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Method execution timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.max_execution_time_seconds))
        
        try:
            result = self.method.estimate_effect(dataset, treatment, outcome, covariates)
            return result
        finally:
            signal.alarm(0)  # Cancel timeout
    
    def analyze_scalability(self) -> Dict[str, Any]:
        """Analyze scalability results."""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        # Group by dataset type
        analysis = {}
        
        for dataset_type in set(r.dataset_type for r in successful_results):
            type_results = [r for r in successful_results if r.dataset_type == dataset_type]
            
            # Calculate average metrics per size
            size_metrics = {}
            for size in set(r.dataset_size for r in type_results):
                size_results = [r for r in type_results if r.dataset_size == size]
                
                size_metrics[size] = {
                    'avg_execution_time': np.mean([r.execution_time for r in size_results]),
                    'avg_memory_usage': np.mean([r.memory_usage_mb for r in size_results]),
                    'std_execution_time': np.std([r.execution_time for r in size_results]),
                    'std_memory_usage': np.std([r.memory_usage_mb for r in size_results]),
                    'success_rate': len(size_results) / self.config.iterations_per_size
                }
            
            # Analyze scaling behavior
            sizes = sorted(size_metrics.keys())
            times = [size_metrics[s]['avg_execution_time'] for s in sizes]
            memories = [size_metrics[s]['avg_memory_usage'] for s in sizes]
            
            # Fit polynomial to determine complexity
            time_complexity = self._analyze_complexity(sizes, times)
            memory_complexity = self._analyze_complexity(sizes, memories)
            
            analysis[dataset_type] = {
                'size_metrics': size_metrics,
                'time_complexity': time_complexity,
                'memory_complexity': memory_complexity,
                'max_successful_size': max(sizes),
                'scalability_rating': self._rate_scalability(time_complexity, memory_complexity)
            }
        
        return analysis
    
    def _analyze_complexity(self, sizes: List[int], values: List[float]) -> Dict[str, Any]:
        """Analyze computational complexity from size vs. performance data."""
        if len(sizes) < 3:
            return {'insufficient_data': True}
        
        # Try different polynomial fits
        complexities = {}
        
        # Linear: O(n)
        linear_coeff = np.polyfit(sizes, values, 1)
        linear_r2 = self._calculate_r_squared(sizes, values, linear_coeff, degree=1)
        complexities['linear'] = {'coefficients': linear_coeff, 'r_squared': linear_r2}
        
        # Quadratic: O(n^2)
        if len(sizes) >= 3:
            quad_coeff = np.polyfit(sizes, values, 2)
            quad_r2 = self._calculate_r_squared(sizes, values, quad_coeff, degree=2)
            complexities['quadratic'] = {'coefficients': quad_coeff, 'r_squared': quad_r2}
        
        # Log-linear: O(n log n)
        log_sizes = [s * np.log(s) for s in sizes]
        loglinear_coeff = np.polyfit(log_sizes, values, 1)
        loglinear_r2 = self._calculate_r_squared(log_sizes, values, loglinear_coeff, degree=1)
        complexities['log_linear'] = {'coefficients': loglinear_coeff, 'r_squared': loglinear_r2}
        
        # Determine best fit
        best_fit = max(complexities.items(), key=lambda x: x[1]['r_squared'])
        
        return {
            'best_fit_complexity': best_fit[0],
            'best_fit_r_squared': best_fit[1]['r_squared'],
            'all_fits': complexities,
            'scaling_factor': best_fit[1]['coefficients'][0] if len(best_fit[1]['coefficients']) > 0 else 0
        }
    
    def _calculate_r_squared(self, x_values: List[float], y_values: List[float], 
                           coefficients: np.ndarray, degree: int) -> float:
        """Calculate R-squared for polynomial fit."""
        if degree == 1:
            y_pred = coefficients[0] * np.array(x_values) + coefficients[1]
        elif degree == 2:
            y_pred = (coefficients[0] * np.array(x_values)**2 + 
                     coefficients[1] * np.array(x_values) + coefficients[2])
        else:
            return 0.0
        
        ss_res = np.sum((np.array(y_values) - y_pred) ** 2)
        ss_tot = np.sum((np.array(y_values) - np.mean(y_values)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def _rate_scalability(self, time_complexity: Dict, memory_complexity: Dict) -> str:
        """Rate the scalability of the method."""
        if 'insufficient_data' in time_complexity:
            return 'insufficient_data'
        
        time_fit = time_complexity['best_fit_complexity']
        memory_fit = memory_complexity['best_fit_complexity']
        
        # Rate based on complexity
        if time_fit == 'linear' and memory_fit == 'linear':
            return 'excellent'
        elif time_fit in ['linear', 'log_linear'] and memory_fit in ['linear', 'log_linear']:
            return 'good'
        elif time_fit == 'quadratic' or memory_fit == 'quadratic':
            return 'fair'
        else:
            return 'poor'
    
    def generate_scalability_plots(self, output_dir: str = "scalability_plots"):
        """Generate scalability visualization plots."""
        if not self.results:
            print("No results to plot")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Create plots for each dataset type
        for dataset_type in set(r.dataset_type for r in successful_results):
            type_results = [r for r in successful_results if r.dataset_type == dataset_type]
            
            # Group by size and calculate averages
            size_groups = {}
            for result in type_results:
                if result.dataset_size not in size_groups:
                    size_groups[result.dataset_size] = {'times': [], 'memories': []}
                size_groups[result.dataset_size]['times'].append(result.execution_time)
                size_groups[result.dataset_size]['memories'].append(result.memory_usage_mb)
            
            sizes = sorted(size_groups.keys())
            avg_times = [np.mean(size_groups[s]['times']) for s in sizes]
            avg_memories = [np.mean(size_groups[s]['memories']) for s in sizes]
            std_times = [np.std(size_groups[s]['times']) for s in sizes]
            std_memories = [np.std(size_groups[s]['memories']) for s in sizes]
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Execution time plot
            ax1.errorbar(sizes, avg_times, yerr=std_times, marker='o', capsize=5)
            ax1.set_xlabel('Dataset Size')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title(f'{self.method_name} - Execution Time Scalability\n({dataset_type})')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            
            # Memory usage plot
            ax2.errorbar(sizes, avg_memories, yerr=std_memories, marker='s', capsize=5, color='orange')
            ax2.set_xlabel('Dataset Size')
            ax2.set_ylabel('Peak Memory Usage (MB)')
            ax2.set_title(f'{self.method_name} - Memory Usage Scalability\n({dataset_type})')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{self.method_name}_{dataset_type}_scalability.png"
            plot_path = Path(output_dir) / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved scalability plot: {plot_path}")


# Pytest test classes
class TestScalability:
    """Test suite for scalability testing."""
    
    @pytest.mark.performance
    @pytest.mark.scalability
    def test_linear_regression_scalability(self):
        """Test scalability of linear regression method."""
        try:
            from causal_agent.methods.linear_regression.estimator import LinearRegressionEstimator
            
            method = LinearRegressionEstimator()
            config = ScalabilityTestConfig(
                base_size=50,
                size_multipliers=[1, 2, 4, 8],
                iterations_per_size=2
            )
            
            tester = ScalabilityTester(method, "LinearRegression", config)
            results = tester.run_scalability_test()
            
            # Validate results
            successful_results = [r for r in results if r.success]
            assert len(successful_results) > 0, "Should have successful results"
            
            # Analyze scalability
            analysis = tester.analyze_scalability()
            assert 'error' not in analysis, f"Analysis failed: {analysis.get('error')}"
            
            # Check that method scales reasonably
            for dataset_type, type_analysis in analysis.items():
                scalability_rating = type_analysis['scalability_rating']
                assert scalability_rating in ['excellent', 'good', 'fair'], \
                    f"Poor scalability rating: {scalability_rating}"
            
        except ImportError:
            pytest.skip("LinearRegressionEstimator not available")
    
    @pytest.mark.performance
    @pytest.mark.scalability
    def test_difference_in_means_scalability(self):
        """Test scalability of difference in means method."""
        try:
            from causal_agent.methods.diff_in_means.estimator import DifferenceInMeansEstimator
            
            method = DifferenceInMeansEstimator()
            config = ScalabilityTestConfig(
                base_size=100,
                size_multipliers=[1, 2, 5, 10],
                iterations_per_size=2
            )
            
            tester = ScalabilityTester(method, "DifferenceInMeans", config)
            results = tester.run_scalability_test()
            
            # Validate results
            successful_results = [r for r in results if r.success]
            assert len(successful_results) > 0, "Should have successful results"
            
            # Check that largest dataset was handled successfully
            max_size = max(r.dataset_size for r in successful_results)
            assert max_size >= 200, f"Should handle at least 200 samples, got {max_size}"
            
        except ImportError:
            pytest.skip("DifferenceInMeansEstimator not available")
    
    @pytest.mark.performance
    @pytest.mark.scalability
    @pytest.mark.slow
    def test_comprehensive_scalability_comparison(self):
        """Compare scalability across multiple methods."""
        
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
        
        config = ScalabilityTestConfig(
            base_size=50,
            size_multipliers=[1, 2, 4],
            iterations_per_size=2
        )
        
        scalability_results = {}
        
        for method_class, method_name in methods_to_test:
            try:
                method = method_class()
                tester = ScalabilityTester(method, method_name, config)
                results = tester.run_scalability_test()
                analysis = tester.analyze_scalability()
                
                scalability_results[method_name] = {
                    'results': results,
                    'analysis': analysis
                }
                
                # Basic validation
                successful_results = [r for r in results if r.success]
                assert len(successful_results) > 0, f"{method_name} had no successful results"
                
            except Exception as e:
                scalability_results[method_name] = {'error': str(e)}
        
        # Should have tested at least one method successfully
        successful_methods = [name for name, results in scalability_results.items() 
                             if 'error' not in results]
        assert len(successful_methods) > 0, "No methods were successfully tested"
        
        # Compare scalability ratings
        ratings = {}
        for method_name, results in scalability_results.items():
            if 'error' not in results and 'analysis' in results:
                analysis = results['analysis']
                for dataset_type, type_analysis in analysis.items():
                    rating = type_analysis.get('scalability_rating', 'unknown')
                    ratings[f"{method_name}_{dataset_type}"] = rating
        
        print(f"Scalability ratings: {ratings}")


if __name__ == "__main__":
    # Run a quick scalability test
    print("Running scalability test...")
    
    try:
        from causal_agent.methods.diff_in_means.estimator import DifferenceInMeansEstimator
        
        method = DifferenceInMeansEstimator()
        config = ScalabilityTestConfig(
            base_size=50,
            size_multipliers=[1, 2, 4],
            iterations_per_size=2
        )
        
        tester = ScalabilityTester(method, "DifferenceInMeans", config)
        results = tester.run_scalability_test()
        analysis = tester.analyze_scalability()
        
        print(f"Scalability test completed:")
        print(f"- Total results: {len(results)}")
        print(f"- Successful results: {len([r for r in results if r.success])}")
        
        if 'error' not in analysis:
            for dataset_type, type_analysis in analysis.items():
                rating = type_analysis['scalability_rating']
                max_size = type_analysis['max_successful_size']
                print(f"- {dataset_type}: {rating} scalability, max size {max_size}")
        
    except ImportError:
        print("DifferenceInMeansEstimator not available for testing")
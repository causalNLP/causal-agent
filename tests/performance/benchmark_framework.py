"""Benchmark comparison framework for causal inference methods."""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tests.performance.test_performance_base import (
    PerformanceMetrics, 
    PerformanceTestBase,
    benchmark_method_performance
)
from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig, DatasetType


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    dataset_sizes: List[int]
    dataset_types: List[str]
    iterations_per_test: int
    memory_stress_iterations: int
    output_directory: str
    save_raw_data: bool = True
    generate_plots: bool = True
    compare_with_baseline: bool = False
    baseline_file: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a method."""
    method_name: str
    timestamp: str
    config: BenchmarkConfig
    performance_metrics: List[PerformanceMetrics]
    summary_statistics: Dict[str, Any]
    comparison_results: Optional[Dict[str, Any]] = None


class BenchmarkRunner:
    """Main class for running comprehensive benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner with configuration."""
        self.config = config
        self.results: Dict[str, BenchmarkResult] = {}
        
        # Create output directory
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        
    def run_method_benchmark(self, method_class_or_function, method_name: str) -> BenchmarkResult:
        """Run comprehensive benchmark for a single method."""
        print(f"Running benchmark for {method_name}...")
        
        # Run the benchmark
        benchmark_data = benchmark_method_performance(
            method_class_or_function, 
            method_name, 
            self.config.dataset_sizes
        )
        
        # Create benchmark result
        result = BenchmarkResult(
            method_name=method_name,
            timestamp=datetime.now().isoformat(),
            config=self.config,
            performance_metrics=benchmark_data['scalability_metrics'] + benchmark_data['memory_stress_metrics'],
            summary_statistics=benchmark_data['summary']
        )
        
        # Store result
        self.results[method_name] = result
        
        # Save individual result
        if self.config.save_raw_data:
            self._save_result(result)
        
        return result
    
    def run_comparative_benchmark(self, methods: List[Tuple[Any, str]]) -> Dict[str, BenchmarkResult]:
        """Run benchmark across multiple methods for comparison."""
        print(f"Running comparative benchmark for {len(methods)} methods...")
        
        all_results = {}
        
        for method_class_or_function, method_name in methods:
            try:
                result = self.run_method_benchmark(method_class_or_function, method_name)
                all_results[method_name] = result
                print(f"✓ Completed {method_name}")
            except Exception as e:
                print(f"✗ Failed {method_name}: {e}")
                # Create error result
                error_result = BenchmarkResult(
                    method_name=method_name,
                    timestamp=datetime.now().isoformat(),
                    config=self.config,
                    performance_metrics=[],
                    summary_statistics={'error': str(e)}
                )
                all_results[method_name] = error_result
        
        # Generate comparison analysis
        self._generate_comparison_analysis(all_results)
        
        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_comparison_plots(all_results)
        
        return all_results
    
    def _save_result(self, result: BenchmarkResult):
        """Save individual benchmark result to file."""
        filename = f"benchmark_{result.method_name}_{result.timestamp.replace(':', '-')}.json"
        filepath = Path(self.config.output_directory) / filename
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        # Convert PerformanceMetrics to dict
        result_dict['performance_metrics'] = [
            asdict(metric) for metric in result.performance_metrics
        ]
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
    
    def _generate_comparison_analysis(self, results: Dict[str, BenchmarkResult]):
        """Generate comparative analysis across methods."""
        print("Generating comparison analysis...")
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'methods_compared': list(results.keys()),
            'performance_comparison': {},
            'scalability_comparison': {},
            'memory_comparison': {},
            'reliability_comparison': {}
        }
        
        # Extract successful results
        successful_results = {name: result for name, result in results.items() 
                            if 'error' not in result.summary_statistics}
        
        if not successful_results:
            comparison_data['error'] = "No successful benchmark results to compare"
        else:
            # Performance comparison
            for method_name, result in successful_results.items():
                scalability = result.summary_statistics.get('scalability_analysis', {})
                memory = result.summary_statistics.get('memory_analysis', {})
                
                comparison_data['performance_comparison'][method_name] = {
                    'avg_execution_time': scalability.get('avg_execution_time', 0),
                    'max_execution_time': scalability.get('max_execution_time', 0),
                    'avg_memory_usage': scalability.get('avg_memory_usage', 0),
                    'max_memory_usage': scalability.get('max_memory_usage', 0)
                }
                
                comparison_data['memory_comparison'][method_name] = {
                    'memory_stability': memory.get('memory_stability', False),
                    'potential_memory_leak': memory.get('potential_memory_leak', True),
                    'memory_growth': memory.get('memory_growth_over_iterations', 0)
                }
            
            # Find best performing methods
            if comparison_data['performance_comparison']:
                # Fastest method
                fastest_method = min(comparison_data['performance_comparison'].items(),
                                   key=lambda x: x[1]['avg_execution_time'])
                comparison_data['fastest_method'] = {
                    'name': fastest_method[0],
                    'avg_time': fastest_method[1]['avg_execution_time']
                }
                
                # Most memory efficient
                most_efficient = min(comparison_data['performance_comparison'].items(),
                                   key=lambda x: x[1]['avg_memory_usage'])
                comparison_data['most_memory_efficient'] = {
                    'name': most_efficient[0],
                    'avg_memory': most_efficient[1]['avg_memory_usage']
                }
        
        # Save comparison analysis
        comparison_file = Path(self.config.output_directory) / "comparison_analysis.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        return comparison_data
    
    def _generate_comparison_plots(self, results: Dict[str, BenchmarkResult]):
        """Generate comparison plots."""
        print("Generating comparison plots...")
        
        # Extract data for plotting
        plot_data = []
        
        for method_name, result in results.items():
            if 'error' in result.summary_statistics:
                continue
                
            for metric in result.performance_metrics:
                if metric.execution_time > 0:  # Valid metric
                    plot_data.append({
                        'method': method_name,
                        'dataset_size': metric.dataset_size,
                        'dataset_type': metric.dataset_type,
                        'execution_time': metric.execution_time,
                        'memory_usage': metric.peak_memory_mb,
                        'cpu_percent': metric.cpu_percent
                    })
        
        if not plot_data:
            print("No valid data for plotting")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Causal Inference Methods Performance Comparison', fontsize=16)
        
        # Execution time by dataset size
        sns.lineplot(data=df, x='dataset_size', y='execution_time', 
                    hue='method', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time vs Dataset Size')
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        
        # Memory usage by dataset size
        sns.lineplot(data=df, x='dataset_size', y='memory_usage', 
                    hue='method', ax=axes[0, 1])
        axes[0, 1].set_title('Memory Usage vs Dataset Size')
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Peak Memory (MB)')
        
        # Execution time by method (boxplot)
        sns.boxplot(data=df, x='method', y='execution_time', ax=axes[1, 0])
        axes[1, 0].set_title('Execution Time Distribution by Method')
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Execution Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage by method (boxplot)
        sns.boxplot(data=df, x='method', y='memory_usage', ax=axes[1, 1])
        axes[1, 1].set_title('Memory Usage Distribution by Method')
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Peak Memory (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_directory) / "performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scalability plot
        self._create_scalability_plot(df)
    
    def _create_scalability_plot(self, df: pd.DataFrame):
        """Create detailed scalability analysis plot."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability Analysis', fontsize=16)
        
        # Log-log plot for scalability analysis
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            if len(method_data) > 1:
                # Group by dataset size and take mean
                size_groups = method_data.groupby('dataset_size').agg({
                    'execution_time': 'mean',
                    'memory_usage': 'mean'
                }).reset_index()
                
                axes[0].loglog(size_groups['dataset_size'], size_groups['execution_time'], 
                              'o-', label=method)
                axes[1].loglog(size_groups['dataset_size'], size_groups['memory_usage'], 
                              'o-', label=method)
        
        axes[0].set_title('Execution Time Scalability (Log-Log)')
        axes[0].set_xlabel('Dataset Size')
        axes[0].set_ylabel('Execution Time (seconds)')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title('Memory Usage Scalability (Log-Log)')
        axes[1].set_xlabel('Dataset Size')
        axes[1].set_ylabel('Peak Memory (MB)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save scalability plot
        plot_file = Path(self.config.output_directory) / "scalability_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report_lines = [
            "# Causal Agent Performance Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {len(self.config.dataset_sizes)} dataset sizes, "
            f"{len(self.config.dataset_types)} dataset types\n"
        ]
        
        # Summary table
        report_lines.append("## Summary")
        
        if self.results:
            summary_data = []
            for method_name, result in self.results.items():
                if 'error' in result.summary_statistics:
                    summary_data.append([method_name, "FAILED", "-", "-", "-"])
                else:
                    scalability = result.summary_statistics.get('scalability_analysis', {})
                    memory = result.summary_statistics.get('memory_analysis', {})
                    
                    avg_time = scalability.get('avg_execution_time', 0)
                    avg_memory = scalability.get('avg_memory_usage', 0)
                    memory_stable = "Yes" if memory.get('memory_stability', False) else "No"
                    
                    summary_data.append([
                        method_name, 
                        "SUCCESS", 
                        f"{avg_time:.3f}s", 
                        f"{avg_memory:.1f}MB",
                        memory_stable
                    ])
            
            # Create markdown table
            report_lines.extend([
                "| Method | Status | Avg Time | Avg Memory | Memory Stable |",
                "|--------|--------|----------|------------|---------------|"
            ])
            
            for row in summary_data:
                report_lines.append(f"| {' | '.join(row)} |")
        
        report_lines.append("")
        
        # Detailed results
        report_lines.append("## Detailed Results")
        
        for method_name, result in self.results.items():
            report_lines.append(f"### {method_name}")
            
            if 'error' in result.summary_statistics:
                report_lines.append(f"**Status:** Failed - {result.summary_statistics['error']}")
            else:
                scalability = result.summary_statistics.get('scalability_analysis', {})
                memory = result.summary_statistics.get('memory_analysis', {})
                
                report_lines.extend([
                    f"**Status:** Success",
                    f"**Execution Time Range:** {scalability.get('min_execution_time', 0):.3f}s - {scalability.get('max_execution_time', 0):.3f}s",
                    f"**Memory Usage Range:** {scalability.get('min_memory_usage', 0):.1f}MB - {scalability.get('max_memory_usage', 0):.1f}MB",
                    f"**Memory Stability:** {'Good' if memory.get('memory_stability', False) else 'Poor'}",
                    f"**Potential Memory Leak:** {'Yes' if memory.get('potential_memory_leak', True) else 'No'}"
                ])
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = "benchmark_report.md"):
        """Save benchmark report to file."""
        report = self.generate_report()
        report_file = Path(self.config.output_directory) / filename
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_file}")
        return report_file


def run_standard_benchmark(output_dir: str = "benchmark_results") -> Dict[str, BenchmarkResult]:
    """Run standard benchmark with predefined configuration."""
    
    # Import methods (with error handling for missing methods)
    methods_to_test = []
    
    try:
        from causal_agent.methods.linear_regression.estimator import estimate_effect
        methods_to_test.append((estimate_effect, "LinearRegression"))
    except ImportError:
        print("LinearRegression estimate_effect not available")
    
    try:
        from causal_agent.methods.diff_in_means.estimator import estimate_effect as dim_estimate_effect
        methods_to_test.append((dim_estimate_effect, "DifferenceInMeans"))
    except ImportError:
        print("DifferenceInMeans estimate_effect not available")
    
    try:
        from causal_agent.methods.backdoor_adjustment.estimator import estimate_effect as ba_estimate_effect
        methods_to_test.append((ba_estimate_effect, "BackdoorAdjustment"))
    except ImportError:
        print("BackdoorAdjustment estimate_effect not available")
    
    if not methods_to_test:
        raise RuntimeError("No causal inference methods available for benchmarking")
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_sizes=[100, 200, 500, 1000],
        dataset_types=['observational', 'rct'],
        iterations_per_test=3,
        memory_stress_iterations=5,
        output_directory=output_dir,
        save_raw_data=True,
        generate_plots=True
    )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_comparative_benchmark(methods_to_test)
    
    # Generate and save report
    runner.save_report()
    
    return results


if __name__ == "__main__":
    # Run standard benchmark when script is executed
    print("Running standard benchmark...")
    results = run_standard_benchmark()
    print(f"Benchmark completed. Results for {len(results)} methods.")
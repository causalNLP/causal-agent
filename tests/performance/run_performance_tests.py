"""Comprehensive performance test runner for causal inference methods."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from tests.performance.benchmark_framework import BenchmarkRunner, BenchmarkConfig
from tests.performance.test_memory_validation import MemoryStressTest
from tests.performance.test_scalability import ScalabilityTester, ScalabilityTestConfig


class PerformanceTestSuite:
    """Comprehensive performance test suite runner."""
    
    def __init__(self, output_dir: str = "performance_results"):
        """Initialize the performance test suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available methods (with error handling for imports)
        self.available_methods = self._discover_available_methods()
        
    def _discover_available_methods(self) -> List[tuple]:
        """Discover available causal inference methods."""
        methods = []
        
        method_imports = [
            ("causal_agent.methods.linear_regression.estimator", "estimate_effect", "LinearRegression"),
            ("causal_agent.methods.diff_in_means.estimator", "estimate_effect", "DifferenceInMeans"),
            ("causal_agent.methods.backdoor_adjustment.estimator", "estimate_effect", "BackdoorAdjustment"),
            ("causal_agent.methods.propensity_score.matching", "estimate_effect", "PropensityScoreMatching"),
            ("causal_agent.methods.propensity_score.weighting", "estimate_effect", "PropensityScoreWeighting"),
            ("causal_agent.methods.instrumental_variable.estimator", "estimate_effect", "InstrumentalVariable"),
            ("causal_agent.methods.regression_discontinuity.estimator", "estimate_effect", "RegressionDiscontinuity"),
            ("causal_agent.methods.difference_in_differences.estimator", "estimate_effect", "DifferenceInDifferences"),
        ]
        
        for module_path, function_name, method_name in method_imports:
            try:
                module = __import__(module_path, fromlist=[function_name])
                method_function = getattr(module, function_name)
                methods.append((method_function, method_name))
                print(f"✓ Found method: {method_name}")
            except (ImportError, AttributeError) as e:
                print(f"✗ Method not available: {method_name} ({e})")
        
        return methods
    
    def run_benchmark_tests(self, methods: Optional[List[str]] = None, 
                          dataset_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run benchmark tests for specified methods."""
        print("\n=== Running Benchmark Tests ===")
        
        # Filter methods if specified
        if methods:
            filtered_methods = [(cls, name) for cls, name in self.available_methods 
                              if name in methods]
        else:
            filtered_methods = self.available_methods
        
        if not filtered_methods:
            print("No methods available for benchmarking")
            return {}
        
        # Configure benchmark
        config = BenchmarkConfig(
            dataset_sizes=dataset_sizes or [100, 200, 500, 1000],
            dataset_types=['observational', 'rct'],
            iterations_per_test=3,
            memory_stress_iterations=5,
            output_directory=str(self.output_dir / "benchmarks"),
            save_raw_data=True,
            generate_plots=True
        )
        
        # Run benchmarks
        runner = BenchmarkRunner(config)
        results = runner.run_comparative_benchmark(filtered_methods)
        
        # Save summary
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'methods_tested': list(results.keys()),
                'config': config.__dict__,
                'results_summary': {
                    name: {
                        'success': 'error' not in result.summary_statistics,
                        'summary': result.summary_statistics
                    }
                    for name, result in results.items()
                }
            }, f, indent=2, default=str)
        
        return results
    
    def run_memory_tests(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run memory validation tests for specified methods."""
        print("\n=== Running Memory Tests ===")
        
        # Filter methods if specified
        if methods:
            filtered_methods = [(cls, name) for cls, name in self.available_methods 
                              if name in methods]
        else:
            filtered_methods = self.available_methods
        
        if not filtered_methods:
            print("No methods available for memory testing")
            return {}
        
        memory_results = {}
        
        for method_class, method_name in filtered_methods:
            print(f"\nTesting memory usage for {method_name}...")
            
            try:
                method = method_class()
                stress_test = MemoryStressTest(method, method_name)
                
                # Run stability test
                stability_results = stress_test.run_repeated_execution_test(
                    dataset_size=200, iterations=10
                )
                
                # Run scaling test
                scaling_results = stress_test.run_increasing_dataset_test(
                    base_size=100, size_multipliers=[1, 2, 4, 8]
                )
                
                memory_results[method_name] = {
                    'stability': stability_results,
                    'scaling': scaling_results,
                    'success': True
                }
                
                print(f"  ✓ Memory stable: {stability_results['memory_stable']}")
                print(f"  ✓ Memory growth: {stability_results['total_memory_growth_mb']:.2f} MB")
                
            except Exception as e:
                memory_results[method_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ Memory test failed: {e}")
        
        # Save memory test results
        memory_file = self.output_dir / "memory_test_results.json"
        with open(memory_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': memory_results
            }, f, indent=2, default=str)
        
        return memory_results
    
    def run_scalability_tests(self, methods: Optional[List[str]] = None,
                            max_size_multiplier: int = 20) -> Dict[str, Any]:
        """Run scalability tests for specified methods."""
        print("\n=== Running Scalability Tests ===")
        
        # Filter methods if specified
        if methods:
            filtered_methods = [(cls, name) for cls, name in self.available_methods 
                              if name in methods]
        else:
            filtered_methods = self.available_methods
        
        if not filtered_methods:
            print("No methods available for scalability testing")
            return {}
        
        # Configure scalability tests
        config = ScalabilityTestConfig(
            base_size=100,
            size_multipliers=[1, 2, 5, 10, max_size_multiplier],
            iterations_per_size=3,
            max_execution_time_seconds=300.0,  # 5 minutes
            max_memory_mb=1000.0  # 1GB
        )
        
        scalability_results = {}
        
        for method_class, method_name in filtered_methods:
            print(f"\nTesting scalability for {method_name}...")
            
            try:
                method = method_class()
                tester = ScalabilityTester(method, method_name, config)
                
                # Run scalability test
                results = tester.run_scalability_test()
                analysis = tester.analyze_scalability()
                
                # Generate plots
                plot_dir = self.output_dir / "scalability_plots"
                tester.generate_scalability_plots(str(plot_dir))
                
                scalability_results[method_name] = {
                    'results': results,
                    'analysis': analysis,
                    'success': True
                }
                
                # Print summary
                successful_results = [r for r in results if r.success]
                max_size = max(r.dataset_size for r in successful_results) if successful_results else 0
                print(f"  ✓ Max dataset size: {max_size}")
                
                if 'error' not in analysis:
                    for dataset_type, type_analysis in analysis.items():
                        rating = type_analysis['scalability_rating']
                        print(f"  ✓ {dataset_type}: {rating} scalability")
                
            except Exception as e:
                scalability_results[method_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ Scalability test failed: {e}")
        
        # Save scalability results
        scalability_file = self.output_dir / "scalability_test_results.json"
        with open(scalability_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': config.__dict__,
                'results': scalability_results
            }, f, indent=2, default=str)
        
        return scalability_results
    
    def run_comprehensive_tests(self, methods: Optional[List[str]] = None,
                              quick_mode: bool = False) -> Dict[str, Any]:
        """Run all performance tests."""
        print("=== Comprehensive Performance Test Suite ===")
        print(f"Available methods: {[name for _, name in self.available_methods]}")
        
        if methods:
            print(f"Testing methods: {methods}")
        else:
            print("Testing all available methods")
        
        # Adjust parameters for quick mode
        if quick_mode:
            dataset_sizes = [100, 200]
            max_size_multiplier = 5
            print("Running in quick mode (reduced dataset sizes)")
        else:
            dataset_sizes = [100, 200, 500, 1000]
            max_size_multiplier = 20
            print("Running in full mode")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'quick_mode': quick_mode,
            'methods_tested': methods or [name for _, name in self.available_methods]
        }
        
        # Run benchmark tests
        try:
            benchmark_results = self.run_benchmark_tests(methods, dataset_sizes)
            results['benchmark_tests'] = {
                'success': True,
                'results': benchmark_results
            }
        except Exception as e:
            results['benchmark_tests'] = {
                'success': False,
                'error': str(e)
            }
            print(f"Benchmark tests failed: {e}")
        
        # Run memory tests
        try:
            memory_results = self.run_memory_tests(methods)
            results['memory_tests'] = {
                'success': True,
                'results': memory_results
            }
        except Exception as e:
            results['memory_tests'] = {
                'success': False,
                'error': str(e)
            }
            print(f"Memory tests failed: {e}")
        
        # Run scalability tests
        try:
            scalability_results = self.run_scalability_tests(methods, max_size_multiplier)
            results['scalability_tests'] = {
                'success': True,
                'results': scalability_results
            }
        except Exception as e:
            results['scalability_tests'] = {
                'success': False,
                'error': str(e)
            }
            print(f"Scalability tests failed: {e}")
        
        # Generate comprehensive report
        self._generate_comprehensive_report(results)
        
        return results
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive performance report."""
        report_lines = [
            "# Comprehensive Performance Test Report",
            f"Generated: {results['timestamp']}",
            f"Quick Mode: {results['quick_mode']}",
            f"Methods Tested: {', '.join(results['methods_tested'])}\n"
        ]
        
        # Summary table
        report_lines.append("## Test Summary\n")
        report_lines.extend([
            "| Test Type | Status | Methods Passed | Notes |",
            "|-----------|--------|----------------|-------|"
        ])
        
        for test_type in ['benchmark_tests', 'memory_tests', 'scalability_tests']:
            if test_type in results:
                test_result = results[test_type]
                status = "✓ PASS" if test_result['success'] else "✗ FAIL"
                
                if test_result['success'] and 'results' in test_result:
                    passed_methods = len([name for name, result in test_result['results'].items()
                                        if self._is_method_successful(result)])
                    total_methods = len(test_result['results'])
                    methods_info = f"{passed_methods}/{total_methods}"
                else:
                    methods_info = "0/0"
                
                notes = test_result.get('error', 'Completed successfully')
                report_lines.append(f"| {test_type.replace('_', ' ').title()} | {status} | {methods_info} | {notes} |")
        
        report_lines.append("")
        
        # Detailed results for each test type
        for test_type in ['benchmark_tests', 'memory_tests', 'scalability_tests']:
            if test_type in results and results[test_type]['success']:
                report_lines.append(f"## {test_type.replace('_', ' ').title()}\n")
                
                test_results = results[test_type]['results']
                for method_name, method_result in test_results.items():
                    report_lines.append(f"### {method_name}\n")
                    
                    if self._is_method_successful(method_result):
                        report_lines.append("**Status:** ✓ Success\n")
                        
                        # Add specific details based on test type
                        if test_type == 'benchmark_tests':
                            self._add_benchmark_details(report_lines, method_result)
                        elif test_type == 'memory_tests':
                            self._add_memory_details(report_lines, method_result)
                        elif test_type == 'scalability_tests':
                            self._add_scalability_details(report_lines, method_result)
                    else:
                        error_msg = method_result.get('error', 'Unknown error')
                        report_lines.append(f"**Status:** ✗ Failed - {error_msg}\n")
        
        # Save report
        report_file = self.output_dir / "comprehensive_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nComprehensive report saved to: {report_file}")
    
    def _is_method_successful(self, method_result) -> bool:
        """Check if a method test was successful."""
        if isinstance(method_result, dict):
            return method_result.get('success', True) and 'error' not in method_result
        return hasattr(method_result, 'summary_statistics') and 'error' not in method_result.summary_statistics
    
    def _add_benchmark_details(self, report_lines: List[str], result):
        """Add benchmark-specific details to report."""
        if hasattr(result, 'summary_statistics'):
            summary = result.summary_statistics
            if 'scalability_analysis' in summary:
                scalability = summary['scalability_analysis']
                report_lines.extend([
                    f"- **Average Execution Time:** {scalability.get('avg_execution_time', 0):.3f}s",
                    f"- **Average Memory Usage:** {scalability.get('avg_memory_usage', 0):.1f}MB"
                ])
    
    def _add_memory_details(self, report_lines: List[str], result):
        """Add memory test details to report."""
        if 'stability' in result:
            stability = result['stability']
            report_lines.extend([
                f"- **Memory Stable:** {stability.get('memory_stable', False)}",
                f"- **Memory Growth:** {stability.get('total_memory_growth_mb', 0):.2f}MB"
            ])
    
    def _add_scalability_details(self, report_lines: List[str], result):
        """Add scalability test details to report."""
        if 'analysis' in result and 'error' not in result['analysis']:
            analysis = result['analysis']
            for dataset_type, type_analysis in analysis.items():
                rating = type_analysis.get('scalability_rating', 'unknown')
                max_size = type_analysis.get('max_successful_size', 0)
                report_lines.append(f"- **{dataset_type}:** {rating} scalability (max size: {max_size})")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run performance tests for causal inference methods")
    parser.add_argument('--methods', nargs='+', help='Specific methods to test')
    parser.add_argument('--output-dir', default='performance_results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (smaller datasets)')
    parser.add_argument('--benchmark-only', action='store_true', help='Run only benchmark tests')
    parser.add_argument('--memory-only', action='store_true', help='Run only memory tests')
    parser.add_argument('--scalability-only', action='store_true', help='Run only scalability tests')
    
    args = parser.parse_args()
    
    # Create test suite
    suite = PerformanceTestSuite(args.output_dir)
    
    if not suite.available_methods:
        print("No causal inference methods available for testing")
        sys.exit(1)
    
    # Run specific test types or comprehensive tests
    if args.benchmark_only:
        suite.run_benchmark_tests(args.methods)
    elif args.memory_only:
        suite.run_memory_tests(args.methods)
    elif args.scalability_only:
        suite.run_scalability_tests(args.methods)
    else:
        suite.run_comprehensive_tests(args.methods, args.quick)
    
    print(f"\nPerformance tests completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
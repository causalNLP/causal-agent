#!/usr/bin/env python3
"""
Simplified metrics calculator for baseline causal analysis runs.

Computes:
1. Total queries
2. Successful queries
3. Method Match (final result vs actual method)
4. Total retries (when there are errors)

Error Breakdown:
5. Execution and runtime error percentage
6. Method mismatch
7. Data loading failure
8. Missing Result

USAGE:
    # Process all baseline files in current directory
    python simplified_metrics.py

    # Process all baseline files in a specific directory
    python simplified_metrics.py /path/to/baseline/directory

    # Process a single baseline file
    python simplified_metrics.py /path/to/baseline_file.json

    # Output: Generates simplified_metrics_report.json with detailed metrics
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np


class SimplifiedMetricsCalculator:
    """Calculate simplified metrics for baseline runs."""

    def __init__(self, json_file_path):
        """Initialize the calculator."""
        self.json_file_path = Path(json_file_path)
        self.data = []
        self.queries_data = []

    @staticmethod
    def canonical_method(method):
        """Convert method to canonical form for comparison."""
        if method is None or (isinstance(method, float) and np.isnan(method)):
            return None
        s = str(method).lower()
        if any(k in s for k in ["matching", "propensity", "ipw", "psm"]):
            return "matching"
        if any(k in s for k in ["ols", "regression", "linear"]):
            return "ols"
        if "did" in s:
            return "did"
        if "iv" in s or "instrument" in s:
            return "iv"
        if "rdd" in s:
            return "rdd"
        return s

    @staticmethod
    def has_valid_causal_effect(final_result):
        """Check if final_result has a non-null causal_effect value."""
        if not isinstance(final_result, dict):
            return False

        causal_effect = final_result.get('causal_effect')

        if causal_effect is None:
            return False

        # Check for NaN
        try:
            if isinstance(causal_effect, float) and np.isnan(causal_effect):
                return False
        except (TypeError, ValueError):
            pass

        # Check for numeric value (including dicts with numeric values)
        if isinstance(causal_effect, (int, float)):
            return True

        if isinstance(causal_effect, dict):
            for v in causal_effect.values():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    return True

        return False

    @staticmethod
    def check_method_match(final_result, expected_method='ols'):
        """Check if the method in final_result matches the expected method."""
        if not isinstance(final_result, dict):
            return False

        pred_method = final_result.get('method')

        canon_pred = SimplifiedMetricsCalculator.canonical_method(pred_method)
        canon_expected = SimplifiedMetricsCalculator.canonical_method(expected_method)

        if canon_pred is None or canon_expected is None:
            return False

        return canon_pred == canon_expected

    def load_data(self):
        """Load JSON data from file (supports both JSON array and JSONL formats)."""
        print(f"Loading: {self.json_file_path.name}")

        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except json.JSONDecodeError:
            # Try JSONL format
            self.data = []
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))

        print(f"  Loaded {len(self.data)} queries")

    def has_execution_error(self, code_outputs):
        """Check if any code execution has errors."""
        if not code_outputs:
            return False

        error_patterns = [
            re.compile(r'Error|Exception|Traceback', re.IGNORECASE),
            re.compile(r'Failed|failed', re.IGNORECASE),
        ]

        for output in code_outputs:
            output_str = str(output)
            for pattern in error_patterns:
                if pattern.search(output_str):
                    return True
        return False

    def is_data_loading_failure(self, result):
        """Check if this is a data loading failure."""
        if isinstance(result, str):
            return 'No such file' in result or 'FileNotFound' in result or 'data' in result.lower()

        if not isinstance(result, dict):
            return False

        code_outputs = result.get('code_outputs', [])
        for output in code_outputs:
            output_str = str(output).lower()
            if any(kw in output_str for kw in ['filenotfound', 'no such file', 'cannot load', 'data not found']):
                return True
        return False

    def analyze_query(self, query_data, query_idx):
        """Analyze a single query."""
        result = query_data.get('result', {})

        # Handle null/None result or string result (error messages)
        if result is None or isinstance(result, str):
            is_data_failure = self.is_data_loading_failure(result)
            return {
                'query_index': query_idx,
                'is_successful': False,
                'has_method_match': False,
                'has_execution_error': True,
                'is_method_mismatch': False,
                'is_data_loading_failure': is_data_failure,
                'is_missing_result': True,
                'retries': 0,
            }

        # Get basic info
        retries = result.get('retries', 0)
        code_outputs = result.get('code_outputs', [])
        final_result = result.get('final_result', {})
        expected_method = query_data.get('method', 'ols')

        # Check for execution errors
        has_execution_error = self.has_execution_error(code_outputs)

        # Check for data loading failure
        is_data_loading_failure = self.is_data_loading_failure(result)

        # Check for valid causal effect
        has_valid_effect = self.has_valid_causal_effect(final_result)

        # Check method match
        has_method_match = self.check_method_match(final_result, expected_method)

        # Determine success: valid effect and no execution error in final output
        has_final_exec_error = False
        if code_outputs:
            last_output = code_outputs[-1]
            last_output_str = str(last_output)
            has_final_exec_error = bool(re.search(r'Error|Exception|Traceback', last_output_str, re.IGNORECASE))

        is_successful = has_valid_effect and not has_final_exec_error

        # Method mismatch: has valid effect but wrong method
        is_method_mismatch = has_valid_effect and not has_method_match

        # Missing result: no valid causal effect
        is_missing_result = not has_valid_effect

        # Count retries only if there are errors
        actual_retries = retries if has_execution_error and retries > 0 else 0

        return {
            'query_index': query_idx,
            'is_successful': is_successful,
            'has_method_match': has_method_match,
            'has_execution_error': has_execution_error,
            'is_method_mismatch': is_method_mismatch,
            'is_data_loading_failure': is_data_loading_failure,
            'is_missing_result': is_missing_result,
            'retries': actual_retries,
        }

    def analyze_all_queries(self):
        """Analyze all queries in the dataset."""
        self.queries_data = []
        for idx, query_data in enumerate(self.data):
            analysis = self.analyze_query(query_data, idx)
            self.queries_data.append(analysis)

    def calculate_metrics(self):
        """Calculate simplified metrics."""
        total_queries = len(self.queries_data)

        # Main metrics
        successful_queries = sum(1 for q in self.queries_data if q['is_successful'])
        method_match_queries = sum(1 for q in self.queries_data if q['has_method_match'])
        total_retries = sum(q['retries'] for q in self.queries_data)

        # Error breakdown
        execution_error_queries = sum(1 for q in self.queries_data if q['has_execution_error'])
        method_mismatch_queries = sum(1 for q in self.queries_data if q['is_method_mismatch'])
        data_loading_failure_queries = sum(1 for q in self.queries_data if q['is_data_loading_failure'])
        missing_result_queries = sum(1 for q in self.queries_data if q['is_missing_result'])

        # Calculate percentages
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        method_match_rate = (method_match_queries / total_queries * 100) if total_queries > 0 else 0
        execution_error_rate = (execution_error_queries / total_queries * 100) if total_queries > 0 else 0
        method_mismatch_rate = (method_mismatch_queries / total_queries * 100) if total_queries > 0 else 0
        data_loading_failure_rate = (data_loading_failure_queries / total_queries * 100) if total_queries > 0 else 0
        missing_result_rate = (missing_result_queries / total_queries * 100) if total_queries > 0 else 0

        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'method_match_queries': method_match_queries,
            'total_retries': total_retries,
            'success_rate_percent': round(success_rate, 2),
            'method_match_rate_percent': round(method_match_rate, 2),
            'execution_error_queries': execution_error_queries,
            'execution_error_rate_percent': round(execution_error_rate, 2),
            'method_mismatch_queries': method_mismatch_queries,
            'method_mismatch_rate_percent': round(method_mismatch_rate, 2),
            'data_loading_failure_queries': data_loading_failure_queries,
            'data_loading_failure_rate_percent': round(data_loading_failure_rate, 2),
            'missing_result_queries': missing_result_queries,
            'missing_result_rate_percent': round(missing_result_rate, 2),
        }

    def generate_report(self):
        """Generate text report."""
        self.load_data()
        self.analyze_all_queries()
        metrics = self.calculate_metrics()

        lines = []
        lines.append("=" * 80)
        lines.append("SIMPLIFIED METRICS REPORT")
        lines.append("=" * 80)
        lines.append(f"\nFile: {self.json_file_path.name}")
        lines.append("\n" + "-" * 80)
        lines.append("MAIN METRICS")
        lines.append("-" * 80)
        lines.append(f"1. Total Queries:               {metrics['total_queries']}")
        lines.append(f"2. Successful Queries:          {metrics['successful_queries']} ({metrics['success_rate_percent']}%)")
        lines.append(f"3. Method Match:                {metrics['method_match_queries']} ({metrics['method_match_rate_percent']}%)")
        lines.append(f"4. Total Retries (w/ errors):   {metrics['total_retries']}")

        lines.append("\n" + "-" * 80)
        lines.append("ERROR BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"5. Execution/Runtime Errors:    {metrics['execution_error_queries']} ({metrics['execution_error_rate_percent']}%)")
        lines.append(f"6. Method Mismatch:             {metrics['method_mismatch_queries']} ({metrics['method_mismatch_rate_percent']}%)")
        lines.append(f"7. Data Loading Failure:        {metrics['data_loading_failure_queries']} ({metrics['data_loading_failure_rate_percent']}%)")
        lines.append(f"8. Missing Result:              {metrics['missing_result_queries']} ({metrics['missing_result_rate_percent']}%)")
        lines.append("=" * 80)

        return "\n".join(lines), metrics


def process_all_baseline_files(baseline_dir, exclude_prefix='cais_'):
    """Process all baseline JSON files in a directory."""
    baseline_dir = Path(baseline_dir)
    all_files = list(baseline_dir.glob("*.json"))

    # Filter out excluded files
    excluded_files = {'baseline_error_metrics_report.json', 'error_metrics_report.json', 'simplified_metrics_report.json'}
    json_files = [f for f in all_files
                  if not f.name.startswith(exclude_prefix)
                  and f.name not in excluded_files
                  and not f.name.endswith('.error_metrics.json')
                  and not f.name.endswith('.metrics.json')]

    print(f"\nFound {len(json_files)} baseline files to process")
    print(f"(Excluded {len(all_files) - len(json_files)} files)\n")

    all_results = {}
    summary_data = []

    for json_file in sorted(json_files):
        print(f"\n{'='*80}")
        calculator = SimplifiedMetricsCalculator(json_file)
        report, metrics = calculator.generate_report()
        print(report)

        all_results[json_file.name] = metrics
        summary_data.append({
            'file': json_file.name,
            'total': metrics['total_queries'],
            'successful': metrics['successful_queries'],
            'method_match': metrics['method_match_queries'],
            'retries': metrics['total_retries'],
            'success_rate': metrics['success_rate_percent'],
            'method_match_rate': metrics['method_match_rate_percent'],
            'exec_error_rate': metrics['execution_error_rate_percent'],
            'mismatch_rate': metrics['method_mismatch_rate_percent'],
            'data_fail_rate': metrics['data_loading_failure_rate_percent'],
            'missing_rate': metrics['missing_result_rate_percent'],
        })

    # Print summary table
    print("\n" + "=" * 160)
    print("SUMMARY ACROSS ALL BASELINE FILES")
    print("=" * 160)
    print(f"{'File':<40} {'Total':>6} {'Success':>7} {'Match':>7} {'Retries':>7} "
          f"{'Succ%':>6} {'Match%':>6} {'ExecErr%':>8} {'MethMis%':>8} {'DataFail%':>9} {'MissRes%':>8}")
    print("-" * 160)

    for item in summary_data:
        print(f"{item['file']:<40} {item['total']:>6} {item['successful']:>7} "
              f"{item['method_match']:>7} {item['retries']:>7} "
              f"{item['success_rate']:>5.1f}% {item['method_match_rate']:>5.1f}% "
              f"{item['exec_error_rate']:>7.1f}% {item['mismatch_rate']:>7.1f}% "
              f"{item['data_fail_rate']:>8.1f}% {item['missing_rate']:>7.1f}%")

    # Calculate totals
    total_all = sum(item['total'] for item in summary_data)
    successful_all = sum(item['successful'] for item in summary_data)
    method_match_all = sum(item['method_match'] for item in summary_data)
    retries_all = sum(item['retries'] for item in summary_data)

    print("-" * 160)
    print(f"{'TOTAL':<40} {total_all:>6} {successful_all:>7} "
          f"{method_match_all:>7} {retries_all:>7} "
          f"{successful_all/total_all*100:>5.1f}% {method_match_all/total_all*100:>5.1f}% "
          f"{sum(d['total']*d['exec_error_rate'] for d in summary_data)/total_all:>7.1f}% "
          f"{sum(d['total']*d['mismatch_rate'] for d in summary_data)/total_all:>7.1f}% "
          f"{sum(d['total']*d['data_fail_rate'] for d in summary_data)/total_all:>8.1f}% "
          f"{sum(d['total']*d['missing_rate'] for d in summary_data)/total_all:>7.1f}%")
    print("=" * 160)

    # Save JSON report
    output_file = baseline_dir / "simplified_metrics_report.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed JSON report saved to: {output_file}")

    return all_results, summary_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

        if path.is_dir():
            process_all_baseline_files(path)
        elif path.is_file():
            calculator = SimplifiedMetricsCalculator(path)
            report, metrics = calculator.generate_report()
            print(report)

            # Save JSON
            output_file = path.parent / f"{path.stem}_simplified_metrics.json"
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nJSON saved to: {output_file}")
    else:
        # Default: process current directory
        default_dir = Path(__file__).parent
        process_all_baseline_files(default_dir)

#!/usr/bin/env python3
"""
Simplified metrics calculator for log files.

Computes:
1. Total queries
2. Successful queries (no runtime errors)
3. Failed queries (from _failed.txt)
4. Total errors

Error Breakdown:
5. Runtime error percentage
6. Validation errors
7. Missing variable errors
8. KeyError/Missing data errors

USAGE:
    # Process all log files in current directory
    python simplified_log_metrics.py

    # Process all log files in a specific directory
    python simplified_log_metrics.py /path/to/logs/directory

    # Process a single log file
    python simplified_log_metrics.py /path/to/logfile.log

    # Output: Generates simplified_log_metrics_report.json with detailed metrics
    # Note: Script automatically finds corresponding _failed.txt files
"""

import re
from pathlib import Path
from collections import defaultdict
import json
import sys


class SimplifiedLogMetricsCalculator:
    """Calculate simplified metrics for log files."""

    def __init__(self, log_file_path, failed_file_path=None):
        """Initialize the calculator."""
        self.log_file_path = Path(log_file_path)
        self.failed_file_path = Path(failed_file_path) if failed_file_path else None

        # Auto-detect failed file if not provided
        if not self.failed_file_path:
            base_name = str(self.log_file_path).replace('.log', '').replace('.json', '')
            potential_failed = Path(f"{base_name}_failed.txt")
            if potential_failed.exists():
                self.failed_file_path = potential_failed

        self.queries = []
        self.failed_query_indices = set()
        self.parsed = False

        # Simplified error patterns
        self.error_patterns = {
            'validation_error': re.compile(r'validation error', re.IGNORECASE),
            'missing_variable': re.compile(r'Missing treatment or outcome variable', re.IGNORECASE),
            'pydantic_error': re.compile(r'Pydantic model creation', re.IGNORECASE),
            'keyerror': re.compile(r"KeyError|'method_info'|'results'", re.IGNORECASE),
            'file_error': re.compile(r'FileNotFoundError|No such file', re.IGNORECASE),
            'data_error': re.compile(r'missing required columns|DataFrame', re.IGNORECASE),
        }

    def parse_log_file(self):
        """Parse the log file to extract query information and errors."""
        if self.parsed:
            return

        print(f"Parsing: {self.log_file_path.name}")

        current_query = None
        query_counter = 0

        with open(self.log_file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                # Check if this is the start of a new query
                if 'INFO - Starting causal analysis run...' in line:
                    # Save previous query if exists
                    if current_query is not None:
                        self.queries.append(current_query)

                    # Start new query
                    current_query = {
                        'query_index': query_counter,
                        'start_line': line_num,
                        'has_error': False,
                        'error_types': set(),
                        'error_count': 0,
                    }
                    query_counter += 1

                elif current_query is not None:
                    # Check for errors in current query
                    if ' - ERROR - ' in line:
                        current_query['has_error'] = True
                        current_query['error_count'] += 1

                        # Classify error type
                        for error_type, pattern in self.error_patterns.items():
                            if pattern.search(line):
                                current_query['error_types'].add(error_type)

                    # Check for traceback
                    if 'Traceback (most recent call last)' in line:
                        current_query['has_error'] = True

        # Add the last query
        if current_query is not None:
            self.queries.append(current_query)

        self.parsed = True
        print(f"  Found {len(self.queries)} queries")

    def parse_failed_file(self):
        """Parse the _failed.txt file to get failed query indices."""
        if not self.failed_file_path or not self.failed_file_path.exists():
            return

        with open(self.failed_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse format like "0" or "0 -> 1" or just "1"
                match = re.match(r'(\d+)', line)
                if match:
                    query_idx = int(match.group(1))
                    self.failed_query_indices.add(query_idx)

    def calculate_metrics(self):
        """Calculate simplified metrics."""
        total_queries = len(self.queries)

        # Count queries with errors
        queries_with_errors = sum(1 for q in self.queries if q['has_error'])
        successful_queries = total_queries - queries_with_errors

        # Count failed queries
        failed_queries = len(self.failed_query_indices)

        # Total error count
        total_errors = sum(q['error_count'] for q in self.queries)

        # Error type breakdown
        error_type_counts = defaultdict(int)
        for query in self.queries:
            if query['has_error']:
                for error_type in query['error_types']:
                    error_type_counts[error_type] += 1

        # Calculate percentages
        runtime_error_rate = (queries_with_errors / total_queries * 100) if total_queries > 0 else 0
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        failed_rate = (failed_queries / total_queries * 100) if total_queries > 0 else 0

        # Individual error type rates
        validation_error_rate = (error_type_counts.get('validation_error', 0) / total_queries * 100) if total_queries > 0 else 0
        missing_var_rate = (error_type_counts.get('missing_variable', 0) / total_queries * 100) if total_queries > 0 else 0
        keyerror_rate = (error_type_counts.get('keyerror', 0) / total_queries * 100) if total_queries > 0 else 0
        file_error_rate = (error_type_counts.get('file_error', 0) / total_queries * 100) if total_queries > 0 else 0
        data_error_rate = (error_type_counts.get('data_error', 0) / total_queries * 100) if total_queries > 0 else 0

        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'total_errors': total_errors,
            'success_rate_percent': round(success_rate, 2),
            'runtime_error_queries': queries_with_errors,
            'runtime_error_rate_percent': round(runtime_error_rate, 2),
            'failed_rate_percent': round(failed_rate, 2),
            'validation_error_queries': error_type_counts.get('validation_error', 0),
            'validation_error_rate_percent': round(validation_error_rate, 2),
            'missing_variable_queries': error_type_counts.get('missing_variable', 0),
            'missing_variable_rate_percent': round(missing_var_rate, 2),
            'keyerror_queries': error_type_counts.get('keyerror', 0),
            'keyerror_rate_percent': round(keyerror_rate, 2),
            'file_error_queries': error_type_counts.get('file_error', 0),
            'file_error_rate_percent': round(file_error_rate, 2),
            'data_error_queries': error_type_counts.get('data_error', 0),
            'data_error_rate_percent': round(data_error_rate, 2),
        }

    def generate_report(self):
        """Generate text report."""
        self.parse_log_file()
        self.parse_failed_file()
        metrics = self.calculate_metrics()

        lines = []
        lines.append("=" * 80)
        lines.append("SIMPLIFIED LOG METRICS REPORT")
        lines.append("=" * 80)
        lines.append(f"\nLog File: {self.log_file_path.name}")
        if self.failed_file_path and self.failed_file_path.exists():
            lines.append(f"Failed File: {self.failed_file_path.name}")
        lines.append("\n" + "-" * 80)
        lines.append("MAIN METRICS")
        lines.append("-" * 80)
        lines.append(f"1. Total Queries:               {metrics['total_queries']}")
        lines.append(f"2. Successful Queries:          {metrics['successful_queries']} ({metrics['success_rate_percent']}%)")
        lines.append(f"3. Failed Queries (from file):  {metrics['failed_queries']} ({metrics['failed_rate_percent']}%)")
        lines.append(f"4. Total Errors:                {metrics['total_errors']}")

        lines.append("\n" + "-" * 80)
        lines.append("ERROR BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"5. Runtime Errors:              {metrics['runtime_error_queries']} ({metrics['runtime_error_rate_percent']}%)")
        lines.append(f"6. Validation Errors:           {metrics['validation_error_queries']} ({metrics['validation_error_rate_percent']}%)")
        lines.append(f"7. Missing Variable Errors:     {metrics['missing_variable_queries']} ({metrics['missing_variable_rate_percent']}%)")
        lines.append(f"8. KeyError/Missing Data:       {metrics['keyerror_queries']} ({metrics['keyerror_rate_percent']}%)")
        lines.append(f"   File Errors:                 {metrics['file_error_queries']} ({metrics['file_error_rate_percent']}%)")
        lines.append(f"   Data Errors:                 {metrics['data_error_queries']} ({metrics['data_error_rate_percent']}%)")
        lines.append("=" * 80)

        return "\n".join(lines), metrics


def process_all_log_files(log_dir):
    """Process all log files in a directory."""
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob("*.log"))

    print(f"\nFound {len(log_files)} log files to process\n")

    all_results = {}
    summary_data = []

    for log_file in sorted(log_files):
        print(f"\n{'='*80}")
        calculator = SimplifiedLogMetricsCalculator(log_file)
        report, metrics = calculator.generate_report()
        print(report)

        all_results[log_file.name] = metrics
        summary_data.append({
            'file': log_file.name,
            'total': metrics['total_queries'],
            'successful': metrics['successful_queries'],
            'failed': metrics['failed_queries'],
            'errors': metrics['total_errors'],
            'success_rate': metrics['success_rate_percent'],
            'runtime_error_rate': metrics['runtime_error_rate_percent'],
            'validation_error_rate': metrics['validation_error_rate_percent'],
            'missing_var_rate': metrics['missing_variable_rate_percent'],
            'keyerror_rate': metrics['keyerror_rate_percent'],
        })

    # Print summary table
    print("\n" + "=" * 140)
    print("SUMMARY ACROSS ALL LOG FILES")
    print("=" * 140)
    print(f"{'File':<40} {'Total':>6} {'Success':>7} {'Failed':>7} {'Errors':>7} "
          f"{'Succ%':>6} {'RunErr%':>7} {'ValErr%':>7} {'MissVar%':>8} {'KeyErr%':>7}")
    print("-" * 140)

    for item in summary_data:
        print(f"{item['file']:<40} {item['total']:>6} {item['successful']:>7} "
              f"{item['failed']:>7} {item['errors']:>7} "
              f"{item['success_rate']:>5.1f}% {item['runtime_error_rate']:>6.1f}% "
              f"{item['validation_error_rate']:>6.1f}% {item['missing_var_rate']:>7.1f}% "
              f"{item['keyerror_rate']:>6.1f}%")

    # Calculate totals
    total_all = sum(item['total'] for item in summary_data)
    successful_all = sum(item['successful'] for item in summary_data)
    failed_all = sum(item['failed'] for item in summary_data)
    errors_all = sum(item['errors'] for item in summary_data)

    print("-" * 140)
    print(f"{'TOTAL':<40} {total_all:>6} {successful_all:>7} "
          f"{failed_all:>7} {errors_all:>7} "
          f"{successful_all/total_all*100:>5.1f}% "
          f"{sum(d['total']*d['runtime_error_rate'] for d in summary_data)/total_all:>6.1f}% "
          f"{sum(d['total']*d['validation_error_rate'] for d in summary_data)/total_all:>6.1f}% "
          f"{sum(d['total']*d['missing_var_rate'] for d in summary_data)/total_all:>7.1f}% "
          f"{sum(d['total']*d['keyerror_rate'] for d in summary_data)/total_all:>6.1f}%")
    print("=" * 140)

    # Save JSON report
    output_file = log_dir / "simplified_log_metrics_report.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed JSON report saved to: {output_file}")

    return all_results, summary_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

        if path.is_dir():
            process_all_log_files(path)
        elif path.is_file():
            calculator = SimplifiedLogMetricsCalculator(path)
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
        process_all_log_files(default_dir)

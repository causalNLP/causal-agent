#!/usr/bin/env python3
"""
Coverage analysis script for comprehensive test coverage validation.
This script analyzes coverage reports and identifies gaps in critical code paths.
"""

import os
import sys
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class CoverageAnalyzer:
    """Analyzes test coverage and identifies gaps in critical code paths."""
    
    def __init__(self, target_coverage: float = 80.0):
        self.target_coverage = target_coverage
        self.project_root = Path(__file__).parent.parent
        self.coverage_xml_path = self.project_root / "coverage.xml"
        self.coverage_html_dir = self.project_root / "htmlcov"
        
    def run_tests_with_coverage(self) -> Tuple[bool, str]:
        """Run tests with coverage measurement."""
        print("Running tests with coverage measurement...")
        
        cmd = [
            "python", "-m", "pytest",
            "--cov=causal_agent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-fail-under=0",  # Don't fail on coverage, we'll handle that
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Tests timed out after 10 minutes"
        except Exception as e:
            return False, f"Error running tests: {str(e)}"
    
    def parse_coverage_xml(self) -> Optional[Dict]:
        """Parse coverage XML report."""
        if not self.coverage_xml_path.exists():
            print(f"Coverage XML file not found: {self.coverage_xml_path}")
            return None
            
        try:
            tree = ET.parse(self.coverage_xml_path)
            root = tree.getroot()
            
            coverage_data = {
                'overall': {},
                'packages': {},
                'files': {}
            }
            
            # Overall coverage
            coverage_data['overall'] = {
                'line_rate': float(root.get('line-rate', 0)) * 100,
                'branch_rate': float(root.get('branch-rate', 0)) * 100,
                'lines_covered': int(root.get('lines-covered', 0)),
                'lines_valid': int(root.get('lines-valid', 0)),
                'branches_covered': int(root.get('branches-covered', 0)),
                'branches_valid': int(root.get('branches-valid', 0))
            }
            
            # Package and file level coverage
            for package in root.findall('.//package'):
                package_name = package.get('name')
                package_data = {
                    'line_rate': float(package.get('line-rate', 0)) * 100,
                    'branch_rate': float(package.get('branch-rate', 0)) * 100,
                    'files': {}
                }
                
                for class_elem in package.findall('.//class'):
                    filename = class_elem.get('filename')
                    file_data = {
                        'line_rate': float(class_elem.get('line-rate', 0)) * 100,
                        'branch_rate': float(class_elem.get('branch-rate', 0)) * 100,
                        'lines': {}
                    }
                    
                    # Line coverage details
                    for line in class_elem.findall('.//line'):
                        line_num = int(line.get('number'))
                        hits = int(line.get('hits'))
                        file_data['lines'][line_num] = hits > 0
                    
                    package_data['files'][filename] = file_data
                    coverage_data['files'][filename] = file_data
                
                coverage_data['packages'][package_name] = package_data
            
            return coverage_data
            
        except Exception as e:
            print(f"Error parsing coverage XML: {str(e)}")
            return None
    
    def identify_critical_paths(self) -> List[str]:
        """Identify critical code paths that should have high coverage."""
        critical_patterns = [
            "causal_agent/agent.py",
            "causal_agent/components/",
            "causal_agent/methods/*/estimator.py",
            "causal_agent/methods/*/diagnostics.py",
            "causal_agent/tools/",
            "causal_agent/utils/",
        ]
        
        critical_files = []
        for pattern in critical_patterns:
            if "*" in pattern:
                # Handle glob patterns
                base_path = self.project_root / pattern.replace("*", "**")
                for file_path in self.project_root.glob(pattern):
                    if file_path.is_file() and file_path.suffix == ".py":
                        critical_files.append(str(file_path.relative_to(self.project_root)))
            else:
                file_path = self.project_root / pattern
                if file_path.exists():
                    if file_path.is_file():
                        critical_files.append(pattern)
                    else:
                        # Directory - add all Python files
                        for py_file in file_path.rglob("*.py"):
                            critical_files.append(str(py_file.relative_to(self.project_root)))
        
        return critical_files
    
    def analyze_coverage_gaps(self, coverage_data: Dict) -> Dict:
        """Analyze coverage gaps and prioritize improvements."""
        gaps = {
            'low_coverage_files': [],
            'uncovered_critical_paths': [],
            'missing_tests': [],
            'recommendations': []
        }
        
        critical_files = self.identify_critical_paths()
        
        for filename, file_data in coverage_data['files'].items():
            line_rate = file_data['line_rate']
            
            # Files with low coverage
            if line_rate < self.target_coverage:
                gaps['low_coverage_files'].append({
                    'file': filename,
                    'coverage': line_rate,
                    'is_critical': filename in critical_files
                })
            
            # Critical files with very low coverage
            if filename in critical_files and line_rate < 50:
                gaps['uncovered_critical_paths'].append({
                    'file': filename,
                    'coverage': line_rate
                })
        
        # Sort by priority (critical files first, then by coverage)
        gaps['low_coverage_files'].sort(
            key=lambda x: (not x['is_critical'], x['coverage'])
        )
        
        # Generate recommendations
        gaps['recommendations'] = self._generate_recommendations(gaps)
        
        return gaps
    
    def _generate_recommendations(self, gaps: Dict) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []
        
        if gaps['uncovered_critical_paths']:
            recommendations.append(
                "PRIORITY: Add tests for critical paths with <50% coverage"
            )
        
        if gaps['low_coverage_files']:
            critical_count = sum(1 for f in gaps['low_coverage_files'] if f['is_critical'])
            recommendations.append(
                f"Add tests for {critical_count} critical files below {self.target_coverage}% coverage"
            )
        
        # Specific recommendations based on common patterns
        low_coverage_patterns = {}
        for file_info in gaps['low_coverage_files']:
            file_path = Path(file_info['file'])
            pattern = str(file_path.parent)
            if pattern not in low_coverage_patterns:
                low_coverage_patterns[pattern] = []
            low_coverage_patterns[pattern].append(file_info)
        
        for pattern, files in low_coverage_patterns.items():
            if len(files) > 2:
                recommendations.append(
                    f"Consider comprehensive test suite for {pattern}/ module ({len(files)} files need coverage)"
                )
        
        return recommendations
    
    def generate_coverage_report(self, coverage_data: Dict, gaps: Dict) -> str:
        """Generate a comprehensive coverage report."""
        overall = coverage_data['overall']
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TEST COVERAGE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL COVERAGE STATISTICS:")
        report.append(f"  Line Coverage: {overall['line_rate']:.2f}%")
        report.append(f"  Branch Coverage: {overall['branch_rate']:.2f}%")
        report.append(f"  Lines Covered: {overall['lines_covered']}/{overall['lines_valid']}")
        report.append(f"  Target Coverage: {self.target_coverage}%")
        
        coverage_status = "✅ PASSED" if overall['line_rate'] >= self.target_coverage else "❌ FAILED"
        report.append(f"  Status: {coverage_status}")
        report.append("")
        
        # Coverage gaps
        if gaps['low_coverage_files']:
            report.append("LOW COVERAGE FILES:")
            for file_info in gaps['low_coverage_files'][:10]:  # Top 10
                critical_marker = "🔴" if file_info['is_critical'] else "🟡"
                report.append(f"  {critical_marker} {file_info['file']}: {file_info['coverage']:.1f}%")
            
            if len(gaps['low_coverage_files']) > 10:
                report.append(f"  ... and {len(gaps['low_coverage_files']) - 10} more files")
            report.append("")
        
        # Critical paths
        if gaps['uncovered_critical_paths']:
            report.append("CRITICAL PATHS NEEDING ATTENTION:")
            for path_info in gaps['uncovered_critical_paths']:
                report.append(f"  🔴 {path_info['file']}: {path_info['coverage']:.1f}%")
            report.append("")
        
        # Recommendations
        if gaps['recommendations']:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(gaps['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        # Package-level summary
        report.append("PACKAGE-LEVEL COVERAGE:")
        for package_name, package_data in coverage_data['packages'].items():
            report.append(f"  {package_name}: {package_data['line_rate']:.1f}%")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def create_coverage_badge(self, coverage_percentage: float) -> None:
        """Create a coverage badge."""
        try:
            import subprocess
            
            # Determine badge color based on coverage
            if coverage_percentage >= 90:
                color = "brightgreen"
            elif coverage_percentage >= 80:
                color = "green"
            elif coverage_percentage >= 70:
                color = "yellow"
            elif coverage_percentage >= 60:
                color = "orange"
            else:
                color = "red"
            
            # Create badge using coverage-badge if available
            cmd = [
                "coverage-badge",
                "-o", "coverage.svg",
                "-f"
            ]
            
            subprocess.run(cmd, cwd=self.project_root, check=False)
            print(f"Coverage badge created: coverage.svg ({coverage_percentage:.1f}%)")
            
        except Exception as e:
            print(f"Could not create coverage badge: {str(e)}")
    
    def run_analysis(self, run_tests: bool = True) -> bool:
        """Run complete coverage analysis."""
        print("Starting comprehensive coverage analysis...")
        
        if run_tests:
            success, output = self.run_tests_with_coverage()
            if not success:
                print("Tests failed:")
                print(output)
                return False
        
        # Parse coverage data
        coverage_data = self.parse_coverage_xml()
        if not coverage_data:
            print("Failed to parse coverage data")
            return False
        
        # Analyze gaps
        gaps = self.analyze_coverage_gaps(coverage_data)
        
        # Generate report
        report = self.generate_coverage_report(coverage_data, gaps)
        print(report)
        
        # Save report to file
        report_path = self.project_root / "coverage_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Detailed report saved to: {report_path}")
        
        # Create coverage badge
        self.create_coverage_badge(coverage_data['overall']['line_rate'])
        
        # Return success based on coverage target
        return coverage_data['overall']['line_rate'] >= self.target_coverage


def main():
    parser = argparse.ArgumentParser(description="Analyze test coverage")
    parser.add_argument(
        "--target", 
        type=float, 
        default=80.0,
        help="Target coverage percentage (default: 80.0)"
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip running tests, analyze existing coverage data"
    )
    
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer(target_coverage=args.target)
    success = analyzer.run_analysis(run_tests=not args.no_tests)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
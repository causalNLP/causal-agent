#!/usr/bin/env python3
"""
CI/CD coverage validation script.
This script is designed to be run in CI/CD pipelines to validate coverage requirements.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


class CICoverageValidator:
    """Validates coverage requirements for CI/CD pipelines."""
    
    def __init__(self, min_coverage: float = 80.0, min_critical_coverage: float = 90.0):
        self.min_coverage = min_coverage
        self.min_critical_coverage = min_critical_coverage
        self.project_root = Path(__file__).parent.parent
        self.coverage_xml_path = self.project_root / "coverage.xml"
        
        # Critical files that must have high coverage
        self.critical_files = [
            "causal_agent/agent.py",
            "causal_agent/components/dataset_analyzer.py",
            "causal_agent/components/decision_tree.py",
            "causal_agent/components/input_parser.py",
            "causal_agent/components/query_interpreter.py",
        ]
    
    def run_tests_with_coverage(self) -> bool:
        """Run tests with coverage in CI mode."""
        print("Running tests with coverage measurement...")
        
        # Use environment variables for CI configuration
        test_markers = os.getenv('PYTEST_MARKERS', '')
        parallel_workers = os.getenv('PYTEST_WORKERS', 'auto')
        
        cmd = [
            "python", "-m", "pytest",
            "--cov=causal_agent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-fail-under=0",  # We handle failure ourselves
            "-v",
            "--tb=short"
        ]
        
        # Add parallel execution if specified
        if parallel_workers != '1':
            cmd.extend(["-n", parallel_workers])
        
        # Add test markers if specified
        if test_markers:
            cmd.extend(["-m", test_markers])
        
        # Skip slow tests in CI unless explicitly requested
        if not os.getenv('RUN_SLOW_TESTS'):
            cmd.extend(["-m", "not slow"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                timeout=1800  # 30 minutes timeout for CI
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("ERROR: Tests timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"ERROR: Failed to run tests: {str(e)}")
            return False
    
    def parse_coverage_results(self) -> Optional[Dict]:
        """Parse coverage results from XML."""
        if not self.coverage_xml_path.exists():
            print("ERROR: Coverage XML file not found")
            return None
        
        try:
            tree = ET.parse(self.coverage_xml_path)
            root = tree.getroot()
            
            results = {
                'overall_coverage': float(root.get('line-rate', 0)) * 100,
                'branch_coverage': float(root.get('branch-rate', 0)) * 100,
                'lines_covered': int(root.get('lines-covered', 0)),
                'lines_valid': int(root.get('lines-valid', 0)),
                'file_coverage': {}
            }
            
            # Parse file-level coverage
            for package in root.findall('.//package'):
                for class_elem in package.findall('.//class'):
                    filename = class_elem.get('filename')
                    line_rate = float(class_elem.get('line-rate', 0)) * 100
                    results['file_coverage'][filename] = line_rate
            
            return results
            
        except Exception as e:
            print(f"ERROR: Failed to parse coverage XML: {str(e)}")
            return None
    
    def validate_coverage_requirements(self, results: Dict) -> Dict:
        """Validate coverage against requirements."""
        validation = {
            'overall_passed': False,
            'critical_passed': False,
            'failures': [],
            'warnings': []
        }
        
        # Check overall coverage
        overall_coverage = results['overall_coverage']
        if overall_coverage >= self.min_coverage:
            validation['overall_passed'] = True
            print(f"✅ Overall coverage: {overall_coverage:.2f}% (>= {self.min_coverage}%)")
        else:
            validation['failures'].append(
                f"Overall coverage {overall_coverage:.2f}% is below minimum {self.min_coverage}%"
            )
            print(f"❌ Overall coverage: {overall_coverage:.2f}% (< {self.min_coverage}%)")
        
        # Check critical file coverage
        critical_failures = []
        critical_warnings = []
        
        for critical_file in self.critical_files:
            if critical_file in results['file_coverage']:
                file_coverage = results['file_coverage'][critical_file]
                if file_coverage < self.min_critical_coverage:
                    if file_coverage < 50:  # Very low coverage
                        critical_failures.append(
                            f"{critical_file}: {file_coverage:.1f}% (critical file with very low coverage)"
                        )
                    else:
                        critical_warnings.append(
                            f"{critical_file}: {file_coverage:.1f}% (below {self.min_critical_coverage}%)"
                        )
                else:
                    print(f"✅ Critical file {critical_file}: {file_coverage:.1f}%")
            else:
                critical_failures.append(f"{critical_file}: No coverage data found")
        
        validation['critical_passed'] = len(critical_failures) == 0
        validation['failures'].extend(critical_failures)
        validation['warnings'].extend(critical_warnings)
        
        return validation
    
    def generate_ci_summary(self, results: Dict, validation: Dict) -> str:
        """Generate CI-friendly summary."""
        summary = []
        
        # Status header
        if validation['overall_passed'] and validation['critical_passed']:
            summary.append("🎉 COVERAGE VALIDATION PASSED")
        else:
            summary.append("❌ COVERAGE VALIDATION FAILED")
        
        summary.append("")
        
        # Key metrics
        summary.append("📊 COVERAGE METRICS:")
        summary.append(f"  Overall: {results['overall_coverage']:.2f}% (target: {self.min_coverage}%)")
        summary.append(f"  Branch: {results['branch_coverage']:.2f}%")
        summary.append(f"  Lines: {results['lines_covered']}/{results['lines_valid']}")
        summary.append("")
        
        # Failures
        if validation['failures']:
            summary.append("🚨 FAILURES:")
            for failure in validation['failures']:
                summary.append(f"  • {failure}")
            summary.append("")
        
        # Warnings
        if validation['warnings']:
            summary.append("⚠️  WARNINGS:")
            for warning in validation['warnings']:
                summary.append(f"  • {warning}")
            summary.append("")
        
        # Top files needing attention
        low_coverage_files = [
            (filename, coverage) 
            for filename, coverage in results['file_coverage'].items()
            if coverage < 50
        ]
        
        if low_coverage_files:
            summary.append("📝 FILES NEEDING ATTENTION:")
            for filename, coverage in sorted(low_coverage_files, key=lambda x: x[1])[:5]:
                summary.append(f"  • {filename}: {coverage:.1f}%")
            summary.append("")
        
        return "\n".join(summary)
    
    def export_results_for_ci(self, results: Dict, validation: Dict) -> None:
        """Export results in CI-friendly formats."""
        # JSON for programmatic consumption
        ci_data = {
            'coverage': {
                'overall': results['overall_coverage'],
                'branch': results['branch_coverage'],
                'target': self.min_coverage
            },
            'validation': {
                'passed': validation['overall_passed'] and validation['critical_passed'],
                'overall_passed': validation['overall_passed'],
                'critical_passed': validation['critical_passed']
            },
            'failures': validation['failures'],
            'warnings': validation['warnings']
        }
        
        # Save JSON results
        json_path = self.project_root / "coverage_results.json"
        with open(json_path, 'w') as f:
            json.dump(ci_data, f, indent=2)
        
        # Set GitHub Actions outputs if running in GitHub Actions
        if os.getenv('GITHUB_ACTIONS'):
            self._set_github_outputs(ci_data)
        
        # Set environment variables for other CI systems
        os.environ['COVERAGE_PERCENTAGE'] = str(results['overall_coverage'])
        os.environ['COVERAGE_PASSED'] = str(validation['overall_passed']).lower()
    
    def _set_github_outputs(self, ci_data: Dict) -> None:
        """Set GitHub Actions outputs."""
        github_output = os.getenv('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"coverage-percentage={ci_data['coverage']['overall']:.2f}\n")
                f.write(f"coverage-passed={str(ci_data['validation']['passed']).lower()}\n")
                f.write(f"coverage-target={ci_data['coverage']['target']}\n")
    
    def run_validation(self) -> bool:
        """Run complete coverage validation."""
        print("Starting CI coverage validation...")
        
        # Run tests
        if not self.run_tests_with_coverage():
            print("❌ Tests failed")
            return False
        
        # Parse results
        results = self.parse_coverage_results()
        if not results:
            print("❌ Failed to parse coverage results")
            return False
        
        # Validate requirements
        validation = self.validate_coverage_requirements(results)
        
        # Generate summary
        summary = self.generate_ci_summary(results, validation)
        print(summary)
        
        # Export results
        self.export_results_for_ci(results, validation)
        
        # Return overall success
        success = validation['overall_passed'] and validation['critical_passed']
        
        if success:
            print("✅ Coverage validation passed!")
        else:
            print("❌ Coverage validation failed!")
        
        return success


def main():
    """Main entry point for CI coverage validation."""
    # Get configuration from environment
    min_coverage = float(os.getenv('MIN_COVERAGE', '80.0'))
    min_critical_coverage = float(os.getenv('MIN_CRITICAL_COVERAGE', '90.0'))
    
    # Create validator
    validator = CICoverageValidator(
        min_coverage=min_coverage,
        min_critical_coverage=min_critical_coverage
    )
    
    # Run validation
    success = validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
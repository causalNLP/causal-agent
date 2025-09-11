#!/usr/bin/env python3
"""
Script to identify and help fill coverage gaps in critical code paths.
This script analyzes uncovered lines and suggests test improvements.
"""

import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import xml.etree.ElementTree as ET


class CoverageGapFiller:
    """Identifies coverage gaps and suggests test improvements."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.coverage_xml_path = self.project_root / "coverage.xml"
        self.source_dir = self.project_root / "causal_agent"
        self.test_dir = self.project_root / "tests"
        
    def get_uncovered_lines(self) -> Dict[str, List[int]]:
        """Get uncovered lines from coverage XML."""
        if not self.coverage_xml_path.exists():
            print("Coverage XML not found. Run tests with coverage first.")
            return {}
        
        uncovered_lines = {}
        
        try:
            tree = ET.parse(self.coverage_xml_path)
            root = tree.getroot()
            
            for package in root.findall('.//package'):
                for class_elem in package.findall('.//class'):
                    filename = class_elem.get('filename')
                    uncovered = []
                    
                    for line in class_elem.findall('.//line'):
                        line_num = int(line.get('number'))
                        hits = int(line.get('hits'))
                        if hits == 0:
                            uncovered.append(line_num)
                    
                    if uncovered:
                        uncovered_lines[filename] = uncovered
            
            return uncovered_lines
            
        except Exception as e:
            print(f"Error parsing coverage XML: {str(e)}")
            return {}
    
    def analyze_uncovered_code(self, filename: str, uncovered_lines: List[int]) -> Dict:
        """Analyze uncovered code to understand what needs testing."""
        file_path = self.project_root / filename
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
                lines = source_code.split('\n')
            
            # Parse AST to understand code structure
            tree = ast.parse(source_code)
            
            analysis = {
                'functions': [],
                'classes': [],
                'error_handling': [],
                'edge_cases': [],
                'imports': []
            }
            
            # Analyze uncovered lines
            for line_num in uncovered_lines:
                if line_num <= len(lines):
                    line_content = lines[line_num - 1].strip()
                    
                    # Categorize uncovered code
                    if line_content.startswith('def '):
                        analysis['functions'].append({
                            'line': line_num,
                            'content': line_content,
                            'type': 'function_definition'
                        })
                    elif line_content.startswith('class '):
                        analysis['classes'].append({
                            'line': line_num,
                            'content': line_content,
                            'type': 'class_definition'
                        })
                    elif 'except' in line_content or 'raise' in line_content:
                        analysis['error_handling'].append({
                            'line': line_num,
                            'content': line_content,
                            'type': 'error_handling'
                        })
                    elif 'if' in line_content and ('not' in line_content or 'None' in line_content):
                        analysis['edge_cases'].append({
                            'line': line_num,
                            'content': line_content,
                            'type': 'edge_case'
                        })
                    elif line_content.startswith('import') or line_content.startswith('from'):
                        analysis['imports'].append({
                            'line': line_num,
                            'content': line_content,
                            'type': 'import'
                        })
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {filename}: {str(e)}")
            return {}
    
    def find_existing_tests(self, source_file: str) -> List[str]:
        """Find existing test files for a source file."""
        # Convert source path to potential test paths
        rel_path = Path(source_file)
        
        # Remove causal_agent prefix if present
        if rel_path.parts[0] == 'causal_agent':
            rel_path = Path(*rel_path.parts[1:])
        
        # Potential test file patterns
        test_patterns = [
            f"tests/unit/causal_agent/{rel_path.parent}/test_{rel_path.stem}.py",
            f"tests/unit/causal_agent/test_{rel_path.stem}.py",
            f"tests/integration/test_{rel_path.stem}.py",
            f"tests/cais/{rel_path.parent}/test_{rel_path.stem}.py",
        ]
        
        existing_tests = []
        for pattern in test_patterns:
            test_path = self.project_root / pattern
            if test_path.exists():
                existing_tests.append(str(test_path))
        
        return existing_tests
    
    def suggest_test_improvements(self, filename: str, analysis: Dict) -> List[str]:
        """Suggest specific test improvements based on uncovered code."""
        suggestions = []
        
        # Function coverage suggestions
        if analysis.get('functions'):
            suggestions.append(f"Add tests for {len(analysis['functions'])} uncovered functions:")
            for func in analysis['functions'][:3]:  # Show first 3
                suggestions.append(f"  - Line {func['line']}: {func['content']}")
        
        # Error handling suggestions
        if analysis.get('error_handling'):
            suggestions.append(f"Add tests for {len(analysis['error_handling'])} error handling paths:")
            for error in analysis['error_handling'][:3]:
                suggestions.append(f"  - Line {error['line']}: {error['content']}")
        
        # Edge case suggestions
        if analysis.get('edge_cases'):
            suggestions.append(f"Add tests for {len(analysis['edge_cases'])} edge cases:")
            for edge in analysis['edge_cases'][:3]:
                suggestions.append(f"  - Line {edge['line']}: {edge['content']}")
        
        # Class coverage suggestions
        if analysis.get('classes'):
            suggestions.append(f"Add tests for {len(analysis['classes'])} class definitions:")
            for cls in analysis['classes']:
                suggestions.append(f"  - Line {cls['line']}: {cls['content']}")
        
        if not any([analysis.get('functions'), analysis.get('error_handling'), 
                   analysis.get('edge_cases'), analysis.get('classes')]):
            suggestions.append("Add basic unit tests to cover uncovered lines")
        
        return suggestions
    
    def generate_test_template(self, filename: str, analysis: Dict) -> str:
        """Generate a basic test template for uncovered code."""
        rel_path = Path(filename)
        module_name = str(rel_path.with_suffix('')).replace('/', '.')
        
        template = f'''"""
Test template for {filename}
Generated automatically to improve coverage.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

# Import the module under test
from {module_name} import *


class Test{rel_path.stem.title()}(unittest.TestCase):
    """Test cases for {rel_path.stem} module."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
'''
        
        # Add test methods for uncovered functions
        for func in analysis.get('functions', []):
            func_name = func['content'].split('(')[0].replace('def ', '').strip()
            template += f'''
    def test_{func_name}(self):
        """Test {func_name} function."""
        # TODO: Implement test for line {func['line']}
        # {func['content']}
        pass
'''
        
        # Add test methods for error handling
        if analysis.get('error_handling'):
            template += '''
    def test_error_handling(self):
        """Test error handling paths."""
        # TODO: Add tests for exception handling
        pass
'''
        
        # Add test methods for edge cases
        if analysis.get('edge_cases'):
            template += '''
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # TODO: Add tests for edge cases
        pass
'''
        
        template += '''

if __name__ == '__main__':
    unittest.main()
'''
        
        return template
    
    def create_missing_test_files(self, uncovered_files: Dict[str, List[int]]) -> None:
        """Create missing test files for uncovered code."""
        created_files = []
        
        for filename, uncovered_lines in uncovered_files.items():
            existing_tests = self.find_existing_tests(filename)
            
            if not existing_tests:
                # Create new test file
                analysis = self.analyze_uncovered_code(filename, uncovered_lines)
                template = self.generate_test_template(filename, analysis)
                
                # Determine test file path
                rel_path = Path(filename)
                if rel_path.parts[0] == 'causal_agent':
                    rel_path = Path(*rel_path.parts[1:])
                
                test_path = self.project_root / f"tests/unit/causal_agent/{rel_path.parent}/test_{rel_path.stem}.py"
                test_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write test file
                with open(test_path, 'w') as f:
                    f.write(template)
                
                created_files.append(str(test_path))
                print(f"Created test template: {test_path}")
        
        return created_files
    
    def run_gap_analysis(self) -> Dict:
        """Run comprehensive gap analysis."""
        print("Analyzing coverage gaps...")
        
        uncovered_lines = self.get_uncovered_lines()
        if not uncovered_lines:
            print("No uncovered lines found or coverage data unavailable.")
            return {}
        
        gap_analysis = {}
        
        for filename, lines in uncovered_lines.items():
            if len(lines) > 5:  # Focus on files with significant gaps
                analysis = self.analyze_uncovered_code(filename, lines)
                existing_tests = self.find_existing_tests(filename)
                suggestions = self.suggest_test_improvements(filename, analysis)
                
                gap_analysis[filename] = {
                    'uncovered_lines': lines,
                    'line_count': len(lines),
                    'analysis': analysis,
                    'existing_tests': existing_tests,
                    'suggestions': suggestions
                }
        
        return gap_analysis
    
    def generate_gap_report(self, gap_analysis: Dict) -> str:
        """Generate a detailed gap analysis report."""
        report = []
        report.append("=" * 80)
        report.append("COVERAGE GAP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_files = len(gap_analysis)
        total_uncovered = sum(data['line_count'] for data in gap_analysis.values())
        
        report.append(f"Files with significant coverage gaps: {total_files}")
        report.append(f"Total uncovered lines analyzed: {total_uncovered}")
        report.append("")
        
        # Detailed analysis for each file
        for filename, data in sorted(gap_analysis.items(), key=lambda x: x[1]['line_count'], reverse=True):
            report.append(f"FILE: {filename}")
            report.append(f"Uncovered lines: {data['line_count']}")
            report.append(f"Existing tests: {len(data['existing_tests'])}")
            
            if data['existing_tests']:
                for test_file in data['existing_tests']:
                    report.append(f"  - {test_file}")
            else:
                report.append("  - No existing tests found")
            
            report.append("")
            report.append("SUGGESTIONS:")
            for suggestion in data['suggestions']:
                report.append(f"  {suggestion}")
            
            report.append("")
            report.append("-" * 40)
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    gap_filler = CoverageGapFiller()
    
    # Run gap analysis
    gap_analysis = gap_filler.run_gap_analysis()
    
    if not gap_analysis:
        print("No significant coverage gaps found.")
        return
    
    # Generate and display report
    report = gap_filler.generate_gap_report(gap_analysis)
    print(report)
    
    # Save report
    report_path = gap_filler.project_root / "coverage_gaps_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Gap analysis report saved to: {report_path}")
    
    # Ask if user wants to create test templates
    response = input("\nCreate test templates for files without tests? (y/n): ")
    if response.lower() == 'y':
        uncovered_files = {
            filename: data['uncovered_lines'] 
            for filename, data in gap_analysis.items()
            if not data['existing_tests']
        }
        
        if uncovered_files:
            created_files = gap_filler.create_missing_test_files(uncovered_files)
            print(f"Created {len(created_files)} test template files.")
        else:
            print("All files already have existing tests.")


if __name__ == "__main__":
    main()
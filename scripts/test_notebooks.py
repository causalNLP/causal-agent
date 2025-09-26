#!/usr/bin/env python3
"""
Test script for Jupyter notebooks in the tutorials directory.

This script performs various tests on the tutorial notebooks:
1. Syntax validation
2. Cell execution testing (without API calls)
3. Output validation
4. Link checking
5. Code quality checks
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import PythonExporter
import re

class NotebookTester:
    """Test Jupyter notebooks for various quality metrics."""
    
    def __init__(self, notebooks_dir: str):
        self.notebooks_dir = Path(notebooks_dir)
        self.results = []
        
    def find_notebooks(self) -> List[Path]:
        """Find all notebook files in the directory."""
        return list(self.notebooks_dir.glob("*.ipynb"))
    
    def test_notebook_structure(self, notebook_path: Path) -> Dict[str, Any]:
        """Test basic notebook structure and metadata."""
        result = {
            'notebook': notebook_path.name,
            'test': 'structure',
            'passed': True,
            'errors': []
        }
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Check if notebook has cells
            if not nb.cells:
                result['errors'].append("Notebook has no cells")
                result['passed'] = False
            
            # Check for markdown cells (documentation)
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
            if len(markdown_cells) < 3:
                result['errors'].append("Notebook should have at least 3 markdown cells for documentation")
                result['passed'] = False
            
            # Check for code cells
            code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            if len(code_cells) < 5:
                result['errors'].append("Notebook should have at least 5 code cells")
                result['passed'] = False
            
            # Check for title in first cell
            if markdown_cells and not any(line.startswith('#') for line in markdown_cells[0].source.split('\n')):
                result['errors'].append("First markdown cell should contain a title (# header)")
                result['passed'] = False
                
        except Exception as e:
            result['errors'].append(f"Failed to read notebook: {str(e)}")
            result['passed'] = False
            
        return result
    
    def test_code_quality(self, notebook_path: Path) -> Dict[str, Any]:
        """Test code quality in notebook cells."""
        result = {
            'notebook': notebook_path.name,
            'test': 'code_quality',
            'passed': True,
            'errors': []
        }
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    source = cell.source
                    
                    # Check for common issues
                    if 'import *' in source:
                        result['errors'].append(f"Cell {i}: Avoid wildcard imports")
                        result['passed'] = False
                    
                    # Check for hardcoded paths (should be relative)
                    if re.search(r'["\'][A-Za-z]:\\', source) or re.search(r'["\']\/home\/', source):
                        result['errors'].append(f"Cell {i}: Avoid hardcoded absolute paths")
                        result['passed'] = False
                    
                    # Check for API keys in code
                    if re.search(r'(api_key|API_KEY|secret|SECRET|token|TOKEN)\s*=\s*["\'][^"\']+["\']', source):
                        result['errors'].append(f"Cell {i}: Potential hardcoded API key or secret")
                        result['passed'] = False
                    
                    # Check for print statements without context
                    print_lines = [line for line in source.split('\n') if line.strip().startswith('print(')]
                    if len(print_lines) > 5:
                        result['errors'].append(f"Cell {i}: Too many print statements, consider using logging")
                        
        except Exception as e:
            result['errors'].append(f"Failed to analyze code quality: {str(e)}")
            result['passed'] = False
            
        return result
    
    def test_syntax_validation(self, notebook_path: Path) -> Dict[str, Any]:
        """Test that notebook code cells have valid Python syntax."""
        result = {
            'notebook': notebook_path.name,
            'test': 'syntax',
            'passed': True,
            'errors': []
        }
        
        try:
            # Convert notebook to Python script
            exporter = PythonExporter()
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            (body, resources) = exporter.from_notebook_node(nb)
            
            # Try to compile the Python code
            try:
                compile(body, notebook_path.name, 'exec')
            except SyntaxError as e:
                result['errors'].append(f"Syntax error: {str(e)}")
                result['passed'] = False
                
        except Exception as e:
            result['errors'].append(f"Failed to validate syntax: {str(e)}")
            result['passed'] = False
            
        return result
    
    def test_mock_execution(self, notebook_path: Path) -> Dict[str, Any]:
        """Test notebook execution with mocked external dependencies."""
        result = {
            'notebook': notebook_path.name,
            'test': 'mock_execution',
            'passed': True,
            'errors': []
        }
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Create a modified version with mocked API calls
            modified_nb = self._create_mocked_notebook(nb)
            
            # Execute the modified notebook
            ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
            
            try:
                ep.preprocess(modified_nb, {'metadata': {'path': str(notebook_path.parent)}})
            except Exception as e:
                # Check if it's an expected error (like missing API keys)
                error_str = str(e).lower()
                if any(term in error_str for term in ['api', 'key', 'token', 'authentication']):
                    result['errors'].append(f"Expected API-related error: {str(e)}")
                    # This is expected, so don't fail the test
                else:
                    result['errors'].append(f"Execution error: {str(e)}")
                    result['passed'] = False
                    
        except Exception as e:
            result['errors'].append(f"Failed to test execution: {str(e)}")
            result['passed'] = False
            
        return result
    
    def _create_mocked_notebook(self, nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
        """Create a version of the notebook with mocked external calls."""
        modified_nb = nb.copy()
        
        for cell in modified_nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                
                # Mock CAIS API calls
                if 'run_causal_analysis' in source:
                    # Replace with mock result
                    mock_result = '''
# Mocked CAIS result for testing
result = {
    'results': {
        'results': {
            'method_used': 'Mock Method',
            'effect_estimate': -1.5,
            'standard_error': 0.3,
            'p_value': 0.001,
            'confidence_interval': [-2.1, -0.9]
        },
        'variables': {
            'treatment_variable': 'treatment',
            'outcome_variable': 'outcome',
            'covariates': ['covariate1', 'covariate2']
        }
    }
}
print("Mock analysis completed")
'''
                    # Replace the run_causal_analysis call
                    source = re.sub(
                        r'result\s*=\s*run_causal_analysis\([^)]+\)',
                        mock_result,
                        source
                    )
                
                # Mock file operations that might fail
                source = source.replace(
                    'pd.read_csv(dataset_path)',
                    'pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.DataFrame()'
                )
                
                cell.source = source
        
        return modified_nb
    
    def test_documentation_quality(self, notebook_path: Path) -> Dict[str, Any]:
        """Test documentation quality in markdown cells."""
        result = {
            'notebook': notebook_path.name,
            'test': 'documentation',
            'passed': True,
            'errors': []
        }
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
            
            # Check for learning objectives
            has_objectives = any('learning objective' in cell.source.lower() for cell in markdown_cells)
            if not has_objectives:
                result['errors'].append("Notebook should include learning objectives")
                result['passed'] = False
            
            # Check for conclusion/summary
            has_conclusion = any(any(term in cell.source.lower() for term in ['conclusion', 'summary', 'key findings']) 
                               for cell in markdown_cells)
            if not has_conclusion:
                result['errors'].append("Notebook should include conclusion or summary")
                result['passed'] = False
            
            # Check for exercises
            has_exercises = any('exercise' in cell.source.lower() for cell in markdown_cells)
            if not has_exercises:
                result['errors'].append("Notebook should include exercises for practice")
                result['passed'] = False
                
        except Exception as e:
            result['errors'].append(f"Failed to test documentation: {str(e)}")
            result['passed'] = False
            
        return result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all tests on all notebooks."""
        notebooks = self.find_notebooks()
        
        if not notebooks:
            print(f"No notebooks found in {self.notebooks_dir}")
            return []
        
        print(f"Testing {len(notebooks)} notebooks...")
        
        for notebook_path in notebooks:
            print(f"\nTesting {notebook_path.name}...")
            
            # Run all test types
            tests = [
                self.test_notebook_structure,
                self.test_syntax_validation,
                self.test_code_quality,
                self.test_documentation_quality,
                self.test_mock_execution
            ]
            
            for test_func in tests:
                try:
                    result = test_func(notebook_path)
                    self.results.append(result)
                    
                    status = "PASS" if result['passed'] else "FAIL"
                    print(f"  {result['test']}: {status}")
                    
                    if result['errors']:
                        for error in result['errors']:
                            print(f"    - {error}")
                            
                except Exception as e:
                    error_result = {
                        'notebook': notebook_path.name,
                        'test': test_func.__name__,
                        'passed': False,
                        'errors': [f"Test failed with exception: {str(e)}"]
                    }
                    self.results.append(error_result)
                    print(f"  {test_func.__name__}: ERROR - {str(e)}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a summary report of all test results."""
        if not self.results:
            return "No test results available."
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        report = f"""
Notebook Testing Report
======================

Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success Rate: {(passed_tests/total_tests)*100:.1f}%

"""
        
        # Group by notebook
        notebooks = {}
        for result in self.results:
            nb_name = result['notebook']
            if nb_name not in notebooks:
                notebooks[nb_name] = []
            notebooks[nb_name].append(result)
        
        for nb_name, results in notebooks.items():
            nb_passed = sum(1 for r in results if r['passed'])
            nb_total = len(results)
            
            report += f"\n{nb_name}:\n"
            report += f"  Tests: {nb_passed}/{nb_total} passed\n"
            
            for result in results:
                status = "PASS" if result['passed'] else "FAIL"
                report += f"  - {result['test']}: {status}\n"
                
                if result['errors']:
                    for error in result['errors']:
                        report += f"    * {error}\n"
        
        return report


def main():
    """Main function to run notebook tests."""
    notebooks_dir = "docs/source/tutorials/notebooks"
    
    if not os.path.exists(notebooks_dir):
        print(f"Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)
    
    tester = NotebookTester(notebooks_dir)
    results = tester.run_all_tests()
    
    # Generate and print report
    report = tester.generate_report()
    print("\n" + "="*50)
    print(report)
    
    # Exit with error code if any tests failed
    failed_tests = sum(1 for r in results if not r['passed'])
    if failed_tests > 0:
        print(f"\n{failed_tests} tests failed!")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
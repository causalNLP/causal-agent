#!/usr/bin/env python3
"""
Docstring validation script for CAIS project.

This script validates that all public functions and classes have proper docstrings
following Google style conventions.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any


class DocstringValidator(ast.NodeVisitor):
    """Validates docstrings in Python source files."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Dict[str, Any]] = []
        self.current_class = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions and validate docstrings."""
        old_class = self.current_class
        self.current_class = node.name
        
        if not node.name.startswith('_'):  # Public class
            self._validate_docstring(node, 'class', node.name)
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and validate docstrings."""
        if not node.name.startswith('_') or node.name == '__init__':  # Public function or __init__
            context = f"{self.current_class}.{node.name}" if self.current_class else node.name
            self._validate_docstring(node, 'function', context)
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions and validate docstrings."""
        if not node.name.startswith('_'):  # Public function
            context = f"{self.current_class}.{node.name}" if self.current_class else node.name
            self._validate_docstring(node, 'async function', context)
        
        self.generic_visit(node)
    
    def _validate_docstring(self, node: ast.AST, node_type: str, name: str) -> None:
        """Validate docstring for a given node."""
        docstring = ast.get_docstring(node)
        
        if not docstring:
            self.issues.append({
                'type': 'missing_docstring',
                'severity': 'error',
                'line': node.lineno,
                'name': name,
                'node_type': node_type,
                'message': f"Missing docstring for {node_type} '{name}'"
            })
            return
        
        # Check for basic docstring quality
        if len(docstring.strip()) < 10:
            self.issues.append({
                'type': 'short_docstring',
                'severity': 'warning',
                'line': node.lineno,
                'name': name,
                'node_type': node_type,
                'message': f"Docstring for {node_type} '{name}' is too short"
            })
        
        # Check for Google-style docstring sections for functions
        if node_type in ['function', 'async function'] and hasattr(node, 'args'):
            self._validate_function_docstring(node, docstring, name)
    
    def _validate_function_docstring(self, node: ast.FunctionDef, docstring: str, name: str) -> None:
        """Validate function-specific docstring requirements."""
        # Check if function has parameters (excluding self)
        args = [arg.arg for arg in node.args.args if arg.arg != 'self']
        
        if args and 'Args:' not in docstring and 'Arguments:' not in docstring:
            self.issues.append({
                'type': 'missing_args_section',
                'severity': 'warning',
                'line': node.lineno,
                'name': name,
                'node_type': 'function',
                'message': f"Function '{name}' has parameters but no Args section in docstring"
            })
        
        # Check for return statement and Returns section
        has_return = any(isinstance(n, ast.Return) and n.value is not None 
                        for n in ast.walk(node))
        
        if has_return and 'Returns:' not in docstring and 'Return:' not in docstring:
            self.issues.append({
                'type': 'missing_returns_section',
                'severity': 'warning',
                'line': node.lineno,
                'name': name,
                'node_type': 'function',
                'message': f"Function '{name}' has return statement but no Returns section in docstring"
            })


def validate_file(filepath: Path) -> List[Dict[str, Any]]:
    """Validate docstrings in a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(filepath))
        validator = DocstringValidator(str(filepath))
        validator.visit(tree)
        
        return validator.issues
    except SyntaxError as e:
        return [{
            'type': 'syntax_error',
            'severity': 'error',
            'line': e.lineno or 0,
            'name': str(filepath),
            'node_type': 'file',
            'message': f"Syntax error in file: {e.msg}"
        }]
    except Exception as e:
        return [{
            'type': 'validation_error',
            'severity': 'error',
            'line': 0,
            'name': str(filepath),
            'node_type': 'file',
            'message': f"Error validating file: {str(e)}"
        }]


def validate_package(package_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Validate docstrings in all Python files in a package."""
    results = {}
    
    for py_file in package_path.rglob('*.py'):
        if py_file.name.startswith('test_'):
            continue  # Skip test files
        
        relative_path = py_file.relative_to(package_path.parent)
        issues = validate_file(py_file)
        
        if issues:
            results[str(relative_path)] = issues
    
    return results


def print_results(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print validation results in a readable format."""
    total_errors = 0
    total_warnings = 0
    
    for filepath, issues in results.items():
        print(f"\n{filepath}:")
        print("-" * len(filepath))
        
        for issue in issues:
            severity_symbol = "❌" if issue['severity'] == 'error' else "⚠️"
            print(f"  {severity_symbol} Line {issue['line']}: {issue['message']}")
            
            if issue['severity'] == 'error':
                total_errors += 1
            else:
                total_warnings += 1
    
    print(f"\n📊 Summary:")
    print(f"   Errors: {total_errors}")
    print(f"   Warnings: {total_warnings}")
    print(f"   Files with issues: {len(results)}")
    
    return total_errors, total_warnings


def main():
    """Main function to run docstring validation."""
    if len(sys.argv) > 1:
        package_path = Path(sys.argv[1])
    else:
        # Default to causal_agent package
        package_path = Path(__file__).parent.parent / 'causal_agent'
    
    if not package_path.exists():
        print(f"❌ Package path does not exist: {package_path}")
        sys.exit(1)
    
    print(f"🔍 Validating docstrings in: {package_path}")
    results = validate_package(package_path)
    
    if not results:
        print("✅ All docstrings are valid!")
        sys.exit(0)
    
    errors, warnings = print_results(results)
    
    # Exit with error code if there are errors
    if errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
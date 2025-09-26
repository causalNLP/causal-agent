#!/usr/bin/env python3
"""
Code example execution testing for documentation.
Extracts and tests code examples from RST and HTML files.
"""

import os
import sys
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast
import traceback

class CodeExampleTester:
    """Test code examples in documentation."""
    
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.source_dir = self.docs_dir / "source"
        self.html_dir = self.docs_dir / "build" / "html"
        self.test_results = []
        self.code_examples = []
        
        # Common imports that should be available
        self.common_imports = [
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import sys",
            "import os",
            "from pathlib import Path",
            "import json",
            "import warnings",
            "warnings.filterwarnings('ignore')",
            "# Add project root to path for imports",
            "sys.path.insert(0, os.path.abspath('../../'))",
        ]
    
    def run_all_tests(self) -> bool:
        """Run all code example tests."""
        print("Starting code example testing...")
        
        # Extract code examples from source files
        self._extract_code_examples()
        
        if not self.code_examples:
            print("No code examples found to test")
            return True
            
        # Test code examples
        success = self._test_code_examples()
        
        # Generate report
        self._generate_report()
        
        return success
    
    def _extract_code_examples(self):
        """Extract code examples from documentation files."""
        print("Extracting code examples...")
        
        # Extract from RST files
        if self.source_dir.exists():
            rst_files = list(self.source_dir.rglob("*.rst"))
            for rst_file in rst_files:
                self._extract_from_rst(rst_file)
                
        # Extract from Python files in docs
        py_files = list(self.docs_dir.rglob("*.py"))
        for py_file in py_files:
            if "test_" not in py_file.name:  # Skip test files themselves
                self._extract_from_python(py_file)
        
        print(f"Found {len(self.code_examples)} code examples")
    
    def _extract_from_rst(self, rst_file: Path):
        """Extract code examples from RST file."""
        try:
            with open(rst_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find code-block directives
            code_blocks = re.finditer(
                r'^\.\. code-block::\s*(\w+)\s*\n(.*?)(?=^\S|\Z)',
                content,
                re.MULTILINE | re.DOTALL
            )
            
            for match in code_blocks:
                language = match.group(1)
                code_content = match.group(2)
                
                # Only test Python code
                if language.lower() in ['python', 'py']:
                    # Remove indentation
                    lines = code_content.split('\n')
                    # Find minimum indentation (excluding empty lines)
                    min_indent = float('inf')
                    for line in lines:
                        if line.strip():
                            indent = len(line) - len(line.lstrip())
                            min_indent = min(min_indent, indent)
                    
                    if min_indent == float('inf'):
                        min_indent = 0
                        
                    # Remove common indentation
                    cleaned_lines = []
                    for line in lines:
                        if line.strip():
                            cleaned_lines.append(line[min_indent:])
                        else:
                            cleaned_lines.append('')
                    
                    code = '\n'.join(cleaned_lines).strip()
                    
                    if code:
                        self.code_examples.append({
                            'source_file': str(rst_file.relative_to(self.docs_dir)),
                            'language': language,
                            'code': code,
                            'type': 'rst_code_block'
                        })
                        
            # Find literal code blocks (::)
            literal_blocks = re.finditer(
                r'::\s*\n\n((?:[ \t]+.*\n?)+)',
                content,
                re.MULTILINE
            )
            
            for match in literal_blocks:
                code_content = match.group(1)
                
                # Check if it looks like Python code
                if self._looks_like_python(code_content):
                    # Remove indentation
                    lines = code_content.split('\n')
                    min_indent = float('inf')
                    for line in lines:
                        if line.strip():
                            indent = len(line) - len(line.lstrip())
                            min_indent = min(min_indent, indent)
                    
                    if min_indent == float('inf'):
                        min_indent = 0
                        
                    cleaned_lines = []
                    for line in lines:
                        if line.strip():
                            cleaned_lines.append(line[min_indent:])
                        else:
                            cleaned_lines.append('')
                    
                    code = '\n'.join(cleaned_lines).strip()
                    
                    if code:
                        self.code_examples.append({
                            'source_file': str(rst_file.relative_to(self.docs_dir)),
                            'language': 'python',
                            'code': code,
                            'type': 'rst_literal_block'
                        })
                        
        except Exception as e:
            print(f"WARNING: Could not extract from {rst_file}: {e}")
    
    def _extract_from_python(self, py_file: Path):
        """Extract code examples from Python docstrings."""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the Python file
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return  # Skip files with syntax errors
                
            # Extract docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Look for code examples in docstring
                        examples = self._extract_docstring_examples(docstring)
                        for example in examples:
                            self.code_examples.append({
                                'source_file': str(py_file.relative_to(self.docs_dir)),
                                'language': 'python',
                                'code': example,
                                'type': 'docstring_example',
                                'function': getattr(node, 'name', 'module')
                            })
                            
        except Exception as e:
            print(f"WARNING: Could not extract from {py_file}: {e}")
    
    def _extract_docstring_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        
        # Look for Examples section
        examples_match = re.search(
            r'Examples?\s*:?\s*\n\s*-+\s*\n(.*?)(?=\n\s*\w+\s*:?\s*\n\s*-+|\Z)',
            docstring,
            re.DOTALL | re.IGNORECASE
        )
        
        if examples_match:
            examples_text = examples_match.group(1)
            
            # Find code blocks (>>> or indented blocks)
            code_blocks = re.finditer(
                r'(?:>>>.*(?:\n(?:\.\.\.|\s+).*)*)|(?:^\s{4,}.*(?:\n\s{4,}.*)*)',
                examples_text,
                re.MULTILINE
            )
            
            for match in code_blocks:
                code_block = match.group(0)
                
                # Clean up doctest format
                if '>>>' in code_block:
                    lines = code_block.split('\n')
                    code_lines = []
                    for line in lines:
                        if line.strip().startswith('>>>'):
                            code_lines.append(line.strip()[4:])
                        elif line.strip().startswith('...'):
                            code_lines.append(line.strip()[4:])
                    code = '\n'.join(code_lines).strip()
                else:
                    # Remove common indentation
                    lines = code_block.split('\n')
                    min_indent = float('inf')
                    for line in lines:
                        if line.strip():
                            indent = len(line) - len(line.lstrip())
                            min_indent = min(min_indent, indent)
                    
                    if min_indent == float('inf'):
                        min_indent = 0
                        
                    code_lines = []
                    for line in lines:
                        if line.strip():
                            code_lines.append(line[min_indent:])
                        else:
                            code_lines.append('')
                    code = '\n'.join(code_lines).strip()
                
                if code and self._looks_like_python(code):
                    examples.append(code)
        
        return examples
    
    def _looks_like_python(self, code: str) -> bool:
        """Check if code looks like Python."""
        python_keywords = [
            'import', 'from', 'def', 'class', 'if', 'for', 'while',
            'try', 'except', 'with', 'return', 'yield', 'print'
        ]
        
        # Check for Python keywords
        for keyword in python_keywords:
            if re.search(rf'\b{keyword}\b', code):
                return True
                
        # Check for Python-like syntax
        python_patterns = [
            r'^\s*#',  # Comments
            r'=\s*\[',  # List assignment
            r'=\s*\{',  # Dict assignment
            r'\.append\(',  # Method calls
            r'\.format\(',
            r'f["\']',  # f-strings
        ]
        
        for pattern in python_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return True
                
        return False
    
    def _test_code_examples(self) -> bool:
        """Test all extracted code examples."""
        print(f"Testing {len(self.code_examples)} code examples...")
        
        all_passed = True
        
        for i, example in enumerate(self.code_examples):
            print(f"Testing example {i+1}/{len(self.code_examples)}: {example['source_file']}")
            
            result = self._test_single_example(example)
            self.test_results.append(result)
            
            if not result['passed']:
                all_passed = False
                print(f"  ❌ FAILED: {result['error']}")
            else:
                print(f"  ✅ PASSED")
        
        return all_passed
    
    def _test_single_example(self, example: Dict) -> Dict:
        """Test a single code example."""
        result = {
            'source_file': example['source_file'],
            'type': example['type'],
            'passed': False,
            'error': None,
            'output': None
        }
        
        try:
            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add common imports
                f.write('\n'.join(self.common_imports) + '\n\n')
                
                # Add the example code
                f.write(example['code'])
                
                temp_file = f.name
            
            try:
                # Run the code
                process = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout
                    cwd=self.docs_dir  # Run from docs directory
                )
                
                if process.returncode == 0:
                    result['passed'] = True
                    result['output'] = process.stdout
                else:
                    result['error'] = process.stderr or f"Exit code: {process.returncode}"
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            result['error'] = "Code execution timed out"
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _generate_report(self):
        """Generate code example testing report."""
        report_file = self.docs_dir / "code_examples_report.json"
        
        passed = sum(1 for result in self.test_results if result['passed'])
        total = len(self.test_results)
        
        report = {
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_examples': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': (passed / total * 100) if total > 0 else 0
            },
            'results': self.test_results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 Code examples report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("CODE EXAMPLES TEST SUMMARY")
        print("="*50)
        print(f"Total examples: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        
        # Show failed examples
        failed_results = [r for r in self.test_results if not r['passed']]
        if failed_results:
            print(f"\nFailed examples:")
            for result in failed_results[:5]:  # Show first 5
                print(f"  {result['source_file']}: {result['error']}")
        
        if passed == total:
            print("🎉 All code examples passed!")
        else:
            print(f"⚠️  {total - passed} code examples failed")

def main():
    """Main function to run code example tests."""
    if len(sys.argv) < 2:
        print("Usage: python test_code_examples.py <docs_directory>")
        sys.exit(1)
        
    docs_dir = sys.argv[1]
    
    tester = CodeExampleTester(docs_dir)
    success = tester.run_all_tests()
    
    # Don't fail build for code example failures, just report them
    sys.exit(0)

if __name__ == "__main__":
    main()
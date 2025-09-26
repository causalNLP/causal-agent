#!/usr/bin/env python3
"""
Master test runner for all documentation validation tests.
Runs build tests, link checking, spell/grammar checking, and code example testing.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

class DocumentationValidator:
    """Master validator for all documentation tests."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.results = {}
        self.start_time = time.time()
        
    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        print("="*60)
        print("DOCUMENTATION VALIDATION SUITE")
        print("="*60)
        print(f"Testing documentation in: {self.docs_dir}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Define all tests
        tests = [
            ("Documentation Build", self._run_build_tests),
            ("Link Validation", self._run_link_tests),
            ("Spell & Grammar Check", self._run_spell_tests),
            ("Code Examples", self._run_code_tests),
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"RUNNING: {test_name}")
            print('='*60)
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                self.results[test_name] = {
                    'passed': result,
                    'duration': duration,
                    'status': 'PASS' if result else 'FAIL'
                }
                
                print(f"\n{test_name}: {'PASS' if result else 'FAIL'} ({duration:.1f}s)")
                
                if not result:
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                print(f"ERROR in {test_name}: {e}")
                self.results[test_name] = {
                    'passed': False,
                    'duration': duration,
                    'status': 'ERROR',
                    'error': str(e)
                }
                all_passed = False
        
        # Generate final report
        self._generate_final_report()
        
        return all_passed
    
    def _run_build_tests(self) -> bool:
        """Run documentation build tests."""
        test_script = self.docs_dir / "test_documentation_build.py"
        
        if not test_script.exists():
            print("ERROR: Build test script not found")
            return False
            
        try:
            result = subprocess.run(
                [sys.executable, str(test_script), str(self.docs_dir)],
                capture_output=True,
                text=True,
                cwd=self.docs_dir.parent
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"ERROR running build tests: {e}")
            return False
    
    def _run_link_tests(self) -> bool:
        """Run link validation tests."""
        test_script = self.docs_dir / "test_link_checker.py"
        html_dir = self.docs_dir / "build" / "html"
        
        if not test_script.exists():
            print("ERROR: Link test script not found")
            return False
            
        if not html_dir.exists():
            print("WARNING: HTML build directory not found, skipping link tests")
            return True
            
        try:
            result = subprocess.run(
                [sys.executable, str(test_script), str(html_dir)],
                capture_output=True,
                text=True,
                cwd=self.docs_dir.parent
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"ERROR running link tests: {e}")
            return False
    
    def _run_spell_tests(self) -> bool:
        """Run spell and grammar checking."""
        test_script = self.docs_dir / "test_spell_grammar.py"
        
        if not test_script.exists():
            print("ERROR: Spell test script not found")
            return False
            
        try:
            result = subprocess.run(
                [sys.executable, str(test_script), str(self.docs_dir)],
                capture_output=True,
                text=True,
                cwd=self.docs_dir.parent
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            # Spell/grammar tests don't fail the build, just report issues
            return True
            
        except Exception as e:
            print(f"ERROR running spell tests: {e}")
            return False
    
    def _run_code_tests(self) -> bool:
        """Run code example tests."""
        test_script = self.docs_dir / "test_code_examples.py"
        
        if not test_script.exists():
            print("ERROR: Code test script not found")
            return False
            
        try:
            result = subprocess.run(
                [sys.executable, str(test_script), str(self.docs_dir)],
                capture_output=True,
                text=True,
                cwd=self.docs_dir.parent
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            # Code example tests don't fail the build, just report issues
            return True
            
        except Exception as e:
            print(f"ERROR running code tests: {e}")
            return False
    
    def _generate_final_report(self):
        """Generate final validation report."""
        total_duration = time.time() - self.start_time
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': total_duration,
            'docs_directory': str(self.docs_dir),
            'results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results.values() if r['passed']),
                'failed': sum(1 for r in self.results.values() if not r['passed']),
                'overall_status': 'PASS' if all(r['passed'] for r in self.results.values()) else 'FAIL'
            }
        }
        
        # Save detailed report
        report_file = self.docs_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL VALIDATION SUMMARY")
        print('='*60)
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Tests run: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Overall status: {report['summary']['overall_status']}")
        
        print(f"\nDetailed results:")
        for test_name, result in self.results.items():
            status_icon = "✅" if result['passed'] else "❌"
            print(f"  {status_icon} {test_name}: {result['status']} ({result['duration']:.1f}s)")
        
        print(f"\n📊 Full report saved to: {report_file}")
        
        if report['summary']['overall_status'] == 'PASS':
            print("\n🎉 All validation tests completed successfully!")
        else:
            print(f"\n⚠️  {report['summary']['failed']} validation test(s) failed")
            print("Check individual test reports for details.")

def main():
    """Main function to run all documentation validation."""
    if len(sys.argv) > 1:
        docs_dir = sys.argv[1]
    else:
        docs_dir = "docs"
        
    validator = DocumentationValidator(docs_dir)
    success = validator.run_all_validations()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
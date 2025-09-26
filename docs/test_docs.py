#!/usr/bin/env python3
"""
Simple documentation test runner.
Quick way to run essential documentation validation tests.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run essential documentation tests."""
    docs_dir = Path(__file__).parent
    
    print("🚀 Running essential documentation tests...")
    print(f"📁 Documentation directory: {docs_dir}")
    
    # Install test dependencies if needed
    try:
        import requests
        import spellchecker
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"
        ], cwd=docs_dir, check=True)
    
    # Run the comprehensive validation suite
    print("\n🔍 Running validation tests...")
    result = subprocess.run([
        sys.executable, "test_all_validation.py", str(docs_dir)
    ], cwd=docs_dir)
    
    if result.returncode == 0:
        print("\n✅ All tests completed successfully!")
        print("📊 Check the generated reports in the docs directory:")
        print("  - validation_report.json")
        print("  - link_check_report.json")
        print("  - spell_grammar_report.json")
        print("  - code_examples_report.json")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
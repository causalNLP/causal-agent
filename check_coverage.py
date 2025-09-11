#!/usr/bin/env python3
"""Coverage checking script that works around current test issues."""

import subprocess
import sys
import os

def main():
    """Run coverage analysis on working tests."""
    
    print("🔍 Checking Coverage on Working Tests")
    print("=" * 50)
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['CAUSAL_AGENT_TEST_MODE'] = 'true'
    
    # List of working test files (excluding broken ones)
    working_tests = [
        'tests/unit/causal_agent/test_base_classes.py',
        'tests/unit/causal_agent/components/test_dataset_analyzer.py',
        'tests/unit/causal_agent/components/test_decision_tree.py',
        'tests/unit/causal_agent/components/test_state_manager.py',
        'tests/unit/causal_agent/tools/test_dataset_analyzer_tool.py',
        'tests/fixtures/test_config.py',
        'tests/test_fixtures_integration.py',
    ]
    
    # Filter to only existing files
    existing_tests = [test for test in working_tests if os.path.exists(test)]
    
    if not existing_tests:
        print("❌ No working test files found")
        return 1
    
    print(f"📋 Running coverage on {len(existing_tests)} test files:")
    for test in existing_tests:
        print(f"  - {test}")
    
    # Run pytest with coverage
    cmd = [
        sys.executable, '-m', 'pytest',
        *existing_tests,
        '--cov=causal_agent',
        '--cov-report=term-missing',
        '--cov-report=html',
        '--cov-report=xml',
        '-v'
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("\n✅ Coverage analysis completed successfully!")
            print("📊 Check htmlcov/index.html for detailed coverage report")
            print("📄 XML report saved to coverage.xml")
        else:
            print(f"\n⚠️  Coverage analysis completed with return code: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running coverage: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
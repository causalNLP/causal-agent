#!/usr/bin/env python3
"""
Script to run real LLM integration tests.

This script checks for API key availability and runs the appropriate tests.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def check_api_key():
    """Check if OpenAI API key is available."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI API key not found in environment variables")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    if api_key.startswith("sk-"):
        print("✅ OpenAI API key found")
        return True
    else:
        print("⚠️  OpenAI API key found but format looks incorrect")
        print("Expected format: sk-...")
        return False


def run_basic_tests():
    """Run basic LLM integration tests."""
    print("\n🧪 Running basic LLM integration tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_llm_integration_basic.py",
        "-v",
        "-m", "requires_llm",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Basic LLM tests passed!")
            return True
        else:
            print(f"❌ Basic LLM tests failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_comprehensive_tests():
    """Run comprehensive LLM integration tests."""
    print("\n🧪 Running comprehensive LLM integration tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_real_llm_workflows.py",
        "-v",
        "-m", "requires_llm",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Comprehensive LLM tests passed!")
            return True
        else:
            print(f"❌ Comprehensive LLM tests failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_single_test():
    """Run a single quick test to verify LLM integration."""
    print("\n🧪 Running single quick test...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_llm_integration_basic.py::TestBasicLLMIntegration::test_simple_rct_analysis_real_llm",
        "-v",
        "-s",  # Don't capture output
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=120)
        
        if result.returncode == 0:
            print("✅ Single test passed!")
            return True
        else:
            print(f"❌ Single test failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    """Main function to run LLM tests."""
    print("🚀 Real LLM Integration Test Runner")
    print("=" * 50)
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    # Get test mode from command line
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "basic"
    
    print(f"\n📋 Test mode: {mode}")
    
    success = False
    
    if mode == "single":
        success = run_single_test()
    elif mode == "basic":
        success = run_basic_tests()
    elif mode == "comprehensive" or mode == "full":
        success = run_comprehensive_tests()
    else:
        print(f"❌ Unknown test mode: {mode}")
        print("Available modes: single, basic, comprehensive")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
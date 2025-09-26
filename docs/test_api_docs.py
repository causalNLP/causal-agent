#!/usr/bin/env python3
"""
Test script for API documentation generation.

This script tests that the API documentation system is working correctly
by checking that modules can be imported and documented.
"""

import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any


def test_module_imports() -> Dict[str, Any]:
    """Test that all modules can be imported successfully."""
    
    results = {
        'success': [],
        'failed': [],
        'total': 0
    }
    
    # List of modules to test
    modules_to_test = [
        'causal_agent',
        'causal_agent.agent',
        'causal_agent.config',
        'causal_agent.models',
        'causal_agent.components.dataset_analyzer',
        'causal_agent.components.decision_tree',
        'causal_agent.components.explanation_generator',
        'causal_agent.methods.causal_method',
        'causal_agent.methods.diff_in_means.estimator',
        'causal_agent.methods.difference_in_differences.estimator',
        'causal_agent.tools.data_analyzer',
        'causal_agent.utils.llm_helpers',
    ]
    
    for module_name in modules_to_test:
        results['total'] += 1
        try:
            module = importlib.import_module(module_name)
            results['success'].append({
                'name': module_name,
                'docstring': getattr(module, '__doc__', None),
                'has_docstring': bool(getattr(module, '__doc__', None))
            })
        except ImportError as e:
            results['failed'].append({
                'name': module_name,
                'error': str(e)
            })
        except Exception as e:
            results['failed'].append({
                'name': module_name,
                'error': f"Unexpected error: {str(e)}"
            })
    
    return results


def test_api_documentation_files() -> Dict[str, Any]:
    """Test that API documentation files exist and are properly formatted."""
    
    docs_dir = Path(__file__).parent
    api_modules_dir = docs_dir / 'source' / 'api' / 'modules'
    
    results = {
        'existing_files': [],
        'missing_files': [],
        'malformed_files': [],
        'total_files': 0
    }
    
    expected_files = [
        'index.rst',
        'causal_agent.rst',
        'components.rst',
        'methods.rst',
        'tools.rst',
        'utils.rst',
    ]
    
    for filename in expected_files:
        filepath = api_modules_dir / filename
        results['total_files'] += 1
        
        if not filepath.exists():
            results['missing_files'].append(filename)
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic RST structure
            if '.. automodule::' not in content and filename != 'index.rst':
                results['malformed_files'].append({
                    'file': filename,
                    'issue': 'Missing automodule directive'
                })
            else:
                results['existing_files'].append({
                    'file': filename,
                    'size': len(content),
                    'lines': len(content.splitlines())
                })
                
        except Exception as e:
            results['malformed_files'].append({
                'file': filename,
                'issue': f"Error reading file: {str(e)}"
            })
    
    return results


def test_sphinx_configuration() -> Dict[str, Any]:
    """Test that Sphinx configuration is properly set up for API documentation."""
    
    docs_dir = Path(__file__).parent
    conf_path = docs_dir / 'source' / 'conf.py'
    
    results = {
        'config_exists': False,
        'required_extensions': [],
        'missing_extensions': [],
        'autodoc_settings': {},
        'issues': []
    }
    
    if not conf_path.exists():
        results['issues'].append("conf.py file not found")
        return results
    
    results['config_exists'] = True
    
    try:
        with open(conf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required extensions
        required_extensions = [
            'sphinx.ext.autodoc',
            'sphinx.ext.autosummary',
            'sphinx.ext.viewcode',
            'sphinx.ext.napoleon',
        ]
        
        for ext in required_extensions:
            if ext in content:
                results['required_extensions'].append(ext)
            else:
                results['missing_extensions'].append(ext)
        
        # Check for autodoc settings
        autodoc_settings = [
            'autodoc_default_options',
            'autosummary_generate',
            'napoleon_google_docstring',
        ]
        
        for setting in autodoc_settings:
            if setting in content:
                results['autodoc_settings'][setting] = True
            else:
                results['autodoc_settings'][setting] = False
                
    except Exception as e:
        results['issues'].append(f"Error reading conf.py: {str(e)}")
    
    return results


def print_test_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Print test results in a readable format."""
    
    print("🧪 API Documentation Test Results")
    print("=" * 50)
    
    # Module import results
    import_results = results['imports']
    print(f"\n📦 Module Import Test:")
    print(f"   ✅ Successfully imported: {len(import_results['success'])}")
    print(f"   ❌ Failed to import: {len(import_results['failed'])}")
    print(f"   📊 Total modules tested: {import_results['total']}")
    
    if import_results['failed']:
        print("\n   Failed imports:")
        for failed in import_results['failed']:
            print(f"     - {failed['name']}: {failed['error']}")
    
    # Documentation files results
    files_results = results['files']
    print(f"\n📄 Documentation Files Test:")
    print(f"   ✅ Existing files: {len(files_results['existing_files'])}")
    print(f"   ❌ Missing files: {len(files_results['missing_files'])}")
    print(f"   ⚠️  Malformed files: {len(files_results['malformed_files'])}")
    print(f"   📊 Total files expected: {files_results['total_files']}")
    
    if files_results['missing_files']:
        print("\n   Missing files:")
        for filename in files_results['missing_files']:
            print(f"     - {filename}")
    
    if files_results['malformed_files']:
        print("\n   Malformed files:")
        for file_info in files_results['malformed_files']:
            print(f"     - {file_info['file']}: {file_info['issue']}")
    
    # Sphinx configuration results
    config_results = results['config']
    print(f"\n⚙️  Sphinx Configuration Test:")
    print(f"   ✅ Config file exists: {config_results['config_exists']}")
    print(f"   ✅ Required extensions: {len(config_results['required_extensions'])}")
    print(f"   ❌ Missing extensions: {len(config_results['missing_extensions'])}")
    
    if config_results['missing_extensions']:
        print("\n   Missing extensions:")
        for ext in config_results['missing_extensions']:
            print(f"     - {ext}")
    
    if config_results['issues']:
        print("\n   Configuration issues:")
        for issue in config_results['issues']:
            print(f"     - {issue}")
    
    # Overall assessment
    total_issues = (
        len(import_results['failed']) +
        len(files_results['missing_files']) +
        len(files_results['malformed_files']) +
        len(config_results['missing_extensions']) +
        len(config_results['issues'])
    )
    
    print(f"\n🎯 Overall Assessment:")
    if total_issues == 0:
        print("   ✅ All tests passed! API documentation system is ready.")
    elif total_issues <= 3:
        print("   ⚠️  Minor issues found. API documentation should work with some limitations.")
    else:
        print("   ❌ Significant issues found. API documentation may not work properly.")
    
    print(f"   📊 Total issues: {total_issues}")


def main():
    """Main function to run API documentation tests."""
    
    print("🚀 Starting API documentation tests...")
    
    # Add the parent directory to Python path for imports
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    # Run tests
    results = {
        'imports': test_module_imports(),
        'files': test_api_documentation_files(),
        'config': test_sphinx_configuration(),
    }
    
    # Print results
    print_test_results(results)
    
    # Exit with appropriate code
    total_critical_issues = (
        len(results['imports']['failed']) +
        len(results['files']['missing_files']) +
        len(results['config']['issues'])
    )
    
    if total_critical_issues > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
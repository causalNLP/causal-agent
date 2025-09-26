#!/usr/bin/env python3
"""
Validation script for ReadTheDocs configuration.
This script checks that all necessary files and configurations are in place.
"""

import os
import sys
import yaml
from pathlib import Path

def validate_readthedocs_config():
    """Validate ReadTheDocs configuration files and structure."""
    
    print("🔍 Validating ReadTheDocs configuration...")
    
    # Check .readthedocs.yaml exists and is valid
    rtd_config_path = Path(".readthedocs.yaml")
    if not rtd_config_path.exists():
        print("❌ .readthedocs.yaml not found")
        return False
    
    try:
        with open(rtd_config_path, 'r') as f:
            rtd_config = yaml.safe_load(f)
        print("✅ .readthedocs.yaml is valid YAML")
    except yaml.YAMLError as e:
        print(f"❌ .readthedocs.yaml has invalid YAML: {e}")
        return False
    
    # Check required fields in .readthedocs.yaml
    required_fields = ['version', 'build', 'sphinx', 'python']
    for field in required_fields:
        if field not in rtd_config:
            print(f"❌ Missing required field '{field}' in .readthedocs.yaml")
            return False
    print("✅ All required fields present in .readthedocs.yaml")
    
    # Check Sphinx configuration path
    sphinx_config = rtd_config.get('sphinx', {}).get('configuration')
    if not sphinx_config:
        print("❌ Sphinx configuration path not specified")
        return False
    
    sphinx_config_path = Path(sphinx_config)
    if not sphinx_config_path.exists():
        print(f"❌ Sphinx configuration file not found: {sphinx_config}")
        return False
    print(f"✅ Sphinx configuration file exists: {sphinx_config}")
    
    # Check docs requirements file
    docs_requirements = Path("docs/requirements.txt")
    if not docs_requirements.exists():
        print("❌ docs/requirements.txt not found")
        return False
    print("✅ docs/requirements.txt exists")
    
    # Check main requirements file
    main_requirements = Path("requirements.txt")
    if not main_requirements.exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt exists")
    
    # Check index.rst exists
    index_rst = Path("docs/source/index.rst")
    if not index_rst.exists():
        print("❌ docs/source/index.rst not found")
        return False
    print("✅ docs/source/index.rst exists")
    
    # Validate Python configuration
    try:
        sys.path.insert(0, str(Path("docs/source").absolute()))
        import conf
        print("✅ Sphinx conf.py is importable")
    except ImportError as e:
        print(f"❌ Cannot import Sphinx conf.py: {e}")
        return False
    
    # Check for essential directories
    essential_dirs = [
        "docs/source",
        "docs/source/_static",
        "causal_agent"  # Main package directory
    ]
    
    for dir_path in essential_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Essential directory missing: {dir_path}")
            return False
    print("✅ All essential directories exist")
    
    print("\n🎉 ReadTheDocs configuration validation passed!")
    print("\nNext steps:")
    print("1. Import project on ReadTheDocs.org")
    print("2. Configure project settings")
    print("3. Test automated builds")
    print("4. Set up custom domain (optional)")
    
    return True

if __name__ == "__main__":
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    success = validate_readthedocs_config()
    sys.exit(0 if success else 1)
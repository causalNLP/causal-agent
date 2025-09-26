#!/usr/bin/env python3
"""
Generate API documentation for CAIS project.

This script automatically generates RST files for all modules in the causal_agent package
using Sphinx's apidoc functionality with custom templates.
"""

import os
import sys
import subprocess
from pathlib import Path


def generate_api_docs():
    """Generate API documentation by scanning the package structure."""
    
    # Get paths
    docs_dir = Path(__file__).parent
    source_dir = docs_dir / 'source'
    api_dir = source_dir / 'api' / 'modules'
    package_dir = docs_dir.parent / 'causal_agent'
    
    print(f"📚 Generating API documentation...")
    print(f"   Package: {package_dir}")
    print(f"   Output: {api_dir}")
    
    # Ensure the API modules directory exists
    api_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the package directory exists
    if not package_dir.exists():
        print(f"❌ Package directory not found: {package_dir}")
        return False
    
    # Generate documentation for the main package and subpackages
    modules_generated = []
    
    # Generate main package documentation
    main_rst = generate_package_rst('causal_agent', package_dir)
    if main_rst:
        with open(api_dir / 'causal_agent.rst', 'w') as f:
            f.write(main_rst)
        modules_generated.append('causal_agent.rst')
    
    # Generate subpackage documentation
    subpackages = ['components', 'methods', 'tools', 'utils', 'synthetic', 'prompts']
    
    for subpackage in subpackages:
        subpackage_dir = package_dir / subpackage
        if subpackage_dir.exists() and subpackage_dir.is_dir():
            rst_content = generate_subpackage_rst(f'causal_agent.{subpackage}', subpackage_dir)
            if rst_content:
                with open(api_dir / f'{subpackage}.rst', 'w') as f:
                    f.write(rst_content)
                modules_generated.append(f'{subpackage}.rst')
    
    print(f"✅ Generated {len(modules_generated)} API documentation files:")
    for module in modules_generated:
        print(f"   - {module}")
    
    return True


def generate_package_rst(package_name: str, package_dir: Path) -> str:
    """Generate RST content for a package."""
    
    # Find Python modules in the package
    modules = []
    for py_file in package_dir.glob('*.py'):
        if py_file.name != '__init__.py' and not py_file.name.startswith('test_'):
            module_name = py_file.stem
            modules.append(f'{package_name}.{module_name}')
    
    # Create RST content
    title = f"{package_name} package"
    underline = "=" * len(title)
    
    rst_content = f"""{title}
{underline}

.. automodule:: {package_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    
    if modules:
        rst_content += "\nSubmodules\n----------\n\n"
        
        for module in sorted(modules):
            module_title = f"{module} module"
            module_underline = "-" * len(module_title)
            
            rst_content += f"""{module_title}
{module_underline}

.. automodule:: {module}
   :members:
   :undoc-members:
   :show-inheritance:

"""
    
    return rst_content


def generate_subpackage_rst(package_name: str, package_dir: Path) -> str:
    """Generate RST content for a subpackage."""
    
    # Find Python modules in the subpackage
    modules = []
    subpackages = []
    
    for item in package_dir.iterdir():
        if item.is_file() and item.suffix == '.py' and item.name != '__init__.py' and not item.name.startswith('test_'):
            module_name = item.stem
            modules.append(f'{package_name}.{module_name}')
        elif item.is_dir() and (item / '__init__.py').exists():
            subpackages.append(f'{package_name}.{item.name}')
    
    # Create RST content
    title = f"{package_name} package"
    underline = "=" * len(title)
    
    rst_content = f"""{title}
{underline}

.. automodule:: {package_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    
    if modules:
        rst_content += "\nSubmodules\n----------\n\n"
        
        for module in sorted(modules):
            module_title = f"{module} module"
            module_underline = "-" * len(module_title)
            
            rst_content += f"""{module_title}
{module_underline}

.. automodule:: {module}
   :members:
   :undoc-members:
   :show-inheritance:

"""
    
    if subpackages:
        rst_content += "\nSubpackages\n-----------\n\n.. toctree::\n   :maxdepth: 4\n\n"
        
        for subpackage in sorted(subpackages):
            # Convert package name to file name
            subpackage_file = subpackage.replace('causal_agent.', '')
            rst_content += f"   {subpackage_file}\n"
    
    return rst_content


def update_api_index():
    """Update the API index file with autosummary directives."""
    
    api_index_path = Path(__file__).parent / 'source' / 'api' / 'index.rst'
    
    # Read the current content
    with open(api_index_path, 'r') as f:
        content = f.read()
    
    # Check if autosummary is already present
    if '.. autosummary::' in content:
        print("✅ API index already contains autosummary directives")
        return
    
    # Add autosummary section
    autosummary_section = """

Module Reference
----------------

.. autosummary::
   :toctree: modules
   :template: autosummary/module.rst
   :recursive:

   causal_agent
"""
    
    # Insert before the Quick Reference section
    if 'Quick Reference' in content:
        content = content.replace('Quick Reference', autosummary_section + '\nQuick Reference')
    else:
        content += autosummary_section
    
    # Write back the updated content
    with open(api_index_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated API index with autosummary directives")


def validate_generated_docs():
    """Validate that the generated documentation files exist and are properly formatted."""
    
    api_modules_dir = Path(__file__).parent / 'source' / 'api' / 'modules'
    
    expected_files = [
        'causal_agent.rst',
        'causal_agent.components.rst',
        'causal_agent.methods.rst',
        'causal_agent.tools.rst',
        'causal_agent.utils.rst',
    ]
    
    missing_files = []
    for filename in expected_files:
        filepath = api_modules_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"⚠️  Missing expected API documentation files:")
        for filename in missing_files:
            print(f"   - {filename}")
    else:
        print("✅ All expected API documentation files are present")
    
    # Count total generated files
    rst_files = list(api_modules_dir.glob('*.rst'))
    print(f"📊 Generated {len(rst_files)} API documentation files")


def main():
    """Main function to generate API documentation."""
    
    print("🚀 Starting API documentation generation...")
    
    # Generate API docs using sphinx-apidoc
    generate_api_docs()
    
    # Update the API index file
    update_api_index()
    
    # Validate the generated documentation
    validate_generated_docs()
    
    print("\n✅ API documentation generation complete!")
    print("\n📖 To build the documentation, run:")
    print("   cd docs && make html")


if __name__ == '__main__':
    main()
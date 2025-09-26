# API Documentation System

This document describes the API documentation system for the Causal AI Scientist (CAIS) project.

## Overview

The API documentation system automatically generates comprehensive documentation for all Python modules, classes, and functions in the CAIS codebase using Sphinx with the following features:

- **Automatic API documentation generation** from docstrings
- **Cross-references** between modules and external libraries
- **Type hint integration** for better parameter documentation
- **Source code links** for implementation details
- **Docstring validation** to ensure documentation quality
- **Responsive design** that works on all devices

## Architecture

### Core Components

1. **Sphinx Configuration** (`source/conf.py`)
   - Configures autodoc, autosummary, and other extensions
   - Sets up cross-references to external libraries
   - Defines documentation generation options

2. **API Documentation Generator** (`generate_api_docs.py`)
   - Scans the causal_agent package structure
   - Generates RST files for all modules and subpackages
   - Creates proper cross-references and navigation

3. **Docstring Validator** (`validate_docstrings.py`)
   - Validates that public functions and classes have docstrings
   - Checks for Google-style docstring formatting
   - Reports missing Args and Returns sections

4. **API Documentation Tester** (`test_api_docs.py`)
   - Tests that all modules can be imported
   - Validates that documentation files exist and are properly formatted
   - Checks Sphinx configuration completeness

### Generated Documentation Structure

```
docs/source/api/
├── index.rst                    # Main API reference page
└── modules/
    ├── index.rst                # Module index with navigation
    ├── causal_agent.rst         # Main package documentation
    ├── components.rst           # Core components
    ├── methods.rst              # Causal inference methods
    ├── tools.rst                # Analysis tools
    ├── utils.rst                # Utility functions
    ├── synthetic.rst            # Synthetic data generation
    └── prompts.rst              # LLM prompt templates
```

## Usage

### Building API Documentation

```bash
# Generate API documentation and build HTML
make html

# Generate only API documentation files
make apidoc

# Build HTML without regenerating API docs
make html-noapi

# Full clean build
make full-build
```

### Validation and Testing

```bash
# Validate docstring quality
make validate-docstrings

# Test API documentation system
make test-api

# Run all documentation checks
make check
```

### Manual API Documentation Generation

```bash
# Generate API documentation files
python generate_api_docs.py

# Validate docstrings in the codebase
python validate_docstrings.py

# Test the API documentation system
python test_api_docs.py
```

## Configuration

### Sphinx Extensions

The following Sphinx extensions are configured for API documentation:

- `sphinx.ext.autodoc` - Automatic documentation from docstrings
- `sphinx.ext.autosummary` - Generate summary tables
- `sphinx.ext.viewcode` - Add source code links
- `sphinx.ext.napoleon` - Support for Google/NumPy style docstrings
- `sphinx.ext.intersphinx` - Cross-references to external documentation

### Autodoc Configuration

Key autodoc settings in `conf.py`:

```python
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'inherited-members': True,
}

autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autosummary_generate = True
```

### Cross-References

The system includes cross-references to:

- Python standard library
- NumPy, Pandas, SciPy, Scikit-learn
- Matplotlib, Seaborn
- Other CAIS modules and classes

## Docstring Standards

### Required Elements

All public functions and classes should have docstrings with:

1. **Brief description** of what the function/class does
2. **Args section** for functions with parameters
3. **Returns section** for functions that return values
4. **Raises section** for functions that raise exceptions (optional)
5. **Examples section** with usage examples (recommended)

### Google Style Format

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose and behavior
    of the function in more detail.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter. Defaults to 10.
    
    Returns:
        Description of the return value.
    
    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.
    
    Examples:
        Basic usage:
        
        >>> result = example_function("hello", 5)
        >>> print(result)
        True
        
        With default parameter:
        
        >>> result = example_function("world")
        >>> print(result)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    if not isinstance(param2, int):
        raise TypeError("param2 must be an integer")
    return len(param1) > param2
```

## Maintenance

### Adding New Modules

When adding new modules to the codebase:

1. Ensure all public functions and classes have proper docstrings
2. Run `python generate_api_docs.py` to update documentation files
3. Run `make validate-docstrings` to check docstring quality
4. Run `make test-api` to verify everything works

### Updating Documentation

The API documentation is automatically regenerated when:

1. Running `make html` (includes `make apidoc`)
2. Running `make apidoc` directly
3. Running `python generate_api_docs.py`

### Quality Assurance

Regular maintenance tasks:

1. **Weekly**: Run `make validate-docstrings` to check for missing documentation
2. **Before releases**: Run `make check` to validate all documentation
3. **After major changes**: Run `make full-build` to ensure everything works

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and the package is importable
2. **Missing Docstrings**: Run the validator to identify undocumented functions
3. **Broken Cross-References**: Check that referenced modules and functions exist
4. **Build Failures**: Check Sphinx configuration and ensure all RST files are valid

### Debug Commands

```bash
# Check if modules can be imported
python -c "import causal_agent; print('Import successful')"

# Test specific module documentation
python -c "import causal_agent.agent; help(causal_agent.agent.CausalAgent)"

# Validate RST syntax
sphinx-build -b dummy source build/dummy -W
```

## Integration with ReadTheDocs

The API documentation system is designed to work seamlessly with ReadTheDocs:

1. **Automatic builds** triggered by GitHub commits
2. **Version management** for different releases
3. **Search integration** across all documentation
4. **Mobile-responsive** design

### ReadTheDocs Configuration

Key settings in `.readthedocs.yaml`:

```yaml
sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: false

python:
  install:
    - requirements: docs/requirements.txt
    - requirements: requirements.txt
```

## Future Enhancements

Planned improvements to the API documentation system:

1. **Interactive examples** using Jupyter notebooks
2. **Performance benchmarks** in API documentation
3. **Automated docstring generation** for missing documentation
4. **API usage analytics** to identify popular functions
5. **Multi-language support** for international users

## Contributing

When contributing to the API documentation system:

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update this README for significant changes
4. Ensure all validation checks pass

For more information, see the [Contributing Guide](../development/contributing.rst).
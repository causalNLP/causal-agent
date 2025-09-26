# Documentation

This directory contains the Sphinx documentation for the Causal AI Scientist (CAIS) project.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

To build the HTML documentation:

```bash
make html
```

The built documentation will be available in `build/html/index.html`.

### Live Reload Development

For development with automatic rebuilding:

```bash
make livehtml
```

This will start a local server with live reload functionality.

### Other Build Targets

- `make clean` - Clean the build directory
- `make linkcheck` - Check for broken links
- `make doctest` - Run doctests
- `make coverage` - Generate documentation coverage report

## Directory Structure

- `source/` - Source files for documentation
- `source/conf.py` - Sphinx configuration
- `source/_static/` - Static files (CSS, images, etc.)
- `build/` - Built documentation output
- `requirements.txt` - Documentation dependencies

## Contributing

When adding new documentation:

1. Create `.rst` files in the appropriate `source/` subdirectory
2. Update the relevant `index.rst` file to include your new content
3. Build and test the documentation locally
4. Ensure all links work with `make linkcheck`
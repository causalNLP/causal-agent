# Scripts Directory

This directory contains utility scripts for managing the causal-agent project.

## Release Management (`release.py`)

A comprehensive script for managing versions and releases.

### Usage

```bash
# Show current version
python scripts/release.py current

# Update version manually
python scripts/release.py update --version 1.2.0

# Increment version automatically
python scripts/release.py update --increment patch   # 0.1.1 -> 0.1.2
python scripts/release.py update --increment minor   # 0.1.1 -> 0.2.0
python scripts/release.py update --increment major   # 0.1.1 -> 1.0.0

# Create a release (dry run first recommended)
python scripts/release.py release --increment minor --dry-run
python scripts/release.py release --increment minor --push

# Create release with specific version
python scripts/release.py release --version 1.0.0 --push
```

### Features

- **Version Management**: Update versions in both `causal_agent/__init__.py` and `pyproject.toml`
- **Semantic Versioning**: Automatic increment of major, minor, or patch versions
- **Git Integration**: Create and push git tags for releases
- **Validation**: Ensure version format follows semantic versioning
- **Dry Run**: Preview changes before applying them

### Integration with CI/CD

The release script works seamlessly with the automated release workflow:

1. Use the script to update versions and create tags locally
2. Push tags to trigger the automated release workflow
3. The workflow handles testing, building, and publishing to PyPI

See `docs/RELEASE_PROCESS.md` for complete release documentation.
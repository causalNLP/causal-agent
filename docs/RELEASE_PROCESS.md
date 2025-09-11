# Release Process

This document describes the automated release process for the causal-agent library.

## Overview

The release process is fully automated using GitHub Actions and includes:

1. **Pre-release validation** - Comprehensive testing and quality checks
2. **PyPI publishing** - Automated publishing to both Test PyPI and PyPI
3. **GitHub release creation** - Automated release notes and asset uploads
4. **Post-release validation** - Multi-platform installation testing

## Release Methods

### Method 1: Tag-based Release (Recommended)

1. **Update version** (if needed):
   ```bash
   # Using the release script
   python scripts/release.py update --increment patch  # or minor, major
   # Or manually update causal_agent/__init__.py and pyproject.toml
   ```

2. **Commit and push changes**:
   ```bash
   git add causal_agent/__init__.py pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

3. **Create and push tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

4. **Monitor the release workflow** at: https://github.com/causalNLP/causal-agent/actions

### Method 2: Manual Workflow Dispatch

1. Go to the [Actions tab](https://github.com/causalNLP/causal-agent/actions)
2. Select "Release & Publish" workflow
3. Click "Run workflow"
4. Enter the version number (e.g., `1.0.0`)
5. Optionally mark as pre-release
6. Click "Run workflow"

### Method 3: Using the Release Script

The release script provides a convenient way to manage versions and releases:

```bash
# Show current version
python scripts/release.py current

# Update version
python scripts/release.py update --version 1.0.0
python scripts/release.py update --increment patch

# Create a release (dry run first)
python scripts/release.py release --increment minor --dry-run
python scripts/release.py release --increment minor --push
```

## Release Workflow Steps

### 1. Pre-release Validation (`validate-release`)

- ✅ Version format validation
- ✅ Package version consistency check
- ✅ Code quality checks (Black, isort, flake8, bandit)
- ✅ Comprehensive test suite execution
- ✅ Package build validation
- ✅ Coverage threshold validation (≥80%)

### 2. PyPI Publishing (`publish-pypi`)

- 📦 Package building with `python -m build`
- 🧪 Test PyPI publication (for validation)
- 🧪 Installation test from Test PyPI
- 🚀 Production PyPI publication
- 📋 Package metadata validation

### 3. GitHub Release (`create-github-release`)

- 📝 Automated release notes generation
- 🏷️ GitHub release creation
- 📎 Build artifacts attachment
- 🔗 Links to PyPI and documentation

### 4. Post-release Validation (`post-release-validation`)

- 🧪 Multi-platform installation testing
- 🐍 Multi-Python version compatibility
- ✅ Basic functionality verification
- 📊 Installation success confirmation

## Prerequisites

### GitHub Secrets

The following secrets must be configured in the GitHub repository:

1. **`PYPI_API_TOKEN`** - PyPI API token for publishing
   - Go to https://pypi.org/manage/account/token/
   - Create a new token with "Entire account" scope
   - Add to GitHub repository secrets

2. **`TEST_PYPI_API_TOKEN`** - Test PyPI API token (optional but recommended)
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new token with "Entire account" scope
   - Add to GitHub repository secrets

3. **`GITHUB_TOKEN`** - Automatically provided by GitHub Actions

### GitHub Environment (Optional)

For additional security, create a `release` environment in GitHub:

1. Go to repository Settings → Environments
2. Create new environment named `release`
3. Add protection rules (e.g., required reviewers)
4. The workflow will use this environment for PyPI publishing

## Version Management

### Version Format

Versions must follow semantic versioning: `X.Y.Z` or `X.Y.Z-suffix`

Examples:
- `1.0.0` - Major release
- `1.1.0` - Minor release  
- `1.1.1` - Patch release
- `1.0.0-alpha` - Pre-release
- `1.0.0-beta.1` - Pre-release with build number

### Version Locations

Versions must be updated in two places:
1. `causal_agent/__init__.py` - `__version__ = "X.Y.Z"`
2. `pyproject.toml` - `version = "X.Y.Z"`

The release workflow validates that both locations have matching versions.

## Quality Gates

The release process includes several quality gates that must pass:

### Code Quality
- ✅ Black formatting compliance
- ✅ isort import sorting compliance  
- ✅ flake8 linting (no critical errors)
- ✅ Bandit security scanning (no high-severity issues)

### Testing
- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ All end-to-end tests pass
- ✅ Code coverage ≥ 80%

### Package Quality
- ✅ Package builds successfully
- ✅ Package metadata is valid
- ✅ Installation works on multiple platforms
- ✅ Basic functionality verification

## Troubleshooting

### Common Issues

1. **Version mismatch error**
   - Ensure `causal_agent/__init__.py` and `pyproject.toml` have the same version
   - Use the release script to update both automatically

2. **Test failures**
   - All tests must pass before release
   - Check the test workflow logs for details
   - Fix issues and re-run the release

3. **Coverage below threshold**
   - Ensure test coverage is ≥ 80%
   - Add tests for uncovered code
   - Check coverage report in workflow artifacts

4. **PyPI upload fails**
   - Check that the version doesn't already exist on PyPI
   - Verify PyPI API token is valid and has correct permissions
   - Check for package name conflicts

5. **Tag already exists**
   - Delete the existing tag if needed: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`
   - Use a different version number

### Manual Recovery

If the automated release fails partway through:

1. **Check workflow logs** to identify the failure point
2. **Fix the underlying issue** (tests, code quality, etc.)
3. **Delete the failed tag** (if created):
   ```bash
   git tag -d vX.Y.Z
   git push origin :refs/tags/vX.Y.Z
   ```
4. **Re-run the release** with the same or updated version

### Getting Help

- Check the [GitHub Actions logs](https://github.com/causalNLP/causal-agent/actions) for detailed error messages
- Review this documentation for common solutions
- Open an issue if you encounter persistent problems

## Release Checklist

Before creating a release:

- [ ] All planned features/fixes are merged to main
- [ ] Version number is updated (if not using auto-increment)
- [ ] All tests pass locally
- [ ] Code quality checks pass locally
- [ ] Documentation is updated (if needed)
- [ ] CHANGELOG is updated (if maintained)
- [ ] PyPI API tokens are configured in GitHub secrets

After release:

- [ ] Verify package is available on PyPI
- [ ] Test installation: `pip install causal-agent==X.Y.Z`
- [ ] Verify GitHub release is created with correct assets
- [ ] Update any dependent projects/documentation
- [ ] Announce the release (if applicable)

## Security Considerations

- API tokens are stored as GitHub secrets (encrypted)
- Release environment can be protected with required reviewers
- Test PyPI is used for validation before production publishing
- Multi-platform testing ensures broad compatibility
- Security scanning is included in quality checks

## Monitoring

Monitor the release process through:

- **GitHub Actions**: Real-time workflow execution logs
- **PyPI**: Package availability and download statistics  
- **GitHub Releases**: Release notes and asset downloads
- **Issues**: User-reported problems with new releases
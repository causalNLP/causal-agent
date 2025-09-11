# Code Quality Gates

This document describes the code quality gates implemented in the project to ensure high code quality, security, and maintainability.

## Overview

The project implements automated code quality checks through GitHub Actions workflows that run on every push and pull request. These checks ensure:

- Code formatting consistency
- Import organization
- Linting compliance
- Type safety
- Security vulnerability detection
- Test coverage requirements

## Quality Checks

### 1. Code Formatting (Black)

**Tool:** [Black](https://black.readthedocs.io/)
**Configuration:** `.pyproject.toml`
**Requirements:**
- Line length: 88 characters
- Python 3.10+ target version
- Consistent formatting across all Python files

**Local Usage:**
```bash
# Check formatting
black --check --diff causal_agent/ tests/

# Auto-format
black causal_agent/ tests/
```

### 2. Import Sorting (isort)

**Tool:** [isort](https://pycqa.github.io/isort/)
**Configuration:** `.pyproject.toml`
**Requirements:**
- Black-compatible profile
- Consistent import organization
- Proper grouping of first-party, third-party, and standard library imports

**Local Usage:**
```bash
# Check import sorting
isort --check-only --diff causal_agent/ tests/

# Auto-sort imports
isort causal_agent/ tests/
```

### 3. Linting (flake8)

**Tool:** [flake8](https://flake8.pycqa.org/)
**Configuration:** `.flake8`
**Requirements:**
- No syntax errors or undefined names
- Maximum complexity: 10
- Line length: 88 characters (compatible with Black)

**Local Usage:**
```bash
flake8 causal_agent/ tests/
```

### 4. Type Checking (mypy)

**Tool:** [mypy](https://mypy.readthedocs.io/)
**Configuration:** `mypy.ini`
**Requirements:**
- Type hints where appropriate
- No critical type errors
- Ignore missing imports for external libraries

**Local Usage:**
```bash
mypy causal_agent/
```

### 5. Security Scanning (bandit)

**Tool:** [bandit](https://bandit.readthedocs.io/)
**Configuration:** `.bandit`
**Requirements:**
- No high-severity security issues
- Medium-severity issues reviewed and justified
- Exclusion of test files from certain checks

**Local Usage:**
```bash
bandit -r causal_agent/
```

### 6. Dependency Security (safety)

**Tool:** [safety](https://pyup.io/safety/)
**Requirements:**
- No known security vulnerabilities in dependencies
- Regular updates of vulnerable packages

**Local Usage:**
```bash
safety check
```

### 7. Test Coverage

**Tool:** [pytest-cov](https://pytest-cov.readthedocs.io/)
**Configuration:** `.pyproject.toml`
**Requirements:**
- Minimum 80% code coverage
- Coverage reports generated for all test runs
- HTML and XML reports for detailed analysis

**Local Usage:**
```bash
pytest tests/ --cov=causal_agent --cov-report=html --cov-report=term-missing
```

## Workflows

### Code Quality Workflow (`.github/workflows/quality.yml`)

This workflow runs on every push and pull request and includes:

1. **Code Quality Job:**
   - Black formatting check
   - isort import sorting check
   - flake8 linting
   - mypy type checking (non-blocking)
   - pylint analysis (non-blocking)
   - bandit security scanning
   - safety dependency check

2. **Coverage Job:**
   - Run full test suite with coverage
   - Generate coverage reports (XML, HTML, badge)
   - Upload to Codecov
   - Enforce 80% minimum coverage

3. **Quality Gates Job:**
   - Verify all quality checks passed
   - Block merge if any critical checks fail

4. **Dependency Review Job:**
   - Review new dependencies in PRs
   - Check for license compatibility
   - Identify security vulnerabilities

## Local Development

### Pre-commit Hooks

Install pre-commit hooks to run quality checks automatically:

```bash
pip install pre-commit
pre-commit install
```

This will run quality checks on every commit, preventing issues from reaching CI.

### Quality Check Script

Run all quality checks locally:

```bash
./scripts/quality_check.sh
```

This script runs the same checks as the CI pipeline.

## Configuration Files

- `.flake8` - flake8 linting configuration
- `mypy.ini` - mypy type checking configuration
- `.pylintrc` - pylint analysis configuration
- `.bandit` - bandit security scanning configuration
- `.pre-commit-config.yaml` - pre-commit hooks configuration
- `codecov.yml` - Codecov coverage reporting configuration

## Quality Thresholds

| Check | Threshold | Blocking |
|-------|-----------|----------|
| Black formatting | 100% compliance | Yes |
| isort import sorting | 100% compliance | Yes |
| flake8 linting | No syntax/undefined errors | Yes |
| Test coverage | 80% minimum | Yes |
| Security (bandit) | No high severity | Yes |
| Dependencies (safety) | No known vulnerabilities | Yes |
| mypy type checking | Best effort | No |
| pylint analysis | Best effort | No |

## Troubleshooting

### Common Issues

1. **Black formatting failures:**
   ```bash
   black causal_agent/ tests/
   ```

2. **Import sorting issues:**
   ```bash
   isort causal_agent/ tests/
   ```

3. **Coverage below threshold:**
   - Add tests for uncovered code
   - Use `# pragma: no cover` for unreachable code

4. **Security issues:**
   - Update vulnerable dependencies
   - Add exclusions to `.bandit` if false positive

### Getting Help

- Check the [GitHub Actions logs](../../actions) for detailed error messages
- Review the quality gate failure issue template
- Run quality checks locally before pushing
- Use the quality check script for comprehensive validation

## Continuous Improvement

The quality gates are regularly reviewed and updated to:
- Incorporate new tools and best practices
- Adjust thresholds based on project maturity
- Add new security and quality checks
- Improve developer experience
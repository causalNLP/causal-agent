# GitHub Actions Workflows

This directory contains the CI/CD workflows for the causal-agent project.

## Workflows

### test.yml - Test Suite

This workflow runs the comprehensive test suite on every push and pull request.

**Features:**
- **Multi-platform testing**: Ubuntu, macOS, and Windows
- **Multi-version Python support**: Python 3.10, 3.11, and 3.12
- **Comprehensive test coverage**: Unit, integration, end-to-end, and performance tests
- **Coverage reporting**: Generates coverage reports and uploads to Codecov
- **Test artifacts**: Uploads test results and coverage reports as artifacts

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Test Structure:**
- **Unit tests**: `tests/unit/` - Tests individual components in isolation
- **Integration tests**: `tests/integration/` - Tests component interactions
- **End-to-end tests**: `tests/end_to_end/` - Tests complete workflows
- **Performance tests**: `tests/performance/` - Tests performance and scalability (Linux/macOS only)

**Environment Variables:**
- `PYTHONPATH`: Set to workspace root for proper imports
- `CAUSAL_AGENT_TEST_MODE`: Set to `true` for test mode
- `CAUSAL_AGENT_LOG_LEVEL`: Set to `WARNING` to reduce log noise

**Coverage:**
- Coverage reports are generated for Ubuntu + Python 3.11 combination
- Reports are uploaded to Codecov for tracking
- HTML and XML coverage reports are available as artifacts

## Local Testing

To run the same tests locally that run in CI:

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist pytest-mock "coverage[toml]" pytest-html pytest-json-report

# Run unit tests
pytest tests/unit/ -v -m "unit"

# Run integration tests  
pytest tests/integration/ -v -m "integration"

# Run end-to-end tests
pytest tests/end_to_end/ -v -m "e2e"

# Run performance tests
pytest tests/performance/ -v -m "performance"

# Generate coverage report
pytest tests/unit/ tests/integration/ --cov=causal_agent --cov-report=html --cov-report=term-missing
```

## Configuration

The test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
addopts = [
    "-ra",
    "--strict-markers", 
    "--strict-config",
    "--cov=causal_agent",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=0",
]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests", 
    "performance: Performance tests",
    "slow: Slow running tests",
    "requires_llm: Tests requiring LLM API access",
    "memory: Memory usage tests",
]
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure `PYTHONPATH` includes the project root
2. **Missing Dependencies**: Install all test dependencies listed above
3. **Coverage Failures**: Coverage threshold is set to 0% for CI, but can be increased locally
4. **Platform-specific Failures**: Some tests may behave differently on Windows vs Unix systems

### Debugging Failed Tests

1. Check the workflow logs in the GitHub Actions tab
2. Download test artifacts for detailed reports
3. Run tests locally with the same Python version and OS
4. Use `pytest -v --tb=long` for detailed error information

## Future Enhancements

Planned improvements to the CI/CD pipeline:

- [ ] Add code quality checks (linting, formatting)
- [ ] Add security scanning
- [ ] Add automated release workflow
- [ ] Add performance benchmarking
- [ ] Add dependency vulnerability scanning
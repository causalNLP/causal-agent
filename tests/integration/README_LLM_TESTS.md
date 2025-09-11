# Real LLM Integration Tests

This directory contains integration tests that use actual LLM API calls to test the causal agent workflows end-to-end.

## Prerequisites

1. **OpenAI API Key**: You need a valid OpenAI API key set in your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

2. **Dependencies**: Ensure all project dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Test Files

### `test_llm_integration_basic.py`
- **Purpose**: Basic integration tests with real LLM calls
- **Duration**: Fast (2-5 minutes)
- **Coverage**: Core component integration, simple workflows
- **Recommended for**: Quick validation that LLM integration works

### `test_real_llm_workflows.py`
- **Purpose**: Comprehensive end-to-end workflow testing
- **Duration**: Slower (10-20 minutes)
- **Coverage**: Multiple dataset types, complex queries, error handling
- **Recommended for**: Full validation before releases

### `conftest_llm.py`
- **Purpose**: Test configuration and fixtures for LLM tests
- **Features**: API key validation, environment setup, test helpers

## Running the Tests

### Option 1: Using the Test Runner (Recommended)

```bash
# Quick single test
python tests/integration/run_llm_tests.py single

# Basic test suite (recommended for development)
python tests/integration/run_llm_tests.py basic

# Comprehensive test suite (for CI/releases)
python tests/integration/run_llm_tests.py comprehensive
```

### Option 2: Using pytest directly

```bash
# Run all LLM integration tests
pytest tests/integration/ -m requires_llm -v

# Run only basic tests
pytest tests/integration/test_llm_integration_basic.py -v

# Run specific test
pytest tests/integration/test_llm_integration_basic.py::TestBasicLLMIntegration::test_simple_rct_analysis_real_llm -v -s

# Skip LLM tests (if no API key)
pytest tests/integration/ -m "not requires_llm"
```

### Option 3: Individual test execution

```bash
# Run basic LLM integration tests
python -m pytest tests/integration/test_llm_integration_basic.py -v -s

# Run comprehensive workflow tests
python -m pytest tests/integration/test_real_llm_workflows.py -v -s
```

## Test Markers

- `@pytest.mark.requires_llm`: Tests that require actual LLM API access
- `@pytest.mark.slow`: Tests that take longer to run (>30 seconds)

## Environment Configuration

The tests automatically configure the LLM environment:
- **Provider**: OpenAI
- **Model**: gpt-3.5-turbo
- **Temperature**: 0.1 (for consistent results)

You can override these by setting environment variables:
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
```

## Test Data

Tests create synthetic datasets with known causal structures:

1. **RCT Data**: Randomized treatment assignment with clear treatment effects
2. **Observational Data**: Confounded treatment assignment requiring adjustment
3. **IV Data**: Instrumental variable scenarios for causal identification
4. **Problematic Data**: Edge cases for error handling validation

## Expected Behavior

### Successful Tests
- Tests should complete without exceptions
- Results should have proper structure (`results` key present)
- Method selection should be appropriate for data type
- Effect estimates should have reasonable magnitudes

### Common Issues

1. **API Key Missing**: Tests will be skipped automatically
2. **Rate Limiting**: Tests include delays to respect API limits
3. **LLM Response Variability**: Tests are designed to handle response variation
4. **Network Issues**: Tests may fail due to connectivity problems

## Debugging

### Verbose Output
Add `-s` flag to see detailed test output:
```bash
pytest tests/integration/test_llm_integration_basic.py -v -s
```

### Single Test Debugging
Run individual tests for focused debugging:
```bash
pytest tests/integration/test_llm_integration_basic.py::TestBasicLLMIntegration::test_simple_rct_analysis_real_llm -v -s
```

### Environment Validation
Check LLM configuration:
```bash
pytest tests/integration/test_llm_integration_basic.py::TestBasicLLMIntegration::test_llm_configuration_real -v -s
```

## Cost Considerations

- **Basic tests**: ~10-20 API calls (~$0.01-0.05)
- **Comprehensive tests**: ~50-100 API calls (~$0.10-0.25)
- Tests use gpt-3.5-turbo for cost efficiency
- Small datasets minimize token usage

## CI/CD Integration

For continuous integration:

```yaml
# Example GitHub Actions step
- name: Run LLM Integration Tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    python tests/integration/run_llm_tests.py basic
```

## Troubleshooting

### Test Failures
1. Check API key validity and format
2. Verify network connectivity
3. Check for rate limiting (429 errors)
4. Review LLM response format changes

### Performance Issues
1. Use basic test suite for development
2. Run comprehensive tests only before releases
3. Consider using faster models for development

### API Errors
- **401 Unauthorized**: Invalid API key
- **429 Rate Limited**: Too many requests
- **500 Server Error**: OpenAI service issues

## Contributing

When adding new LLM integration tests:

1. Mark with `@pytest.mark.requires_llm`
2. Use small datasets for efficiency
3. Include proper error handling
4. Add descriptive test output
5. Consider API cost implications

## Support

For issues with LLM integration tests:
1. Check this README first
2. Verify API key setup
3. Run individual tests for debugging
4. Check OpenAI service status
# Makefile for causal-agent project

.PHONY: help install test test-unit test-integration test-e2e coverage coverage-report coverage-html clean lint format type-check quality-check ci-test

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install the package and dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e         Run end-to-end tests only"
	@echo "  coverage         Run tests with coverage analysis"
	@echo "  coverage-report  Generate detailed coverage report"
	@echo "  coverage-html    Generate HTML coverage report"
	@echo "  coverage-gaps    Analyze coverage gaps"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  quality-check    Run all quality checks"
	@echo "  ci-test          Run CI-style tests with coverage validation"
	@echo "  clean            Clean up generated files"

# Installation
install:
	pip install -e ".[dev]"

# Testing targets
test:
	python -m pytest -v

test-unit:
	python -m pytest tests/unit/ -v -m "unit"

test-integration:
	python -m pytest tests/integration/ -v -m "integration"

test-e2e:
	python -m pytest tests/end_to_end/ -v -m "e2e"

# Coverage targets
coverage:
	python -m pytest --cov=causal_agent --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=80

coverage-report:
	python scripts/coverage_analysis.py --target 80.0

coverage-html:
	python -m pytest --cov=causal_agent --cov-report=html --cov-fail-under=0
	@echo "HTML coverage report generated in htmlcov/"

coverage-gaps:
	python scripts/fill_coverage_gaps.py

# Code quality targets
lint:
	flake8 causal_agent/ tests/
	pylint causal_agent/

format:
	black causal_agent/ tests/ scripts/
	isort causal_agent/ tests/ scripts/

type-check:
	mypy causal_agent/

quality-check: lint type-check
	bandit -r causal_agent/
	safety check

# CI target
ci-test:
	python scripts/ci_coverage_check.py

# Cleanup
clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f coverage.xml
	rm -f coverage.svg
	rm -f coverage_report.txt
	rm -f coverage_gaps_report.txt
	rm -f coverage_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

# Development helpers
dev-setup: install
	pre-commit install

test-watch:
	python -m pytest --cov=causal_agent --cov-report=term-missing -f

# Performance testing
test-performance:
	python -m pytest tests/performance/ -v -m "performance"

# Security checks
security-check:
	bandit -r causal_agent/
	safety check

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Release helpers
check-release: quality-check coverage-report
	@echo "Release checks completed"

# Docker targets (if needed in future)
docker-build:
	@echo "Docker build not yet implemented"

docker-test:
	@echo "Docker test not yet implemented"
#!/bin/bash

# Quality Check Script
# Run all quality checks locally before pushing

set -e

echo "🔍 Running code quality checks..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing quality tools..."
pip install black isort flake8 mypy pylint bandit safety coverage pytest pytest-cov

echo "🎨 Checking code formatting with Black..."
black --check --diff causal_agent/ tests/ --line-length 88

echo "📋 Checking import sorting with isort..."
isort --check-only --diff causal_agent/ tests/ --profile black

echo "🔍 Linting with flake8..."
flake8 causal_agent/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 causal_agent/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

echo "🔒 Security check with bandit..."
bandit -r causal_agent/ --severity-level medium

echo "🛡️ Checking dependencies for vulnerabilities..."
safety check

echo "📊 Running tests with coverage..."
pytest tests/ --cov=causal_agent --cov-report=term-missing --cov-fail-under=80

echo "✅ All quality checks passed!"
echo "🚀 Ready to push!"
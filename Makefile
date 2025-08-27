.PHONY: help install install-dev test test-cov lint format type-check clean build docs pre-commit
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,test,docs]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=ladder --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	flake8 src/ tests/
	bandit -r src/ -x tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	mypy src/ladder/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

docs: ## Build documentation
	cd docs && make html

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

check: format type-check lint test ## Run all checks (format, type-check, lint, test)

ci: check ## Run CI pipeline locally
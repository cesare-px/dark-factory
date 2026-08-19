.DEFAULT_GOAL := help
.PHONY: help install lint lint-fix format format-check typecheck test test-cov check ci clean

PYTHON ?= python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies (ruff, mypy, pytest)
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Lint (pycodestyle, pyflakes, isort, pydocstyle, bugbear, ...)
	ruff check .

lint-fix: ## Lint and auto-fix what's safe to fix
	ruff check . --fix

format: ## Format code (100-char lines)
	ruff format .

format-check: ## Check formatting without modifying files (what CI runs)
	ruff format --check .

typecheck: ## Strict type-check src/, lighter check on tests/
	mypy

test: ## Run the test suite
	pytest

test-cov: ## Run the test suite with a coverage report
	pytest --cov=dark_factory --cov-report=term-missing

check: lint format-check typecheck test ## Run everything CI runs, in the same order

ci: check ## Alias for `check`, matching the CI job name

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +

.PHONY: help install install-dev install-docs install-all test test-fast coverage lint format fix check clean build docs docs-serve

# Tools (run inside the uv-managed venv)
PYTEST := uv run pytest
BLACK  := uv run black
RUFF   := uv run ruff
MYPY   := uv run mypy
SPHINX := uv run sphinx-build

# Directories
SRC_DIR        := pagexml_hf
TEST_DIR       := tests
DOCS_DIR       := docs
DOCS_BUILD_DIR := $(DOCS_DIR)/_build

help:
	@echo "Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make install-docs  - Install documentation dependencies"
	@echo "  make install-all   - Install all dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run tests"
	@echo "  make test-fast     - Run tests in parallel"
	@echo "  make coverage      - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run all linters (ruff, mypy)"
	@echo "  make format        - Format code with black"
	@echo "  make fix           - Auto-fix linting issues"
	@echo "  make check         - Run format check without modifying files"
	@echo ""
	@echo "Build & Distribution:"
	@echo "  make clean         - Remove build artifacts and cache"
	@echo "  make build         - Build distribution packages"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Generate HTML documentation"
	@echo "  make docs-serve    - Serve documentation locally"

install:
	uv sync

install-dev:
	uv sync --extra dev

install-docs:
	uv sync --extra docs

install-all:
	uv sync --all-extras

test:
	$(PYTEST) $(TEST_DIR)

test-fast:
	$(PYTEST) $(TEST_DIR) -n auto

coverage:
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term --cov-report=xml

lint:
	@echo "Running ruff..."
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	@echo ""
	@echo "Running mypy..."
	$(MYPY) $(SRC_DIR)

format:
	@echo "Running black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

fix:
	@echo "Auto-fixing with ruff..."
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)
	@echo ""
	@echo "Running black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

check:
	@echo "Checking format with black..."
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	@echo ""
	@echo "Checking with ruff..."
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf logs/*.log 2>/dev/null || true
	@echo "Clean complete!"

build: clean
	uv build

docs:
	@echo "Generating documentation..."
	$(SPHINX) -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
	@echo "Documentation generated in $(DOCS_BUILD_DIR)/html"

docs-serve: docs
	@echo "Serving documentation on http://localhost:8000"
	cd $(DOCS_BUILD_DIR)/html && uv run python -m http.server 8000

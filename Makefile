.DEFAULT_GOAL := all

.PHONY: all
all: format lint

.PHONY: format
format:
	@echo "Running code formatting..."
	uv run black .
	uv run isort .
	uv run ruff check --fix .
	@echo "Code formatting completed."

.PHONY: lint
lint:
	@echo "Running code linting..."
	uv run black --check .
	uv run isort -c .
	uv run ruff check .
	@echo "Code linting completed."

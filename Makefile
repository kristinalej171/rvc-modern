.PHONY: install install-dev format lint type-check test clean run

install:
	pip install --upgrade pip uv
	uv pip install -e .

install-dev:
	pip install --upgrade pip uv
	uv pip install -e ".[dev]"
	pre-commit install

format:
	ruff check --fix src tests
	ruff format src tests

lint:
	ruff check src tests
	ruff format --check src tests

type-check:
	mypy src

test:
	pytest tests --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist .coverage htmlcov

run:
	python -m rvc.main

download-models:
	python -m rvc.core.utils --download-models

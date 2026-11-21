.PHONY: setup lint test test-cov run-dashboard clean help

help:
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies and setup package"
	@echo "  make lint         - Run ruff code linter"
	@echo "  make test         - Run pytest test suite"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make run-dashboard- Launch the interactive Streamlit research lab"
	@echo "  make clean        - Remove caches and build artifacts"

setup:
	python -m pip install --upgrade pip
	pip install -e .[dev,deep-learning]

lint:
	ruff check src tests

test:
	pytest tests -v

test-cov:
	pytest --cov=src/volatility_lab --cov-report=term-missing tests

run-dashboard:
	streamlit run app/streamlit_app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

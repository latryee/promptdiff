.PHONY: install test test-cov lint format clean run-demo

install:
	pip install -e ".[dev]"

test:
	pytest -v

test-cov:
	pytest --cov=promptdiff --cov-report=term-missing --cov-report=html

lint:
	ruff check .
	mypy promptdiff

format:
	ruff format .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov .mypy_cache .promptdiff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

run-demo:
	python -m promptdiff.cli.app test examples/prompts/support_bot_v1.txt examples/prompts/support_bot_v2.txt --eval "json_validity,latency,cost,similarity" --mock

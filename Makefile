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
	python -c "import shutil, pathlib, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('build') + glob.glob('dist') + glob.glob('*.egg-info') + glob.glob('.pytest_cache') + glob.glob('.mypy_cache') + glob.glob('.ruff_cache') + glob.glob('.promptdiff_cache') + glob.glob('htmlcov')]; [p.unlink() for p in pathlib.Path('.').glob('.coverage*')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('**/__pycache__')]"

run-demo:
	python -m promptdiff.cli.app test examples/prompts/support_bot_v1.txt examples/prompts/support_bot_v2.txt --eval "json_validity,latency,cost,similarity" --mock

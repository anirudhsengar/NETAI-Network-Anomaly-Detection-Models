.PHONY: install dev test lint clean data train evaluate serve docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=netai_anomaly --cov-report=term-missing

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

data:
	python scripts/generate_data.py

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

serve:
	python scripts/serve.py --config configs/default.yaml

docker:
	docker build -f docker/Dockerfile -t netai-anomaly:latest .

docker-gpu:
	docker build -f docker/Dockerfile.gpu -t netai-anomaly:gpu .

.PHONY: install test train run-api docker-build docker-run clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -v

train:
	python -m mode train

run-api:
	uvicorn serving.app:app --reload --port 8000

docker-build:
	docker build -t quant-ml-crypto:latest .

docker-run:
	docker-compose up --build -d

docker-stop:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
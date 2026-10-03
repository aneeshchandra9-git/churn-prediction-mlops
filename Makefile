.PHONY: help install install-dev clean test format lint run-notebook run-api docker-build docker-run airflow-start airflow-stop mlflow-ui

help:
	@echo "==================================="
	@echo "MLOps Project - Available Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install           - Install dependencies"
	@echo "  make install-dev       - Install dev dependencies"
	@echo "  make create-dirs       - Create project directories"
	@echo ""
	@echo "Development:"
	@echo "  make test              - Run all tests"
	@echo "  make test-cov          - Run tests with coverage"
	@echo "  make format            - Format code with black"
	@echo "  make lint              - Run linter (flake8)"
	@echo "  make type-check        - Run mypy type checking"
	@echo ""
	@echo "Running Components:"
	@echo "  make run-notebook      - Start Jupyter notebook"
	@echo "  make run-api           - Start FastAPI server"
	@echo "  make mlflow-ui         - Start MLflow UI"
	@echo "  make airflow-init      - Initialize Airflow"
	@echo "  make airflow-start     - Start Airflow scheduler"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  make docker-build      - Build Docker image"
	@echo "  make docker-run        - Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove build artifacts"
	@echo "  make clean-cache       - Remove Python cache files"
	@echo ""

# Setup Commands
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black flake8 pylint mypy

create-dirs:
	@echo "Creating project directories..."
	mkdir -p data/raw data/processed data/external
	mkdir -p notebooks
	mkdir -p src
	mkdir -p tests
	mkdir -p models
	mkdir -p logs
	mkdir -p dags
	mkdir -p docker
	mkdir -p deployment
	mkdir -p config
	mkdir -p scripts
	mkdir -p monitoring
	@echo "✓ Directories created successfully"

# Testing Commands
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code Quality Commands
format:
	black src/ tests/ scripts/

lint:
	flake8 src/ tests/ scripts/ --max-line-length=100

type-check:
	mypy src/ --ignore-missing-imports

quality: format lint type-check
	@echo "✓ Code quality checks complete"

# Running Components
run-notebook:
	jupyter lab notebooks/

run-api:
	@echo "Starting FastAPI server on http://localhost:8000"
	uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000"
	mlflow ui

airflow-init:
	@echo "Initializing Airflow..."
	airflow db init

airflow-start:
	@echo "Starting Airflow scheduler..."
	airflow scheduler

# Docker Commands
docker-build:
	docker build -f docker/Dockerfile -t mlops-project:latest .

docker-run:
	docker run -p 8000:8000 mlops-project:latest

# Cleanup Commands
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	rm -rf build/ dist/
	@echo "✓ Clean complete"

clean-cache:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✓ Cache cleaned"

# Combined Commands
setup: create-dirs install
	@echo "✓ Setup complete! Run 'make' to see available commands"

dev-setup: create-dirs install-dev
	@echo "✓ Development setup complete!"
	@echo "Now run: make test  (to verify installation)"

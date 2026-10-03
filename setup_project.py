#!/usr/bin/env python3
"""
Auto-setup script for MLOps project structure
Run this after cloning/downloading the project
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create all necessary project directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src",
        "tests",
        "models",
        "logs",
        "dags",
        "docker",
        "deployment",
        "config",
        "scripts",
        "monitoring",
        "monitoring/prometheus",
        "monitoring/grafana",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep to ensure empty directories are tracked
        gitkeep = Path(directory) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("✓ Created all project directories")

def create_init_files():
    """Create __init__.py files for Python packages"""
    packages = ["src", "tests"]
    
    for package in packages:
        init_file = Path(package) / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization"""\n')
    
    print("✓ Created __init__.py files")

def create_sample_files():
    """Create sample Python files with basic structure"""
    
    # src/logger.py
    logger_file = Path("src/logger.py")
    if not logger_file.exists():
        logger_file.write_text("""import logging
import sys
from pathlib import Path

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
""")
    
    # tests/__init__.py
    tests_init = Path("tests/__init__.py")
    if not tests_init.exists():
        tests_init.write_text('"""Test suite for MLOps project"""\n')
    
    # conftest.py for pytest
    conftest = Path("tests/conftest.py")
    if not conftest.exists():
        conftest.write_text("""import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_data():
    \"\"\"Fixture providing sample data for tests\"\"\"
    return {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    }
""")
    
    print("✓ Created sample Python files")

def create_config_files():
    """Create configuration files"""
    
    # config/config.yaml
    config_file = Path("config/config.yaml")
    if not config_file.exists():
        config_file.write_text("""# MLOps Project Configuration

# Data Settings
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  train_test_split: 0.8
  random_state: 42

# Model Settings
model:
  type: "xgboost"
  hyperparameters:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1

# Training Settings
training:
  batch_size: 32
  epochs: 10
  validation_split: 0.2

# MLflow Settings
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "mlops-experiment"

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

# Logging Settings
logging:
  level: "INFO"
  log_file: "logs/app.log"
""")
    
    print("✓ Created config files")

def create_docker_files():
    """Create Docker-related files"""
    
    # docker/Dockerfile
    dockerfile = Path("docker/Dockerfile")
    if not dockerfile.exists():
        dockerfile.write_text("""FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run the application
CMD ["python", "-m", "uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
""")
    
    # docker/.dockerignore
    dockerignore = Path("docker/.dockerignore")
    if not dockerignore.exists():
        dockerignore.write_text("""__pycache__
*.pyc
.git
.gitignore
*.md
.env
venv
.vscode
.idea
""")
    
    print("✓ Created Docker files")

def create_github_actions():
    """Create GitHub Actions CI/CD workflow"""
    
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    ci_file = workflows_dir / "ci.yml"
    if not ci_file.exists():
        ci_file.write_text("""name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 src tests --max-line-length=100
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
""")
    
    print("✓ Created GitHub Actions workflows")

def print_summary():
    """Print summary of what was created"""
    print()
    print("=" * 50)
    print("✓ MLOps Project Setup Complete!")
    print("=" * 50)
    print()
    print("Created:")
    print("  ✓ Project directories (data, notebooks, src, etc.)")
    print("  ✓ Python package structure (__init__.py files)")
    print("  ✓ Sample configuration files")
    print("  ✓ Docker configuration")
    print("  ✓ GitHub Actions CI/CD workflow")
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Verify installation: pytest --version")
    print("  3. Start Jupyter: make run-notebook")
    print("  4. Check available commands: make help")
    print()
    print("For detailed instructions, read: README_SETUP.md")
    print()

def main():
    """Run all setup tasks"""
    try:
        print("Setting up MLOps project structure...")
        print()
        
        create_directories()
        create_init_files()
        create_sample_files()
        create_config_files()
        create_docker_files()
        create_github_actions()
        
        print_summary()
        
    except Exception as e:
        print(f"✗ Error during setup: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

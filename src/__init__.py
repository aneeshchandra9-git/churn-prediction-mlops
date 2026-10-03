"""
MLOps Project - Machine Learning Operations Pipeline

This package contains the complete MLOps pipeline including:
- Data processing and validation
- Model training and evaluation
- Experiment tracking
- Model serving via API
- Monitoring and alerting
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"MLOps Project v{__version__} initialized")

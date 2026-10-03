from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlops-project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A complete MLOps pipeline for production ML systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlops-project",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.7.0",
        "fastapi>=0.103.0",
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "black>=23.7.0",
            "flake8>=6.0.0",
            "pylint>=2.17.5",
            "mypy>=1.4.1",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.82.0",
        ],
    },
)

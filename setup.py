#!/usr/bin/env python3
"""
Setup script for MZTrain - Motor de Entrenamiento en Espacio Comprimido
"""

from setuptools import setup, find_packages
import os


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as fh:
            return [
                line.strip() for line in fh
                if line.strip() and not line.startswith("#")
            ]
    return ["torch>=2.0.0", "numpy>=1.21.0"]


setup(
    name="mztrain",
    version="1.3.1",
    author="Esraderey and Raul Cruz Acosta",
    author_email="msc.framework@gmail.com",
    description=(
        "Motor de Entrenamiento en Espacio Comprimido - "
        "Entrenamiento de modelos de IA con factorizacion SVD y compresion via MNEME"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/esraderey/mztrain",
    project_urls={
        "Bug Reports": "https://github.com/esraderey/mztrain/issues",
        "Source": "https://github.com/esraderey/mztrain",
        "Documentation": "https://github.com/esraderey/mztrain/wiki",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "training",
        "compression",
        "svd",
        "factorization",
        "pytorch",
        "machine-learning",
        "memory-efficient",
        "low-rank",
        "optimizer",
        "gradient-compression",
    ],
)

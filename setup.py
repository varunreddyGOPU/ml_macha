"""
KFP ML Library - setup configuration.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="kfp-ml-library",
    version="1.0.0",
    author="KFP ML Library Team",
    description="A comprehensive Kubeflow Pipelines ML model deployment library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/varunreddyGOPU/ml_macha",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "tensorflow": ["tensorflow>=2.15.0"],
        "pytorch": ["torch>=2.1.0", "torchvision>=0.16.0"],
        "keras": ["keras>=3.0.0", "tensorflow>=2.15.0"],
        "automl": ["flaml>=2.1.0"],
        "automl-sklearn": ["auto-sklearn>=0.15.0"],
        "gcp": [
            "google-cloud-aiplatform>=1.38.0",
            "google-cloud-build>=3.22.0",
            "google-cloud-storage>=2.13.0",
            "google-cloud-bigquery>=3.13.0",
            "google-cloud-artifactregistry>=1.9.0",
        ],
        "all": [
            "tensorflow>=2.15.0",
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "flaml>=2.1.0",
            "google-cloud-aiplatform>=1.38.0",
            "google-cloud-build>=3.22.0",
            "google-cloud-storage>=2.13.0",
            "google-cloud-bigquery>=3.13.0",
            "google-cloud-artifactregistry>=1.9.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "kfp-ml=kfp_ml_library:__version__",
        ],
    },
)

"""
KFP ML Library - A comprehensive Kubeflow Pipelines ML Model Deployment Library.

This library provides reusable KFP components for building end-to-end ML pipelines
including data preparation, training, evaluation, deployment, and monitoring.

Supported frameworks:
    - Scikit-learn
    - XGBoost
    - Keras
    - TensorFlow
    - PyTorch
    - AutoML
"""

__version__ = "1.0.0"
__author__ = "KFP ML Library Team"

from kfp_ml_library.configs.constants import (
    DEFAULT_IMAGE,
    PIPELINE_NAME,
    FrameworkType,
)
from kfp_ml_library.pipelines.full_pipeline import create_full_ml_pipeline
from kfp_ml_library.pipelines.training_pipeline import create_training_pipeline
from kfp_ml_library.pipelines.deployment_pipeline import create_deployment_pipeline
from kfp_ml_library.pipelines.monitoring_pipeline import create_monitoring_pipeline

__all__ = [
    "DEFAULT_IMAGE",
    "PIPELINE_NAME",
    "FrameworkType",
    "create_full_ml_pipeline",
    "create_training_pipeline",
    "create_deployment_pipeline",
    "create_monitoring_pipeline",
]

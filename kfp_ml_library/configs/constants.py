"""
Constants for the KFP ML Library.

Central location for all constants, enums, and default values used throughout
the pipeline library.
"""

from enum import Enum
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------
PIPELINE_NAME = "kfp-ml-pipeline"
PIPELINE_DESCRIPTION = "End-to-end ML model deployment pipeline"
PIPELINE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Container / image defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE = "python:3.10-slim"
DEFAULT_BASE_IMAGE = "python:3.10-slim"
GPU_IMAGE = "nvidia/cuda:12.2.0-runtime-ubuntu22.04"
DOCKER_REGISTRY = "gcr.io"
DEFAULT_CONTAINER_REGISTRY = "gcr.io/my-project"

# ---------------------------------------------------------------------------
# GCS / artifact defaults
# ---------------------------------------------------------------------------
DEFAULT_ARTIFACT_BUCKET = "gs://ml-pipeline-artifacts"
DEFAULT_MODEL_REGISTRY = "gs://ml-model-registry"
DEFAULT_DATA_PATH = "gs://ml-pipeline-data"

# ---------------------------------------------------------------------------
# Compute defaults
# ---------------------------------------------------------------------------
DEFAULT_CPU_LIMIT = "4"
DEFAULT_MEMORY_LIMIT = "16Gi"
DEFAULT_GPU_LIMIT = "1"
DEFAULT_GPU_TYPE = "nvidia-tesla-t4"
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_RETRY_COUNT = 2

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_N_TRIALS_HYPERPARAMETER = 50

# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------
DEFAULT_EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc",
]
DEFAULT_REGRESSION_METRICS = [
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "explained_variance",
]
MODEL_PERFORMANCE_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Monitoring defaults
# ---------------------------------------------------------------------------
DRIFT_THRESHOLD = 0.05
MONITORING_INTERVAL_SECONDS = 3600
ALERT_EMAIL = "ml-alerts@company.com"
MONITORING_WINDOW_SIZE = 1000

# ---------------------------------------------------------------------------
# Deployment defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_REPLICAS = 1
DEFAULT_MAX_REPLICAS = 5
DEFAULT_TARGET_CPU_UTILIZATION = 70
CANARY_TRAFFIC_PERCENTAGE = 10
DEFAULT_SERVING_PORT = 8080
HEALTH_CHECK_PATH = "/health"
PREDICTION_PATH = "/predict"


class FrameworkType(str, Enum):
    """Supported ML framework types."""

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    KERAS = "keras"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    AUTOML = "automl"


class TaskType(str, Enum):
    """Supported ML task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    RECOMMENDATION = "recommendation"


class DeploymentStrategy(str, Enum):
    """Supported deployment strategies."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


class DataFormat(str, Enum):
    """Supported data formats."""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    TFRECORD = "tfrecord"
    AVRO = "avro"
    BIGQUERY = "bigquery"


class ModelStatus(str, Enum):
    """Model lifecycle status."""

    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class MonitoringMetricType(str, Enum):
    """Types of monitoring metrics."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_IMPORTANCE = "feature_importance"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


# ---------------------------------------------------------------------------
# Default packages per framework
# ---------------------------------------------------------------------------
FRAMEWORK_PACKAGES: Dict[str, list] = {
    FrameworkType.SKLEARN: ["scikit-learn>=1.3.0", "joblib>=1.3.0"],
    FrameworkType.XGBOOST: ["xgboost>=2.0.0", "scikit-learn>=1.3.0"],
    FrameworkType.KERAS: ["keras>=3.0.0", "tensorflow>=2.15.0"],
    FrameworkType.TENSORFLOW: ["tensorflow>=2.15.0"],
    FrameworkType.PYTORCH: ["torch>=2.1.0", "torchvision>=0.16.0"],
    FrameworkType.AUTOML: [
        "auto-sklearn>=0.15.0",
        "scikit-learn>=1.3.0",
        "flaml>=2.1.0",
    ],
}

# ---------------------------------------------------------------------------
# Docker file templates
# ---------------------------------------------------------------------------
DOCKERFILE_TEMPLATE = """
FROM {base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

{extra_commands}

ENTRYPOINT ["python", "{entrypoint}"]
"""

DOCKERFILE_GPU_TEMPLATE = """
FROM {base_image}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip && \\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

{extra_commands}

ENTRYPOINT ["python3", "{entrypoint}"]
"""

# KFP ML Library — Complete Usage Guide, Scenarios & Code Review

> **Version**: 1.0.0  
> **Last Reviewed**: February 2026  
> **Scope**: Every file, every function, all usage scenarios, code review findings, and modern improvement recommendations.

---

## Table of Contents

1. [Installation & Quick Start](#1-installation--quick-start)
2. [Configs — File-by-File Usage](#2-configs--file-by-file-usage)
   - [constants.py](#21-constantspy)
   - [compute_constraints.py](#22-compute_constraintspy)
   - [trainer_config.py](#23-trainer_configpy)
   - [evaluator_config.py](#24-evaluator_configpy)
   - [monitoring_config.py](#25-monitoring_configpy)
3. [Components — File-by-File Usage](#3-components--file-by-file-usage)
   - [data_prep/data_ingestion.py](#31-data_prepdata_ingestionpy)
   - [data_prep/data_validation.py](#32-data_prepdata_validationpy)
   - [data_prep/data_transformation.py](#33-data_prepdata_transformationpy)
   - [data_prep/feature_engineering.py](#34-data_prepfeature_engineeringpy)
   - [training/trainer_base.py](#35-trainingtrainer_basepy)
   - [training/hyperparameter_tuning.py](#36-traininghyperparameter_tuningpy)
   - [training/generic_trainer.py](#37-traininggeneric_trainerpy)
   - [evaluation/evaluator_base.py](#38-evaluationevaluator_basepy)
   - [evaluation/model_evaluation.py](#39-evaluationmodel_evaluationpy)
   - [evaluation/metrics.py](#310-evaluationmetricspy)
   - [deployment/model_deployer.py](#311-deploymentmodel_deployerpy)
   - [deployment/endpoint_manager.py](#312-deploymentendpoint_managerpy)
   - [monitoring/model_monitor.py](#313-monitoringmodel_monitorpy)
   - [monitoring/drift_detection.py](#314-monitoringdrift_detectionpy)
   - [container/docker_builder.py](#315-containerdocker_builderpy)
   - [container/cpr_manager.py](#316-containercpr_managerpy)
   - [container/containerized_component.py](#317-containercontainerized_componentpy)
   - [generic/generic_component.py](#318-genericgeneric_componentpy)
4. [Frameworks — File-by-File Usage](#4-frameworks--file-by-file-usage)
   - [sklearn_impl.py](#41-sklearn_implpy)
   - [xgboost_impl.py](#42-xgboost_implpy)
   - [keras_impl.py](#43-keras_implpy)
   - [tensorflow_impl.py](#44-tensorflow_implpy)
   - [pytorch_impl.py](#45-pytorch_implpy)
   - [automl_impl.py](#46-automl_implpy)
5. [Pipelines — File-by-File Usage](#5-pipelines--file-by-file-usage)
   - [training_pipeline.py](#51-training_pipelinepy)
   - [deployment_pipeline.py](#52-deployment_pipelinepy)
   - [monitoring_pipeline.py](#53-monitoring_pipelinepy)
   - [full_pipeline.py](#54-full_pipelinepy)
6. [Utils — File-by-File Usage](#6-utils--file-by-file-usage)
   - [logging_utils.py](#61-logging_utilspy)
   - [io_utils.py](#62-io_utilspy)
   - [validation_utils.py](#63-validation_utilspy)
7. [End-to-End Scenarios](#7-end-to-end-scenarios)
8. [Code Review Findings](#8-code-review-findings)
9. [Modern Features & Improvements](#9-modern-features--improvements)

---

## 1. Installation & Quick Start

### Install the library

```bash
# Core install
pip install -e .

# With specific ML framework
pip install -e ".[tensorflow]"
pip install -e ".[pytorch]"
pip install -e ".[automl]"
pip install -e ".[gcp]"

# Everything
pip install -e ".[all]"

# Development (includes testing, linting)
pip install -e ".[dev]"
```

### Minimal pipeline run

```python
from kfp import compiler
from kfp_ml_library import create_full_ml_pipeline

compiler.Compiler().compile(
    pipeline_func=create_full_ml_pipeline,
    package_path="full_pipeline.yaml",
)
```

### Submit to Vertex AI

```python
from google.cloud import aiplatform
aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="my-ml-pipeline",
    template_path="full_pipeline.yaml",
    parameter_values={
        "source_type": "gcs",
        "source_path": "gs://my-bucket/data.csv",
        "target_column": "label",
        "framework": "sklearn",
        "project_id": "my-project",
    },
)
job.submit()
```

---

## 2. Configs — File-by-File Usage

### 2.1 `constants.py`

**Location**: `kfp_ml_library/configs/constants.py`

**What it provides**: All enums, default values, Dockerfile templates, and framework package mappings.

#### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `FrameworkType` | `SKLEARN`, `XGBOOST`, `KERAS`, `TENSORFLOW`, `PYTORCH`, `AUTOML` | Identify which ML framework to use |
| `TaskType` | `CLASSIFICATION`, `REGRESSION`, `CLUSTERING`, `TIME_SERIES`, `NLP`, `COMPUTER_VISION`, `RECOMMENDATION` | Define the ML task |
| `DeploymentStrategy` | `BLUE_GREEN`, `CANARY`, `ROLLING`, `SHADOW`, `A_B_TEST` | Choose deployment strategy |
| `DataFormat` | `CSV`, `PARQUET`, `JSON`, `TFRECORD`, `AVRO`, `BIGQUERY` | Declare input data format |
| `ModelStatus` | `TRAINING`, `VALIDATING`, `STAGING`, `PRODUCTION`, `DEPRECATED`, `ARCHIVED` | Track model lifecycle |
| `MonitoringMetricType` | `DATA_DRIFT`, `CONCEPT_DRIFT`, `PREDICTION_DRIFT`, `FEATURE_IMPORTANCE`, `LATENCY`, `THROUGHPUT`, `ERROR_RATE` | Select monitoring metrics |

#### Scenarios

```python
# Scenario 1: Use FrameworkType to configure a trainer
from kfp_ml_library.configs.constants import FrameworkType, TaskType

framework = FrameworkType.XGBOOST
task = TaskType.REGRESSION

# Scenario 2: Look up default packages for a framework
from kfp_ml_library.configs.constants import FRAMEWORK_PACKAGES
pkgs = FRAMEWORK_PACKAGES[FrameworkType.PYTORCH]
# Returns: ["torch>=2.1.0", "torchvision>=0.16.0"]

# Scenario 3: Use deployment strategy in a conditional
from kfp_ml_library.configs.constants import DeploymentStrategy
strategy = DeploymentStrategy.CANARY

# Scenario 4: Override default training params
from kfp_ml_library.configs.constants import (
    DEFAULT_EPOCHS,           # 100
    DEFAULT_BATCH_SIZE,       # 32
    DEFAULT_LEARNING_RATE,    # 0.001
    DEFAULT_VALIDATION_SPLIT, # 0.2
)

# Scenario 5: Access container defaults
from kfp_ml_library.configs.constants import (
    DEFAULT_IMAGE,              # "python:3.10-slim"
    GPU_IMAGE,                  # "nvidia/cuda:12.2.0-runtime-ubuntu22.04"
    DEFAULT_CONTAINER_REGISTRY, # "gcr.io/my-project"
)

# Scenario 6: Use Dockerfile templates
from kfp_ml_library.configs.constants import DOCKERFILE_TEMPLATE
dockerfile = DOCKERFILE_TEMPLATE.format(
    base_image="python:3.10-slim",
    extra_commands="RUN apt-get update",
    entrypoint="serve.py",
)

# Scenario 7: Check model performance threshold
from kfp_ml_library.configs.constants import MODEL_PERFORMANCE_THRESHOLD
# Default: 0.80 — model must score ≥ 80% on primary metric to be blessed
```

---

### 2.2 `compute_constraints.py`

**Location**: `kfp_ml_library/configs/compute_constraints.py`

**What it provides**: Dataclasses for Kubernetes resource requests/limits, accelerator configs, and pre-built compute profiles.

#### Classes

| Class | Fields | Purpose |
|-------|--------|---------|
| `AcceleratorConfig` | `accelerator_type`, `accelerator_count`, `require_gpu` | GPU/TPU config |
| `ResourceRequests` | `cpu`, `memory`, `ephemeral_storage` | K8s resource requests |
| `ResourceLimits` | `cpu`, `memory`, `gpu`, `ephemeral_storage` | K8s resource limits |
| `NodeConfig` | `node_selector`, `tolerations`, `affinity` | Node placement |
| `ComputeConstraints` | All of the above + `timeout_seconds`, `retry_count`, `service_account` | Full compute profile |

#### Pre-built Profiles

| Profile | CPU Request | Memory Request | GPU |
|---------|-------------|----------------|-----|
| `SMALL_CPU` | 1 | 2Gi | — |
| `MEDIUM_CPU` | 2 | 8Gi | — |
| `LARGE_CPU` | 8 | 32Gi | — |
| `SMALL_GPU` | 4 | 16Gi | 1x T4 |
| `LARGE_GPU` | 8 | 64Gi | 4x A100 |

#### Scenarios

```python
# Scenario 1: Use a pre-built profile for a training step
from kfp_ml_library.configs.compute_constraints import LARGE_GPU, ComputeConstraints
constraints = LARGE_GPU
print(constraints.to_dict())

# Scenario 2: Create a custom profile
custom = ComputeConstraints(
    requests=ResourceRequests(cpu="4", memory="16Gi"),
    limits=ResourceLimits(cpu="8", memory="32Gi", gpu="2"),
    accelerator=AcceleratorConfig(
        accelerator_type="nvidia-tesla-v100",
        accelerator_count=2,
        require_gpu=True,
    ),
    timeout_seconds=7200,
    retry_count=3,
    service_account="ml-training@project.iam.gserviceaccount.com",
)

# Scenario 3: Serialize for KFP consumption
config_dict = custom.to_dict()
# Pass to a KFP component as JSON parameter

# Scenario 4: Use role-based aliases
from kfp_ml_library.configs.compute_constraints import (
    TRAINING_DEFAULT,    # = MEDIUM_CPU
    EVALUATION_DEFAULT,  # = SMALL_CPU
    DEPLOYMENT_DEFAULT,  # = SMALL_CPU
    DATA_PREP_DEFAULT,   # = MEDIUM_CPU
)

# Scenario 5: Add node selector for on-prem GPU nodes
from kfp_ml_library.configs.compute_constraints import NodeConfig
node_cfg = NodeConfig(
    node_selector={"gpu-type": "a100", "zone": "us-east1-b"},
    tolerations=[{"key": "gpu", "operator": "Exists", "effect": "NoSchedule"}],
)
constraints = ComputeConstraints(node_config=node_cfg)
```

---

### 2.3 `trainer_config.py`

**Location**: `kfp_ml_library/configs/trainer_config.py`

**What it provides**: Base trainer config + framework-specific configs + hyperparameter config.

#### Classes

| Class | Extends | Key Extra Fields |
|-------|---------|------------------|
| `TrainerConfig` | — | `framework`, `task_type`, `epochs`, `batch_size`, `learning_rate`, `custom_params` |
| `HyperparameterConfig` | — | `search_algorithm`, `n_trials`, `parameter_space`, `pruner`, `sampler` |
| `SklearnTrainerConfig` | `TrainerConfig` | `model_class`, `n_estimators`, `max_depth`, `n_jobs`, `class_weight` |
| `XGBoostTrainerConfig` | `TrainerConfig` | `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `use_gpu` |
| `KerasTrainerConfig` | `TrainerConfig` | `hidden_layers`, `dropout_rate`, `activation`, `use_batch_norm`, `callbacks` |
| `TensorFlowTrainerConfig` | `TrainerConfig` | `strategy`, `mixed_precision`, `xla_compilation`, `distribute`, `save_format` |
| `PyTorchTrainerConfig` | `TrainerConfig` | `optimizer`, `scheduler`, `weight_decay`, `gradient_clip_value`, `mixed_precision` |
| `AutoMLTrainerConfig` | `TrainerConfig` | `engine`, `time_budget`, `max_models`, `include_estimators`, `ensemble_size` |

#### Scenarios

```python
# Scenario 1: Sklearn Random Forest for classification
from kfp_ml_library.configs.trainer_config import SklearnTrainerConfig
config = SklearnTrainerConfig(
    model_class="RandomForestClassifier",
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
)
print(config.to_dict())

# Scenario 2: XGBoost with GPU for regression
from kfp_ml_library.configs.trainer_config import XGBoostTrainerConfig
from kfp_ml_library.configs.constants import TaskType
config = XGBoostTrainerConfig(
    task_type=TaskType.REGRESSION,
    n_estimators=2000,
    max_depth=8,
    use_gpu=True,
    subsample=0.9,
    colsample_bytree=0.8,
)

# Scenario 3: Keras deep learning model
from kfp_ml_library.configs.trainer_config import KerasTrainerConfig
config = KerasTrainerConfig(
    hidden_layers=[256, 128, 64],
    dropout_rate=0.3,
    use_batch_norm=True,
    epochs=200,
    batch_size=64,
    callbacks=["early_stopping", "reduce_lr", "model_checkpoint"],
)

# Scenario 4: TensorFlow with distributed training
from kfp_ml_library.configs.trainer_config import TensorFlowTrainerConfig
config = TensorFlowTrainerConfig(
    strategy="mirrored",
    distribute=True,
    mixed_precision=True,
    xla_compilation=True,
    save_format="tflite",  # Export to TFLite for mobile
)

# Scenario 5: PyTorch with AMP and cosine scheduler
from kfp_ml_library.configs.trainer_config import PyTorchTrainerConfig
config = PyTorchTrainerConfig(
    optimizer="AdamW",
    scheduler="cosine",
    weight_decay=0.01,
    gradient_clip_value=1.0,
    mixed_precision=True,
    epochs=50,
)

# Scenario 6: AutoML with FLAML
from kfp_ml_library.configs.trainer_config import AutoMLTrainerConfig
config = AutoMLTrainerConfig(
    engine="flaml",
    time_budget=7200,
    max_models=50,
    metric="f1",
    include_estimators=["xgboost", "lgbm", "rf"],
)

# Scenario 7: Hyperparameter tuning configuration
from kfp_ml_library.configs.trainer_config import HyperparameterConfig
hp_config = HyperparameterConfig(
    search_algorithm="bayesian",
    n_trials=100,
    objective_metric="f1_score",
    direction="maximize",
    parameter_space={
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    },
    pruner="hyperband",
    sampler="tpe",
)

# Scenario 8: Use custom_params to pass framework-specific args
config = SklearnTrainerConfig(
    model_class="GradientBoostingClassifier",
    custom_params={"min_samples_split": 5, "min_samples_leaf": 2},
)
```

---

### 2.4 `evaluator_config.py`

**Location**: `kfp_ml_library/configs/evaluator_config.py`

**What it provides**: Evaluation thresholds, slicing config, and evaluator config with blessing decisions.

#### Scenarios

```python
# Scenario 1: Basic classification evaluation
from kfp_ml_library.configs.evaluator_config import EvaluatorConfig
config = EvaluatorConfig(
    primary_metric="f1_score",
    blessing_threshold=0.85,
)

# Scenario 2: With per-metric thresholds
from kfp_ml_library.configs.evaluator_config import EvaluatorConfig, EvaluationThreshold
config = EvaluatorConfig(
    thresholds=[
        EvaluationThreshold("accuracy", 0.90, comparison="gte"),
        EvaluationThreshold("f1_score", 0.85),
        EvaluationThreshold("precision", 0.80),
    ],
)
# Each threshold.check(actual_value) → bool

# Scenario 3: Evaluate with slicing
from kfp_ml_library.configs.evaluator_config import SlicingConfig
config = EvaluatorConfig(
    slicing_configs=[
        SlicingConfig(
            slicing_column="country",
            slicing_values=["US", "UK", "DE"],
            min_samples_per_slice=50,
        ),
        SlicingConfig(
            slicing_column="device_type",
            slicing_values=["mobile", "desktop"],
        ),
    ],
)

# Scenario 4: Regression evaluation
from kfp_ml_library.configs.constants import TaskType
config = EvaluatorConfig(
    task_type=TaskType.REGRESSION,
    primary_metric="r2_score",
    blessing_threshold=0.75,
)
print(config.get_metrics_for_task())
# → ["mse", "rmse", "mae", "r2_score", "explained_variance"]

# Scenario 5: Baseline comparison
config = EvaluatorConfig(
    compare_with_baseline=True,
    baseline_model_path="gs://models/baseline_v1/model.pkl",
)

# Scenario 6: Serialize for KFP component
import json
config_json = json.dumps(config.to_dict())
```

---

### 2.5 `monitoring_config.py`

**Location**: `kfp_ml_library/configs/monitoring_config.py`

**What it provides**: Drift detection config, alerting config, latency thresholds, and full monitoring config.

#### Scenarios

```python
# Scenario 1: Basic monitoring setup
from kfp_ml_library.configs.monitoring_config import MonitoringConfig
config = MonitoringConfig(
    model_name="fraud_detector_v2",
    endpoint_name="fraud-endpoint",
)

# Scenario 2: Custom drift detection
from kfp_ml_library.configs.monitoring_config import MonitoringConfig, DriftDetectionConfig
config = MonitoringConfig(
    drift_config=DriftDetectionConfig(
        method="ks_test",  # or "psi", "chi_squared", "wasserstein"
        threshold=0.05,
        features_to_monitor=["amount", "merchant_category", "hour_of_day"],
        window_size=5000,
    ),
)

# Scenario 3: Alerting with Slack and email
from kfp_ml_library.configs.monitoring_config import AlertConfig
config = MonitoringConfig(
    alert_config=AlertConfig(
        email_recipients=["team@company.com", "oncall@company.com"],
        slack_webhook="https://hooks.slack.com/services/...",
        auto_rollback_on_critical=True,
        cooldown_seconds=1800,
    ),
)

# Scenario 4: Latency monitoring thresholds
from kfp_ml_library.configs.monitoring_config import LatencyConfig
config = MonitoringConfig(
    latency_config=LatencyConfig(
        p50_threshold_ms=50.0,
        p95_threshold_ms=200.0,
        p99_threshold_ms=500.0,
    ),
)

# Scenario 5: Full config serialized
import json
config_json = json.dumps(config.to_dict())

# Scenario 6: Custom monitoring metrics
config = MonitoringConfig(
    custom_metrics={"business_kpi": "conversion_rate", "threshold": 0.03},
    log_predictions=True,
    log_features=True,
    max_log_retention_days=180,
)
```

---

## 3. Components — File-by-File Usage

### 3.1 `data_prep/data_ingestion.py`

**Function**: `data_ingestion_component`

**Purpose**: Loads data from GCS, BigQuery, local files, or CSV URLs and writes a standardized Parquet artifact.

#### Scenarios

```python
# Scenario 1: Ingest from GCS (CSV)
ingest = data_ingestion_component(
    source_type="gcs",
    source_path="gs://bucket/data.csv",
    file_format="csv",
)

# Scenario 2: Ingest from BigQuery
ingest = data_ingestion_component(
    source_type="bigquery",
    query="SELECT * FROM `project.dataset.table` WHERE date > '2025-01-01'",
)

# Scenario 3: Ingest from local Parquet file
ingest = data_ingestion_component(
    source_type="local",
    source_path="/data/train.parquet",
    file_format="parquet",
)

# Scenario 4: Sample 10% of data for experimentation
ingest = data_ingestion_component(
    source_type="gcs",
    source_path="gs://bucket/large_dataset.csv",
    sample_fraction=0.1,
    random_state=42,
)

# Scenario 5: Ingest from public CSV URL
ingest = data_ingestion_component(
    source_type="csv_url",
    source_path="https://raw.githubusercontent.com/.../data.csv",
)
```

**Output**: Parquet `Dataset` artifact with metadata containing `stats` (num_rows, num_columns, columns, dtypes).

---

### 3.2 `data_prep/data_validation.py`

**Function**: `data_validation_component`

**Purpose**: Validates a dataset for null rates, duplicates, schema conformance, and outliers.

#### Scenarios

```python
# Scenario 1: Basic validation with default thresholds
validate = data_validation_component(
    input_dataset=ingest.outputs["output_dataset"],
)

# Scenario 2: Strict null threshold (max 1%)
validate = data_validation_component(
    input_dataset=ingest.outputs["output_dataset"],
    max_null_fraction=0.01,
    max_duplicate_fraction=0.02,
)

# Scenario 3: With schema enforcement
import json
schema = json.dumps({
    "columns": ["feature_1", "feature_2", "target"],
    "dtypes": {"feature_1": "float64", "target": "int64"},
})
validate = data_validation_component(
    input_dataset=ingest.outputs["output_dataset"],
    schema_json=schema,
)

# Scenario 4: Adjust outlier sensitivity
validate = data_validation_component(
    input_dataset=ingest.outputs["output_dataset"],
    outlier_std_threshold=2.5,  # Stricter than default 3.0
)
```

**Outputs**: `validation_report` artifact (JSON) + `validation_metrics` (KFP Metrics with num_rows, issues count, etc.).

---

### 3.3 `data_prep/data_transformation.py`

**Function**: `data_transformation_component`

**Purpose**: Imputes missing values, encodes categoricals, scales numericals, and splits into train/val/test.

#### Scenarios

```python
# Scenario 1: Standard preprocessing pipeline
transform = data_transformation_component(
    input_dataset=ingest.outputs["output_dataset"],
    target_column="label",
    numerical_strategy="standard",   # StandardScaler
    categorical_strategy="onehot",   # One-hot encoding
    handle_missing="median",
)

# Scenario 2: MinMax scaling with label encoding
transform = data_transformation_component(
    input_dataset=ingest.outputs["output_dataset"],
    target_column="price",
    numerical_strategy="minmax",
    categorical_strategy="label",
    handle_missing="mean",
)

# Scenario 3: Custom split ratios
transform = data_transformation_component(
    input_dataset=ingest.outputs["output_dataset"],
    target_column="target",
    validation_split=0.15,
    test_split=0.15,
)

# Scenario 4: Drop unnecessary columns
import json
transform = data_transformation_component(
    input_dataset=ingest.outputs["output_dataset"],
    target_column="target",
    drop_columns=json.dumps(["id", "timestamp", "row_hash"]),
)

# Scenario 5: Robust scaling (outlier-resistant)
transform = data_transformation_component(
    input_dataset=ingest.outputs["output_dataset"],
    target_column="target",
    numerical_strategy="robust",
    handle_missing="drop",
)
```

**Outputs**: Three `Dataset` artifacts: `train_dataset`, `val_dataset`, `test_dataset`.

---

### 3.4 `data_prep/feature_engineering.py`

**Function**: `feature_engineering_component`

**Purpose**: Generates polynomial features, interaction features, bins columns, and selects top-k features.

#### Scenarios

```python
# Scenario 1: Add polynomial features (degree=2)
fe = feature_engineering_component(
    input_dataset=transform.outputs["train_dataset"],
    target_column="target",
    polynomial_degree=2,
)

# Scenario 2: Interaction features between all numeric columns
fe = feature_engineering_component(
    input_dataset=transform.outputs["train_dataset"],
    target_column="target",
    interaction_features=True,
)

# Scenario 3: Bin specific columns into quantiles
import json
fe = feature_engineering_component(
    input_dataset=transform.outputs["train_dataset"],
    target_column="target",
    binning_columns=json.dumps(["age", "income"]),
    n_bins=5,
)

# Scenario 4: Select top-20 features using univariate selection
fe = feature_engineering_component(
    input_dataset=transform.outputs["train_dataset"],
    target_column="target",
    select_k_best=20,
)

# Scenario 5: Remove low-variance features
fe = feature_engineering_component(
    input_dataset=transform.outputs["train_dataset"],
    target_column="target",
    drop_low_variance=True,
    variance_threshold=0.01,
)

# Scenario 6: Combined feature engineering
fe = feature_engineering_component(
    input_dataset=transform.outputs["train_dataset"],
    target_column="target",
    polynomial_degree=2,
    interaction_features=True,
    drop_low_variance=True,
    select_k_best=50,
)
```

---

### 3.5 `training/trainer_base.py`

**Classes**: `TrainerBase` (abstract), `train_model_component` (KFP component)

**Purpose**: Abstract base for all framework trainers with a factory-pattern KFP component.

#### Scenarios

```python
# Scenario 1: Use TrainerBase subclass directly (outside KFP)
from kfp_ml_library.frameworks.sklearn_impl import SklearnTrainer
from kfp_ml_library.configs.trainer_config import SklearnTrainerConfig

config = SklearnTrainerConfig(model_class="GradientBoostingClassifier")
trainer = SklearnTrainer(config)

# Build and train in one call
history = trainer.build_and_train(X_train, y_train, X_val, y_val)

# Evaluate and save in one call
metrics, model_path = trainer.evaluate_and_save(X_test, y_test, "/tmp/model.pkl")

# Scenario 2: Use the KFP component (framework-agnostic)
import json
config = {"framework": "sklearn", "target_column": "label", "model_class": "RandomForestClassifier"}
train = train_model_component(
    train_data=transform.outputs["train_dataset"],
    val_data=transform.outputs["val_dataset"],
    config_json=json.dumps(config),
)

# Scenario 3: Implement a custom trainer
from kfp_ml_library.components.training.trainer_base import TrainerBase

class MyCustomTrainer(TrainerBase):
    def _build_model(self, input_shape=None, **kwargs):
        # your model construction
        ...
    def _train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # your training loop
        ...
    def _evaluate(self, X_test, y_test, **kwargs):
        # your evaluation logic
        ...
    def _save_model(self, output_path):
        # your save logic
        ...
    def _load_model(self, model_path):
        # your load logic
        ...
```

---

### 3.6 `training/hyperparameter_tuning.py`

**Functions**: `hyperparameter_tuning_component`, `retrain_with_best_params_component`

**Purpose**: Optuna-based hyperparameter optimization + retrain with best params.

#### Scenarios

```python
# Scenario 1: Basic HP tuning for Random Forest
import json

trainer_config = json.dumps({
    "framework": "sklearn",
    "model_class": "RandomForestClassifier",
    "target_column": "label",
})
hp_config = json.dumps({
    "n_trials": 100,
    "objective_metric": "accuracy",
    "direction": "maximize",
    "parameter_space": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 20},
    },
    "cross_validation_folds": 5,
    "pruner": "median",
})

hp_task = hyperparameter_tuning_component(
    train_data=fe_train.outputs["output_dataset"],
    val_data=fe_val.outputs["output_dataset"],
    trainer_config_json=trainer_config,
    hp_config_json=hp_config,
)

# Scenario 2: Retrain using the best params found
retrain = retrain_with_best_params_component(
    train_data=fe_train.outputs["output_dataset"],
    val_data=fe_val.outputs["output_dataset"],
    best_params_artifact=hp_task.outputs["best_params_artifact"],
    trainer_config_json=trainer_config,
)

# Scenario 3: XGBoost HP tuning with mixed parameter types
hp_config = json.dumps({
    "n_trials": 200,
    "objective_metric": "f1_weighted",
    "direction": "maximize",
    "parameter_space": {
        "n_estimators": {"type": "int", "low": 100, "high": 2000},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    },
    "pruner": "hyperband",
})
```

---

### 3.7 `training/generic_trainer.py`

**Function**: `generic_train_component`

**Purpose**: Single component that trains any sklearn/XGBoost model via a config dict.

#### Scenarios

```python
# Scenario 1: Train sklearn classifier
import json
config = json.dumps({
    "framework": "sklearn",
    "task_type": "classification",
    "model_class": "GradientBoostingClassifier",
    "target_column": "target",
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
})
train = generic_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    config_json=config,
)

# Scenario 2: Train XGBoost regressor
config = json.dumps({
    "framework": "xgboost",
    "task_type": "regression",
    "target_column": "price",
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
})
train = generic_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    config_json=config,
)
```

**Output**: `model_artifact` (pickled model) + `metrics_artifact` (accuracy/f1 or mse/r2).

---

### 3.8 `evaluation/evaluator_base.py`

**Class**: `EvaluatorBase` (abstract)

**Purpose**: Abstract evaluator with full evaluate pipeline: metrics → plots → blessing → comparison.

#### Scenarios

```python
# Scenario 1: Use the evaluate() method
evaluator = MyEvaluator(config)
report = evaluator.evaluate(
    y_true=y_test,
    y_pred=predictions,
    y_prob=probabilities,
    feature_names=feature_list,
    model=trained_model,
    baseline_metrics={"accuracy": 0.85, "f1_score": 0.82},
)
# report = {
#     "metrics": {...},
#     "threshold_results": [...],
#     "is_blessed": True/False,
#     "primary_metric": "accuracy",
#     "primary_value": 0.91,
#     "comparison": {...},
# }

# Scenario 2: Check blessing after evaluation
is_good = evaluator.check_blessing()

# Scenario 3: Implement a custom evaluator
class FraudEvaluator(EvaluatorBase):
    def _compute_metrics(self, y_true, y_pred, y_prob=None, **kwargs):
        return {"precision": ..., "recall": ..., "auc_pr": ...}
    def _generate_plots(self, y_true, y_pred, y_prob=None, **kwargs):
        return {"pr_curve": ..., "confusion_matrix": ...}
    def _compare_with_baseline(self, current, baseline):
        return {"improved": current["precision"] > baseline["precision"]}
```

---

### 3.9 `evaluation/model_evaluation.py`

**Functions**: `model_evaluation_component`, `model_blessing_gate_component`

**Purpose**: Evaluate a trained model on test data and produce blessing decision.

#### Scenarios

```python
# Scenario 1: Evaluate classification model
import json
eval_config = json.dumps({
    "task_type": "classification",
    "target_column": "label",
    "primary_metric": "f1_score",
    "blessing_threshold": 0.85,
})
eval_task = model_evaluation_component(
    test_data=test_dataset,
    model_artifact=train_task.outputs["model_artifact"],
    eval_config_json=eval_config,
)

# Scenario 2: Use blessing gate for conditional deployment
gate = model_blessing_gate_component(
    eval_report=eval_task.outputs["eval_report"],
)
# gate.output → True/False

# Scenario 3: Evaluate regression model
eval_config = json.dumps({
    "task_type": "regression",
    "target_column": "price",
    "primary_metric": "r2_score",
    "blessing_threshold": 0.75,
})

# Scenario 4: Use in a conditional branch
with dsl.If(gate.output == True):
    deploy_model(...)
```

---

### 3.10 `evaluation/metrics.py`

**Functions**: `compute_classification_metrics`, `compute_regression_metrics`, `compute_drift_psi`, `compute_ks_statistic`, `compute_feature_importance_drift`

**Purpose**: Centralized metric computation functions.

#### Scenarios

```python
# Scenario 1: Classification metrics
from kfp_ml_library.components.evaluation.metrics import compute_classification_metrics
metrics = compute_classification_metrics(y_true, y_pred, y_prob, average="weighted")
# → {"accuracy": 0.92, "f1_score": 0.91, "precision": 0.90, "recall": 0.93, "auc_roc": 0.96}

# Scenario 2: Regression metrics
from kfp_ml_library.components.evaluation.metrics import compute_regression_metrics
metrics = compute_regression_metrics(y_true, y_pred)
# → {"mse": 0.025, "rmse": 0.158, "mae": 0.12, "r2_score": 0.94, "explained_variance": 0.95}

# Scenario 3: PSI drift detection between distributions
from kfp_ml_library.components.evaluation.metrics import compute_drift_psi
import numpy as np
psi = compute_drift_psi(reference_data, current_data, n_bins=10)
# PSI < 0.1 → No drift
# PSI 0.1-0.2 → Moderate drift
# PSI > 0.2 → Significant drift

# Scenario 4: KS statistic for distribution comparison
from kfp_ml_library.components.evaluation.metrics import compute_ks_statistic
ks = compute_ks_statistic(reference_data, current_data)

# Scenario 5: Feature importance drift
from kfp_ml_library.components.evaluation.metrics import compute_feature_importance_drift
drift = compute_feature_importance_drift(
    old_importances={"feature_a": 0.3, "feature_b": 0.5},
    new_importances={"feature_a": 0.1, "feature_b": 0.6},
)
# → {"feature_a": 0.2, "feature_b": 0.1}
```

---

### 3.11 `deployment/model_deployer.py`

**Function**: `deploy_model_component`

**Purpose**: Deploy a blessed model to Vertex AI with configurable deployment strategy.

#### Scenarios

```python
# Scenario 1: Rolling deployment (default)
deploy = deploy_model_component(
    model_artifact=train_task.outputs["model_artifact"],
    eval_report=eval_task.outputs["eval_report"],
    project_id="my-project",
    region="us-central1",
    endpoint_name="fraud-detector",
    deployment_strategy="rolling",
    min_replicas=2,
    max_replicas=10,
    machine_type="n1-standard-4",
)

# Scenario 2: Canary deployment (10% traffic)
deploy = deploy_model_component(
    model_artifact=model,
    eval_report=report,
    project_id="my-project",
    endpoint_name="recommendation-engine",
    deployment_strategy="canary",
    canary_percentage=10,
)

# Scenario 3: GPU deployment
deploy = deploy_model_component(
    model_artifact=model,
    eval_report=report,
    project_id="my-project",
    endpoint_name="image-classifier",
    machine_type="n1-standard-8",
    accelerator_type="nvidia-tesla-t4",
    accelerator_count=1,
)

# Scenario 4: Custom serving container
deploy = deploy_model_component(
    model_artifact=model,
    eval_report=report,
    project_id="my-project",
    endpoint_name="custom-model",
    serving_container_image="us-docker.pkg.dev/my-project/ml-repo/serving:latest",
)

# Scenario 5: Auto-skip if model is not blessed
# The component checks eval_report["is_blessed"] — if False, deployment is skipped automatically
```

---

### 3.12 `deployment/endpoint_manager.py`

**Functions**: `manage_endpoint_component`, `rollback_deployment_component`

**Purpose**: List, update traffic, delete endpoints, and rollback deployments.

#### Scenarios

```python
# Scenario 1: List all endpoints
endpoints = manage_endpoint_component(
    project_id="my-project",
    region="us-central1",
    action="list",
)

# Scenario 2: Update traffic split
import json
manage_endpoint_component(
    project_id="my-project",
    endpoint_name="fraud-detector",
    action="update_traffic",
    traffic_split_json=json.dumps({"model-v2": 80, "model-v1": 20}),
)

# Scenario 3: Delete an endpoint
manage_endpoint_component(
    project_id="my-project",
    endpoint_name="deprecated-model",
    action="delete",
)

# Scenario 4: Rollback to previous model
rollback = rollback_deployment_component(
    deployment_artifact=deploy_task.outputs["deployment_artifact"],
    project_id="my-project",
    endpoint_name="fraud-detector",
    previous_model_id="projects/my-project/locations/us-central1/models/123",
)
```

---

### 3.13 `monitoring/model_monitor.py`

**Functions**: `model_monitoring_component`, `prediction_logging_component`

**Purpose**: Monitor model predictions for drift/degradation and log prediction statistics.

#### Scenarios

```python
# Scenario 1: Monitor with PSI drift detection
import json
monitor = model_monitoring_component(
    reference_data=ref_dataset,
    current_data=cur_dataset,
    monitoring_config_json=json.dumps({
        "drift_method": "psi",
        "drift_threshold": 0.1,
        "features_to_monitor": ["feature_1", "feature_2", "feature_3"],
    }),
)

# Scenario 2: Monitor with KS test
monitor = model_monitoring_component(
    reference_data=ref_dataset,
    current_data=cur_dataset,
    monitoring_config_json=json.dumps({
        "drift_method": "ks_test",
        "drift_threshold": 0.05,
    }),
)

# Scenario 3: Log prediction statistics
log = prediction_logging_component(
    predictions_data=predictions_dataset,
    model_name="fraud_detector_v2",
    endpoint_name="fraud-endpoint",
)
# Logs: total_predictions, prediction distribution, latency stats (p50, p95, p99)
```

---

### 3.14 `monitoring/drift_detection.py`

**Functions**: `drift_detection_component`, `alerting_component`

**Purpose**: Dedicated per-feature drift detection with multiple methods + Slack/email alerting.

#### Scenarios

```python
# Scenario 1: PSI drift detection
drift = drift_detection_component(
    reference_data=ref_dataset,
    current_data=cur_dataset,
    method="psi",
    threshold=0.1,
    n_bins=10,
)

# Scenario 2: KS test with specific features
import json
drift = drift_detection_component(
    reference_data=ref_dataset,
    current_data=cur_dataset,
    features_json=json.dumps(["amount", "merchant_id", "transaction_hour"]),
    method="ks_test",
    threshold=0.05,
)

# Scenario 3: Wasserstein distance
drift = drift_detection_component(
    reference_data=ref_dataset,
    current_data=cur_dataset,
    method="wasserstein",
    threshold=0.5,
)

# Scenario 4: Send Slack alert on drift
alert = alerting_component(
    drift_report=drift.outputs["drift_report"],
    slack_webhook="https://hooks.slack.com/services/...",
    severity="critical",
)

# Scenario 5: Email alert
alert = alerting_component(
    drift_report=drift.outputs["drift_report"],
    email_recipients=json.dumps(["ml-team@company.com"]),
    severity="warning",
)
```

---

### 3.15 `container/docker_builder.py`

**Functions**: `build_docker_image_kaniko`, `build_docker_image_cloud_build`

**Purpose**: Build Docker images inside pipelines using Kaniko (no daemon) or Cloud Build.

#### Scenarios

```python
# Scenario 1: Kaniko build (in-cluster, no Docker daemon)
build = build_docker_image_kaniko(
    dockerfile_content="FROM python:3.10-slim\nCOPY . /app\n...",
    context_gcs_path="gs://bucket/build-context/",
    image_name="ml-training",
    image_tag="v1.0",
    registry="gcr.io",
    project_id="my-project",
)

# Scenario 2: Cloud Build (managed GCP build)
build = build_docker_image_cloud_build(
    source_gcs_uri="gs://bucket/source.tar.gz",
    image_name="ml-serving",
    image_tag="latest",
    project_id="my-project",
    registry="us-docker.pkg.dev",
    timeout_seconds=1800,
    machine_type="N1_HIGHCPU_32",
)

# Scenario 3: Kaniko with build args
import json
build = build_docker_image_kaniko(
    dockerfile_content=dockerfile,
    context_gcs_path="gs://bucket/context/",
    image_name="custom-ml",
    image_tag="nightly",
    project_id="my-project",
    build_args=json.dumps({"PIP_INDEX_URL": "https://my-pypi/simple"}),
)
```

---

### 3.16 `container/cpr_manager.py`

**Functions**: `cpr_list_images_component`, `cpr_tag_image_component`, `cpr_cleanup_images_component`

**Purpose**: List, tag, and clean up container images in Artifact Registry.

#### Scenarios

```python
# Scenario 1: List images in a repository
images = cpr_list_images_component(
    project_id="my-project",
    region="us-central1",
    repository="ml-models",
)

# Scenario 2: Tag an image for production promotion
tag = cpr_tag_image_component(
    project_id="my-project",
    repository="ml-models",
    image_name="fraud-detector",
    source_tag="v2.1",
    target_tag="production",
)

# Scenario 3: Clean up old images (keep last 5, older than 30 days)
cleanup = cpr_cleanup_images_component(
    project_id="my-project",
    repository="ml-models",
    keep_latest_n=5,
    older_than_days=30,
)
```

---

### 3.17 `container/containerized_component.py`

**Functions**: `generate_component_yaml`, `generate_dockerfile_component`

**Purpose**: Dynamically generate KFP component YAML specs and Dockerfiles.

#### Scenarios

```python
# Scenario 1: Generate component YAML
import json
spec = generate_component_yaml(
    component_name="my-custom-component",
    description="Does custom processing",
    base_image="python:3.10-slim",
    command="python",
    args_json=json.dumps(["main.py", "--mode", "train"]),
    packages_json=json.dumps(["pandas", "scikit-learn"]),
    input_specs_json=json.dumps([
        {"name": "data_path", "type": "String"},
    ]),
    output_specs_json=json.dumps([
        {"name": "model_path", "type": "String"},
    ]),
)

# Scenario 2: Generate Dockerfile for CPU serving
dockerfile = generate_dockerfile_component(
    base_image="python:3.10-slim",
    entrypoint="serve.py",
    extra_commands="ENV MODEL_PATH=/models",
)

# Scenario 3: Generate Dockerfile for GPU training
dockerfile = generate_dockerfile_component(
    base_image="nvidia/cuda:12.2.0-runtime-ubuntu22.04",
    entrypoint="train.py",
    use_gpu=True,
)
```

---

### 3.18 `generic/generic_component.py`

**Functions**: `echo_component`, `send_notification_component`, `copy_dataset_component`, `conditional_gate_component`, `merge_datasets_component`, `wait_component`, `data_profiling_component`

**Purpose**: Reusable utility components for any pipeline.

#### Scenarios

```python
# Scenario 1: Debug logging
echo = echo_component(message="Pipeline started at 2025-01-01")

# Scenario 2: Slack notification
notify = send_notification_component(
    message="Training complete! Accuracy: 94.5%",
    channel="slack",
    webhook_url="https://hooks.slack.com/services/...",
)

# Scenario 3: Copy + sample a dataset
copy = copy_dataset_component(
    source_dataset=full_dataset,
    sample_fraction=0.1,
)

# Scenario 4: Conditional gate
import json
gate = conditional_gate_component(
    condition_json=json.dumps({"result": True}),
)

# Scenario 5: Merge two datasets
merged = merge_datasets_component(
    dataset_a=dataset_1,
    dataset_b=dataset_2,
    merge_strategy="concat",
)

# Scenario 6: Merge with join key
merged = merge_datasets_component(
    dataset_a=users,
    dataset_b=transactions,
    merge_strategy="inner",
    merge_key="user_id",
)

# Scenario 7: Wait between deployment stages (canary)
wait = wait_component(seconds=300)  # 5 min observation window

# Scenario 8: Data profiling
profile = data_profiling_component(
    input_dataset=dataset,
)
# Returns: shape, per-column stats (mean, std, min, max, nulls, unique count, top values)
```

---

## 4. Frameworks — File-by-File Usage

### 4.1 `sklearn_impl.py`

**Class**: `SklearnTrainer` | **Component**: `sklearn_train_component`

**Supported models**: RandomForest, GradientBoosting, Logistic, Linear, SVC, SVR, DecisionTree, KNeighbors, AdaBoost, Ridge, Lasso, ElasticNet (17 total)

#### Scenarios

```python
# Scenario 1: Use SklearnTrainer class directly
from kfp_ml_library.frameworks.sklearn_impl import SklearnTrainer
from kfp_ml_library.configs.trainer_config import SklearnTrainerConfig

config = SklearnTrainerConfig(
    model_class="GradientBoostingClassifier",
    n_estimators=200,
    max_depth=5,
)
trainer = SklearnTrainer(config)
history = trainer.build_and_train(X_train, y_train, X_val, y_val)
metrics, path = trainer.evaluate_and_save(X_test, y_test, "/tmp/model.pkl")

# Scenario 2: Use the KFP component
train = sklearn_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    model_class="RandomForestClassifier",
    task_type="classification",
    target_column="label",
    n_estimators=100,
    max_depth=10,
)

# Scenario 3: Regression with Ridge
train = sklearn_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    model_class="Ridge",           # Not in standalone component's model_map — use class directly
    task_type="regression",
    target_column="price",
)

# Scenario 4: Load and use a saved model
trainer.model = trainer._load_model("/tmp/model.pkl")
predictions = trainer.model.predict(X_new)
```

---

### 4.2 `xgboost_impl.py`

**Class**: `XGBoostTrainer` | **Component**: `xgboost_train_component`

#### Scenarios

```python
# Scenario 1: XGBoost classifier
train = xgboost_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="fraud",
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
)

# Scenario 2: XGBoost regressor with extra params
import json
train = xgboost_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="regression",
    target_column="price",
    extra_params_json=json.dumps({
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
    }),
)

# Scenario 3: GPU training using the Trainer class
from kfp_ml_library.frameworks.xgboost_impl import XGBoostTrainer
from kfp_ml_library.configs.trainer_config import XGBoostTrainerConfig

config = XGBoostTrainerConfig(use_gpu=True, tree_method="hist")
trainer = XGBoostTrainer(config)
trainer.build_and_train(X_train, y_train, X_val, y_val)

# Scenario 4: Save/load XGBoost native format
trainer._save_model("/tmp/xgb_model.json")
trainer._load_model("/tmp/xgb_model.json")
```

---

### 4.3 `keras_impl.py`

**Class**: `KerasTrainer` | **Component**: `keras_train_component`

#### Scenarios

```python
# Scenario 1: Binary classification
train = keras_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="label",
    hidden_layers_json="[256, 128, 64]",
    epochs=100,
    dropout_rate=0.3,
    early_stopping_patience=15,
)

# Scenario 2: Multi-class classification
train = keras_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="category",
    hidden_layers_json="[512, 256, 128]",
    learning_rate=0.0005,
)

# Scenario 3: Regression
train = keras_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="regression",
    target_column="price",
    hidden_layers_json="[128, 64]",
    batch_size=64,
)

# Scenario 4: Using the Trainer class with callbacks
from kfp_ml_library.frameworks.keras_impl import KerasTrainer
from kfp_ml_library.configs.trainer_config import KerasTrainerConfig

config = KerasTrainerConfig(
    hidden_layers=[256, 128, 64, 32],
    dropout_rate=0.2,
    use_batch_norm=True,
    callbacks=["early_stopping", "reduce_lr", "model_checkpoint"],
    output_model_path="/tmp/keras_model",
)
trainer = KerasTrainer(config)
trainer.build_and_train(X_train, y_train, X_val, y_val, num_classes=5, input_shape=(X_train.shape[1],))
```

---

### 4.4 `tensorflow_impl.py`

**Class**: `TensorFlowTrainer` | **Component**: `tensorflow_train_component`

#### Scenarios

```python
# Scenario 1: Standard TF training
train = tensorflow_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="label",
    epochs=50,
    batch_size=64,
)

# Scenario 2: Mixed precision training
train = tensorflow_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="label",
    use_mixed_precision=True,
)

# Scenario 3: Distributed training with MirroredStrategy
from kfp_ml_library.frameworks.tensorflow_impl import TensorFlowTrainer
from kfp_ml_library.configs.trainer_config import TensorFlowTrainerConfig

config = TensorFlowTrainerConfig(
    strategy="mirrored",
    distribute=True,
    mixed_precision=True,
    xla_compilation=True,
    tf_function=True,
    epochs=100,
)
trainer = TensorFlowTrainer(config)
trainer.build_and_train(X_train, y_train, X_val, y_val, num_classes=10, input_shape=(784,))

# Scenario 4: Export to TFLite for mobile deployment
config = TensorFlowTrainerConfig(save_format="tflite")
trainer = TensorFlowTrainer(config)
# After training...
trainer._save_model("/tmp/tf_model")  # → creates model.tflite

# Scenario 5: Export as H5
config = TensorFlowTrainerConfig(save_format="h5")
```

---

### 4.5 `pytorch_impl.py`

**Class**: `PyTorchTrainer` | **Component**: `pytorch_train_component`

#### Scenarios

```python
# Scenario 1: Classification with AdamW + cosine scheduler
from kfp_ml_library.frameworks.pytorch_impl import PyTorchTrainer
from kfp_ml_library.configs.trainer_config import PyTorchTrainerConfig

config = PyTorchTrainerConfig(
    optimizer="AdamW",
    scheduler="cosine",
    weight_decay=0.01,
    gradient_clip_value=1.0,
    epochs=50,
    batch_size=64,
)
trainer = PyTorchTrainer(config)
trainer.build_and_train(X_train, y_train, X_val, y_val, num_classes=5, input_shape=(X_train.shape[1],))

# Scenario 2: Mixed precision (AMP) for GPU training
config = PyTorchTrainerConfig(
    mixed_precision=True,
    optimizer="Adam",
    scheduler="step",
)

# Scenario 3: Regression with linear scheduler
config = PyTorchTrainerConfig(
    task_type=TaskType.REGRESSION,
    optimizer="SGD",
    scheduler="linear",
    learning_rate=0.01,
    weight_decay=0.001,
)

# Scenario 4: Early stopping behavior
# Built-in: stops when validation loss doesn't improve for `early_stopping_patience` epochs
config = PyTorchTrainerConfig(early_stopping_patience=15)
```

---

### 4.6 `automl_impl.py`

**Class**: `AutoMLTrainer` | **Component**: `automl_train_component`

**Engines**: FLAML, auto-sklearn

#### Scenarios

```python
# Scenario 1: FLAML AutoML (default)
train = automl_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="label",
    time_budget=3600,
    metric="accuracy",
)

# Scenario 2: FLAML with specific estimators only
import json
train = automl_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="classification",
    target_column="label",
    time_budget=7200,
    max_models=50,
    estimator_list_json=json.dumps(["xgboost", "lgbm", "rf", "extra_tree"]),
)

# Scenario 3: Auto-sklearn (using the Trainer class)
from kfp_ml_library.frameworks.automl_impl import AutoMLTrainer
from kfp_ml_library.configs.trainer_config import AutoMLTrainerConfig

config = AutoMLTrainerConfig(
    engine="auto-sklearn",
    time_budget=7200,
    ensemble_size=5,
    seed=42,
)
trainer = AutoMLTrainer(config)
trainer.build_and_train(X_train, y_train)

# Scenario 4: Regression with FLAML
train = automl_train_component(
    train_data=train_dataset,
    val_data=val_dataset,
    task_type="regression",
    target_column="price",
    metric="r2",
    time_budget=1800,
)
```

---

## 5. Pipelines — File-by-File Usage

### 5.1 `training_pipeline.py`

**Function**: `create_training_pipeline`

**Steps**: Ingest → Validate → Transform → Feature Engineer → (optional HP Tuning) → Train → Evaluate

#### Scenarios

```python
from kfp import compiler
from kfp_ml_library.pipelines.training_pipeline import create_training_pipeline
import json

# Scenario 1: Simple sklearn training
compiler.Compiler().compile(
    pipeline_func=create_training_pipeline,
    package_path="training.yaml",
)
# Submit with:
params = {
    "source_type": "gcs",
    "source_path": "gs://data/train.csv",
    "target_column": "label",
    "framework": "sklearn",
    "trainer_config_json": json.dumps({
        "framework": "sklearn",
        "model_class": "RandomForestClassifier",
        "target_column": "label",
    }),
}

# Scenario 2: With hyperparameter tuning
params = {
    "source_type": "gcs",
    "source_path": "gs://data/train.csv",
    "target_column": "label",
    "enable_hyperparameter_tuning": True,
    "trainer_config_json": json.dumps({
        "framework": "sklearn",
        "model_class": "GradientBoostingClassifier",
        "target_column": "label",
    }),
    "hp_config_json": json.dumps({
        "n_trials": 100,
        "objective_metric": "f1_weighted",
        "parameter_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
        },
    }),
}

# Scenario 3: BigQuery source with custom preprocessing
params = {
    "source_type": "bigquery",
    "source_path": "",
    "target_column": "churn",
    "numerical_strategy": "robust",
    "categorical_strategy": "label",
    "sample_fraction": 0.5,
    "schema_json": json.dumps({"columns": ["feature_1", "feature_2", "churn"]}),
}
```

---

### 5.2 `deployment_pipeline.py`

**Function**: `create_deployment_pipeline`

**Steps**: (Optional) Build Image → Deploy → Validate Endpoint

#### Scenarios

```python
from kfp_ml_library.pipelines.deployment_pipeline import create_deployment_pipeline

# Scenario 1: Deploy with existing serving image
params = {
    "project_id": "my-project",
    "endpoint_name": "fraud-detector",
    "deployment_strategy": "rolling",
    "machine_type": "n1-standard-4",
    "min_replicas": 2,
    "max_replicas": 10,
}

# Scenario 2: Build custom image + deploy
params = {
    "project_id": "my-project",
    "build_custom_image": True,
    "source_gcs_uri": "gs://bucket/serving-code.tar.gz",
    "image_name": "fraud-serving",
    "image_tag": "v2.0",
    "endpoint_name": "fraud-detector-v2",
    "deployment_strategy": "canary",
}

# Scenario 3: Blue/green deployment
params = {
    "project_id": "my-project",
    "endpoint_name": "recommendations",
    "deployment_strategy": "blue_green",
    "traffic_percentage": 100,
}
```

---

### 5.3 `monitoring_pipeline.py`

**Functions**: `create_monitoring_pipeline`, `create_monitoring_with_data_pipeline`

#### Scenarios

```python
from kfp_ml_library.pipelines.monitoring_pipeline import create_monitoring_with_data_pipeline

# Scenario 1: Monitor with reference/current data from GCS
params = {
    "reference_data_path": "gs://bucket/reference_data.parquet",
    "current_data_path": "gs://bucket/current_batch.parquet",
    "drift_method": "psi",
    "drift_threshold": 0.1,
    "model_name": "fraud_detector_v2",
    "slack_webhook": "https://hooks.slack.com/services/...",
}

# Scenario 2: KS test with email alerts
params = {
    "reference_data_path": "gs://bucket/training_data.parquet",
    "current_data_path": "gs://bucket/production_batch_20250228.parquet",
    "drift_method": "ks_test",
    "drift_threshold": 0.05,
    "email_recipients": json.dumps(["team@company.com"]),
}

# Scenario 3: Monitor specific features only
params = {
    "reference_data_path": "gs://bucket/ref.parquet",
    "current_data_path": "gs://bucket/cur.parquet",
    "features_json": json.dumps(["amount", "merchant_category", "hour"]),
    "drift_method": "wasserstein",
    "drift_threshold": 0.5,
}
```

---

### 5.4 `full_pipeline.py`

**Function**: `create_full_ml_pipeline`

**Steps**: Ingest → Validate → Transform → Feature Eng → Train → Evaluate → Blessing Gate → Conditional Deploy → Notify

#### Scenarios

```python
from kfp import compiler
from kfp_ml_library.pipelines.full_pipeline import create_full_ml_pipeline

compiler.Compiler().compile(
    pipeline_func=create_full_ml_pipeline,
    package_path="full_pipeline.yaml",
)

# Scenario 1: End-to-end from GCS to Vertex AI deployment
params = {
    "source_type": "gcs",
    "source_path": "gs://data/fraud_data.csv",
    "target_column": "is_fraud",
    "framework": "sklearn",
    "trainer_config_json": json.dumps({
        "framework": "sklearn",
        "model_class": "GradientBoostingClassifier",
        "target_column": "is_fraud",
        "n_estimators": 300,
    }),
    "eval_config_json": json.dumps({
        "task_type": "classification",
        "primary_metric": "f1_score",
        "blessing_threshold": 0.90,
    }),
    "project_id": "my-project",
    "endpoint_name": "fraud-detector",
    "deployment_strategy": "canary",
    "notification_channel": "slack",
    "notification_webhook": "https://hooks.slack.com/services/...",
}

# Scenario 2: Regression pipeline with BigQuery source
params = {
    "source_type": "bigquery",
    "target_column": "price",
    "task_type": "regression",
    "framework": "xgboost",
    "trainer_config_json": json.dumps({
        "framework": "xgboost",
        "task_type": "regression",
        "target_column": "price",
    }),
    "eval_config_json": json.dumps({
        "task_type": "regression",
        "primary_metric": "r2_score",
        "blessing_threshold": 0.80,
    }),
    "project_id": "my-project",
    "endpoint_name": "price-predictor",
}

# Scenario 3: Pipeline with custom data processing
params = {
    "source_path": "gs://data/customers.csv",
    "target_column": "churn",
    "numerical_strategy": "robust",
    "categorical_strategy": "label",
    "validation_split": 0.15,
    "test_split": 0.15,
    "sample_fraction": 0.5,  # Train on 50% for faster iteration
}
```

---

## 6. Utils — File-by-File Usage

### 6.1 `logging_utils.py`

**Functions**: `get_logger`, `log_dict`

```python
from kfp_ml_library.utils.logging_utils import get_logger, log_dict

# Scenario 1: Create a consistent logger
logger = get_logger(__name__)
logger.info("Training started")

# Scenario 2: Log a metrics dictionary
metrics = {"accuracy": 0.95, "f1_score": 0.93}
log_dict(logger, metrics, prefix="  → ")
# Output:
#   → accuracy: 0.950000
#   → f1_score: 0.930000

# Scenario 3: Custom format
logger = get_logger("my_component", fmt="%(levelname)s: %(message)s")
```

### 6.2 `io_utils.py`

**Functions**: `save_json`, `load_json`, `save_pickle`, `load_pickle`, `ensure_dir`, `is_gcs_path`, `gcs_upload`, `gcs_download`

```python
from kfp_ml_library.utils.io_utils import *

# Scenario 1: Save/load JSON config
save_json({"accuracy": 0.95}, "/tmp/metrics.json")
data = load_json("/tmp/metrics.json")

# Scenario 2: Save/load pickled model
save_pickle(trained_model, "/tmp/model.pkl")
model = load_pickle("/tmp/model.pkl")

# Scenario 3: GCS operations
gcs_upload("/tmp/model.pkl", "gs://models/v2/model.pkl")
gcs_download("gs://models/v2/model.pkl", "/tmp/local_model.pkl")

# Scenario 4: Check if path is GCS
if is_gcs_path(path):
    gcs_download(path, local_path)
else:
    # read from local

# Scenario 5: Ensure output directory exists
ensure_dir("/tmp/outputs/models/v2")
```

### 6.3 `validation_utils.py`

**Functions**: `validate_config`, `validate_dataframe_columns`, `validate_metric_threshold`, `validate_framework`, `validate_task_type`

```python
from kfp_ml_library.utils.validation_utils import *

# Scenario 1: Validate config has required fields
validate_config(config_dict, required_fields=["framework", "task_type", "target_column"])
# Raises ValidationError if missing

# Scenario 2: Validate DataFrame has expected columns
validate_dataframe_columns(df, ["feature_1", "feature_2", "target"], name="training data")

# Scenario 3: Check metric threshold
passed = validate_metric_threshold("accuracy", 0.92, 0.90, comparison="gte")
# → True (0.92 >= 0.90)

# Scenario 4: Validate framework name
validate_framework("sklearn")   # OK
validate_framework("catboost")  # Raises ValidationError

# Scenario 5: Validate task type
validate_task_type("classification")  # OK
validate_task_type("segmentation")    # Raises ValidationError
```

---

## 7. End-to-End Scenarios

### Scenario A: Fraud Detection Model (sklearn)

```python
from kfp import compiler
from kfp_ml_library import create_full_ml_pipeline
import json

compiler.Compiler().compile(create_full_ml_pipeline, "fraud_pipeline.yaml")

params = {
    "source_type": "bigquery",
    "target_column": "is_fraud",
    "framework": "sklearn",
    "trainer_config_json": json.dumps({
        "framework": "sklearn",
        "model_class": "GradientBoostingClassifier",
        "target_column": "is_fraud",
        "n_estimators": 500,
    }),
    "eval_config_json": json.dumps({
        "task_type": "classification",
        "primary_metric": "f1_score",
        "blessing_threshold": 0.92,
    }),
    "project_id": "my-project",
    "endpoint_name": "fraud-detector",
    "deployment_strategy": "canary",
}
```

### Scenario B: Price Prediction (XGBoost) with HP Tuning

```python
from kfp_ml_library.pipelines.training_pipeline import create_training_pipeline

params = {
    "source_path": "gs://data/house_prices.csv",
    "target_column": "price",
    "enable_hyperparameter_tuning": True,
    "trainer_config_json": json.dumps({
        "framework": "xgboost",
        "task_type": "regression",
        "target_column": "price",
    }),
    "hp_config_json": json.dumps({
        "n_trials": 200,
        "objective_metric": "neg_mean_squared_error",
        "direction": "minimize",
        "parameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 3000},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
        },
    }),
}
```

### Scenario C: Image Classification (TensorFlow) with GPU

```python
from kfp_ml_library.frameworks.tensorflow_impl import TensorFlowTrainer
from kfp_ml_library.configs.trainer_config import TensorFlowTrainerConfig
from kfp_ml_library.configs.compute_constraints import LARGE_GPU

config = TensorFlowTrainerConfig(
    strategy="mirrored",
    distribute=True,
    mixed_precision=True,
    xla_compilation=True,
    epochs=100,
    batch_size=128,
)
trainer = TensorFlowTrainer(config)
# Use LARGE_GPU profile for the pipeline step
```

### Scenario D: Continuous Monitoring with Auto-Rollback

```python
from kfp_ml_library.pipelines.monitoring_pipeline import create_monitoring_with_data_pipeline

params = {
    "reference_data_path": "gs://data/training_snapshot.parquet",
    "current_data_path": "gs://data/production_batch_latest.parquet",
    "drift_method": "psi",
    "drift_threshold": 0.15,
    "slack_webhook": "https://hooks.slack.com/services/...",
    "email_recipients": json.dumps(["oncall@company.com"]),
}
# Combine with AlertConfig(auto_rollback_on_critical=True) for automatic rollback
```

### Scenario E: Container Image Lifecycle in Pipeline

```python
from kfp import dsl
from kfp_ml_library.components.container.docker_builder import build_docker_image_cloud_build
from kfp_ml_library.components.container.cpr_manager import (
    cpr_tag_image_component,
    cpr_cleanup_images_component,
)

@dsl.pipeline(name="image-lifecycle")
def image_pipeline():
    build = build_docker_image_cloud_build(
        source_gcs_uri="gs://bucket/code.tar.gz",
        image_name="ml-serving",
        image_tag="v3.0",
        project_id="my-project",
    )
    tag = cpr_tag_image_component(
        project_id="my-project",
        image_name="ml-serving",
        source_tag="v3.0",
        target_tag="production",
    )
    tag.after(build)
    cleanup = cpr_cleanup_images_component(
        project_id="my-project",
        keep_latest_n=3,
        older_than_days=60,
    )
    cleanup.after(tag)
```

---

## 8. Code Review Findings

### Critical Issues

| # | File | Issue | Severity | Description |
|---|------|-------|----------|-------------|
| 1 | `training/hyperparameter_tuning.py` | Limited framework support | **High** | `objective()` only supports `sklearn` and `xgboost`. Keras, TF, PyTorch models are not tunable via this component. |
| 2 | `training/hyperparameter_tuning.py` | Limited model class mapping | **High** | Only maps `RandomForestClassifier` and `GradientBoostingClassifier` — missing regression models, SVM, etc. |
| 3 | `training/generic_trainer.py` | Only sklearn + XGBoost | **High** | Despite being "generic", it cannot train Keras/TF/PyTorch models. The name is misleading. |
| 4 | `deployment/endpoint_manager.py` | Wrong type annotation | **High** | `rollback_deployment_component` uses `deployment_artifact: Artifact` instead of `Input[Artifact]`, which will fail in KFP v2. |
| 5 | `monitoring/monitoring_pipeline.py` | Empty pipeline body | **Medium** | `create_monitoring_pipeline` has `pass` as its body — it does nothing. |
| 6 | `deployment/deployment_pipeline.py` | Incomplete pipeline | **Medium** | Pipeline only lists endpoints but never calls `deploy_model_component` because the model/eval_report artifacts are passed as URIs, not proper artifacts. |

### Moderate Issues

| # | File | Issue | Severity | Description |
|---|------|-------|----------|-------------|
| 7 | `sklearn_impl.py` | `_build_model` instantiates model twice | **Medium** | Uses `hasattr(cls(), "n_estimators")` which unnecessarily creates a temporary instance to check attribute existence. Should use `inspect` or check class params directly. |
| 8 | `data_prep/data_transformation.py` | `mode()` may fail on empty DataFrame | **Medium** | `df[cat_cols].mode().iloc[0]` will raise `IndexError` if any column has all NaN values. |
| 9 | `evaluation/model_evaluation.py` | Confusion matrix logged as KFP metric | **Low** | `confusion_matrix` is a list not a scalar, but the code tries to log it as a metric. The `isinstance(v, (int, float))` guard saves it, but the matrix is lost from KFP UI. |
| 10 | `constants.py` | Hardcoded `my-project` | **Low** | `DEFAULT_CONTAINER_REGISTRY = "gcr.io/my-project"` — placeholder should be replaced or parameterized. |
| 11 | `configs/` files | No `from_dict()` factory methods | **Medium** | All configs have `to_dict()` but no `from_dict()` deserialization, making it awkward to reconstruct configs from saved state. |
| 12 | `container/cpr_manager.py` | `cpr_tag_image_component` is a stub | **Medium** | Returns a hardcoded "tagged" status without actually performing the tagging API call. |
| 13 | `frameworks/pytorch_impl.py` | `_load_model` requires model already built | **Medium** | `load_state_dict` needs the model architecture to exist first. If you call `_load_model` without `_build_model`, it will crash. |
| 14 | `training/trainer_base.py` | `train_model_component` only supports sklearn+xgboost | **Medium** | Despite being in `trainer_base.py`, the factory pattern only handles sklearn and xgboost, not keras/tf/pytorch/automl. |

### Code Quality Issues

| # | File | Issue | Description |
|---|------|-------|-------------|
| 15 | Multiple | Duplicate metric computation | Classification metrics are computed independently in `model_evaluation.py`, `generic_trainer.py`, `sklearn_impl.py`, `xgboost_impl.py` etc. Should use `metrics.py` centralized functions everywhere. |
| 16 | Multiple | PSI implementation duplicated | PSI calculation appears in `metrics.py`, `model_monitor.py`, and `drift_detection.py` — three copies of the same algorithm. |
| 17 | Multiple | `import json` inside component functions | Required by KFP v2 (@dsl.component), but imports are also at module level outside components creating unnecessary import overhead. |
| 18 | `trainer_config.py` | No validation on dataclass fields | Config accepts any string for `model_class` without validating it exists in the framework's model map. |
| 19 | `data_prep/feature_engineering.py` | `f_classif` hard-coded | Feature selection uses `f_classif` (classification-only). For regression tasks, should use `f_regression`. |
| 20 | `frameworks/keras_impl.py` | `model.summary()` in logger | `model.summary()` returns `None` and prints directly to stdout. The `logger.info("Built Keras model:\n%s", model.summary())` logs `None`. |

### Security Concerns

| # | File | Issue | Description |
|---|------|-------|-------------|
| 21 | `io_utils.py` | `pickle.load` without safety | Unpickling arbitrary files is a known security risk. Consider using `joblib` with `mmap_mode` or ONNX for model serialization. |
| 22 | `constants.py` | Email hardcoded | `ALERT_EMAIL = "ml-alerts@company.com"` should be externalized to environment variables or config files, not source code. |

---

## 9. Modern Features & Improvements

### 9.1 Architecture Improvements

| Improvement | Description | Priority |
|-------------|-------------|----------|
| **Pydantic v2 configs** | Replace `@dataclass` configs with Pydantic `BaseModel` for runtime validation, JSON schema generation, and `.model_validate()` deserialization. | **High** |
| **Plugin/Registry pattern** | Replace if/elif framework chains in `generic_trainer.py` and `train_model_component` with a registry that auto-discovers framework implementations. | **High** |
| **Centralized metric functions** | All components should import from `metrics.py` instead of inline metric computation. Extract PSI/KS into a single shared utility. | **High** |
| **Config `from_dict()` factory** | Add `from_dict()` / `from_json()` classmethods on all config dataclasses for proper deserialization. | **Medium** |
| **Type-safe pipeline parameters** | Use KFP v2 `NamedTuple` pipeline outputs and typed `Input`/`Output` artifacts consistently. | **Medium** |

### 9.2 Framework & Feature Additions

| Feature | Description | Priority |
|---------|-------------|----------|
| **LightGBM support** | Add `LGBMTrainer` — widely used alongside XGBoost. | **High** |
| **CatBoost support** | Add `CatBoostTrainer` — excellent for categorical features. | **Medium** |
| **ONNX export** | Add ONNX export for all frameworks for portable inference. | **High** |
| **MLflow/W&B integration** | Add experiment tracking via MLflow or Weights & Biases for metric logging, artifact versioning, and model registry. | **High** |
| **Feature Store integration** | Add Feast/Vertex AI Feature Store components for feature retrieval. | **Medium** |
| **Model explainability** | Add SHAP/LIME components for model interpretation. | **Medium** |
| **Data versioning** | Add DVC or Delta Lake integration for data version tracking. | **Medium** |

### 9.3 Testing & Quality

| Improvement | Description | Priority |
|-------------|-------------|----------|
| **Unit tests** | No tests exist. Add pytest tests for all configs, trainers, metrics, and validation utils. | **Critical** |
| **Integration tests** | Test full pipeline compilation and component connectivity. | **High** |
| **Property-based testing** | Use Hypothesis for testing metric functions, config validation, and data transformations. | **Medium** |
| **Type checking** | Run `mypy --strict` — current code has inconsistent type hints (e.g., `Dict` vs `dict` mixed). | **Medium** |
| **Linting** | Run `ruff check` — several style issues (unused imports at module level inside component functions). | **Low** |

### 9.4 DevOps & Production Readiness

| Improvement | Description | Priority |
|-------------|-------------|----------|
| **CI/CD pipeline** | Add GitHub Actions / Cloud Build pipeline for lint → test → build → publish. | **Critical** |
| **Container image caching** | Pre-build and cache component base images with dependencies. Currently every component does `pip install` at runtime. | **High** |
| **Secret management** | Replace hardcoded emails/webhooks with Secret Manager, environment variables, or KFP ConfigMaps. | **High** |
| **Retry/error handling** | Add structured error handling with custom exceptions, retry decorators, and dead letter queues. | **Medium** |
| **Pipeline versioning** | Tag pipeline versions and support rollback to previous pipeline definitions. | **Medium** |
| **Cost estimation** | Add a component that estimates compute cost before execution based on data size and compute profile. | **Low** |

### 9.5 Modern Python Features

| Feature | Description |
|---------|-------------|
| **`match` statements** (Python 3.10+) | Replace `if/elif` framework chains with `match framework:` pattern matching. |
| **`slots=True`** | Add `@dataclass(slots=True)` for memory optimization on configs. |
| **`kw_only=True`** | Add `@dataclass(kw_only=True)` to prevent positional argument bugs. |
| **`TypeAlias`** | Use `type` aliases for common types like `MetricsDict = dict[str, float]`. |
| **`typing.Protocol`** | Replace `ABC` with `Protocol` for structural subtyping where appropriate. |
| **`functools.cache`** | Cache expensive operations like model class resolution. |

### 9.6 Observability & Monitoring Enhancements

| Feature | Description |
|---------|-------------|
| **OpenTelemetry tracing** | Add distributed tracing spans for each pipeline component. |
| **Prometheus metrics** | Export pipeline execution metrics (duration, success rate, resource usage). |
| **Grafana dashboards** | Pre-built dashboards for model performance, drift, and latency. |
| **Evidently AI integration** | Use Evidently for production-grade data/model monitoring reports. |
| **Great Expectations** | Replace custom validation with Great Expectations for data quality. |

### 9.7 Scalability Improvements

| Feature | Description |
|---------|-------------|
| **Dask/Ray integration** | Add distributed data processing for large datasets that don't fit in memory. |
| **Streaming data support** | Add Kafka/Pub-Sub connectors for real-time prediction logging. |
| **Multi-model serving** | Support serving multiple models on a single endpoint with routing. |
| **A/B testing framework** | Full A/B testing with statistical significance testing (not just traffic splitting). |
| **Model caching** | Add Redis/Memcached model caching layer for frequently accessed models. |

---

## Summary

This library provides a solid foundation for KFP-based ML pipelines with good modularity and framework coverage. The **highest priority improvements** are:

1. **Add unit tests** — nothing is tested, making refactoring dangerous
2. **Fix the broken/incomplete components** (monitoring pipeline, deployment pipeline, rollback type annotation)
3. **Adopt Pydantic** for config validation and serialization
4. **Add experiment tracking** (MLflow/W&B) for proper ML lifecycle management
5. **Centralize duplicate code** (metrics computation, PSI calculation)
6. **Extend HP tuning** to support all frameworks, not just sklearn/xgboost
7. **Add ONNX export** for portable, framework-agnostic model serving
8. **Pre-build container images** to eliminate runtime pip installs

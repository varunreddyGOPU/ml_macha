# ML Macha — KFP ML Library

[![Build & Publish Wheel](https://github.com/varunreddyGOPU/ml_macha/actions/workflows/build-wheel.yml/badge.svg)](https://github.com/varunreddyGOPU/ml_macha/actions/workflows/build-wheel.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A comprehensive **Kubeflow Pipelines (KFP)** library for building end-to-end ML model deployment pipelines. Provides reusable, composable pipeline components for data preparation, model training, evaluation, deployment, and monitoring.

## Repository

```
https://github.com/varunreddyGOPU/ml_macha.git
```

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases — wheel is built & released here |
| `develop` | Feature development — all new work happens here |

> **Workflow:** `feature/*` → PR into `develop` → PR into `main` → auto-build wheel + GitHub Release

## Supported Frameworks

| Framework | Module | Training | HP Tuning | Deployment |
|-----------|--------|----------|-----------|------------|
| **Scikit-learn** | `frameworks.sklearn_impl` | ✅ | ✅ | ✅ |
| **XGBoost** | `frameworks.xgboost_impl` | ✅ | ✅ | ✅ |
| **Keras** | `frameworks.keras_impl` | ✅ | ✅ | ✅ |
| **TensorFlow** | `frameworks.tensorflow_impl` | ✅ | ✅ | ✅ |
| **PyTorch** | `frameworks.pytorch_impl` | ✅ | ✅ | ✅ |
| **AutoML (FLAML)** | `frameworks.automl_impl` | ✅ | auto | ✅ |

## Project Structure

```
kfp_ml_library/
├── __init__.py
├── components/
│   ├── data_prep/                  # Data pipeline components
│   │   ├── data_ingestion.py       # Load from GCS, BigQuery, CSV, etc.
│   │   ├── data_validation.py      # Schema checks, nulls, duplicates, outliers
│   │   ├── data_transformation.py  # Scale, encode, impute, train/val/test split
│   │   └── feature_engineering.py  # Polynomial, interaction, binning, selection
│   ├── training/                   # Model training components
│   │   ├── trainer_base.py         # Abstract TrainerBase class + factory
│   │   ├── hyperparameter_tuning.py # Optuna-based HP tuning component
│   │   └── generic_trainer.py      # Framework-agnostic training component
│   ├── evaluation/                 # Model evaluation components
│   │   ├── evaluator_base.py       # Abstract EvaluatorBase class
│   │   ├── model_evaluation.py     # Full eval + blessing gate component
│   │   └── metrics.py              # Classification, regression, drift metrics
│   ├── deployment/                 # Model deployment components
│   │   ├── model_deployer.py       # Deploy with blue/green, canary, rolling
│   │   └── endpoint_manager.py     # Manage endpoints + rollback
│   ├── monitoring/                 # Model monitoring components
│   │   ├── model_monitor.py        # Drift detection + prediction logging
│   │   └── drift_detection.py      # PSI, KS, Wasserstein + alerting
│   ├── container/                  # Container/Docker/CPR components
│   │   ├── docker_builder.py       # Kaniko + Cloud Build image building
│   │   ├── cpr_manager.py          # Container registry management
│   │   └── containerized_component.py # Generate component YAML + Dockerfile
│   └── generic/                    # Generic utility components
│       └── generic_component.py    # Echo, notify, copy, merge, profile, gate
├── configs/                        # Configuration dataclasses
│   ├── constants.py                # All constants, enums, defaults
│   ├── compute_constraints.py      # Resource requests/limits, GPU, node config
│   ├── trainer_config.py           # Per-framework trainer configs
│   ├── evaluator_config.py         # Evaluation thresholds, metrics, slicing
│   └── monitoring_config.py        # Drift detection, alerting, latency config
├── frameworks/                     # Framework-specific implementations
│   ├── sklearn_impl.py             # SklearnTrainer + KFP component
│   ├── xgboost_impl.py             # XGBoostTrainer + KFP component
│   ├── keras_impl.py               # KerasTrainer + KFP component
│   ├── tensorflow_impl.py          # TensorFlowTrainer + KFP component
│   ├── pytorch_impl.py             # PyTorchTrainer + KFP component
│   └── automl_impl.py              # AutoMLTrainer + KFP component
├── pipelines/                      # Pre-built pipeline definitions
│   ├── training_pipeline.py        # Data → Train → Evaluate
│   ├── deployment_pipeline.py      # Build → Deploy → Validate
│   ├── monitoring_pipeline.py      # Drift → Alert → Log
│   └── full_pipeline.py            # End-to-end: Data → Deploy → Notify
└── utils/                          # Utility modules
    ├── logging_utils.py            # Consistent logging setup
    ├── io_utils.py                 # JSON/pickle/GCS I/O helpers
    └── validation_utils.py         # Config & schema validation
```

## Installation

```bash
# Core installation (sklearn + xgboost)
pip install -e .

# With TensorFlow
pip install -e ".[tensorflow]"

# With PyTorch
pip install -e ".[pytorch]"

# With AutoML
pip install -e ".[automl]"

# With GCP services (Vertex AI, Cloud Build, etc.)
pip install -e ".[gcp]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

## Quick Start

### 1. Run the Full Pipeline

```python
from kfp import compiler
from kfp_ml_library import create_full_ml_pipeline

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=create_full_ml_pipeline,
    package_path="full_pipeline.yaml",
)
```

### 2. Use Individual Components

```python
from kfp import dsl
from kfp_ml_library.components.data_prep.data_ingestion import data_ingestion_component
from kfp_ml_library.components.training.generic_trainer import generic_train_component
from kfp_ml_library.components.evaluation.model_evaluation import model_evaluation_component

@dsl.pipeline(name="custom-pipeline")
def my_pipeline(source_path: str, target_column: str):
    ingest = data_ingestion_component(
        source_type="gcs",
        source_path=source_path,
        file_format="csv",
    )
    # ... compose your custom pipeline
```

### 3. Use Framework-Specific Trainers

```python
from kfp_ml_library.frameworks.sklearn_impl import sklearn_train_component
from kfp_ml_library.frameworks.xgboost_impl import xgboost_train_component
from kfp_ml_library.frameworks.keras_impl import keras_train_component
from kfp_ml_library.frameworks.pytorch_impl import pytorch_train_component
from kfp_ml_library.frameworks.automl_impl import automl_train_component
```

### 4. Use the Trainer Classes Programmatically

```python
from kfp_ml_library.frameworks.sklearn_impl import SklearnTrainer
from kfp_ml_library.configs.trainer_config import SklearnTrainerConfig

config = SklearnTrainerConfig(
    model_class="RandomForestClassifier",
    n_estimators=200,
    max_depth=10,
)

trainer = SklearnTrainer(config)
history = trainer.build_and_train(X_train, y_train, X_val, y_val)
metrics, path = trainer.evaluate_and_save(X_test, y_test, "/tmp/model.pkl")
```

### 5. Configure Compute Constraints

```python
from kfp_ml_library.configs.compute_constraints import (
    ComputeConstraints,
    LARGE_GPU,
    MEDIUM_CPU,
    TRAINING_DEFAULT,
)

# Use a pre-built profile
constraints = LARGE_GPU

# Or customize
constraints = ComputeConstraints(
    requests=ResourceRequests(cpu="8", memory="32Gi"),
    limits=ResourceLimits(gpu="2"),
    timeout_seconds=7200,
)
```

### 6. Configure Monitoring

```python
from kfp_ml_library.configs.monitoring_config import (
    MonitoringConfig,
    DriftDetectionConfig,
    AlertConfig,
)

monitoring = MonitoringConfig(
    model_name="fraud_detector_v2",
    drift_config=DriftDetectionConfig(
        method="ks_test",
        threshold=0.05,
        features_to_monitor=["amount", "merchant_category"],
    ),
    alert_config=AlertConfig(
        email_recipients=["team@company.com"],
        auto_rollback_on_critical=True,
    ),
)
```

### 7. Hyperparameter Tuning

```python
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
    },
)
```

## Pre-built Pipelines

| Pipeline | Description |
|----------|-------------|
| `create_training_pipeline` | Data ingestion → validation → transformation → feature eng → training → evaluation |
| `create_deployment_pipeline` | Docker build → model deployment → endpoint validation |
| `create_monitoring_pipeline` | Drift detection → alerting → prediction logging |
| `create_full_ml_pipeline` | Complete end-to-end: data → train → evaluate → deploy → notify |

## Key Features

- **Framework Agnostic**: Single interface for sklearn, XGBoost, Keras, TensorFlow, PyTorch, AutoML
- **Composable Components**: Mix and match components to build custom pipelines
- **Configurable**: Dataclass-based configs for compute, training, evaluation, monitoring
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization with pruning
- **Model Blessing**: Automated model quality gates before deployment
- **Deployment Strategies**: Blue/green, canary, rolling, shadow deployments
- **Drift Detection**: PSI, KS test, Wasserstein distance, Chi-squared
- **Container Management**: Kaniko / Cloud Build image building, registry cleanup
- **Alerting**: Slack, email notifications on drift or deployment events
- **Pre-built Compute Profiles**: SMALL_CPU, MEDIUM_CPU, LARGE_CPU, SMALL_GPU, LARGE_GPU

## License

Apache License 2.0

---

## Download Wheel

Pre-built wheel files are available from:

1. **GitHub Actions Artifacts** — every push to `main` or `develop` produces a downloadable `.whl` artifact
2. **GitHub Releases** — tagged versions (e.g. `v1.0.0`) automatically create a Release with the wheel attached

```bash
# Install directly from GitHub Release (replace TAG)
pip install https://github.com/varunreddyGOPU/ml_macha/releases/download/v1.0.0/kfp_ml_library-1.0.0-py3-none-any.whl

# Or install from local wheel
pip install dist/kfp_ml_library-1.0.0-py3-none-any.whl
```

## Contributing

1. Fork → create `feature/your-feature` from `develop`
2. Make changes, add tests
3. Open a PR into `develop`
4. After review, `develop` is merged into `main` for release

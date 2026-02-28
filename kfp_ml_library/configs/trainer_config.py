"""
Trainer configuration module.

Defines configuration dataclasses for all supported training frameworks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kfp_ml_library.configs.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_TRIALS_HYPERPARAMETER,
    DEFAULT_VALIDATION_SPLIT,
    FrameworkType,
    TaskType,
)


@dataclass
class TrainerConfig:
    """Base trainer configuration shared across all frameworks."""

    framework: FrameworkType = FrameworkType.SKLEARN
    task_type: TaskType = TaskType.CLASSIFICATION
    model_name: str = "model"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    random_state: int = 42
    verbose: int = 1
    output_model_path: str = "/tmp/model"
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework.value if isinstance(self.framework, FrameworkType) else self.framework,
            "task_type": self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
            "model_name": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "output_model_path": self.output_model_path,
            **self.custom_params,
        }


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter tuning runs."""

    search_algorithm: str = "bayesian"  # bayesian | grid | random | hyperband
    n_trials: int = DEFAULT_N_TRIALS_HYPERPARAMETER
    objective_metric: str = "accuracy"
    direction: str = "maximize"  # maximize | minimize
    parameter_space: Dict[str, Any] = field(default_factory=dict)
    cross_validation_folds: int = 5
    parallel_trials: int = 1
    timeout_per_trial: int = 600
    early_stopping: bool = True
    pruner: str = "median"  # median | hyperband | none
    sampler: str = "tpe"  # tpe | cmaes | random | grid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_algorithm": self.search_algorithm,
            "n_trials": self.n_trials,
            "objective_metric": self.objective_metric,
            "direction": self.direction,
            "parameter_space": self.parameter_space,
            "cross_validation_folds": self.cross_validation_folds,
            "parallel_trials": self.parallel_trials,
            "timeout_per_trial": self.timeout_per_trial,
            "early_stopping": self.early_stopping,
            "pruner": self.pruner,
            "sampler": self.sampler,
        }


@dataclass
class SklearnTrainerConfig(TrainerConfig):
    """Scikit-learn specific config."""

    framework: FrameworkType = FrameworkType.SKLEARN
    model_class: str = "RandomForestClassifier"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    n_jobs: int = -1
    class_weight: Optional[str] = None


@dataclass
class XGBoostTrainerConfig(TrainerConfig):
    """XGBoost specific config."""

    framework: FrameworkType = FrameworkType.XGBOOST
    n_estimators: int = 1000
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    eval_metric: str = "logloss"
    use_gpu: bool = False


@dataclass
class KerasTrainerConfig(TrainerConfig):
    """Keras specific config."""

    framework: FrameworkType = FrameworkType.KERAS
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    activation: str = "relu"
    output_activation: str = "softmax"
    use_batch_norm: bool = True
    callbacks: List[str] = field(
        default_factory=lambda: ["early_stopping", "reduce_lr", "model_checkpoint"]
    )


@dataclass
class TensorFlowTrainerConfig(TrainerConfig):
    """TensorFlow specific config."""

    framework: FrameworkType = FrameworkType.TENSORFLOW
    strategy: str = "mirrored"  # mirrored | multi_worker | tpu | parameter_server
    mixed_precision: bool = False
    xla_compilation: bool = False
    tf_function: bool = True
    distribute: bool = False
    num_workers: int = 1
    save_format: str = "saved_model"  # saved_model | h5 | tflite


@dataclass
class PyTorchTrainerConfig(TrainerConfig):
    """PyTorch specific config."""

    framework: FrameworkType = FrameworkType.PYTORCH
    optimizer: str = "AdamW"
    scheduler: str = "cosine"  # cosine | step | exponential | linear
    weight_decay: float = 0.01
    gradient_clip_value: float = 1.0
    mixed_precision: bool = False
    distributed: bool = False
    num_workers_dataloader: int = 4
    pin_memory: bool = True
    accumulation_steps: int = 1


@dataclass
class AutoMLTrainerConfig(TrainerConfig):
    """AutoML specific config."""

    framework: FrameworkType = FrameworkType.AUTOML
    engine: str = "flaml"  # flaml | auto-sklearn | h2o | tpot
    time_budget: int = 3600
    max_models: int = 20
    metric: str = "accuracy"
    include_estimators: Optional[List[str]] = None
    exclude_estimators: Optional[List[str]] = None
    ensemble_size: int = 1
    seed: int = 42

"""
Trainer base module.

Provides the abstract ``TrainerBase`` class that every framework-specific
trainer must implement.  Also provides a factory for creating trainers
by framework type.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, FrameworkType
from kfp_ml_library.configs.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class TrainerBase(ABC):
    """
    Abstract base class for all ML framework trainers.

    Sub-classes must implement ``_build_model``, ``_train``, ``_evaluate``
    and ``_save_model``.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.model: Any = None
        self.history: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        """Construct the model architecture / estimator."""
        ...

    @abstractmethod
    def _train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the training loop; return a history dict."""
        ...

    @abstractmethod
    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        """Return evaluation metrics dict."""
        ...

    @abstractmethod
    def _save_model(self, output_path: str) -> str:
        """Persist model to *output_path*; return the saved path."""
        ...

    @abstractmethod
    def _load_model(self, model_path: str) -> Any:
        """Load a previously saved model."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def build_and_train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        input_shape: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """High-level convenience: build → train → return history."""
        logger.info("Building model with config: %s", self.config.to_dict())
        self.model = self._build_model(input_shape=input_shape, **kwargs)
        logger.info("Starting training…")
        self.history = self._train(X_train, y_train, X_val, y_val, **kwargs)
        logger.info("Training complete.")
        return self.history

    def evaluate_and_save(
        self,
        X_test,
        y_test,
        output_path: str,
        **kwargs,
    ) -> Tuple[Dict[str, float], str]:
        """Evaluate → save → return (metrics, model_path)."""
        metrics = self._evaluate(X_test, y_test, **kwargs)
        logger.info("Evaluation metrics: %s", metrics)
        model_path = self._save_model(output_path)
        logger.info("Model saved to %s", model_path)
        return metrics, model_path


# -----------------------------------------------------------------------
# KFP component wrappers
# -----------------------------------------------------------------------

@dsl.component(base_image=DEFAULT_BASE_IMAGE, packages_to_install=["scikit-learn>=1.3.0"])
def train_model_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    config_json: str,
) -> str:
    """
    Generic KFP training component.

    Reads *config_json*, instantiates the appropriate trainer via
    ``TrainerFactory``, trains, evaluates, and writes outputs.
    """
    import json
    import pickle
    import pandas as pd

    config = json.loads(config_json)
    framework = config.get("framework", "sklearn")

    # Load datasets
    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    target_col = config.get("target_column", train_df.columns[-1])
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Dynamically import the right framework trainer
    if framework == "sklearn":
        from kfp_ml_library.frameworks.sklearn_impl import SklearnTrainer
        from kfp_ml_library.configs.trainer_config import SklearnTrainerConfig
        trainer_cfg = SklearnTrainerConfig(**{k: v for k, v in config.items() if k in SklearnTrainerConfig.__dataclass_fields__})
        trainer = SklearnTrainer(trainer_cfg)
    elif framework == "xgboost":
        from kfp_ml_library.frameworks.xgboost_impl import XGBoostTrainer
        from kfp_ml_library.configs.trainer_config import XGBoostTrainerConfig
        trainer_cfg = XGBoostTrainerConfig(**{k: v for k, v in config.items() if k in XGBoostTrainerConfig.__dataclass_fields__})
        trainer = XGBoostTrainer(trainer_cfg)
    else:
        raise ValueError(f"Unsupported framework inside this component: {framework}")

    history = trainer.build_and_train(X_train, y_train, X_val, y_val)
    eval_metrics, saved_path = trainer.evaluate_and_save(X_val, y_val, model_artifact.path)

    for k, v in eval_metrics.items():
        metrics_artifact.log_metric(k, v)

    model_artifact.metadata["framework"] = framework
    model_artifact.metadata["config"] = config

    return json.dumps(eval_metrics)

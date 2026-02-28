"""
AutoML framework implementation.

Provides ``AutoMLTrainer`` (extends ``TrainerBase``) and a standalone
KFP component using FLAML / auto-sklearn.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from kfp_ml_library.components.training.trainer_base import TrainerBase
from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, TaskType
from kfp_ml_library.configs.trainer_config import AutoMLTrainerConfig

logger = logging.getLogger(__name__)


class AutoMLTrainer(TrainerBase):
    """AutoML trainer using FLAML or auto-sklearn."""

    def __init__(self, config: AutoMLTrainerConfig) -> None:
        super().__init__(config)
        self.config: AutoMLTrainerConfig = config

    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        engine = self.config.engine

        if engine == "flaml":
            from flaml import AutoML

            self.model = AutoML()
        elif engine == "auto-sklearn":
            import autosklearn.classification
            import autosklearn.regression

            task_str = self.config.task_type.value if isinstance(self.config.task_type, TaskType) else self.config.task_type
            if task_str in ("classification",):
                self.model = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=self.config.time_budget,
                    seed=self.config.seed,
                    ensemble_size=self.config.ensemble_size,
                )
            else:
                self.model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=self.config.time_budget,
                    seed=self.config.seed,
                    ensemble_size=self.config.ensemble_size,
                )
        else:
            raise ValueError(f"Unsupported AutoML engine: {engine}")

        logger.info("Built AutoML model with engine=%s", engine)
        return self.model

    def _train(
        self, X_train, y_train, X_val=None, y_val=None, **kwargs
    ) -> Dict[str, Any]:
        engine = self.config.engine
        task_str = self.config.task_type.value if isinstance(self.config.task_type, TaskType) else self.config.task_type

        if engine == "flaml":
            settings = {
                "time_budget": self.config.time_budget,
                "metric": self.config.metric,
                "task": task_str,
                "seed": self.config.seed,
                "verbose": self.config.verbose,
                "max_iter": self.config.max_models,
            }
            if self.config.include_estimators:
                settings["estimator_list"] = self.config.include_estimators

            self.model.fit(X_train, y_train, **settings)

            self.best_params = self.model.best_config or {}
            return {
                "best_estimator": str(getattr(self.model, "best_estimator", "unknown")),
                "best_loss": float(getattr(self.model, "best_loss", 0.0)),
                "best_config": self.best_params,
            }

        elif engine == "auto-sklearn":
            self.model.fit(X_train, y_train)
            return {"status": "completed", "engine": "auto-sklearn"}

        return {"status": "completed"}

    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_squared_error,
            r2_score,
        )

        y_pred = self.model.predict(X_test)
        task_str = self.config.task_type.value if isinstance(self.config.task_type, TaskType) else self.config.task_type

        if task_str == "classification":
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            mse = float(mean_squared_error(y_test, y_pred))
            return {
                "mse": mse,
                "rmse": mse**0.5,
                "r2_score": float(r2_score(y_test, y_pred)),
            }

    def _save_model(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)
        return output_path

    def _load_model(self, model_path: str) -> Any:
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        return self.model


# ---------------------------------------------------------------------------
# Standalone KFP component
# ---------------------------------------------------------------------------
@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "flaml>=2.1.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "pyarrow>=14.0.0",
    ],
)
def automl_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    task_type: str = "classification",
    target_column: str = "",
    time_budget: int = 3600,
    metric: str = "accuracy",
    max_models: int = 20,
    estimator_list_json: str = "[]",
) -> str:
    """Standalone AutoML (FLAML) training component."""
    import json
    import pickle
    import pandas as pd
    from flaml import AutoML
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if not target_column:
        target_column = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": metric,
        "task": task_type,
        "max_iter": max_models,
        "verbose": 1,
    }

    estimators = json.loads(estimator_list_json) if estimator_list_json else []
    if estimators:
        settings["estimator_list"] = estimators

    automl.fit(X_train, y_train, **settings)

    y_pred = automl.predict(X_val)
    eval_metrics: dict = {}

    if task_type == "classification":
        eval_metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
        eval_metrics["f1_score"] = float(f1_score(y_val, y_pred, average="weighted"))
    else:
        eval_metrics["mse"] = float(mean_squared_error(y_val, y_pred))
        eval_metrics["r2_score"] = float(r2_score(y_val, y_pred))

    eval_metrics["best_estimator"] = str(getattr(automl, "best_estimator", "unknown"))
    eval_metrics["best_loss"] = float(getattr(automl, "best_loss", 0.0))

    for k, v in eval_metrics.items():
        if isinstance(v, (int, float)):
            metrics_artifact.log_metric(k, v)

    with open(model_artifact.path, "wb") as f:
        pickle.dump(automl, f)

    model_artifact.metadata["framework"] = "automl"
    model_artifact.metadata["engine"] = "flaml"
    model_artifact.metadata["best_estimator"] = eval_metrics.get("best_estimator", "")

    return json.dumps(eval_metrics)

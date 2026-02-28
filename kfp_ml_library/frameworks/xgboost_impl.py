"""
XGBoost framework implementation.

Provides ``XGBoostTrainer`` (extends ``TrainerBase``) and a standalone
KFP component for XGBoost training.
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
from kfp_ml_library.configs.trainer_config import XGBoostTrainerConfig

logger = logging.getLogger(__name__)


class XGBoostTrainer(TrainerBase):
    """XGBoost trainer implementation."""

    def __init__(self, config: XGBoostTrainerConfig) -> None:
        super().__init__(config)
        self.config: XGBoostTrainerConfig = config

    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        import xgboost as xgb

        params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "tree_method": self.config.tree_method,
            "random_state": self.config.random_state,
            "verbosity": self.config.verbose,
        }

        if self.config.use_gpu:
            params["device"] = "cuda"

        params.update(self.config.custom_params)

        if self.config.task_type in (TaskType.REGRESSION, "regression"):
            self.model = xgb.XGBRegressor(**params)
        else:
            params["eval_metric"] = self.config.eval_metric
            params["use_label_encoder"] = False
            self.model = xgb.XGBClassifier(**params)

        logger.info("Built XGBoost model with params: %s", params)
        return self.model

    def _train(
        self, X_train, y_train, X_val=None, y_val=None, **kwargs
    ) -> Dict[str, Any]:
        fit_params: dict = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = self.config.verbose > 0

        self.model.fit(X_train, y_train, **fit_params)

        history: Dict[str, Any] = {"status": "completed"}
        if hasattr(self.model, "evals_result"):
            try:
                history["evals_result"] = self.model.evals_result()
            except Exception:
                pass
        if hasattr(self.model, "feature_importances_"):
            history["feature_importances"] = self.model.feature_importances_.tolist()

        return history

    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        y_pred = self.model.predict(X_test)

        if self.config.task_type in (TaskType.CLASSIFICATION, "classification"):
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            mse = float(mean_squared_error(y_test, y_pred))
            return {
                "mse": mse,
                "rmse": mse**0.5,
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2_score": float(r2_score(y_test, y_pred)),
            }

    def _save_model(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.model.save_model(output_path)
        logger.info("XGBoost model saved to %s", output_path)
        return output_path

    def _load_model(self, model_path: str) -> Any:
        import xgboost as xgb

        if self.config.task_type in (TaskType.REGRESSION, "regression"):
            self.model = xgb.XGBRegressor()
        else:
            self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        return self.model


# ---------------------------------------------------------------------------
# Standalone KFP component
# ---------------------------------------------------------------------------
@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "pyarrow>=14.0.0",
    ],
)
def xgboost_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    task_type: str = "classification",
    target_column: str = "",
    n_estimators: int = 1000,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    extra_params_json: str = "{}",
) -> str:
    """Standalone XGBoost training component."""
    import json
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if not target_column:
        target_column = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    extra = json.loads(extra_params_json) if extra_params_json else {}

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        **extra,
    }

    if task_type == "regression":
        model = xgb.XGBRegressor(**params)
    else:
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)

    eval_metrics: dict = {}
    if task_type == "classification":
        eval_metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
        eval_metrics["f1_score"] = float(f1_score(y_val, y_pred, average="weighted"))
    else:
        eval_metrics["mse"] = float(mean_squared_error(y_val, y_pred))
        eval_metrics["r2_score"] = float(r2_score(y_val, y_pred))

    for k, v in eval_metrics.items():
        metrics_artifact.log_metric(k, v)

    model.save_model(model_artifact.path)
    model_artifact.metadata["framework"] = "xgboost"

    return json.dumps(eval_metrics)

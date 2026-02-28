"""
Scikit-learn framework implementation.

Provides ``SklearnTrainer`` (extends ``TrainerBase``) and a standalone
KFP component for sklearn training.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from kfp_ml_library.components.training.trainer_base import TrainerBase
from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, TaskType
from kfp_ml_library.configs.trainer_config import SklearnTrainerConfig

logger = logging.getLogger(__name__)


class SklearnTrainer(TrainerBase):
    """Scikit-learn trainer implementation."""

    def __init__(self, config: SklearnTrainerConfig) -> None:
        super().__init__(config)
        self.config: SklearnTrainerConfig = config

    def _get_model_class(self):
        """Dynamically resolve the sklearn estimator class."""
        from sklearn.ensemble import (
            AdaBoostClassifier,
            AdaBoostRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import (
            ElasticNet,
            Lasso,
            LinearRegression,
            LogisticRegression,
            Ridge,
        )
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "LogisticRegression": LogisticRegression,
            "LinearRegression": LinearRegression,
            "SVC": SVC,
            "SVR": SVR,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "KNeighborsClassifier": KNeighborsClassifier,
            "KNeighborsRegressor": KNeighborsRegressor,
            "AdaBoostClassifier": AdaBoostClassifier,
            "AdaBoostRegressor": AdaBoostRegressor,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
        }
        return model_map.get(self.config.model_class, RandomForestClassifier)

    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        cls = self._get_model_class()
        params = {
            "random_state": self.config.random_state,
        }
        if hasattr(cls(), "n_estimators"):
            params["n_estimators"] = self.config.n_estimators
        if self.config.max_depth is not None and hasattr(cls(), "max_depth"):
            params["max_depth"] = self.config.max_depth
        if hasattr(cls(), "n_jobs"):
            params["n_jobs"] = self.config.n_jobs

        # Merge custom params
        params.update(self.config.custom_params)

        self.model = cls(**params)
        logger.info("Built %s with params: %s", self.config.model_class, params)
        return self.model

    def _train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        self.model.fit(X_train, y_train)

        history: Dict[str, Any] = {"status": "completed"}

        if hasattr(self.model, "feature_importances_"):
            history["feature_importances"] = self.model.feature_importances_.tolist()

        return history

    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )

        y_pred = self.model.predict(X_test)

        if self.config.task_type in (TaskType.CLASSIFICATION, "classification"):
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
                "precision": float(precision_score(y_test, y_pred, average="weighted")),
                "recall": float(recall_score(y_test, y_pred, average="weighted")),
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
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Sklearn model saved to %s", output_path)
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
    packages_to_install=["scikit-learn>=1.3.0", "pandas>=2.1.0", "pyarrow>=14.0.0"],
)
def sklearn_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    model_class: str = "RandomForestClassifier",
    task_type: str = "classification",
    target_column: str = "",
    n_estimators: int = 100,
    max_depth: int = -1,
    random_state: int = 42,
    extra_params_json: str = "{}",
) -> str:
    """Standalone sklearn training component."""
    import json
    import pickle
    import pandas as pd
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if not target_column:
        target_column = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    # Resolve model class
    model_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
    }
    cls = model_map.get(model_class, RandomForestClassifier)

    params: dict = {"random_state": random_state}
    extra = json.loads(extra_params_json) if extra_params_json else {}
    if hasattr(cls(), "n_estimators"):
        params["n_estimators"] = n_estimators
    if max_depth > 0 and hasattr(cls(), "max_depth"):
        params["max_depth"] = max_depth
    params.update(extra)

    model = cls(**params)
    model.fit(X_train, y_train)
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

    with open(model_artifact.path, "wb") as f:
        pickle.dump(model, f)

    model_artifact.metadata["framework"] = "sklearn"
    model_artifact.metadata["model_class"] = model_class

    return json.dumps(eval_metrics)

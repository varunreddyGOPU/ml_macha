"""
Generic trainer KFP component.

A single component that can train any registered framework model
by accepting a config dictionary.
"""

from __future__ import annotations

import json
import logging

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE

logger = logging.getLogger(__name__)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
    ],
)
def generic_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    config_json: str,
) -> str:
    """
    Framework-agnostic training component.

    *config_json* must include at minimum ``framework`` and ``task_type``.
    """
    import json
    import pickle
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
    )

    config = json.loads(config_json)
    framework = config.get("framework", "sklearn")
    task_type = config.get("task_type", "classification")
    target_col = config.get("target_column", None)

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if target_col is None:
        target_col = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # ---------- Build model ----------
    model = None
    if framework == "sklearn":
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )

        model_class_name = config.get("model_class", "RandomForestClassifier")
        model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }
        cls = model_map.get(model_class_name, RandomForestClassifier)
        model_params = {
            k: v
            for k, v in config.items()
            if k in ("n_estimators", "max_depth", "n_jobs", "random_state", "learning_rate")
        }
        model = cls(**model_params)

    elif framework == "xgboost":
        import xgboost as xgb

        xgb_params = {
            k: v for k, v in config.items()
            if k in (
                "n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "reg_alpha",
                "reg_lambda", "tree_method",
            )
        }
        if task_type == "regression":
            model = xgb.XGBRegressor(**xgb_params, random_state=42)
        else:
            model = xgb.XGBClassifier(
                **xgb_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
    else:
        raise ValueError(f"Framework {framework} not supported in generic_train_component")

    # ---------- Train ----------
    model.fit(X_train, y_train)

    # ---------- Evaluate ----------
    y_pred = model.predict(X_val)
    eval_metrics: dict = {}
    if task_type in ("classification",):
        eval_metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
        eval_metrics["f1_score"] = float(f1_score(y_val, y_pred, average="weighted"))
    else:
        eval_metrics["mse"] = float(mean_squared_error(y_val, y_pred))
        eval_metrics["r2_score"] = float(r2_score(y_val, y_pred))

    for k, v in eval_metrics.items():
        metrics_artifact.log_metric(k, v)

    # ---------- Save ----------
    with open(model_artifact.path, "wb") as f:
        pickle.dump(model, f)

    model_artifact.metadata["framework"] = framework
    model_artifact.metadata["task_type"] = task_type

    return json.dumps(eval_metrics)

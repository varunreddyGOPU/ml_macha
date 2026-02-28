"""
Hyperparameter tuning KFP component.

Supports Optuna-based Bayesian optimisation, grid search, and random search
for any registered framework trainer.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE

logger = logging.getLogger(__name__)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "optuna>=3.4.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "xgboost>=2.0.0",
    ],
)
def hyperparameter_tuning_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    best_params_artifact: Output[Artifact],
    tuning_metrics: Output[Metrics],
    trainer_config_json: str,
    hp_config_json: str,
) -> str:
    """
    Run hyperparameter tuning using Optuna.

    Parameters
    ----------
    trainer_config_json : str
        JSON-serialized ``TrainerConfig``.
    hp_config_json : str
        JSON-serialized ``HyperparameterConfig`` including ``parameter_space``.

    Returns
    -------
    str
        JSON-serialised best parameters.
    """
    import json
    import optuna
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import cross_val_score

    trainer_config = json.loads(trainer_config_json)
    hp_config = json.loads(hp_config_json)

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    target_col = trainer_config.get("target_column", train_df.columns[-1])
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values

    framework = trainer_config.get("framework", "sklearn")
    param_space = hp_config.get("parameter_space", {})
    n_trials = hp_config.get("n_trials", 50)
    direction = hp_config.get("direction", "maximize")
    cv_folds = hp_config.get("cross_validation_folds", 5)
    objective_metric = hp_config.get("objective_metric", "accuracy")

    def _suggest(trial: optuna.Trial, name: str, spec: dict):
        ptype = spec.get("type", "float")
        if ptype == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif ptype == "int":
            return trial.suggest_int(name, spec["low"], spec["high"])
        elif ptype == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        return spec.get("default")

    def objective(trial: optuna.Trial) -> float:
        params = {k: _suggest(trial, k, v) for k, v in param_space.items()}

        if framework == "sklearn":
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            model_class = trainer_config.get("model_class", "RandomForestClassifier")
            cls = {"RandomForestClassifier": RandomForestClassifier,
                   "GradientBoostingClassifier": GradientBoostingClassifier}.get(model_class, RandomForestClassifier)
            model = cls(**params, random_state=42)
        elif framework == "xgboost":
            import xgboost as xgb
            task = trainer_config.get("task_type", "classification")
            if task == "regression":
                model = xgb.XGBRegressor(**params, random_state=42)
            else:
                model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric="logloss")
        else:
            raise ValueError(f"HP tuning not implemented for {framework} inside this component")

        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=objective_metric)
        return float(np.mean(scores))

    # Optuna study
    pruner = (
        optuna.pruners.MedianPruner()
        if hp_config.get("pruner", "median") == "median"
        else optuna.pruners.HyperbandPruner()
    )
    study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=hp_config.get("timeout_per_trial", 600) * n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Write outputs
    with open(best_params_artifact.path, "w") as f:
        json.dump(best_params, f, indent=2)

    tuning_metrics.log_metric("best_score", best_value)
    tuning_metrics.log_metric("n_trials_completed", len(study.trials))
    for k, v in best_params.items():
        if isinstance(v, (int, float)):
            tuning_metrics.log_metric(f"best_{k}", v)

    best_params_artifact.metadata["best_value"] = best_value
    best_params_artifact.metadata["n_trials"] = len(study.trials)

    return json.dumps({"best_params": best_params, "best_score": best_value})


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["scikit-learn>=1.3.0", "pandas>=2.1.0"],
)
def retrain_with_best_params_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    best_params_artifact: Input[Artifact],
    model_artifact: Output[Artifact],
    metrics_artifact: Output[Metrics],
    trainer_config_json: str,
) -> str:
    """Retrain using the best hyperparameters found during tuning."""
    import json
    import pickle
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    trainer_config = json.loads(trainer_config_json)
    with open(best_params_artifact.path, "r") as f:
        best_params = json.load(f)

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    target_col = trainer_config.get("target_column", train_df.columns[-1])
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values

    framework = trainer_config.get("framework", "sklearn")

    if framework == "sklearn":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**best_params, random_state=42)
    elif framework == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError(f"Retrain not supported for {framework}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")

    metrics_artifact.log_metric("accuracy", acc)
    metrics_artifact.log_metric("f1_score", f1)

    with open(model_artifact.path, "wb") as f:
        pickle.dump(model, f)

    model_artifact.metadata["framework"] = framework
    model_artifact.metadata["best_params"] = best_params

    return json.dumps({"accuracy": acc, "f1_score": f1})

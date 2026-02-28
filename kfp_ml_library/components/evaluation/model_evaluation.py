"""
Model evaluation KFP component.

Produces comprehensive evaluation reports including metrics,
confusion matrix, ROC curves, and model blessing decision.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=14.0.0",
        "matplotlib>=3.8.0",
    ],
)
def model_evaluation_component(
    test_data: Input[Dataset],
    model_artifact: Input[Model],
    eval_report: Output[Artifact],
    eval_metrics: Output[Metrics],
    eval_config_json: str = "{}",
) -> str:
    """
    Evaluate a trained model on the test dataset.

    Returns JSON with all metrics and the blessing decision.
    """
    import json
    import pickle
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    config = json.loads(eval_config_json) if eval_config_json else {}
    task_type = config.get("task_type", "classification")
    target_col = config.get("target_column", None)
    blessing_threshold = config.get("blessing_threshold", 0.80)
    primary_metric = config.get("primary_metric", "accuracy")

    # Load model
    with open(model_artifact.path, "rb") as f:
        model = pickle.load(f)

    # Load data
    df = pd.read_parquet(test_data.path)
    if target_col is None:
        target_col = df.columns[-1]
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    y_pred = model.predict(X_test)
    metrics_dict: dict = {}

    if task_type == "classification":
        metrics_dict["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics_dict["f1_score"] = float(f1_score(y_test, y_pred, average="weighted"))
        metrics_dict["precision"] = float(precision_score(y_test, y_pred, average="weighted"))
        metrics_dict["recall"] = float(recall_score(y_test, y_pred, average="weighted"))

        # ROC AUC (binary only)
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics_dict["auc_roc"] = float(roc_auc_score(y_test, y_prob))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        metrics_dict["confusion_matrix"] = cm

        # Classification report
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        metrics_dict["classification_report"] = cls_report

    else:  # regression
        metrics_dict["mse"] = float(mean_squared_error(y_test, y_pred))
        metrics_dict["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_dict["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics_dict["r2_score"] = float(r2_score(y_test, y_pred))

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        feature_names = list(X_test.columns)
        metrics_dict["feature_importances"] = dict(zip(feature_names, importances))

    # Blessing decision
    primary_value = metrics_dict.get(primary_metric, 0.0)
    is_blessed = primary_value >= blessing_threshold

    report = {
        "metrics": metrics_dict,
        "is_blessed": is_blessed,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "blessing_threshold": blessing_threshold,
        "task_type": task_type,
        "test_samples": len(df),
    }

    with open(eval_report.path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Log KFP metrics
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            eval_metrics.log_metric(k, v)
    eval_metrics.log_metric("is_blessed", int(is_blessed))

    return json.dumps({"is_blessed": is_blessed, "primary_value": primary_value})


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0"],
)
def model_blessing_gate_component(
    eval_report: Input[Artifact],
) -> bool:
    """
    Gate component: reads evaluation report and returns True
    if the model is blessed (meets performance thresholds).
    """
    import json

    with open(eval_report.path, "r") as f:
        report = json.load(f)

    return report.get("is_blessed", False)

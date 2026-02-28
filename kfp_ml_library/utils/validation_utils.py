"""
Validation utilities.

Schema validation, config validation, and runtime assertions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_config(config: Dict[str, Any], required_fields: List[str]) -> None:
    """Raise ``ValidationError`` if any *required_fields* are missing."""
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValidationError(f"Missing required config fields: {missing}")


def validate_dataframe_columns(
    df,
    required_columns: List[str],
    name: str = "DataFrame",
) -> None:
    """Raise ``ValidationError`` if any required columns are absent."""
    actual = set(df.columns)
    missing = set(required_columns) - actual
    if missing:
        raise ValidationError(f"{name} missing columns: {missing}")


def validate_metric_threshold(
    metric_name: str,
    value: float,
    threshold: float,
    comparison: str = "gte",
) -> bool:
    """Check if a metric meets the threshold."""
    ops = {
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
    }
    op = ops.get(comparison, ops["gte"])
    return op(value, threshold)


def validate_framework(framework: str) -> None:
    """Raise if framework is not supported."""
    supported = {"sklearn", "xgboost", "keras", "tensorflow", "pytorch", "automl"}
    if framework not in supported:
        raise ValidationError(
            f"Unsupported framework '{framework}'. Supported: {sorted(supported)}"
        )


def validate_task_type(task_type: str) -> None:
    """Raise if task type is not supported."""
    supported = {
        "classification", "regression", "clustering",
        "time_series", "nlp", "computer_vision", "recommendation",
    }
    if task_type not in supported:
        raise ValidationError(
            f"Unsupported task_type '{task_type}'. Supported: {sorted(supported)}"
        )

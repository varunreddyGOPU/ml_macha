"""
Data validation KFP component.

Validates schemas, checks for nulls, duplicates, outliers,
and generates a data quality report.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "pyarrow>=14.0.0", "scipy>=1.11.0"],
)
def data_validation_component(
    input_dataset: Input[Dataset],
    validation_report: Output[Artifact],
    validation_metrics: Output[Metrics],
    schema_json: str = "{}",
    max_null_fraction: float = 0.1,
    max_duplicate_fraction: float = 0.05,
    outlier_std_threshold: float = 3.0,
) -> str:
    """
    Validate the input dataset and produce a quality report.

    Returns a JSON string with the validation summary.
    """
    import json
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(input_dataset.path)
    schema = json.loads(schema_json) if schema_json else {}

    report: dict = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "issues": [],
        "passed": True,
    }

    # --- Null checks ---
    null_fractions = df.isnull().mean().to_dict()
    for col, frac in null_fractions.items():
        if frac > max_null_fraction:
            report["issues"].append(
                {"type": "high_null_rate", "column": col, "value": round(frac, 4)}
            )
            report["passed"] = False
    report["null_fractions"] = {k: round(v, 4) for k, v in null_fractions.items()}

    # --- Duplicate checks ---
    dup_fraction = df.duplicated().mean()
    report["duplicate_fraction"] = round(float(dup_fraction), 4)
    if dup_fraction > max_duplicate_fraction:
        report["issues"].append(
            {"type": "high_duplicate_rate", "value": round(float(dup_fraction), 4)}
        )
        report["passed"] = False

    # --- Schema checks ---
    if schema:
        expected_cols = set(schema.get("columns", []))
        actual_cols = set(df.columns)
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            report["issues"].append({"type": "missing_columns", "columns": list(missing)})
            report["passed"] = False
        if extra:
            report["issues"].append({"type": "extra_columns", "columns": list(extra)})

    # --- Outlier checks (numeric columns) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary: dict = {}
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            outliers = ((df[col] - mean).abs() > outlier_std_threshold * std).sum()
            outlier_summary[col] = int(outliers)
    report["outlier_counts"] = outlier_summary

    # --- Statistics ---
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "median": round(float(df[col].median()), 4),
        }
    report["numeric_stats"] = stats

    # Write outputs
    with open(validation_report.path, "w") as f:
        json.dump(report, f, indent=2)

    validation_metrics.log_metric("num_rows", report["num_rows"])
    validation_metrics.log_metric("num_columns", report["num_columns"])
    validation_metrics.log_metric("duplicate_fraction", report["duplicate_fraction"])
    validation_metrics.log_metric("num_issues", len(report["issues"]))
    validation_metrics.log_metric("validation_passed", int(report["passed"]))

    return json.dumps(report)

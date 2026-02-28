"""
Model monitor KFP component.

Runs periodic health checks and logs prediction metrics for
a deployed model endpoint.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pyarrow>=14.0.0",
    ],
)
def model_monitoring_component(
    reference_data: Input[Dataset],
    current_data: Input[Dataset],
    monitoring_report: Output[Artifact],
    monitoring_metrics: Output[Metrics],
    monitoring_config_json: str = "{}",
) -> str:
    """
    Monitor model predictions for drift and degradation.

    Compares ``current_data`` against ``reference_data`` using
    configurable statistical tests.
    """
    import json
    import logging
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp, chi2_contingency

    logger = logging.getLogger(__name__)

    config = json.loads(monitoring_config_json) if monitoring_config_json else {}
    drift_method = config.get("drift_method", "psi")
    drift_threshold = config.get("drift_threshold", 0.05)
    features_to_monitor = config.get("features_to_monitor", [])

    ref_df = pd.read_parquet(reference_data.path)
    cur_df = pd.read_parquet(current_data.path)

    if not features_to_monitor:
        features_to_monitor = list(
            ref_df.select_dtypes(include=[np.number]).columns
        )

    results: dict = {
        "feature_drift": {},
        "alerts": [],
        "overall_drift_detected": False,
        "summary": {},
    }

    for feat in features_to_monitor:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue

        ref_vals = ref_df[feat].dropna().values
        cur_vals = cur_df[feat].dropna().values

        drift_score = 0.0

        if drift_method == "ks_test":
            stat, p_value = ks_2samp(ref_vals, cur_vals)
            drift_score = stat
            drift_detected = p_value < drift_threshold
        elif drift_method == "psi":
            # Population Stability Index
            eps = 1e-6
            n_bins = 10
            breakpoints = np.quantile(ref_vals, np.linspace(0, 1, n_bins + 1))
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
            ref_counts = np.histogram(ref_vals, bins=breakpoints)[0] / len(ref_vals) + eps
            cur_counts = np.histogram(cur_vals, bins=breakpoints)[0] / len(cur_vals) + eps
            drift_score = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
            drift_detected = drift_score > drift_threshold
        else:
            stat, p_value = ks_2samp(ref_vals, cur_vals)
            drift_score = stat
            drift_detected = p_value < drift_threshold

        results["feature_drift"][feat] = {
            "drift_score": round(drift_score, 6),
            "drift_detected": drift_detected,
            "method": drift_method,
        }

        if drift_detected:
            results["alerts"].append({
                "type": "data_drift",
                "feature": feat,
                "drift_score": round(drift_score, 6),
                "threshold": drift_threshold,
            })
            results["overall_drift_detected"] = True

    # Summary statistics
    drift_scores = [v["drift_score"] for v in results["feature_drift"].values()]
    results["summary"] = {
        "total_features_monitored": len(features_to_monitor),
        "features_with_drift": sum(1 for v in results["feature_drift"].values() if v["drift_detected"]),
        "max_drift_score": round(max(drift_scores), 6) if drift_scores else 0.0,
        "mean_drift_score": round(float(np.mean(drift_scores)), 6) if drift_scores else 0.0,
        "reference_samples": len(ref_df),
        "current_samples": len(cur_df),
    }

    # Write outputs
    with open(monitoring_report.path, "w") as f:
        json.dump(results, f, indent=2)

    monitoring_metrics.log_metric("features_with_drift", results["summary"]["features_with_drift"])
    monitoring_metrics.log_metric("max_drift_score", results["summary"]["max_drift_score"])
    monitoring_metrics.log_metric("overall_drift_detected", int(results["overall_drift_detected"]))

    return json.dumps(results)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "pyarrow>=14.0.0"],
)
def prediction_logging_component(
    predictions_data: Input[Dataset],
    logging_report: Output[Artifact],
    model_name: str = "",
    endpoint_name: str = "",
) -> str:
    """
    Log prediction statistics: distribution, latency, volume.
    """
    import json
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(predictions_data.path)

    report = {
        "model_name": model_name,
        "endpoint_name": endpoint_name,
        "total_predictions": len(df),
        "prediction_stats": {},
    }

    if "prediction" in df.columns:
        pred_col = df["prediction"]
        report["prediction_stats"] = {
            "unique_values": int(pred_col.nunique()),
            "distribution": pred_col.value_counts().head(20).to_dict()
            if pred_col.dtype == "object" or pred_col.nunique() < 50
            else {
                "mean": round(float(pred_col.mean()), 4),
                "std": round(float(pred_col.std()), 4),
                "min": round(float(pred_col.min()), 4),
                "max": round(float(pred_col.max()), 4),
            },
        }

    if "latency_ms" in df.columns:
        report["latency_stats"] = {
            "p50": round(float(df["latency_ms"].quantile(0.5)), 2),
            "p95": round(float(df["latency_ms"].quantile(0.95)), 2),
            "p99": round(float(df["latency_ms"].quantile(0.99)), 2),
            "mean": round(float(df["latency_ms"].mean()), 2),
        }

    with open(logging_report.path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return json.dumps(report, default=str)

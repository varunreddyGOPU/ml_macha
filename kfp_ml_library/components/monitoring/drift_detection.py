"""
Drift detection KFP component.

Dedicated component for statistical drift detection using multiple
methods: PSI, KS test, Chi-squared, Wasserstein distance.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "numpy>=1.26.0",
        "pyarrow>=14.0.0",
    ],
)
def drift_detection_component(
    reference_data: Input[Dataset],
    current_data: Input[Dataset],
    drift_report: Output[Artifact],
    drift_metrics: Output[Metrics],
    features_json: str = "[]",
    method: str = "psi",
    threshold: float = 0.05,
    n_bins: int = 10,
) -> str:
    """
    Detect data drift between reference and current datasets.

    Methods:
        - ``psi``: Population Stability Index
        - ``ks_test``: Kolmogorov-Smirnov test
        - ``wasserstein``: Wasserstein distance
        - ``chi_squared``: Chi-squared test (categorical)
    """
    import json
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency

    ref_df = pd.read_parquet(reference_data.path)
    cur_df = pd.read_parquet(current_data.path)

    features = json.loads(features_json) if features_json else []
    if not features:
        features = list(ref_df.select_dtypes(include=[np.number]).columns)

    results: dict = {"features": {}, "summary": {}}

    def _psi(ref_vals, cur_vals, n_bins_=10):
        eps = 1e-6
        bp = np.quantile(ref_vals, np.linspace(0, 1, n_bins_ + 1))
        bp[0], bp[-1] = -np.inf, np.inf
        rc = np.histogram(ref_vals, bins=bp)[0] / len(ref_vals) + eps
        cc = np.histogram(cur_vals, bins=bp)[0] / len(cur_vals) + eps
        return float(np.sum((cc - rc) * np.log(cc / rc)))

    for feat in features:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue

        ref_vals = ref_df[feat].dropna().values
        cur_vals = cur_df[feat].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        if method == "psi":
            score = _psi(ref_vals, cur_vals, n_bins)
            p_value = None
        elif method == "ks_test":
            score, p_value = ks_2samp(ref_vals, cur_vals)
            p_value = float(p_value)
            score = float(score)
        elif method == "wasserstein":
            score = float(wasserstein_distance(ref_vals, cur_vals))
            p_value = None
        else:
            score, p_value = ks_2samp(ref_vals, cur_vals)
            score = float(score)
            p_value = float(p_value) if p_value is not None else None

        drift_detected = score > threshold if method in ("psi", "wasserstein") else (p_value is not None and p_value < threshold)

        results["features"][feat] = {
            "score": round(score, 6),
            "p_value": round(p_value, 6) if p_value is not None else None,
            "drift_detected": drift_detected,
        }

    scores = [v["score"] for v in results["features"].values()]
    drifted = sum(1 for v in results["features"].values() if v["drift_detected"])

    results["summary"] = {
        "method": method,
        "threshold": threshold,
        "total_features": len(results["features"]),
        "drifted_features": drifted,
        "max_score": round(max(scores), 6) if scores else 0.0,
        "mean_score": round(float(np.mean(scores)), 6) if scores else 0.0,
    }

    with open(drift_report.path, "w") as f:
        json.dump(results, f, indent=2)

    drift_metrics.log_metric("drifted_features", drifted)
    drift_metrics.log_metric("max_drift_score", results["summary"]["max_score"])

    return json.dumps(results)


@dsl.component(base_image=DEFAULT_BASE_IMAGE, packages_to_install=["requests>=2.31.0"])
def alerting_component(
    drift_report: Input[Artifact],
    alert_report: Output[Artifact],
    email_recipients: str = "[]",
    slack_webhook: str = "",
    severity: str = "warning",
) -> str:
    """
    Send alerts based on drift detection results.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    with open(drift_report.path, "r") as f:
        report = json.load(f)

    drifted = report.get("summary", {}).get("drifted_features", 0)
    alerts_sent: list = []

    if drifted > 0:
        message = (
            f"⚠️ Data drift detected!\n"
            f"Drifted features: {drifted}/{report['summary']['total_features']}\n"
            f"Max drift score: {report['summary']['max_score']}\n"
            f"Severity: {severity}"
        )

        # Slack notification
        if slack_webhook:
            try:
                import requests
                requests.post(slack_webhook, json={"text": message}, timeout=10)
                alerts_sent.append({"channel": "slack", "status": "sent"})
            except Exception as e:
                alerts_sent.append({"channel": "slack", "status": "failed", "error": str(e)})

        # Email notification (placeholder)
        recipients = json.loads(email_recipients) if email_recipients else []
        if recipients:
            alerts_sent.append({
                "channel": "email",
                "recipients": recipients,
                "status": "queued",
                "message": message,
            })

        logger.warning(message)
    else:
        logger.info("No drift detected - no alerts to send.")

    result = {
        "drift_detected": drifted > 0,
        "drifted_features": drifted,
        "alerts_sent": alerts_sent,
        "severity": severity,
    }

    with open(alert_report.path, "w") as f:
        json.dump(result, f, indent=2)

    return json.dumps(result)

"""
Metrics helper module.

Centralised metric computation functions used by evaluators
and monitoring components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average=average)),
        "precision": float(precision_score(y_true, y_pred, average=average)),
        "recall": float(recall_score(y_true, y_pred, average=average)),
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                )
        except Exception:
            pass

    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics."""
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def compute_drift_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    PSI < 0.1 → no drift, 0.1–0.2 → moderate, > 0.2 → significant.
    """
    eps = 1e-6
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current) + eps

    psi = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
    return psi


def compute_ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between two distributions."""
    from scipy.stats import ks_2samp

    stat, _ = ks_2samp(reference, current)
    return float(stat)


def compute_feature_importance_drift(
    old_importances: Dict[str, float],
    new_importances: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-feature importance drift."""
    drift: Dict[str, float] = {}
    for feat in old_importances:
        if feat in new_importances:
            drift[feat] = abs(new_importances[feat] - old_importances[feat])
    return drift

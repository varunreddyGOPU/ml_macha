"""
Monitoring configuration module.

Defines drift detection, alerting, and observability configurations
for deployed model monitoring.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kfp_ml_library.configs.constants import (
    ALERT_EMAIL,
    DRIFT_THRESHOLD,
    MONITORING_INTERVAL_SECONDS,
    MONITORING_WINDOW_SIZE,
    MonitoringMetricType,
)


@dataclass
class DriftDetectionConfig:
    """Configuration for a drift detection check."""

    method: str = "psi"  # psi | ks_test | chi_squared | wasserstein | js_divergence
    threshold: float = DRIFT_THRESHOLD
    features_to_monitor: List[str] = field(default_factory=list)
    reference_dataset_path: Optional[str] = None
    window_size: int = MONITORING_WINDOW_SIZE
    min_samples: int = 100


@dataclass
class AlertConfig:
    """Alerting configuration for monitoring triggers."""

    email_recipients: List[str] = field(default_factory=lambda: [ALERT_EMAIL])
    slack_webhook: Optional[str] = None
    pagerduty_key: Optional[str] = None
    severity_levels: Dict[str, str] = field(
        default_factory=lambda: {
            "data_drift": "warning",
            "concept_drift": "critical",
            "latency_spike": "warning",
            "error_rate_high": "critical",
            "model_degradation": "critical",
        }
    )
    cooldown_seconds: int = 3600
    auto_rollback_on_critical: bool = False


@dataclass
class LatencyConfig:
    """Latency monitoring configuration."""

    p50_threshold_ms: float = 100.0
    p95_threshold_ms: float = 500.0
    p99_threshold_ms: float = 1000.0
    sampling_rate: float = 1.0  # 0..1


@dataclass
class MonitoringConfig:
    """
    Full monitoring configuration for a deployed model.

    Example::

        config = MonitoringConfig(
            model_name="fraud_detector_v2",
            drift_config=DriftDetectionConfig(
                method="ks_test",
                threshold=0.05,
                features_to_monitor=["amount", "merchant_category"],
            ),
            alert_config=AlertConfig(
                email_recipients=["team@company.com"],
                auto_rollback_on_critical=True,
            ),
        )
    """

    model_name: str = ""
    endpoint_name: str = ""
    monitoring_interval_seconds: int = MONITORING_INTERVAL_SECONDS
    metric_types: List[MonitoringMetricType] = field(
        default_factory=lambda: [
            MonitoringMetricType.DATA_DRIFT,
            MonitoringMetricType.PREDICTION_DRIFT,
            MonitoringMetricType.LATENCY,
            MonitoringMetricType.ERROR_RATE,
        ]
    )
    drift_config: DriftDetectionConfig = field(default_factory=DriftDetectionConfig)
    alert_config: AlertConfig = field(default_factory=AlertConfig)
    latency_config: LatencyConfig = field(default_factory=LatencyConfig)
    log_predictions: bool = True
    log_features: bool = True
    max_log_retention_days: int = 90
    dashboard_enabled: bool = True
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "endpoint_name": self.endpoint_name,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "metric_types": [m.value for m in self.metric_types],
            "drift_config": {
                "method": self.drift_config.method,
                "threshold": self.drift_config.threshold,
                "features_to_monitor": self.drift_config.features_to_monitor,
                "window_size": self.drift_config.window_size,
            },
            "log_predictions": self.log_predictions,
            "log_features": self.log_features,
            "dashboard_enabled": self.dashboard_enabled,
        }

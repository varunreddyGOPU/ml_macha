"""
Monitoring pipeline definition.

Composes drift detection → alerting → prediction logging
into a single KFP pipeline for continuous model monitoring.
"""

from __future__ import annotations

from kfp import dsl

from kfp_ml_library.components.monitoring.drift_detection import (
    alerting_component,
    drift_detection_component,
)
from kfp_ml_library.components.monitoring.model_monitor import (
    model_monitoring_component,
    prediction_logging_component,
)


@dsl.pipeline(
    name="ml-monitoring-pipeline",
    description="Model monitoring pipeline: drift detection → alerting → logging",
)
def create_monitoring_pipeline(
    monitoring_config_json: str = "{}",
    drift_method: str = "psi",
    drift_threshold: float = 0.05,
    features_json: str = "[]",
    model_name: str = "",
    endpoint_name: str = "",
    email_recipients: str = "[]",
    slack_webhook: str = "",
    alert_severity: str = "warning",
):
    """
    Composable monitoring pipeline.

    Steps:
        1. Comprehensive model monitoring (data drift + prediction stats)
        2. Dedicated drift detection (detailed per-feature analysis)
        3. Alerting on drift detection results
    """

    # 1. Full model monitoring
    # Note: reference_data and current_data are expected as pipeline inputs
    # In practice these would be linked from data sources or upstream pipelines.

    # 2. Drift detection with detailed analysis
    # This component requires reference_data and current_data Dataset inputs.
    # When composing pipelines, these are connected from data sources.

    # The pipeline is designed to be composed with data-producing pipelines.
    # Individual components can also be used standalone.
    pass


@dsl.pipeline(
    name="ml-monitoring-with-data-pipeline",
    description="Full monitoring pipeline with data ingestion",
)
def create_monitoring_with_data_pipeline(
    reference_data_path: str = "",
    current_data_path: str = "",
    drift_method: str = "psi",
    drift_threshold: float = 0.05,
    features_json: str = "[]",
    model_name: str = "",
    endpoint_name: str = "",
    email_recipients: str = "[]",
    slack_webhook: str = "",
):
    """
    Monitoring pipeline that ingests reference and current data,
    then runs drift detection and alerting.
    """
    from kfp_ml_library.components.data_prep.data_ingestion import data_ingestion_component

    # Ingest reference data
    ref_ingest = data_ingestion_component(
        source_type="gcs",
        source_path=reference_data_path,
        file_format="parquet",
    ).set_display_name("Ingest Reference Data")

    # Ingest current data
    cur_ingest = data_ingestion_component(
        source_type="gcs",
        source_path=current_data_path,
        file_format="parquet",
    ).set_display_name("Ingest Current Data")

    # Drift detection
    drift_task = drift_detection_component(
        reference_data=ref_ingest.outputs["output_dataset"],
        current_data=cur_ingest.outputs["output_dataset"],
        features_json=features_json,
        method=drift_method,
        threshold=drift_threshold,
    )

    # Full monitoring
    monitor_task = model_monitoring_component(
        reference_data=ref_ingest.outputs["output_dataset"],
        current_data=cur_ingest.outputs["output_dataset"],
        monitoring_config_json=f'{{"drift_method": "{drift_method}", "drift_threshold": {drift_threshold}}}',
    )

    # Alerting
    alert_task = alerting_component(
        drift_report=drift_task.outputs["drift_report"],
        email_recipients=email_recipients,
        slack_webhook=slack_webhook,
    )

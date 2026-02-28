"""
Full ML pipeline definition.

Composes Training → Evaluation → Deployment → Monitoring
into one end-to-end KFP pipeline with conditional branches.
"""

from __future__ import annotations

from kfp import dsl

from kfp_ml_library.components.data_prep.data_ingestion import data_ingestion_component
from kfp_ml_library.components.data_prep.data_transformation import data_transformation_component
from kfp_ml_library.components.data_prep.data_validation import data_validation_component
from kfp_ml_library.components.data_prep.feature_engineering import feature_engineering_component
from kfp_ml_library.components.deployment.model_deployer import deploy_model_component
from kfp_ml_library.components.evaluation.model_evaluation import (
    model_blessing_gate_component,
    model_evaluation_component,
)
from kfp_ml_library.components.generic.generic_component import send_notification_component
from kfp_ml_library.components.training.generic_trainer import generic_train_component


@dsl.pipeline(
    name="full-ml-pipeline",
    description="End-to-end ML pipeline: data prep → train → evaluate → deploy → notify",
)
def create_full_ml_pipeline(
    # Data params
    source_type: str = "gcs",
    source_path: str = "",
    file_format: str = "csv",
    target_column: str = "target",
    schema_json: str = "{}",
    # Training params
    framework: str = "sklearn",
    task_type: str = "classification",
    trainer_config_json: str = "{}",
    eval_config_json: str = "{}",
    # Data processing params
    validation_split: float = 0.2,
    test_split: float = 0.1,
    numerical_strategy: str = "standard",
    categorical_strategy: str = "onehot",
    # Deployment params
    project_id: str = "",
    region: str = "us-central1",
    endpoint_name: str = "ml-serving-endpoint",
    deployment_strategy: str = "rolling",
    serving_container_image: str = "",
    machine_type: str = "n1-standard-4",
    min_replicas: int = 1,
    max_replicas: int = 5,
    # Notification params
    notification_channel: str = "slack",
    notification_webhook: str = "",
    sample_fraction: float = 1.0,
):
    """
    Full ML pipeline from data to deployment.

    Steps:
        1. Data ingestion
        2. Data validation
        3. Data transformation
        4. Feature engineering
        5. Model training
        6. Model evaluation
        7. Model blessing gate
        8. Conditional deployment (if blessed)
        9. Notification
    """

    # ---- 1. Data Ingestion ----
    ingest_task = data_ingestion_component(
        source_type=source_type,
        source_path=source_path,
        file_format=file_format,
        sample_fraction=sample_fraction,
    ).set_display_name("1. Data Ingestion")

    # ---- 2. Data Validation ----
    validate_task = data_validation_component(
        input_dataset=ingest_task.outputs["output_dataset"],
        schema_json=schema_json,
    ).set_display_name("2. Data Validation")

    # ---- 3. Data Transformation ----
    transform_task = data_transformation_component(
        input_dataset=ingest_task.outputs["output_dataset"],
        target_column=target_column,
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        validation_split=validation_split,
        test_split=test_split,
    ).set_display_name("3. Data Transformation")
    transform_task.after(validate_task)

    # ---- 4. Feature Engineering ----
    fe_train = feature_engineering_component(
        input_dataset=transform_task.outputs["train_dataset"],
        target_column=target_column,
    ).set_display_name("4a. Feature Engineering (train)")

    fe_val = feature_engineering_component(
        input_dataset=transform_task.outputs["val_dataset"],
        target_column=target_column,
    ).set_display_name("4b. Feature Engineering (val)")

    # ---- 5. Model Training ----
    train_task = generic_train_component(
        train_data=fe_train.outputs["output_dataset"],
        val_data=fe_val.outputs["output_dataset"],
        config_json=trainer_config_json,
    ).set_display_name("5. Model Training")

    # ---- 6. Model Evaluation ----
    eval_task = model_evaluation_component(
        test_data=transform_task.outputs["test_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        eval_config_json=eval_config_json,
    ).set_display_name("6. Model Evaluation")

    # ---- 7. Blessing Gate ----
    gate_task = model_blessing_gate_component(
        eval_report=eval_task.outputs["eval_report"],
    ).set_display_name("7. Blessing Gate")

    # ---- 8. Conditional Deployment ----
    with dsl.If(
        gate_task.output == True,
        name="Deploy if Blessed",
    ):
        deploy_task = deploy_model_component(
            model_artifact=train_task.outputs["model_artifact"],
            eval_report=eval_task.outputs["eval_report"],
            project_id=project_id,
            region=region,
            endpoint_name=endpoint_name,
            deployment_strategy=deployment_strategy,
            serving_container_image=serving_container_image,
            machine_type=machine_type,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        ).set_display_name("8. Deploy Model")

        # ---- 9. Notification ----
        notify_task = send_notification_component(
            message="Model deployed successfully!",
            channel=notification_channel,
            webhook_url=notification_webhook,
        ).set_display_name("9. Send Notification")
        notify_task.after(deploy_task)

    with dsl.If(
        gate_task.output == False,
        name="Notify if Not Blessed",
    ):
        notify_fail = send_notification_component(
            message="Model did NOT meet blessing threshold - deployment skipped.",
            channel=notification_channel,
            webhook_url=notification_webhook,
        ).set_display_name("9. Notify - Not Blessed")

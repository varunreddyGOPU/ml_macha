"""
Training pipeline definition.

Composes data ingestion → validation → transformation → feature engineering
→ training → hyperparameter tuning → evaluation into a single KFP pipeline.
"""

from __future__ import annotations

from kfp import dsl

from kfp_ml_library.components.data_prep.data_ingestion import data_ingestion_component
from kfp_ml_library.components.data_prep.data_transformation import data_transformation_component
from kfp_ml_library.components.data_prep.data_validation import data_validation_component
from kfp_ml_library.components.data_prep.feature_engineering import feature_engineering_component
from kfp_ml_library.components.evaluation.model_evaluation import model_evaluation_component
from kfp_ml_library.components.training.generic_trainer import generic_train_component
from kfp_ml_library.components.training.hyperparameter_tuning import (
    hyperparameter_tuning_component,
    retrain_with_best_params_component,
)


@dsl.pipeline(
    name="ml-training-pipeline",
    description="End-to-end training pipeline: ingest → validate → transform → train → evaluate",
)
def create_training_pipeline(
    source_type: str = "gcs",
    source_path: str = "",
    file_format: str = "csv",
    target_column: str = "target",
    framework: str = "sklearn",
    task_type: str = "classification",
    trainer_config_json: str = "{}",
    hp_config_json: str = "{}",
    eval_config_json: str = "{}",
    schema_json: str = "{}",
    enable_hyperparameter_tuning: bool = False,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    numerical_strategy: str = "standard",
    categorical_strategy: str = "onehot",
    sample_fraction: float = 1.0,
):
    """
    Composable training pipeline.

    Steps:
        1. Data ingestion
        2. Data validation
        3. Data transformation (split + scale + encode)
        4. Feature engineering
        5. Model training (optionally preceded by HP tuning)
        6. Model evaluation + blessing decision
    """

    # 1. Ingest
    ingest_task = data_ingestion_component(
        source_type=source_type,
        source_path=source_path,
        file_format=file_format,
        sample_fraction=sample_fraction,
    )

    # 2. Validate
    validate_task = data_validation_component(
        input_dataset=ingest_task.outputs["output_dataset"],
        schema_json=schema_json,
    )

    # 3. Transform
    transform_task = data_transformation_component(
        input_dataset=ingest_task.outputs["output_dataset"],
        target_column=target_column,
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        validation_split=validation_split,
        test_split=test_split,
    )
    transform_task.after(validate_task)

    # 4. Feature engineering
    fe_train = feature_engineering_component(
        input_dataset=transform_task.outputs["train_dataset"],
        target_column=target_column,
    )
    fe_val = feature_engineering_component(
        input_dataset=transform_task.outputs["val_dataset"],
        target_column=target_column,
    ).set_display_name("Feature Engineering (val)")

    # 5. Training
    with dsl.If(enable_hyperparameter_tuning == True, name="HP Tuning Branch"):
        hp_task = hyperparameter_tuning_component(
            train_data=fe_train.outputs["output_dataset"],
            val_data=fe_val.outputs["output_dataset"],
            trainer_config_json=trainer_config_json,
            hp_config_json=hp_config_json,
        )
        retrain_task = retrain_with_best_params_component(
            train_data=fe_train.outputs["output_dataset"],
            val_data=fe_val.outputs["output_dataset"],
            best_params_artifact=hp_task.outputs["best_params_artifact"],
            trainer_config_json=trainer_config_json,
        )

    with dsl.If(enable_hyperparameter_tuning == False, name="Direct Training Branch"):
        train_task = generic_train_component(
            train_data=fe_train.outputs["output_dataset"],
            val_data=fe_val.outputs["output_dataset"],
            config_json=trainer_config_json,
        )

        # 6. Evaluate
        eval_task = model_evaluation_component(
            test_data=transform_task.outputs["test_dataset"],
            model_artifact=train_task.outputs["model_artifact"],
            eval_config_json=eval_config_json,
        )

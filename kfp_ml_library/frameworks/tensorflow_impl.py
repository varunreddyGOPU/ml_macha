"""
TensorFlow framework implementation.

Provides ``TensorFlowTrainer`` (extends ``TrainerBase``) and a standalone
KFP component for distributed TensorFlow training.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from kfp_ml_library.components.training.trainer_base import TrainerBase
from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, TaskType
from kfp_ml_library.configs.trainer_config import TensorFlowTrainerConfig

logger = logging.getLogger(__name__)


class TensorFlowTrainer(TrainerBase):
    """TensorFlow (low-level / custom) trainer implementation."""

    def __init__(self, config: TensorFlowTrainerConfig) -> None:
        super().__init__(config)
        self.config: TensorFlowTrainerConfig = config
        self._strategy = None

    def _get_strategy(self):
        import tensorflow as tf

        strategies = {
            "mirrored": tf.distribute.MirroredStrategy,
            "multi_worker": tf.distribute.experimental.MultiWorkerMirroredStrategy,
        }
        strategy_cls = strategies.get(self.config.strategy)
        if strategy_cls and self.config.distribute:
            return strategy_cls()
        return tf.distribute.get_strategy()  # default (no-op)

    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        import tensorflow as tf

        self._strategy = self._get_strategy()

        if self.config.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        num_classes = kwargs.get("num_classes", 2)

        with self._strategy.scope():
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Dense(256, activation="relu")(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(64, activation="relu")(x)

            if self.config.task_type in (TaskType.CLASSIFICATION, "classification"):
                if num_classes == 2:
                    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
                    loss = "binary_crossentropy"
                else:
                    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
                    loss = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
            else:
                outputs = tf.keras.layers.Dense(1)(x)
                loss = "mse"
                metrics = ["mae"]

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            if self.config.xla_compilation:
                tf.config.optimizer.set_jit(True)

            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        return self.model

    def _train(
        self, X_train, y_train, X_val=None, y_val=None, **kwargs
    ) -> Dict[str, Any]:
        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        if self.config.tf_function:
            # Wrap in tf.data for better performance
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_ds = train_ds.shuffle(10000).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

            val_ds = None
            if validation_data:
                val_ds = tf.data.Dataset.from_tensor_slices(validation_data)
                val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=self.config.verbose,
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=self.config.verbose,
            )

        return {k: [float(v) for v in vals] for k, vals in history.history.items()}

    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        results = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        return {k: float(v) for k, v in results.items()}

    def _save_model(self, output_path: str) -> str:
        os.makedirs(output_path, exist_ok=True)
        if self.config.save_format == "saved_model":
            self.model.export(output_path)
        elif self.config.save_format == "h5":
            save_path = os.path.join(output_path, "model.h5")
            self.model.save(save_path)
        elif self.config.save_format == "tflite":
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            save_path = os.path.join(output_path, "model.tflite")
            with open(save_path, "wb") as f:
                f.write(tflite_model)
        else:
            self.model.save(output_path)
        return output_path

    def _load_model(self, model_path: str) -> Any:
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        return self.model


# ---------------------------------------------------------------------------
# Standalone KFP component
# ---------------------------------------------------------------------------
@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "tensorflow>=2.15.0",
        "pandas>=2.1.0",
        "pyarrow>=14.0.0",
    ],
)
def tensorflow_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    task_type: str = "classification",
    target_column: str = "",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_mixed_precision: bool = False,
) -> str:
    """Standalone TensorFlow training component."""
    import json
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if not target_column:
        target_column = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_column]).values.astype("float32")
    y_train = train_df[target_column].values
    X_val = val_df.drop(columns=[target_column]).values.astype("float32")
    y_val = val_df[target_column].values

    input_dim = X_train.shape[1]

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    if task_type == "classification":
        num_classes = len(np.unique(y_train))
        if num_classes == 2:
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
        else:
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
            loss = "sparse_categorical_crossentropy"
        metrics_ = ["accuracy"]
    else:
        outputs = tf.keras.layers.Dense(1)(x)
        loss = "mse"
        metrics_ = ["mae"]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics_,
    )

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
        verbose=1,
    )

    eval_results = model.evaluate(val_ds, verbose=0, return_dict=True)
    eval_metrics = {k: float(v) for k, v in eval_results.items()}

    for k, v in eval_metrics.items():
        metrics_artifact.log_metric(k, v)

    model.save(model_artifact.path)
    model_artifact.metadata["framework"] = "tensorflow"

    return json.dumps(eval_metrics)

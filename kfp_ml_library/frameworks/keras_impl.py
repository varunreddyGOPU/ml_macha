"""
Keras framework implementation.

Provides ``KerasTrainer`` (extends ``TrainerBase``) and a standalone
KFP component for Keras model training.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from kfp_ml_library.components.training.trainer_base import TrainerBase
from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, TaskType
from kfp_ml_library.configs.trainer_config import KerasTrainerConfig

logger = logging.getLogger(__name__)


class KerasTrainer(TrainerBase):
    """Keras / TensorFlow-Keras trainer implementation."""

    def __init__(self, config: KerasTrainerConfig) -> None:
        super().__init__(config)
        self.config: KerasTrainerConfig = config

    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        num_classes = kwargs.get("num_classes", 2)

        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))

        for units in self.config.hidden_layers:
            model.add(layers.Dense(units, activation=self.config.activation))
            if self.config.use_batch_norm:
                model.add(layers.BatchNormalization())
            if self.config.dropout_rate > 0:
                model.add(layers.Dropout(self.config.dropout_rate))

        # Output layer
        if self.config.task_type in (TaskType.CLASSIFICATION, "classification"):
            if num_classes == 2:
                model.add(layers.Dense(1, activation="sigmoid"))
            else:
                model.add(layers.Dense(num_classes, activation=self.config.output_activation))
        else:
            model.add(layers.Dense(1, activation="linear"))

        # Compile
        loss = self.config.loss
        if self.config.task_type in (TaskType.REGRESSION, "regression"):
            loss = "mse"
        elif num_classes == 2:
            loss = "binary_crossentropy"

        model.compile(
            optimizer=self.config.optimizer,
            loss=loss,
            metrics=self.config.metrics,
        )

        self.model = model
        logger.info("Built Keras model:\n%s", model.summary())
        return self.model

    def _get_callbacks(self) -> list:
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
        )

        callbacks = []
        callback_map = {
            "early_stopping": EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            "reduce_lr": ReduceLROnPlateau(
                factor=0.5,
                patience=self.config.early_stopping_patience // 2,
                min_lr=1e-7,
            ),
            "model_checkpoint": ModelCheckpoint(
                filepath=os.path.join(self.config.output_model_path, "best_model.keras"),
                save_best_only=True,
                monitor="val_loss",
            ),
        }
        for cb_name in self.config.callbacks:
            if cb_name in callback_map:
                callbacks.append(callback_map[cb_name])

        return callbacks

    def _train(
        self, X_train, y_train, X_val=None, y_val=None, **kwargs
    ) -> Dict[str, Any]:
        callbacks = self._get_callbacks()
        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )

        return {k: [float(v) for v in vals] for k, vals in history.history.items()}

    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        results = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        return {k: float(v) for k, v in results.items()}

    def _save_model(self, output_path: str) -> str:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, "model.keras")
        self.model.save(save_path)
        logger.info("Keras model saved to %s", save_path)
        return save_path

    def _load_model(self, model_path: str) -> Any:
        from tensorflow import keras

        self.model = keras.models.load_model(model_path)
        return self.model


# ---------------------------------------------------------------------------
# Standalone KFP component
# ---------------------------------------------------------------------------
@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "tensorflow>=2.15.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=14.0.0",
    ],
)
def keras_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    task_type: str = "classification",
    target_column: str = "",
    hidden_layers_json: str = "[128, 64, 32]",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2,
    early_stopping_patience: int = 10,
) -> str:
    """Standalone Keras training component."""
    import json
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if not target_column:
        target_column = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_column]).values.astype("float32")
    y_train = train_df[target_column].values
    X_val = val_df.drop(columns=[target_column]).values.astype("float32")
    y_val = val_df[target_column].values

    hidden = json.loads(hidden_layers_json)
    input_dim = X_train.shape[1]

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    if task_type == "classification":
        num_classes = len(np.unique(y_train))
        if num_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            from tensorflow.keras.utils import to_categorical
            y_train = to_categorical(y_train, num_classes)
            y_val = to_categorical(y_val, num_classes)
            model.add(layers.Dense(num_classes, activation="softmax"))
            loss = "categorical_crossentropy"
    else:
        model.add(layers.Dense(1, activation="linear"))
        loss = "mse"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"] if task_type == "classification" else ["mae"],
    )

    callbacks = [
        EarlyStopping(patience=early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=early_stopping_patience // 2),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    eval_results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    eval_metrics = {k: float(v) for k, v in eval_results.items()}

    for k, v in eval_metrics.items():
        metrics_artifact.log_metric(k, v)

    model.save(model_artifact.path)
    model_artifact.metadata["framework"] = "keras"

    return json.dumps(eval_metrics)

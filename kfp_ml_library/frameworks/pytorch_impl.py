"""
PyTorch framework implementation.

Provides ``PyTorchTrainer`` (extends ``TrainerBase``) and a standalone
KFP component for PyTorch model training.
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
from kfp_ml_library.configs.trainer_config import PyTorchTrainerConfig

logger = logging.getLogger(__name__)


class _SimpleNet:
    """Placeholder for the PyTorch nn.Module created at runtime."""
    pass


class PyTorchTrainer(TrainerBase):
    """PyTorch trainer implementation."""

    def __init__(self, config: PyTorchTrainerConfig) -> None:
        super().__init__(config)
        self.config: PyTorchTrainerConfig = config
        self.device = None

    def _build_model(self, input_shape: Optional[Tuple] = None, **kwargs) -> Any:
        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = kwargs.get("num_classes", 2)
        input_dim = input_shape[0] if input_shape else kwargs.get("input_dim", 10)

        class Net(nn.Module):
            def __init__(self, in_features, n_classes, task):
                super().__init__()
                self.task = task
                self.network = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                )
                if task in ("classification",):
                    self.head = nn.Linear(64, n_classes)
                else:
                    self.head = nn.Linear(64, 1)

            def forward(self, x):
                x = self.network(x)
                return self.head(x)

        task_str = self.config.task_type.value if isinstance(self.config.task_type, TaskType) else self.config.task_type
        self.model = Net(input_dim, num_classes, task_str).to(self.device)
        logger.info("Built PyTorch model on device=%s", self.device)
        return self.model

    def _get_optimizer(self):
        import torch.optim as optim

        optimizers = {
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
        }
        opt_cls = optimizers.get(self.config.optimizer, optim.AdamW)
        return opt_cls(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _get_scheduler(self, optimizer, steps_per_epoch: int):
        import torch.optim.lr_scheduler as lr_scheduler

        schedulers = {
            "cosine": lambda: lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            ),
            "step": lambda: lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
            "exponential": lambda: lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            "linear": lambda: lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.config.epochs
            ),
        }
        factory = schedulers.get(self.config.scheduler)
        return factory() if factory else None

    def _train(
        self, X_train, y_train, X_val=None, y_val=None, **kwargs
    ) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np

        task_str = self.config.task_type.value if isinstance(self.config.task_type, TaskType) else self.config.task_type

        X_train_t = torch.FloatTensor(np.array(X_train)).to(self.device)
        if task_str == "classification":
            y_train_t = torch.LongTensor(np.array(y_train)).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_train_t = torch.FloatTensor(np.array(y_train)).unsqueeze(1).to(self.device)
            criterion = nn.MSELoss()

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer, len(train_loader))
        scaler = torch.amp.GradScaler("cuda") if self.config.mixed_precision and self.device.type == "cuda" else None

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_value
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_value
                        )
                    optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            if scheduler:
                scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(np.array(X_val)).to(self.device)
                    if task_str == "classification":
                        y_val_t = torch.LongTensor(np.array(y_val)).to(self.device)
                    else:
                        y_val_t = torch.FloatTensor(np.array(y_val)).unsqueeze(1).to(self.device)
                    val_out = self.model(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        return history

    def _evaluate(self, X_test, y_test, **kwargs) -> Dict[str, float]:
        import torch
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

        task_str = self.config.task_type.value if isinstance(self.config.task_type, TaskType) else self.config.task_type

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X_test)).to(self.device)
            outputs = self.model(X_t).cpu().numpy()

        if task_str == "classification":
            y_pred = outputs.argmax(axis=1)
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            y_pred = outputs.squeeze()
            mse = float(mean_squared_error(y_test, y_pred))
            return {
                "mse": mse,
                "rmse": mse**0.5,
                "r2_score": float(r2_score(y_test, y_pred)),
            }

    def _save_model(self, output_path: str) -> str:
        import torch

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        logger.info("PyTorch model saved to %s", output_path)
        return output_path

    def _load_model(self, model_path: str) -> Any:
        import torch

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        return self.model


# ---------------------------------------------------------------------------
# Standalone KFP component
# ---------------------------------------------------------------------------
@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=[
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "pyarrow>=14.0.0",
    ],
)
def pytorch_train_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    task_type: str = "classification",
    target_column: str = "",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dims_json: str = "[256, 128, 64]",
) -> str:
    """Standalone PyTorch training component."""
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(val_data.path)

    if not target_column:
        target_column = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_column]).values.astype("float32")
    y_train = train_df[target_column].values
    X_val = val_df.drop(columns=[target_column]).values.astype("float32")
    y_val = val_df[target_column].values

    input_dim = X_train.shape[1]
    hidden_dims = json.loads(hidden_dims_json)
    num_classes = len(np.unique(y_train)) if task_type == "classification" else 1

    # Build model
    layers_list = []
    prev = input_dim
    for h in hidden_dims:
        layers_list.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.2)])
        prev = h

    if task_type == "classification":
        layers_list.append(nn.Linear(prev, num_classes))
        criterion = nn.CrossEntropyLoss()
        y_train_t = torch.LongTensor(y_train).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
    else:
        layers_list.append(nn.Linear(prev, 1))
        criterion = nn.MSELoss()
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    model = nn.Sequential(*layers_list).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_artifact.path)
        else:
            counter += 1
            if counter >= patience:
                break

    # Final evaluation
    model.load_state_dict(torch.load(model_artifact.path))
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()

    eval_metrics: dict = {}
    if task_type == "classification":
        y_pred = preds.argmax(axis=1)
        eval_metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
        eval_metrics["f1_score"] = float(f1_score(y_val, y_pred, average="weighted"))
    else:
        y_pred = preds.squeeze()
        eval_metrics["mse"] = float(mean_squared_error(y_val, y_pred))
        eval_metrics["r2_score"] = float(r2_score(y_val, y_pred))

    for k, v in eval_metrics.items():
        metrics_artifact.log_metric(k, v)

    model_artifact.metadata["framework"] = "pytorch"
    return json.dumps(eval_metrics)

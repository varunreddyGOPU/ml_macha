"""
Generic reusable KFP components.

Utility components for notifications, data movement, conditional
logic, and pipeline orchestration helpers.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(base_image=DEFAULT_BASE_IMAGE)
def echo_component(message: str) -> str:
    """Simple echo component for debugging / logging."""
    print(message)
    return message


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["requests>=2.31.0"],
)
def send_notification_component(
    message: str,
    channel: str = "slack",
    webhook_url: str = "",
    email_recipients: str = "[]",
) -> str:
    """
    Send a notification via Slack or email.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)
    result: dict = {"status": "pending"}

    if channel == "slack" and webhook_url:
        try:
            import requests

            resp = requests.post(
                webhook_url,
                json={"text": message},
                timeout=10,
            )
            result = {"status": "sent", "channel": "slack", "response_code": resp.status_code}
        except Exception as e:
            result = {"status": "failed", "channel": "slack", "error": str(e)}

    elif channel == "email":
        recipients = json.loads(email_recipients) if email_recipients else []
        result = {
            "status": "queued",
            "channel": "email",
            "recipients": recipients,
            "message": message,
        }
    else:
        result = {"status": "no_channel_configured"}

    logger.info("Notification result: %s", result)
    return json.dumps(result)


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "pyarrow>=14.0.0"],
)
def copy_dataset_component(
    source_dataset: Input[Dataset],
    target_dataset: Output[Dataset],
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> str:
    """Copy / sample a dataset artifact."""
    import json
    import pandas as pd

    df = pd.read_parquet(source_dataset.path)
    if 0.0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state)

    df.to_parquet(target_dataset.path, index=False)
    return json.dumps({"rows": len(df), "columns": len(df.columns)})


@dsl.component(base_image=DEFAULT_BASE_IMAGE)
def conditional_gate_component(
    condition_json: str,
) -> bool:
    """
    Generic conditional gate.

    Expects a JSON object with ``"result": true/false``.
    """
    import json

    data = json.loads(condition_json)
    return bool(data.get("result", data.get("is_blessed", False)))


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "pyarrow>=14.0.0"],
)
def merge_datasets_component(
    dataset_a: Input[Dataset],
    dataset_b: Input[Dataset],
    merged_dataset: Output[Dataset],
    merge_strategy: str = "concat",
    merge_key: str = "",
) -> str:
    """
    Merge two datasets using concat or join.
    """
    import json
    import pandas as pd

    df_a = pd.read_parquet(dataset_a.path)
    df_b = pd.read_parquet(dataset_b.path)

    if merge_strategy == "concat":
        result_df = pd.concat([df_a, df_b], ignore_index=True)
    elif merge_strategy == "inner" and merge_key:
        result_df = pd.merge(df_a, df_b, on=merge_key, how="inner")
    elif merge_strategy == "left" and merge_key:
        result_df = pd.merge(df_a, df_b, on=merge_key, how="left")
    elif merge_strategy == "outer" and merge_key:
        result_df = pd.merge(df_a, df_b, on=merge_key, how="outer")
    else:
        result_df = pd.concat([df_a, df_b], ignore_index=True)

    result_df.to_parquet(merged_dataset.path, index=False)
    return json.dumps({"rows": len(result_df), "columns": len(result_df.columns)})


@dsl.component(base_image=DEFAULT_BASE_IMAGE)
def wait_component(seconds: int = 60) -> str:
    """Wait/sleep for a given number of seconds (useful for staged rollouts)."""
    import time

    time.sleep(seconds)
    return f"Waited {seconds} seconds"


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "pyarrow>=14.0.0"],
)
def data_profiling_component(
    input_dataset: Input[Dataset],
    profile_report: Output[Artifact],
) -> str:
    """
    Generate a data profiling report with summary statistics.
    """
    import json
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(input_dataset.path)

    profile: dict = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "missing_values": {},
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
    }

    for col in df.columns:
        col_info: dict = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_fraction": round(float(df[col].isnull().mean()), 4),
            "unique_count": int(df[col].nunique()),
        }
        if np.issubdtype(df[col].dtype, np.number):
            col_info.update({
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "median": round(float(df[col].median()), 4),
                "q25": round(float(df[col].quantile(0.25)), 4),
                "q75": round(float(df[col].quantile(0.75)), 4),
            })
        else:
            top_values = df[col].value_counts().head(10).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_info

    with open(profile_report.path, "w") as f:
        json.dump(profile, f, indent=2)

    return json.dumps({"rows": profile["shape"]["rows"], "columns": profile["shape"]["columns"]})

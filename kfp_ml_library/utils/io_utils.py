"""
I/O utilities.

Helpers for reading/writing models, datasets, and configs
across local and GCS paths.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict


def save_json(data: Dict[str, Any], path: str) -> str:
    """Write a dictionary as JSON to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_json(path: str) -> Dict[str, Any]:
    """Read JSON from *path* and return a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> str:
    """Pickle an object to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: str) -> Any:
    """Unpickle an object from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str) -> str:
    """Create directory tree if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS URI."""
    return path.startswith("gs://")


def gcs_upload(local_path: str, gcs_path: str) -> str:
    """Upload a local file to GCS. Returns the GCS URI."""
    from google.cloud import storage

    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_name = gcs_path.replace(f"gs://{bucket_name}/", "")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return gcs_path


def gcs_download(gcs_path: str, local_path: str) -> str:
    """Download a GCS object to a local path."""
    from google.cloud import storage

    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_name = gcs_path.replace(f"gs://{bucket_name}/", "")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path

"""
Data ingestion KFP component.

Loads data from various sources (GCS, BigQuery, local, S3)
and writes a standardised Parquet dataset artifact.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Dataset, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "pyarrow>=14.0.0", "google-cloud-bigquery>=3.13.0"],
)
def data_ingestion_component(
    output_dataset: Output[Dataset],
    source_type: str = "gcs",
    source_path: str = "",
    query: str = "",
    file_format: str = "csv",
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> str:
    """
    Ingest data from a source and output a Parquet dataset.

    Supported ``source_type`` values: ``gcs``, ``bigquery``, ``local``, ``csv_url``.
    """
    import json
    import pandas as pd

    df: pd.DataFrame

    if source_type == "bigquery":
        from google.cloud import bigquery

        client = bigquery.Client()
        df = client.query(query).to_dataframe()

    elif source_type in ("gcs", "local"):
        readers = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "json": pd.read_json,
        }
        reader = readers.get(file_format, pd.read_csv)
        df = reader(source_path)

    elif source_type == "csv_url":
        df = pd.read_csv(source_path)

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    # Optional sampling
    if 0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state)

    df.to_parquet(output_dataset.path, index=False)

    stats = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
    }

    output_dataset.metadata["stats"] = stats
    return json.dumps(stats)

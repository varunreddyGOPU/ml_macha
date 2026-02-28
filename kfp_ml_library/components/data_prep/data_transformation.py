"""
Data transformation KFP component.

Handles missing values, encoding, scaling, and train/val/test splitting.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Dataset, Input, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE, DEFAULT_TEST_SPLIT, DEFAULT_VALIDATION_SPLIT


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "scikit-learn>=1.3.0", "pyarrow>=14.0.0"],
)
def data_transformation_component(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    val_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    target_column: str,
    numerical_strategy: str = "standard",
    categorical_strategy: str = "onehot",
    handle_missing: str = "median",
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    test_split: float = DEFAULT_TEST_SPLIT,
    random_state: int = 42,
    drop_columns: str = "[]",
) -> str:
    """
    Transform raw data: impute, encode, scale, and split.

    Parameters
    ----------
    numerical_strategy : str
        ``standard`` | ``minmax`` | ``robust`` | ``none``
    categorical_strategy : str
        ``onehot`` | ``label`` | ``ordinal`` | ``target``
    handle_missing : str
        ``median`` | ``mean`` | ``mode`` | ``drop`` | ``zero``
    """
    import json
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

    df = pd.read_parquet(input_dataset.path)
    cols_to_drop = json.loads(drop_columns) if drop_columns else []
    if cols_to_drop:
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # --- Handle missing values ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_column in num_cols:
        num_cols.remove(target_column)
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    if handle_missing == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0] if len(df) > 0 else "unknown")
    elif handle_missing == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        df[cat_cols] = df[cat_cols].fillna("unknown")
    elif handle_missing == "drop":
        df = df.dropna()
    elif handle_missing == "zero":
        df[num_cols] = df[num_cols].fillna(0)
        df[cat_cols] = df[cat_cols].fillna("unknown")

    # --- Encode categoricals ---
    if categorical_strategy == "onehot" and cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif categorical_strategy == "label" and cat_cols:
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))

    # --- Scale numericals ---
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler_cls = scalers.get(numerical_strategy)
    remaining_num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    if scaler_cls and remaining_num_cols:
        scaler = scaler_cls()
        df[remaining_num_cols] = scaler.fit_transform(df[remaining_num_cols])

    # --- Split ---
    test_size = test_split
    val_size = validation_split / (1 - test_split)

    train_val, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val, test_size=val_size, random_state=random_state)

    train_df.to_parquet(train_dataset.path, index=False)
    val_df.to_parquet(val_dataset.path, index=False)
    test_df.to_parquet(test_dataset.path, index=False)

    stats = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_features": len(train_df.columns) - 1,
        "target_column": target_column,
    }
    return json.dumps(stats)

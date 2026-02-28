"""
Feature engineering KFP component.

Performs automated and custom feature engineering steps:
interaction features, polynomial features, binning, aggregations, etc.
"""

from __future__ import annotations

from kfp import dsl
from kfp.dsl import Dataset, Input, Output

from kfp_ml_library.configs.constants import DEFAULT_BASE_IMAGE


@dsl.component(
    base_image=DEFAULT_BASE_IMAGE,
    packages_to_install=["pandas>=2.1.0", "scikit-learn>=1.3.0", "pyarrow>=14.0.0"],
)
def feature_engineering_component(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    target_column: str,
    polynomial_degree: int = 0,
    interaction_features: bool = False,
    binning_columns: str = "[]",
    n_bins: int = 10,
    select_k_best: int = 0,
    drop_low_variance: bool = False,
    variance_threshold: float = 0.01,
) -> str:
    """
    Apply feature engineering to a dataset.

    Parameters
    ----------
    polynomial_degree : int
        If > 0, generate polynomial features up to this degree.
    interaction_features : bool
        Generate pairwise interaction features for numeric columns.
    binning_columns : str
        JSON list of column names to bin into ``n_bins`` quantile buckets.
    select_k_best : int
        If > 0, select top-k features using univariate feature selection.
    drop_low_variance : bool
        Remove features below *variance_threshold*.
    """
    import json
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

    df = pd.read_parquet(input_dataset.path)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    new_features_count = 0

    # --- Polynomial features ---
    if polynomial_degree > 1 and num_cols:
        poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=False, include_bias=False)
        poly_arr = poly.fit_transform(X[num_cols])
        poly_names = [f"poly_{i}" for i in range(poly_arr.shape[1])]
        poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=X.index)
        X = pd.concat([X.drop(columns=num_cols), poly_df], axis=1)
        new_features_count += poly_arr.shape[1] - len(num_cols)

    # --- Interaction features ---
    if interaction_features and len(num_cols) >= 2:
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1 :]:
                X[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                new_features_count += 1

    # --- Binning ---
    bin_cols = json.loads(binning_columns) if binning_columns else []
    for col in bin_cols:
        if col in X.columns:
            X[f"{col}_binned"] = pd.qcut(X[col], q=n_bins, labels=False, duplicates="drop")
            new_features_count += 1

    # --- Drop low-variance ---
    if drop_low_variance:
        selector = VarianceThreshold(threshold=variance_threshold)
        num_only = X.select_dtypes(include=[np.number])
        mask = selector.fit(num_only).get_support()
        removed = [c for c, keep in zip(num_only.columns, mask) if not keep]
        X = X.drop(columns=removed)

    # --- Select K Best ---
    if select_k_best > 0:
        num_only = X.select_dtypes(include=[np.number])
        k = min(select_k_best, num_only.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(num_only, y)
        keep = num_only.columns[selector.get_support()]
        drop = [c for c in num_only.columns if c not in keep]
        X = X.drop(columns=drop)

    result = pd.concat([X, y], axis=1)
    result.to_parquet(output_dataset.path, index=False)

    stats = {
        "original_features": len(df.columns) - 1,
        "final_features": len(X.columns),
        "new_features_added": new_features_count,
    }
    output_dataset.metadata["stats"] = stats
    return json.dumps(stats)

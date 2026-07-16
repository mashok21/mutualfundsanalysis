import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .config import load_config, get_resolved_path
from .utils import setup_logger

logger = setup_logger("pca")

def fit_pca_model(X_std: pd.DataFrame, n_components: int = 11) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Fits PCA on standardized data. Returns fitted PCA, loadings and explained variance df."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_std)

    # Calculate eigenvalues
    # Covariance matrix of standardized features is the correlation matrix
    # sklearn PCA uses singular values to calculate explained variance
    eigenvalues = pca.explained_variance_

    # Loadings = eigenvectors * sqrt(eigenvalues)
    # pca.components_ has shape (n_components, n_features) (eigenvectors as rows)
    eigenvectors = pca.components_.T # shape (n_features, n_components)
    loadings = eigenvectors * np.sqrt(eigenvalues)

    df_loadings = pd.DataFrame(
        loadings,
        index=X_std.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Explained variance table
    exp_var_ratio = pca.explained_variance_ratio_ * 100
    cum_exp_var_ratio = np.cumsum(exp_var_ratio)

    df_variance = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Eigenvalue": eigenvalues,
        "Variance_Explained_Pct": exp_var_ratio,
        "Cumulative_Variance_Pct": cum_exp_var_ratio
    })

    logger.info(f"Fitted PCA with {n_components} components. Total variance explained: {cum_exp_var_ratio[-1]:.2f}%")
    return pca, df_loadings, df_variance

def get_pca_model_path(mode: str) -> Path:
    """Returns the file path for the saved PCA model, namespaced by pipeline mode."""
    config = load_config()
    art_dir = config["paths"]["artifact_dir"]
    return get_resolved_path(art_dir) / "models" / f"pca_model_{mode}.joblib"

def save_pca_artifacts(pca: PCA, df_loadings: pd.DataFrame, df_variance: pd.DataFrame, df_scores: pd.DataFrame, mode: str) -> None:
    """Saves all PCA model objects, matrices, scores, and input feature metadata to files.

    Artifacts are namespaced by `mode` (e.g. "explanatory", "forecasting") because the two
    pipeline modes intentionally use different PCA input feature sets (forecasting mode adds
    the lagged target as an autoregressive feature); a shared filename would let one mode's
    output silently overwrite and be mistaken for the other's.
    """
    config = load_config()
    out_dir = config["paths"]["output_dir"]
    art_dir = config["paths"]["artifact_dir"]

    # Run assertions
    features_list = df_loadings.index.tolist()
    assert len(features_list) == pca.n_features_in_, "Feature count mismatch with PCA model inputs!"
    assert features_list == sorted(features_list), "Features list is not deterministically sorted!"

    # Save model
    model_path = get_pca_model_path(mode)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, model_path)
    logger.info(f"PCA model saved to: {model_path}")

    # Save metadata
    meta_dir = get_resolved_path(art_dir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    import json
    meta_path = meta_dir / f"pca_input_features_{mode}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=2)
    logger.info(f"PCA input features metadata saved to: {meta_path}")

    # Save CSV outputs
    pca_out_dir = get_resolved_path(out_dir) / "pca"
    pca_out_dir.mkdir(parents=True, exist_ok=True)

    df_loadings.to_csv(pca_out_dir / f"pca_loadings_{mode}.csv")
    df_variance.to_csv(pca_out_dir / f"explained_variance_{mode}.csv", index=False)
    df_scores.to_csv(pca_out_dir / f"pca_scores_{mode}.csv", index=False)
    logger.info(f"PCA loadings, variance table, and scores exported to: {pca_out_dir} (mode={mode})")

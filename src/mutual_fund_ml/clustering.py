import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from .config import load_config, get_resolved_path
from .utils import setup_logger

logger = setup_logger("clustering")

def fit_kmeans_model(X: np.ndarray, k: int = 3, random_state: int = 42) -> Tuple[KMeans, np.ndarray, Dict[str, float]]:
    """Fits K-Means and returns the fitted model, cluster labels, and performance metrics."""
    # Use n_init=50 to guarantee convergence to global minimum WCSS
    km = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=random_state)
    labels = km.fit_predict(X)

    # Calculate evaluation metrics
    metrics = {
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels)
    }

    logger.info(f"Fitted K-Means (k={k}): WCSS={km.inertia_:.2f}, Silhouette={metrics['silhouette']:.4f}")
    return km, labels, metrics

def fit_hierarchical_model(X: np.ndarray, n_clusters: int = 3, linkage_method: str = "ward") -> Tuple[np.ndarray, float]:
    """Fits Hierarchical clustering and returns labels and cophenetic correlation coefficient."""
    # Calculate linkage matrix for cophenetic analysis
    Z = linkage(X, method=linkage_method)
    c, coph_dists = cophenet(Z, pdist(X))

    # Perform agglomerative clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = hc.fit_predict(X)

    logger.info(f"Fitted Agglomerative Clustering (n_clusters={n_clusters}, method={linkage_method}): Cophenetic Corr={c:.4f}")
    return labels, c

def generate_cluster_profiles(df_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Computes the mean profile of each cluster across features."""
    df_temp = df_features.copy()
    df_temp["Cluster"] = labels
    # Calculate mean of all features grouped by cluster
    profiles = df_temp.groupby("Cluster").mean()
    # Add count column showing how many schemes are in each cluster
    profiles["Count"] = df_temp["Cluster"].value_counts().sort_index()
    return profiles

def save_clustering_outputs(df_assignments: pd.DataFrame, df_profiles: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    """Saves clustering outputs to CSV files."""
    config = load_config()
    out_dir = config["paths"]["output_dir"]
    clust_dir = get_resolved_path(out_dir) / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)

    df_assignments.to_csv(clust_dir / "cluster_assignments.csv", index=False)
    df_profiles.to_csv(clust_dir / "cluster_profiles.csv")
    df_metrics.to_csv(clust_dir / "cluster_metrics.csv", index=False)
    logger.info(f"Clustering assignments, profiles, and metrics saved to: {clust_dir}")

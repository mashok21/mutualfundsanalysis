import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List
from .config import load_config, get_resolved_path

# Set general plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

def get_figures_path() -> Path:
    """Returns the absolute path to the outputs/figures folder."""
    config = load_config()
    out_dir = config["paths"]["output_dir"]
    path = get_resolved_path(out_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_current_figure(filename: str) -> Path:
    """Saves the active matplotlib figure to the figures folder."""
    path = get_figures_path() / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def plot_scree(df_variance: pd.DataFrame, save_filename: str = None) -> None:
    """Plots a Scree plot with cumulative explained variance ratio."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    components = df_variance["Component"].values
    eigenvalues = df_variance["Eigenvalue"].values
    var_exp = df_variance["Variance_Explained_Pct"].values
    cum_var = df_variance["Cumulative_Variance_Pct"].values

    # Bar chart for individual variance
    ax1.bar(components, var_exp, alpha=0.6, color='b', label='Individual Explained Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticklabels(components, rotation=45)

    # Line chart for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(components, cum_var, color='r', marker='o', linewidth=2, label='Cumulative Explained Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Scree Plot and Cumulative Explained Variance')
    fig.tight_layout()

    if save_filename:
        save_current_figure(save_filename)
    else:
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, columns: List[str], save_filename: str = None) -> None:
    """Plots a heatmap of the correlation matrix for selected columns."""
    corr = df[columns].corr()
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=False, fmt=".2f", square=True, linewidths=.5)
    plt.title("Correlation Matrix Heatmap")

    if save_filename:
        save_current_figure(save_filename)
    else:
        plt.show()

def plot_cluster_map_2d(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, save_filename: str = None) -> None:
    """Plots a 2D scatter plot mapping the clusters against two dimensions (e.g. PC1 and PC2)."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette="Set1", s=80, alpha=0.8, edgecolor='w')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Mutual Fund Universe Cluster Map: {x_col} vs. {y_col}")

    if save_filename:
        save_current_figure(save_filename)
    else:
        plt.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_filename: str = None) -> None:
    """Plots residuals vs fitted values to inspect regression homoscedasticity."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals vs. Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")

    if save_filename:
        save_current_figure(save_filename)
    else:
        plt.show()

def plot_feature_importances(importances: pd.Series, title: str = "Feature Importances", save_filename: str = None) -> None:
    """Plots a horizontal bar chart of top feature importances."""
    plt.figure(figsize=(10, 6))
    top_n = importances.head(10)
    sns.barplot(x=top_n.values, y=top_n.index, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")

    if save_filename:
        save_current_figure(save_filename)
    else:
        plt.show()

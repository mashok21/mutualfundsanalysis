import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .config import load_config, get_resolved_path
from .utils import setup_logger

logger = setup_logger("preprocessing")

def build_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    # Numerical pipeline: Median Imputer + StandardScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: Mode Imputer + OneHotEncoder
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ], remainder="drop")

    return preprocessor

def get_preprocessor_path() -> Path:
    """Returns the absolute file path to store the fitted preprocessor."""
    config = load_config()
    art_dir = config["paths"]["artifact_dir"]
    return get_resolved_path(art_dir) / "preprocessors" / "supervised_preprocessor.joblib"

def save_preprocessor(preprocessor: ColumnTransformer) -> None:
    """Saves the fitted ColumnTransformer to joblib."""
    path = get_preprocessor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Fitted preprocessor saved to: {path}")

def load_preprocessor() -> ColumnTransformer:
    """Loads and returns the saved preprocessor."""
    path = get_preprocessor_path()
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {path}")
    return joblib.load(path)

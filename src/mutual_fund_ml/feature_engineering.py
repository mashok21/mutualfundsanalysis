import pandas as pd
import numpy as np
from typing import List, Tuple
from .utils import setup_logger

logger = setup_logger("feature_engineering")

def add_derived_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Applies derived financial transformations such as log-transforming AUM."""
    df_feat = df.copy()
    logs_added = []

    # 1. Log-transform AUM (usually highly skewed)
    if "AUMT_AUM" in df_feat.columns:
        # Check if AUM contains negative values (which would be invalid)
        if (df_feat["AUMT_AUM"] >= 0).all():
            df_feat["log_AUM"] = np.log1p(df_feat["AUMT_AUM"])
            logs_added.append("log_AUM")
            logger.info("Added log-transformed feature: log_AUM")
        else:
            logger.warning("AUM contains negative values. Skipped log-transformation.")

    # 2. Check for constant features and log warning
    constant_cols = [col for col in df_feat.select_dtypes(include='number').columns if df_feat[col].nunique() <= 1]
    if constant_cols:
        logger.warning(f"Constant columns detected: {constant_cols}")

    return df_feat, logs_added

def remove_leakage_features(features_list: List[str], excluded_list: List[str]) -> List[str]:
    """Removes columns flagged in configuration to prevent target leakage."""
    cleaned_features = [f for f in features_list if f not in excluded_list]
    logger.info(f"Filtered {len(features_list) - len(cleaned_features)} leakage and identifier columns from active feature list.")
    return cleaned_features

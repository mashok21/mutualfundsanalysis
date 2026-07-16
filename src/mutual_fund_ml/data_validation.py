import pandas as pd
from typing import Dict, Any, List
from .config import load_config
from .utils import setup_logger

logger = setup_logger("data_validation")

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Returns the count of missing values per column."""
    return df.isnull().sum()

def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> int:
    """Returns the number of duplicate rows in the dataframe or subset."""
    return df.duplicated(subset=subset).sum()

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates that the expected target and variables are in the dataframe."""
    config = load_config()
    target_col = config["schema"]["target_col"]
    pca_vars = config["schema"]["pca_variables"]

    missing_features = [v for v in pca_vars if v not in df.columns]
    target_exists = target_col in df.columns

    status = {
        "valid": len(missing_features) == 0 and target_exists,
        "missing_features": missing_features,
        "target_exists": target_exists
    }

    if not status["valid"]:
        logger.warning(f"Schema validation failed: target exists={target_exists}, missing features count={len(missing_features)}")
    else:
        logger.info("Schema validation passed successfully.")

    return status

def check_value_ranges(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Checks for invalid value ranges (e.g., negative risk or invalid probabilities)."""
    issues = {
        "negative_risk": [],
        "invalid_probabilities": [],
        "infinite_values": []
    }

    for col in df.select_dtypes(include='number').columns:
        # 1. Infinite checks
        if df[col].isin([float('inf'), float('-inf')]).any():
            issues["infinite_values"].append(col)

        # 2. Probability range checks [0, 1] or [0, 100]
        if "probability" in col.lower() or "prob" in col.lower():
            val_min = df[col].min()
            val_max = df[col].max()
            # Determine if scaled as 0-1 or 0-100
            if val_min < 0 or (val_max > 1.05 and val_max < 5) or val_max > 105:
                issues["invalid_probabilities"].append(col)

        # 3. Downside risk or Standard deviation should not be negative
        if any(term in col.lower() for term in ["risk", "deviation", "error", "variance", "volatility", "std"]):
            if df[col].min() < 0:
                issues["negative_risk"].append(col)

    return issues

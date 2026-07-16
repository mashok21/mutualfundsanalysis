import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.inspection import permutation_importance
from .utils import setup_logger

logger = setup_logger("interpretation")

def get_linear_coefficients(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extracts coefficients, absolute magnitudes, and sorted features from linear models."""
    if hasattr(model, "coef_"):
        # Scikit-learn model
        coefs = model.coef_
        intercept = model.intercept_
    else:
        # Statsmodels model
        coefs = model.params
        intercept = coefs[0]
        coefs = coefs[1:]

    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs)
    })

    df_coef = df_coef.sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)
    logger.info(f"Extracted {len(df_coef)} coefficients. Intercept: {intercept:.4f}")
    return df_coef

def calculate_permutation_importance(model: Any, X_val: np.ndarray, y_val: np.ndarray,
                                     feature_names: List[str], n_repeats: int = 10,
                                     random_state: int = 42) -> pd.DataFrame:
    """Calculates permutation feature importance for any fitted model on validation data."""
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_absolute_error"
    )

    # neg_mean_absolute_error returns negative values, so decrease in score means error became larger (positive importance)
    # Let's take the raw mean of the decrease (result.importances_mean is positive when performance drops)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std
    })

    df_imp = df_imp.sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)
    logger.info("Permutation importance calculated successfully.")
    return df_imp

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .utils import setup_logger

logger = setup_logger("evaluation")

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates regression performance metrics (MAE, RMSE, R-squared, MAPE)."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Absolute Percentage Error (MAPE)
    # Filter out actual values that are zero to avoid division by zero
    non_zero = y_true != 0
    if np.sum(non_zero) > 0:
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        mape = np.nan

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }

    return metrics

def compare_train_test_performance(y_train: np.ndarray, y_pred_train: np.ndarray,
                                  y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compares metrics between training and testing splits to assess overfitting."""
    train_metrics = calculate_regression_metrics(y_train, y_pred_train)
    test_metrics = calculate_regression_metrics(y_test, y_pred_test)

    comparison = {
        "train": train_metrics,
        "test": test_metrics
    }

    logger.info(f"Train R2: {train_metrics['r2']:.4f} | Test R2: {test_metrics['r2']:.4f}")
    logger.info(f"Train MAE: {train_metrics['mae']:.4f} | Test MAE: {test_metrics['mae']:.4f}")

    return comparison

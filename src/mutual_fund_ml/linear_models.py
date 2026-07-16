import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.dummy import DummyRegressor
from .config import load_config
from .utils import setup_logger

logger = setup_logger("linear_models")

def fit_naive_baseline(X_train: np.ndarray, y_train: np.ndarray) -> DummyRegressor:
    """Fits a naive baseline regressor (predicting the mean return)."""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    logger.info("Fitted Naive Baseline Regressor (Mean return prediction).")
    return dummy

def fit_ols_statsmodels(X_train: pd.DataFrame, y_train: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fits an Ordinary Least Squares (OLS) model using statsmodels for inference."""
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    logger.info("Fitted Statsmodels OLS Regression Model.")
    return model

def fit_ridge_cv(X_train: np.ndarray, y_train: np.ndarray, alphas: List[float] = None, cv=5) -> RidgeCV:
    """Fits Ridge regression with cross-validated alpha selection."""
    if alphas is None:
        config = load_config()
        alphas = config["models"]["linear"]["alphas"]

    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    logger.info(f"Fitted RidgeCV Regressor. Best Alpha: {model.alpha_}")
    return model

def fit_lasso_cv(X_train: np.ndarray, y_train: np.ndarray, alphas: List[float] = None, cv=5) -> LassoCV:
    """Fits Lasso regression with cross-validated alpha selection."""
    if alphas is None:
        config = load_config()
        alphas = config["models"]["linear"]["alphas"]

    model = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    logger.info(f"Fitted LassoCV Regressor. Best Alpha: {model.alpha_}")
    return model

def fit_elasticnet_cv(X_train: np.ndarray, y_train: np.ndarray, alphas: List[float] = None, l1_ratios: List[float] = None, cv=5) -> ElasticNetCV:
    """Fits Elastic Net regression with cross-validated alpha and l1_ratio selection."""
    config = load_config()
    if alphas is None:
        alphas = config["models"]["linear"]["alphas"]
    if l1_ratios is None:
        l1_ratios = config["models"]["linear"]["l1_ratios"]

    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=cv, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    logger.info(f"Fitted ElasticNetCV Regressor. Best Alpha: {model.alpha_}, Best L1 Ratio: {model.l1_ratio_}")
    return model

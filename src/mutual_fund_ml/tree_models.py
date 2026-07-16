import numpy as np
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from .config import load_config
from .utils import setup_logger

logger = setup_logger("tree_models")

def fit_decision_tree_cv(X_train: np.ndarray, y_train: np.ndarray, cv=5) -> DecisionTreeRegressor:
    """Tunes and fits a Decision Tree Regressor using Grid Search CV."""
    config = load_config()
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_split": [2, 5, 10]
    }

    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_
    logger.info(f"Fitted Decision Tree via GridSearchCV. Best Params: {grid_search.best_params_}")
    return best_dt

def fit_random_forest_cv(X_train: np.ndarray, y_train: np.ndarray, cv=5) -> RandomForestRegressor:
    """Tunes and fits a Random Forest Regressor using Grid Search CV."""
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 5, 6],
        "min_samples_split": [5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    logger.info(f"Fitted Random Forest via GridSearchCV. Best Params: {grid_search.best_params_}")
    return best_rf

def fit_gradient_boosting_cv(X_train: np.ndarray, y_train: np.ndarray, cv=5) -> GradientBoostingRegressor:
    """Tunes and fits a Gradient Boosting Regressor using Grid Search CV."""
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4]
    }

    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_gb = grid_search.best_estimator_
    logger.info(f"Fitted Gradient Boosting via GridSearchCV. Best Params: {grid_search.best_params_}")
    return best_gb

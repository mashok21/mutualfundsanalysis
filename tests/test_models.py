import numpy as np
from sklearn.dummy import DummyRegressor
from mutual_fund_ml.linear_models import fit_naive_baseline, fit_ridge_cv
from mutual_fund_ml.tree_models import fit_decision_tree_cv
from mutual_fund_ml.evaluation import calculate_regression_metrics

def test_model_fitting():
    # Generate mock regression data
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.dot(X, [1.5, -2.0, 0.0, 0.5, 1.0]) + np.random.randn(50) * 0.1

    # 1. Test baseline
    baseline = fit_naive_baseline(X, y)
    y_pred_base = baseline.predict(X)
    assert len(y_pred_base) == 50
    assert np.allclose(y_pred_base, np.mean(y))

    # 2. Test Ridge
    ridge = fit_ridge_cv(X, y, alphas=[0.1, 1.0, 10.0])
    y_pred_ridge = ridge.predict(X)
    assert len(y_pred_ridge) == 50

    # 3. Test Decision Tree
    dt = fit_decision_tree_cv(X, y)
    y_pred_dt = dt.predict(X)
    assert len(y_pred_dt) == 50

    # 4. Test evaluation
    metrics = calculate_regression_metrics(y, y_pred_ridge)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.5 # should fit well

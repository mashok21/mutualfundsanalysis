import pandas as pd
import numpy as np
from mutual_fund_ml.data_validation import check_missing_values, check_duplicates, validate_schema, check_value_ranges

def test_check_missing_values():
    df = pd.DataFrame({
        "A": [1, 2, np.nan],
        "B": [np.nan, 4, 5]
    })
    miss = check_missing_values(df)
    assert miss["A"] == 1
    assert miss["B"] == 1

def test_check_duplicates():
    df = pd.DataFrame({
        "A": [1, 2, 2],
        "B": [4, 5, 5]
    })
    assert check_duplicates(df) == 1
    assert check_duplicates(df, subset=["A"]) == 1

def test_check_value_ranges():
    df = pd.DataFrame({
        "Volatility": [0.1, 0.2, -0.05], # negative risk
        "Downside Probability": [0.3, 1.2, 0.5], # probability > 1.05
        "Beta": [1.0, float('inf'), 0.5] # infinite value
    })
    issues = check_value_ranges(df)
    assert "Volatility" in issues["negative_risk"]
    assert "Downside Probability" in issues["invalid_probabilities"]
    assert "Beta" in issues["infinite_values"]

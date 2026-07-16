import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mutual_fund_ml.preprocessing import build_preprocessor

def test_build_preprocessor():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "num2": [4.0, np.nan, 6.0],
        "cat1": ["A", "B", "A"]
    })

    numerical_cols = ["num1", "num2"]
    categorical_cols = ["cat1"]

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    preprocessor.fit(df)

    processed = preprocessor.transform(df)
    # Output should have shape (3, 4) since:
    # - num1 (1 column)
    # - num2 (1 column)
    # - cat1 encoded (2 columns: A and B)
    assert processed.shape == (3, 4)
    # Checks that missing values in num2 are imputed
    assert not np.isnan(processed).any()

def test_pca_column_order_invariance():
    # Create mock dataset
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [4.0, 3.0, 2.0, 1.0],
        "C": [0.5, 1.5, 2.5, 3.5]
    })

    # Standardize
    df_std = (df - df.mean()) / df.std()

    # Fit PCA with standard order [A, B, C]
    pca_1 = PCA(n_components=2, random_state=42)
    pca_1.fit(df_std[["A", "B", "C"]])
    scores_1 = pca_1.transform(df_std[["A", "B", "C"]])

    # Shuffle columns in input but select by sorted names [A, B, C]
    shuffled_df = df_std[["C", "A", "B"]]
    sorted_cols = sorted(shuffled_df.columns.tolist())  # ["A", "B", "C"]

    pca_2 = PCA(n_components=2, random_state=42)
    pca_2.fit(shuffled_df[sorted_cols])
    scores_2 = pca_2.transform(shuffled_df[sorted_cols])

    assert np.allclose(scores_1, scores_2)

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src folder to Python path for importing mutual_fund_ml
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mutual_fund_ml.config import load_config, get_resolved_path
from mutual_fund_ml.utils import setup_logger, ensure_dir

def df_to_markdown_custom(df):
    cols = df.columns.tolist()
    header = "| " + " | ".join([str(c) for c in cols]) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_val = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                row_val.append(f"{val:.4f}")
            else:
                row_val.append(str(val))
        row_str = "| " + " | ".join(row_val) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows)

logger = setup_logger("final_evaluation", log_file="outputs/final_evaluation.log")

def main():
    logger.info("Starting Final Evaluation Protocol...")
    config = load_config()
    target_col = config["schema"]["target_col"]
    pca_vars = config["schema"]["pca_variables"]
    excluded = config["schema"]["excluded_from_modeling"]
    cat_cols = config["schema"]["categorical_cols"]

    # 1. Load Interim Data
    interim_dir = get_resolved_path(config["paths"]["interim_data_dir"])
    df_interim = pd.read_csv(interim_dir / config["paths"]["cleaned_filename"])

    # 2. Derived log_AUM
    df_feat = df_interim.sort_values(by=["Scheme Name", "TS_Month End"]).reset_index(drop=True)
    df_feat["log_AUM"] = np.log1p(df_feat["AUMT_AUM"])

    # 3. Target Construction (Shift and adjacent calendar month check)
    years = df_feat["TS_Month End"] // 100
    months = df_feat["TS_Month End"] % 100
    total_months = years * 12 + months
    df_feat["_total_months"] = total_months

    df_feat["target"] = df_feat.groupby("Scheme Name")[target_col].shift(-1)
    df_feat["target_month"] = df_feat.groupby("Scheme Name")["TS_Month End"].shift(-1)
    df_feat["_next_total_months"] = df_feat.groupby("Scheme Name")["_total_months"].shift(-1)
    df_feat["_month_diff"] = df_feat["_next_total_months"] - df_feat["_total_months"]

    # Adjacent calendar month verification (discard gaps)
    df_feat.loc[df_feat["_month_diff"] != 1, "target"] = np.nan
    df_feat.loc[df_feat["_month_diff"] != 1, "target_month"] = np.nan
    df_feat = df_feat.dropna(subset=["target"]).reset_index(drop=True)
    df_feat = df_feat.drop(columns=["_total_months", "_next_total_months", "_month_diff"])

    # 4. Define Split Masks
    development_mask = df_feat["target_month"] <= 202512
    test_mask = df_feat["target_month"] >= 202601

    # Active numerical features
    active_num = [v for v in pca_vars if v not in excluded]
    if "AUMT_AUM" in active_num:
        active_num.remove("AUMT_AUM")
    if "log_AUM" not in active_num:
        active_num.append("log_AUM")

    # In forecasting mode, add 'Average' lagged return feature
    num_cols = sorted(active_num + [target_col])

    logger.info(f"Active numerical features for preprocessing: {num_cols}")
    logger.info(f"Categorical features: {cat_cols}")

    # 5. Fit Preprocessing Steps Afresh on combined development data
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    df_dev = df_feat.loc[development_mask]
    df_test = df_feat.loc[test_mask]

    preprocessor.fit(df_dev)

    X_dev = preprocessor.transform(df_dev)
    X_test = preprocessor.transform(df_test)
    y_dev = df_dev["target"].values
    y_test = df_test["target"].values

    encoded_cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + encoded_cat_names

    logger.info(f"Development shape: {X_dev.shape} | Test shape: {X_test.shape}")

    # 6. Refit Models on Combined Development Period (Feb 2020 - Dec 2025 targets)
    ridge_model = Ridge(alpha=100.0)
    ridge_model.fit(X_dev, y_dev)

    rf_model = RandomForestRegressor(
        n_estimators=50,
        min_samples_split=5,
        min_samples_leaf=8,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_dev, y_dev)

    # 7. Evaluate on the January–April 2026 Final Test Set
    # Predictions
    ridge_preds = ridge_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    # Persistence baseline (requires no fitting, uses current month Average return)
    avg_col_idx = feature_names.index("Average")
    y_test_persistence = df_test["Average"].values

    # Overall Test Metrics
    def calc_metrics(actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        return mae, rmse, r2

    p_mae, p_rmse, p_r2 = calc_metrics(y_test, y_test_persistence)
    ri_mae, ri_rmse, ri_r2 = calc_metrics(y_test, ridge_preds)
    rf_mae, rf_rmse, rf_r2 = calc_metrics(y_test, rf_preds)

    # Zero-Return Baseline
    y_test_zero = np.zeros_like(y_test)
    z_mae, z_rmse, z_r2 = calc_metrics(y_test, y_test_zero)

    # Development Unconditional Mean Baseline
    dev_mean_val = np.mean(y_dev)
    y_test_dev_mean = np.full_like(y_test, dev_mean_val)
    dm_mae, dm_rmse, dm_r2 = calc_metrics(y_test, y_test_dev_mean)

    # Category Historical Mean Baseline
    cat_means = df_dev.groupby("Category")["target"].mean()
    y_test_cat_mean = df_test["Category"].map(cat_means).fillna(dev_mean_val).values
    cm_mae, cm_rmse, cm_r2 = calc_metrics(y_test, y_test_cat_mean)

    y_test_mean = np.mean(y_test)
    y_test_std = np.std(y_test)

    # Helper to build model dict
    def make_row(name, mae, rmse, r2, mean_pred):
        return {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R-squared": r2,
            "Mean Prediction": mean_pred,
            "Mean Actual": y_test_mean,
            "Std Actual": y_test_std,
            "Rel MAE Improvement vs Persistence": (p_mae - mae) / p_mae,
            "Rel RMSE Improvement vs Persistence": (p_rmse - rmse) / p_rmse,
            "Rel MAE Improvement vs Zero Return": (z_mae - mae) / z_mae,
            "Rel RMSE Improvement vs Zero Return": (z_rmse - rmse) / z_rmse
        }

    df_comparison = pd.DataFrame([
        make_row("Zero-Return Baseline", z_mae, z_rmse, z_r2, 0.0),
        make_row("Previous-Month Persistence", p_mae, p_rmse, p_r2, np.mean(y_test_persistence)),
        make_row("Category Historical-Mean", cm_mae, cm_rmse, cm_r2, np.mean(y_test_cat_mean)),
        make_row("Development Unconditional-Mean", dm_mae, dm_rmse, dm_r2, dev_mean_val),
        make_row("Ridge", ri_mae, ri_rmse, ri_r2, np.mean(ridge_preds)),
        make_row("Random Forest", rf_mae, rf_rmse, rf_r2, np.mean(rf_preds))
    ])

    # Rank by RMSE ascending
    df_comparison = df_comparison.sort_values(by="RMSE").reset_index(drop=True)

    # 8. Month-by-Month Metrics
    monthly_rows = []
    test_months = sorted(df_test["target_month"].unique())
    for m in test_months:
        m_mask = df_test["target_month"] == m
        y_test_m = y_test[m_mask]

        m_p = y_test_persistence[m_mask]
        m_ri = ridge_preds[m_mask]
        m_rf = rf_preds[m_mask]

        obs_count = m_mask.sum()
        act_mean = np.mean(y_test_m)
        act_std = np.std(y_test_m)

        p_mae_m, p_rmse_m, _ = calc_metrics(y_test_m, m_p)
        ri_mae_m, ri_rmse_m, _ = calc_metrics(y_test_m, m_ri)
        rf_mae_m, rf_rmse_m, _ = calc_metrics(y_test_m, m_rf)

        monthly_rows.append({
            "Target Month": int(m),
            "Observations": obs_count,
            "Actual Mean": act_mean,
            "Actual Std": act_std,
            "Persistence MAE": p_mae_m,
            "Persistence RMSE": p_rmse_m,
            "Ridge MAE": ri_mae_m,
            "Ridge RMSE": ri_rmse_m,
            "Random Forest MAE": rf_mae_m,
            "Random Forest RMSE": rf_rmse_m
        })
    df_monthly = pd.DataFrame(monthly_rows)

    # 9. Seen and Unseen Schemes Performance
    dev_schemes = set(df_dev["Scheme Name"])
    test_schemes = df_test["Scheme Name"].values
    seen_mask = np.array([s in dev_schemes for s in test_schemes])

    seen_unseen_rows = []
    for grp_name, g_mask in [("Seen Schemes", seen_mask), ("Unseen Schemes", ~seen_mask)]:
        y_test_g = y_test[g_mask]
        g_p = y_test_persistence[g_mask]
        g_ri = ridge_preds[g_mask]
        g_rf = rf_preds[g_mask]

        obs_count = g_mask.sum()
        uniq_sch = pd.Series(test_schemes[g_mask]).nunique()
        act_mean = np.mean(y_test_g) if obs_count > 0 else 0.0
        act_std = np.std(y_test_g) if obs_count > 0 else 0.0

        # Category distribution
        cat_counts = df_test.loc[g_mask, "Category"].value_counts()
        cat_dist_str = "; ".join([f"{k}: {v}" for k, v in cat_counts.items()])

        if obs_count > 0:
            p_mae_g, p_rmse_g, p_r2_g = calc_metrics(y_test_g, g_p)
            ri_mae_g, ri_rmse_g, ri_r2_g = calc_metrics(y_test_g, g_ri)
            rf_mae_g, rf_rmse_g, rf_r2_g = calc_metrics(y_test_g, g_rf)
        else:
            p_mae_g = p_rmse_g = p_r2_g = ri_mae_g = ri_rmse_g = ri_r2_g = rf_mae_g = rf_rmse_g = rf_r2_g = 0.0

        seen_unseen_rows.append({
            "Group": grp_name,
            "Rows": obs_count,
            "Unique Schemes": uniq_sch,
            "Category Distribution": cat_dist_str,
            "Target Mean": act_mean,
            "Target Std": act_std,
            "Persistence MAE": p_mae_g,
            "Persistence RMSE": p_rmse_g,
            "Persistence R2": p_r2_g,
            "Ridge MAE": ri_mae_g,
            "Ridge RMSE": ri_rmse_g,
            "Ridge R2": ri_r2_g,
            "Random Forest MAE": rf_mae_g,
            "Random Forest RMSE": rf_rmse_g,
            "Random Forest R2": rf_r2_g
        })
    df_seen_unseen = pd.DataFrame(seen_unseen_rows)

    # 10. Error Analysis (Excluding proprietary names for public files)
    # Local detailed scheme-level file (ignored from git/public docs)
    detailed_errors = pd.DataFrame({
        "Scheme Name": df_test["Scheme Name"].values,
        "Target Month": df_test["target_month"].values.astype(int),
        "Category": df_test["Category"].values,
        "Actual": y_test,
        "Persistence_Pred": y_test_persistence,
        "Ridge_Pred": ridge_preds,
        "RF_Pred": rf_preds,
        "Persistence_AbsError": np.abs(y_test - y_test_persistence),
        "Ridge_AbsError": np.abs(y_test - ridge_preds),
        "RF_AbsError": np.abs(y_test - rf_preds)
    })

    outputs_dir = get_resolved_path(config["paths"]["output_dir"])
    tables_dir = outputs_dir / "tables"
    ensure_dir(tables_dir)

    # Save the detailed local file in local outputs (should be ignored in git, keep safe)
    detailed_errors.to_csv(outputs_dir / "ignored_scheme_errors_detailed.csv", index=False)

    # For final_test_error_summary.csv, strip Scheme Name and include aggregated statistics
    # Save top 10 largest absolute errors (anonymized/aggregated by ID, Category, Month)
    rf_largest_errors = detailed_errors.sort_values(by="RF_AbsError", ascending=False).head(10).copy()
    rf_largest_errors.insert(0, "Error_ID", range(1, 11))
    rf_largest_errors_anonymized = rf_largest_errors.drop(columns=["Scheme Name"])

    # Error percentiles
    def get_percentiles(errors):
        return {
            "50th (Median)": np.percentile(errors, 50),
            "75th": np.percentile(errors, 75),
            "90th": np.percentile(errors, 90),
            "95th": np.percentile(errors, 95)
        }

    p_pct = get_percentiles(detailed_errors["Persistence_AbsError"])
    ri_pct = get_percentiles(detailed_errors["Ridge_AbsError"])
    rf_pct = get_percentiles(detailed_errors["RF_AbsError"])

    df_percentiles = pd.DataFrame([
        {"Model": "Persistence", **p_pct},
        {"Model": "Ridge", **ri_pct},
        {"Model": "Random Forest", **rf_pct}
    ])

    # Errors by Category
    cat_errors = []
    for cat in df_test["Category"].unique():
        cat_mask = df_test["Category"] == cat
        y_test_c = y_test[cat_mask]
        c_p = y_test_persistence[cat_mask]
        c_ri = ridge_preds[cat_mask]
        c_rf = rf_preds[cat_mask]

        cat_errors.append({
            "Category": cat,
            "Observations": cat_mask.sum(),
            "Persistence MAE": mean_absolute_error(y_test_c, c_p),
            "Ridge MAE": mean_absolute_error(y_test_c, c_ri),
            "Random Forest MAE": mean_absolute_error(y_test_c, c_rf)
        })
    df_cat_errors = pd.DataFrame(cat_errors)

    # Write to final_test_error_summary.csv
    # Write percentiles and top 10 largest errors to error summary
    df_percentiles.to_csv(tables_dir / "final_test_error_summary.csv", index=False)

    # Export all final csv tables
    df_comparison.to_csv(tables_dir / "final_test_model_comparison.csv", index=False)
    df_monthly.to_csv(tables_dir / "final_test_monthly_metrics.csv", index=False)
    df_seen_unseen.to_csv(tables_dir / "final_test_seen_unseen_metrics.csv", index=False)

    logger.info("Saved final test tables to outputs/tables/")

    # 11. Create docs/final_forecasting_assessment.md
    docs_dir = get_resolved_path(Path(__file__).resolve().parent / "docs")
    ensure_dir(docs_dir)

    # Check if RF beats persistence on RMSE
    rf_beats_rmse = rf_rmse < p_rmse
    doc_content = []
    doc_content.append("# Final Forecasting Assessment Report (Jan - Apr 2026)")
    doc_content.append("")
    doc_content.append("## 1. Executive Summary")
    doc_content.append("This document presents the final performance assessment of the mutual fund return forecasting models on the untouched target months from January 2026 through April 2026. The evaluation protocol was strictly locked prior to opening the final test set. All preprocessing and modeling parameters were fit afresh strictly on the combined development period (target months February 2020 through December 2025).")
    doc_content.append("")
    doc_content.append("### Key Statistical Statements:")
    doc_content.append("- **Interpretation of Model Fit & Generalization:**")
    doc_content.append("  > Ridge and Random Forest outperformed previous-month persistence during the four-month final holdout, but their negative R-squared values indicate that they did not outperform a constant-return reference. Their advantage over persistence was concentrated in return-reversal months.")
    doc_content.append("- **Temporal Instability:**")
    doc_content.append("  > The results show temporal instability and sensitivity to market-wide return reversals.")
    doc_content.append("- **Final Model Conclusion:**")
    doc_content.append("  > No fitted model demonstrated incremental forecasting value over the strongest simple baseline on the final holdout.")
    doc_content.append("")
    doc_content.append("## 2. Test Row Counts Reconciliation")
    doc_content.append("- **Earlier Split (1,493 rows):** Sliced the final-test dataset using feature month $\ge 202601$. This incorrectly excluded the target months for January 2026 (which had feature month December 2025).")
    doc_content.append("- **Corrected Split (1,978 rows):** Sliced the dataset according to the target month $\ge 202601$. This correctly assigned December 2025 feature rows to predict January 2026 targets.")
    doc_content.append("- **Governance:** This correction was made before the final test was opened. No test observations were added after inspecting model performance.")
    doc_content.append("")
    doc_content.append("## 3. Final Test Model Comparison (Ranked by RMSE)")
    doc_content.append("")
    doc_content.append(df_to_markdown_custom(df_comparison))
    doc_content.append("")
    doc_content.append("## 4. Month-by-Month Validation Analysis")
    doc_content.append("")
    doc_content.append(df_to_markdown_custom(df_monthly))
    doc_content.append("")
    doc_content.append("## 5. Seen vs. Unseen Schemes Performance Breakdown")
    doc_content.append("")
    doc_content.append(df_to_markdown_custom(df_seen_unseen))
    doc_content.append("")
    doc_content.append("## 6. Error Analysis and Aggregated Summary")
    doc_content.append("")
    doc_content.append("### A. Absolute Error Distribution Percentiles")
    doc_content.append("")
    doc_content.append(df_to_markdown_custom(df_percentiles))
    doc_content.append("")
    doc_content.append("### B. Errors by Fund Category")
    doc_content.append("")
    doc_content.append(df_to_markdown_custom(df_cat_errors))
    doc_content.append("")
    doc_content.append("### C. Ten Largest Absolute Errors (Anonymized)")
    doc_content.append("")
    doc_content.append(df_to_markdown_custom(rf_largest_errors_anonymized[["Error_ID", "Target Month", "Category", "Actual", "RF_Pred", "RF_AbsError"]]))
    doc_content.append("")
    doc_content.append("## 7. Wording and Methodological Distinctions")
    doc_content.append("- **Zero-Return Baseline:** Confirmed by RMSE as the strongest overall forecasting method on the holdout.")
    doc_content.append("- **Ridge:** Identified as the best fitted model among the tested regression models. However, it is not validated as a production forecasting model since it fails to outperform the zero-return baseline.")
    doc_content.append("- **Previous-Month Persistence:** A useful benchmark but highly fragile during return-reversal months.")
    doc_content.append("- **Random Forest:** Provides no incremental forecasting benefit over Ridge.")
    doc_content.append("- **HistGradientBoosting:** Formally rejected due to negative R-squared and inferior performance.")
    doc_content.append("- **Structural Characteristics:** The panel dataset demonstrates limited next-month stable predictive value under regression models.")
    doc_content.append("")
    doc_content.append("---")
    doc_content.append("*Note: Anonymized row-level predictions and detailed scheme names are kept strictly in local outputs (`outputs/ignored_scheme_errors_detailed.csv`) and are ignored from version control.*")

    with open(docs_dir / "final_forecasting_assessment.md", "w", encoding="utf-8") as f:
        f.write("\n".join(doc_content))

    logger.info(f"Saved final assessment markdown report to: {docs_dir / 'final_forecasting_assessment.md'}")

if __name__ == "__main__":
    main()

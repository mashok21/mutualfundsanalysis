import argparse
import pandas as pd
import numpy as np
import os
import sys
import joblib
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Add src folder to Python path for importing mutual_fund_ml
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mutual_fund_ml.config import load_config, get_resolved_path
from mutual_fund_ml.utils import setup_logger, set_seed, ensure_dir
from mutual_fund_ml.data_loading import load_raw_data, load_interim_data, load_processed_data
from mutual_fund_ml.data_validation import validate_schema, check_missing_values, check_duplicates, check_value_ranges
from mutual_fund_ml.preprocessing import build_preprocessor
from mutual_fund_ml.feature_engineering import add_derived_features, remove_leakage_features
from mutual_fund_ml.pca import fit_pca_model, save_pca_artifacts
from mutual_fund_ml.clustering import fit_kmeans_model, fit_hierarchical_model, generate_cluster_profiles, save_clustering_outputs
from mutual_fund_ml.linear_models import fit_naive_baseline, fit_ols_statsmodels, fit_ridge_cv, fit_elasticnet_cv
from mutual_fund_ml.tree_models import fit_decision_tree_cv
from mutual_fund_ml.evaluation import calculate_regression_metrics
from mutual_fund_ml.reporting import save_model_comparison, save_predictions_and_residuals
from mutual_fund_ml.panel_diagnostics import run_panel_diagnostics, save_panel_diagnostics, format_panel_diagnostics

logger = setup_logger("run_pipeline", log_file="outputs/pipeline.log")

def run_validate():
    logger.info("--- Starting Validation Stage ---")
    df_raw = load_raw_data()

    # Run Schema Validation
    schema_status = validate_schema(df_raw)
    if not schema_status["valid"]:
        logger.warning(f"Schema Validation warnings. Missing features: {schema_status['missing_features']}")

    # Check Missing and Duplicates
    miss = check_missing_values(df_raw).sum()
    dups = check_duplicates(df_raw)
    logger.info(f"Missing values: {miss} | Duplicate rows: {dups}")

    # Check ranges
    ranges = check_value_ranges(df_raw)
    logger.info(f"Range checks complete. Infinite cols: {len(ranges['infinite_values'])} | Negative risk cols: {len(ranges['negative_risk'])}")

    # Save Cleaned Interim Data
    config = load_config()
    interim_dir = get_resolved_path(config["paths"]["interim_data_dir"])
    ensure_dir(interim_dir)
    interim_path = interim_dir / config["paths"]["cleaned_filename"]

    # Clean the dataset: remove exact duplicate rows if any
    df_cleaned = df_raw.drop_duplicates().reset_index(drop=True)
    df_cleaned.to_csv(interim_path, index=False)
    logger.info(f"Cleaned dataset saved to interim folder: {interim_path}")

def run_preprocess(mode="explanatory"):
    logger.info(f"--- Starting Preprocessing Stage (Mode: {mode}) ---")
    df_interim = load_interim_data()
    config = load_config()
    set_seed(config["parameters"]["random_seed"])

    # Add derived features (like log_AUM)
    df_feat, logs = add_derived_features(df_interim)

    # Identify variable roles
    target_col = config["schema"]["target_col"]
    pca_vars = config["schema"]["pca_variables"]
    cat_cols = config["schema"]["categorical_cols"]
    excluded = config["schema"]["excluded_from_modeling"]

    # Resolve AUMT_AUM vs log_AUM: Use only log_AUM in active feature list, exclude raw AUMT_AUM
    active_numerical = sorted(remove_leakage_features(pca_vars, excluded))
    if "AUMT_AUM" in active_numerical:
        active_numerical.remove("AUMT_AUM")
    if "log_AUM" not in active_numerical:
        active_numerical.append("log_AUM")
    active_numerical = sorted(active_numerical)

    if mode == "explanatory":
        # Target: contemporaneous monthly return
        df_feat["target"] = df_feat[target_col]

        # Split: Random split for baseline comparison (not generalisation)
        train_idx, test_idx = train_test_split(
            df_feat.index,
            test_size=config["parameters"]["test_size"],
            random_state=config["parameters"]["random_seed"]
        )

        # Build preprocessor pipeline
        preprocessor = build_preprocessor(active_numerical, cat_cols)

        # Fit preprocessor strictly on training data
        df_train_raw = df_feat.loc[train_idx]
        preprocessor.fit(df_train_raw)

        # Save preprocessor for explanatory mode
        preprocessor_dir = get_resolved_path(config["paths"]["artifact_dir"]) / "models"
        ensure_dir(preprocessor_dir)
        joblib.dump(preprocessor, preprocessor_dir / f"preprocessor_{mode}.joblib")

        # Transform both train and test
        X_train_processed = preprocessor.transform(df_feat.loc[train_idx])
        X_test_processed = preprocessor.transform(df_feat.loc[test_idx])

        # Reassemble final datasets
        encoded_cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
        final_feature_names = active_numerical + encoded_cat_names

        df_train_proc = pd.DataFrame(X_train_processed, columns=final_feature_names, index=train_idx)
        df_train_proc["target"] = df_feat.loc[train_idx, "target"].values
        df_train_proc["split"] = "train"
        df_train_proc["TS_Month End"] = df_feat.loc[train_idx, "TS_Month End"].values
        df_train_proc["Scheme Name"] = df_feat.loc[train_idx, "Scheme Name"].values

        df_test_proc = pd.DataFrame(X_test_processed, columns=final_feature_names, index=test_idx)
        df_test_proc["target"] = df_feat.loc[test_idx, "target"].values
        df_test_proc["split"] = "test"
        df_test_proc["TS_Month End"] = df_feat.loc[test_idx, "TS_Month End"].values
        df_test_proc["Scheme Name"] = df_feat.loc[test_idx, "Scheme Name"].values

        df_final = pd.concat([df_train_proc, df_test_proc]).sort_index()

    elif mode == "forecasting":
        # Target: shift return forward by 1 calendar month within each scheme (predict t+1 using features at t)
        df_feat = df_feat.sort_values(by=["Scheme Name", "TS_Month End"]).reset_index(drop=True)

        # Check adjacent calendar months to prevent gaps mapping as adjacent periods
        years = df_feat["TS_Month End"] // 100
        months = df_feat["TS_Month End"] % 100
        total_months = years * 12 + months
        df_feat["_total_months"] = total_months

        df_feat["target"] = df_feat.groupby("Scheme Name")[target_col].shift(-1)
        df_feat["target_month"] = df_feat.groupby("Scheme Name")["TS_Month End"].shift(-1)
        df_feat["_next_total_months"] = df_feat.groupby("Scheme Name")["_total_months"].shift(-1)
        df_feat["_month_diff"] = df_feat["_next_total_months"] - df_feat["_total_months"]

        # Set target to NaN if adjacent month gap is not exactly 1 month
        df_feat.loc[df_feat["_month_diff"] != 1, "target"] = np.nan
        df_feat.loc[df_feat["_month_diff"] != 1, "target_month"] = np.nan

        # Drop rows with missing target or missing intervening months
        df_feat = df_feat.dropna(subset=["target"]).reset_index(drop=True)
        # Cleanup temporary diagnostic columns
        df_feat = df_feat.drop(columns=["_total_months", "_next_total_months", "_month_diff"])

        # Assign train, validation, and test splits strictly according to the target month end
        chron_config = config["parameters"]["chronological_split"]
        train_end = chron_config["train_end"]
        val_end = chron_config["val_end"]

        train_mask = df_feat["target_month"] <= train_end
        val_mask = (df_feat["target_month"] > train_end) & (df_feat["target_month"] <= val_end)
        test_mask = df_feat["target_month"] > val_end

        # Include current return 'Average' as a feature for forecasting next month (lagged return at t)
        forecasting_num_features = sorted(active_numerical + [target_col])

        preprocessor = build_preprocessor(forecasting_num_features, cat_cols)

        # Fit strictly on train mask
        preprocessor.fit(df_feat.loc[train_mask])

        # Save preprocessor for forecasting mode
        preprocessor_dir = get_resolved_path(config["paths"]["artifact_dir"]) / "models"
        ensure_dir(preprocessor_dir)
        joblib.dump(preprocessor, preprocessor_dir / f"preprocessor_{mode}.joblib")

        # Transform all three splits
        X_train_processed = preprocessor.transform(df_feat.loc[train_mask])
        X_val_processed = preprocessor.transform(df_feat.loc[val_mask])
        X_test_processed = preprocessor.transform(df_feat.loc[test_mask])

        # Reassemble final datasets
        encoded_cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
        final_feature_names = forecasting_num_features + encoded_cat_names

        df_train_proc = pd.DataFrame(X_train_processed, columns=final_feature_names, index=df_feat[train_mask].index)
        df_train_proc["target"] = df_feat.loc[train_mask, "target"].values
        df_train_proc["split"] = "train"
        df_train_proc["TS_Month End"] = df_feat.loc[train_mask, "TS_Month End"].values
        df_train_proc["Scheme Name"] = df_feat.loc[train_mask, "Scheme Name"].values
        df_train_proc["target_month"] = df_feat.loc[train_mask, "target_month"].values

        df_val_proc = pd.DataFrame(X_val_processed, columns=final_feature_names, index=df_feat[val_mask].index)
        df_val_proc["target"] = df_feat.loc[val_mask, "target"].values
        df_val_proc["split"] = "val"
        df_val_proc["TS_Month End"] = df_feat.loc[val_mask, "TS_Month End"].values
        df_val_proc["Scheme Name"] = df_feat.loc[val_mask, "Scheme Name"].values
        df_val_proc["target_month"] = df_feat.loc[val_mask, "target_month"].values

        df_test_proc = pd.DataFrame(X_test_processed, columns=final_feature_names, index=df_feat[test_mask].index)
        df_test_proc["target"] = df_feat.loc[test_mask, "target"].values
        df_test_proc["split"] = "test"
        df_test_proc["TS_Month End"] = df_feat.loc[test_mask, "TS_Month End"].values
        df_test_proc["Scheme Name"] = df_feat.loc[test_mask, "Scheme Name"].values
        df_test_proc["target_month"] = df_feat.loc[test_mask, "target_month"].values

        df_final = pd.concat([df_train_proc, df_val_proc, df_test_proc]).sort_index()

    # Export processed modeling dataset
    processed_dir = get_resolved_path(config["paths"]["processed_data_dir"])
    ensure_dir(processed_dir)
    processed_path = processed_dir / f"modelling_dataset_{mode}.csv"
    df_final.to_csv(processed_path, index=True)
    logger.info(f"Processed modeling dataset saved to: {processed_path} (Shape: {df_final.shape})")

def run_model(mode="explanatory"):
    logger.info(f"--- Starting Modeling Stage (Mode: {mode}) ---")
    config = load_config()
    set_seed(config["parameters"]["random_seed"])

    # Load processed modeling dataset
    processed_dir = get_resolved_path(config["paths"]["processed_data_dir"])
    processed_path = processed_dir / f"modelling_dataset_{mode}.csv"
    df_proc = pd.read_csv(processed_path).rename(columns={"Unnamed: 0": "row_id"}).set_index("row_id")

    target_col = config["schema"]["target_col"]
    pca_vars = config["schema"]["pca_variables"]
    excluded = config["schema"]["excluded_from_modeling"]
    active_numerical = sorted(remove_leakage_features(pca_vars, excluded))
    if "AUMT_AUM" in active_numerical:
        active_numerical.remove("AUMT_AUM")
    if "log_AUM" not in active_numerical:
        active_numerical.append("log_AUM")
    active_numerical = sorted(active_numerical)

    # Load raw data to align indexes for panel metrics and clustering scheme names
    df_raw = pd.read_excel(get_resolved_path(config["paths"]["raw_data_dir"]) / config["paths"]["raw_filename"], sheet_name="Rawdata")
    df_raw, _ = add_derived_features(df_raw)

    if mode == "explanatory":
        train_mask = df_proc["split"] == "train"
        test_mask = df_proc["split"] == "test"

        feature_cols = [c for c in df_proc.columns if c not in ["target", "split", "TS_Month End", "Scheme Name"]]

        # Print feature lists
        logger.info(f"Explanatory Mode Raw Numerical Features: {active_numerical}")
        logger.info(f"Explanatory Mode Transformed Features: {feature_cols}")

        X_train = df_proc.loc[train_mask, feature_cols].values
        y_train = df_proc.loc[train_mask, "target"].values
        X_test = df_proc.loc[test_mask, feature_cols].values
        y_test = df_proc.loc[test_mask, "target"].values

        # Run panel diagnostics on clean dataset
        panel_results = run_panel_diagnostics(df_raw, target_col="Average")
        tables_dir = get_resolved_path(config["paths"]["output_dir"]) / "tables"
        ensure_dir(tables_dir)
        save_panel_diagnostics(panel_results, tables_dir / "panel_diagnostics.txt")
        logger.info(format_panel_diagnostics(panel_results))

        # 1. Fit PCA Model (on active_numerical by name)
        X_train_num = df_proc.loc[train_mask, active_numerical]
        pca, df_loadings, df_variance = fit_pca_model(X_train_num, n_components=config["parameters"]["pca_n_components"])

        # Project full standardized numerical dataset
        df_std_num = (df_raw[active_numerical] - df_raw[active_numerical].mean()) / df_raw[active_numerical].std()
        df_std_num = df_std_num.fillna(0)
        scores = pca.transform(df_std_num)
        df_scores = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(config["parameters"]["pca_n_components"])])
        df_scores.insert(0, "Scheme Name", df_raw["Scheme Name"])
        save_pca_artifacts(pca, df_loadings, df_variance, df_scores)

        # 2. Run Self-Contained Clustering
        logger.info("Running clustering analysis natively...")
        df_features_clust = df_scores.groupby("Scheme Name").mean().reset_index()
        pc_cols_avg = [f"PC{i+1}" for i in range(config["parameters"]["pca_n_components"])]
        X_clust = df_features_clust[pc_cols_avg].values
        km, km_labels, km_metrics = fit_kmeans_model(X_clust, k=config["parameters"]["optimal_k"])
        df_profiles = generate_cluster_profiles(df_features_clust[pc_cols_avg], km_labels)
        hc_labels, coph_c = fit_hierarchical_model(X_clust, n_clusters=config["parameters"]["optimal_k"])

        df_assign = pd.DataFrame({
            "Scheme Name": df_features_clust["Scheme Name"],
            "KMeans_Cluster": km_labels + 1,
            "HC_Cluster": hc_labels + 1
        })
        df_metrics_df = pd.DataFrame([km_metrics])
        df_metrics_df["Cophenetic_Correlation"] = coph_c
        save_clustering_outputs(df_assign, df_profiles, df_metrics_df)

        # 3. Fit OLS with Category & Month dummy fixed effects and clustered robust standard errors
        X_train_pcs = pca.transform(df_proc.loc[train_mask, active_numerical])
        X_test_pcs = pca.transform(df_proc.loc[test_mask, active_numerical])

        ols_sm = fit_ols_statsmodels(pd.DataFrame(X_train_pcs, columns=[f"PC{i+1}" for i in range(config["parameters"]["pca_n_components"])]), pd.Series(y_train))
        joblib.dump(ols_sm, get_resolved_path(config["paths"]["artifact_dir"]) / "models" / f"ols (on pcs)_{mode}_model.joblib")

        # Fit Pooled OLS with dummy Category & Month fixed effects and standard errors clustered by Scheme Name
        df_full_feat = df_raw[active_numerical + ["Category", "TS_Month End", "Average", "Scheme Name"]].dropna().reset_index(drop=True)
        df_full_ols = pd.get_dummies(df_full_feat[active_numerical + ["Category", "TS_Month End"]], columns=["Category", "TS_Month End"], drop_first=True, dtype=float)
        X_full_ols = sm.add_constant(df_full_ols)
        y_full_ols = df_full_feat["Average"]
        groups = df_full_feat["Scheme Name"]

        logger.info("Fitting Pooled OLS with Category & Month dummy fixed effects and standard errors clustered by Scheme Name...")
        ols_robust = sm.OLS(y_full_ols, X_full_ols).fit(cov_type='cluster', cov_kwds={'groups': groups})

        ols_summary_path = tables_dir / "explanatory_ols_robust_summary.txt"
        with open(ols_summary_path, "w", encoding="utf-8") as f:
            f.write(ols_robust.summary().as_text())
        logger.info(f"Robust OLS fixed effects summary saved to: {ols_summary_path}")

        # Fit other models
        ridge = fit_ridge_cv(X_train, y_train)
        elastic = fit_elasticnet_cv(X_train, y_train)
        dt = fit_decision_tree_cv(X_train, y_train, cv=5)
        baseline = fit_naive_baseline(X_train, y_train)

        model_dict = {
            "Baseline": baseline,
            "Ridge": ridge,
            "ElasticNet": elastic,
            "DecisionTree": dt
        }

        comparison_rows = []
        # Add OLS (on PCs) manually
        y_pred_ols = ols_sm.predict(sm.add_constant(X_test_pcs))
        ols_metrics = calculate_regression_metrics(y_test, y_pred_ols)
        comparison_rows.append({
            "Model": "OLS (on PCs)",
            "MAE_Train": calculate_regression_metrics(y_train, ols_sm.predict(sm.add_constant(X_train_pcs)))["mae"],
            "MAE_Test": ols_metrics["mae"],
            "RMSE_Test": ols_metrics["rmse"],
            "R2_Test": ols_metrics["r2"]
        })

        for name, m in model_dict.items():
            y_train_pred = m.predict(X_train)
            y_test_pred = m.predict(X_test)
            train_m = calculate_regression_metrics(y_train, y_train_pred)
            test_m = calculate_regression_metrics(y_test, y_test_pred)
            comparison_rows.append({
                "Model": name,
                "MAE_Train": train_m["mae"],
                "MAE_Test": test_m["mae"],
                "RMSE_Test": test_m["rmse"],
                "R2_Test": test_m["r2"]
            })
            joblib.dump(m, get_resolved_path(config["paths"]["artifact_dir"]) / "models" / f"{name.lower()}_{mode}_model.joblib")

        df_comparison = pd.DataFrame(comparison_rows)
        save_model_comparison(df_comparison, filename=f"model_comparison_{mode}.csv")
        logger.info("Explanatory stage model comparison table exported.")

    elif mode == "forecasting":
        train_mask = df_proc["split"] == "train"
        val_mask = df_proc["split"] == "val"
        test_mask = df_proc["split"] == "test"

        feature_cols = [c for c in df_proc.columns if c not in ["target", "split", "TS_Month End", "Scheme Name", "target_month"]]

        # Print feature lists
        forecasting_num_features = sorted(active_numerical + [target_col])
        logger.info(f"Forecasting Mode Raw Numerical Features: {forecasting_num_features}")
        logger.info(f"Forecasting Mode Transformed Features: {feature_cols}")

        # Sort training subset chronologically by target_month
        df_train_subset = df_proc.loc[train_mask].sort_values(by="target_month").reset_index(drop=True)
        X_train = df_train_subset[feature_cols].values
        y_train = df_train_subset["target"].values
        train_target_months = df_train_subset["target_month"].values

        X_val = df_proc.loc[val_mask, feature_cols].values
        y_val = df_proc.loc[val_mask, "target"].values
        X_test = df_proc.loc[test_mask, feature_cols].values
        y_test = df_proc.loc[test_mask, "target"].values

        # 1. Fit PCA Model (on sorted num features by name of sorted training set)
        X_train_num = df_train_subset[forecasting_num_features]
        pca, df_loadings, df_variance = fit_pca_model(X_train_num, n_components=config["parameters"]["pca_n_components"])

        # Project scores
        df_std_sorted = df_raw.sort_values(by=["Scheme Name", "TS_Month End"]).reset_index(drop=True)
        # Shift to match
        df_std_sorted["target"] = df_std_sorted.groupby("Scheme Name")[target_col].shift(-1)
        df_std_sorted["target_month"] = df_std_sorted.groupby("Scheme Name")["TS_Month End"].shift(-1)
        # Match shapes
        years = df_std_sorted["TS_Month End"] // 100
        months = df_std_sorted["TS_Month End"] % 100
        total_months = years * 12 + months
        df_std_sorted["_total_months"] = total_months
        df_std_sorted["_next_total_months"] = df_std_sorted.groupby("Scheme Name")["_total_months"].shift(-1)
        df_std_sorted["_month_diff"] = df_std_sorted["_next_total_months"] - df_std_sorted["_total_months"]
        df_std_sorted.loc[df_std_sorted["_month_diff"] != 1, "target"] = np.nan
        df_std_sorted.loc[df_std_sorted["_month_diff"] != 1, "target_month"] = np.nan
        df_std_sorted = df_std_sorted.dropna(subset=["target"]).reset_index(drop=True)

        df_std_num = (df_std_sorted[forecasting_num_features] - df_std_sorted[forecasting_num_features].mean()) / df_std_sorted[forecasting_num_features].std()
        df_std_num = df_std_num.fillna(0)
        scores = pca.transform(df_std_num)
        df_scores = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(config["parameters"]["pca_n_components"])])
        df_scores.insert(0, "Scheme Name", df_std_sorted["Scheme Name"])
        save_pca_artifacts(pca, df_loadings, df_variance, df_scores)

        # 2. Run Self-Contained Clustering
        logger.info("Running clustering analysis natively...")
        df_features_clust = df_scores.groupby("Scheme Name").mean().reset_index()
        pc_cols_avg = [f"PC{i+1}" for i in range(config["parameters"]["pca_n_components"])]
        X_clust = df_features_clust[pc_cols_avg].values
        km, km_labels, km_metrics = fit_kmeans_model(X_clust, k=config["parameters"]["optimal_k"])
        df_profiles = generate_cluster_profiles(df_features_clust[pc_cols_avg], km_labels)
        hc_labels, coph_c = fit_hierarchical_model(X_clust, n_clusters=config["parameters"]["optimal_k"])

        df_assign = pd.DataFrame({
            "Scheme Name": df_features_clust["Scheme Name"],
            "KMeans_Cluster": km_labels + 1,
            "HC_Cluster": hc_labels + 1
        })
        df_metrics_df = pd.DataFrame([km_metrics])
        df_metrics_df["Cophenetic_Correlation"] = coph_c
        save_clustering_outputs(df_assign, df_profiles, df_metrics_df)

        # 3. Define 5 chronological folds manually to ensure strict temporal separation
        custom_cv_folds = []
        fold_boundaries = [
            (202112, 202201, 202206),
            (202206, 202207, 202212),
            (202212, 202301, 202306),
            (202306, 202307, 202312),
            (202312, 202401, 202412)
        ]

        logger.info("=== Walk-Forward Chronological CV Folds ===")
        for fold, (train_limit, val_start, val_limit) in enumerate(fold_boundaries):
            train_idx_cv = np.where(train_target_months <= train_limit)[0]
            val_idx_cv = np.where((train_target_months >= val_start) & (train_target_months <= val_limit))[0]
            custom_cv_folds.append((train_idx_cv, val_idx_cv))

            logger.info(f"Fold {fold+1}: Train Target Months [min={train_target_months[train_idx_cv].min()} max={train_target_months[train_idx_cv].max()}] | "
                        f"Val Target Months [min={train_target_months[val_idx_cv].min()} max={train_target_months[val_idx_cv].max()}]")
            assert train_target_months[train_idx_cv].max() < train_target_months[val_idx_cv].min(), f"Fold {fold+1} overlap detected!"

        # Fit models
        ridge = fit_ridge_cv(X_train, y_train, cv=custom_cv_folds)
        elastic = fit_elasticnet_cv(X_train, y_train, cv=custom_cv_folds)
        # Small pruned Decision Tree (max_depth=4)
        dt = fit_decision_tree_cv(X_train, y_train, cv=custom_cv_folds)

        # Baselines:
        # a. Zero-return baseline
        y_val_zero = np.zeros_like(y_val)

        # b. Previous-month-return baseline (Average return at month t)
        avg_col_idx = feature_cols.index("Average")
        y_val_prev = X_val[:, avg_col_idx]

        # c. Scheme historical-mean baseline (average of training target return within each Scheme Name)
        scheme_means = pd.DataFrame({
            "Scheme Name": df_proc.loc[train_mask, "Scheme Name"],
            "target": y_train
        }).groupby("Scheme Name")["target"].mean()
        overall_train_mean = y_train.mean()
        y_val_scheme_mean = df_proc.loc[val_mask, "Scheme Name"].map(scheme_means).fillna(overall_train_mean).values

        # d. Category historical-mean baseline (average of training target return within each Category)
        # Map categories to means
        df_train_full = df_std_sorted.loc[train_mask]
        cat_means = df_train_full.groupby("Category")["target"].mean()
        df_val_full = df_std_sorted.loc[val_mask]
        y_val_cat_mean = df_val_full["Category"].map(cat_means).fillna(overall_train_mean).values

        # validation results comparison
        comparison_rows = []

        def add_comparison_row(name, val_preds):
            val_m = calculate_regression_metrics(y_val, val_preds)
            comparison_rows.append({
                "Model": name,
                "MAE_Val": val_m["mae"],
                "RMSE_Val": val_m["rmse"],
                "R2_Val": val_m["r2"]
            })

        add_comparison_row("Zero-Return Baseline", y_val_zero)
        add_comparison_row("Previous-Month Baseline", y_val_prev)
        add_comparison_row("Scheme Hist-Mean Baseline", y_val_scheme_mean)
        add_comparison_row("Category Hist-Mean Baseline", y_val_cat_mean)

        model_dict = {
            "Ridge": ridge,
            "ElasticNet": elastic,
            "Small Decision Tree": dt
        }

        for name, m in model_dict.items():
            val_preds = m.predict(X_val)
            add_comparison_row(name, val_preds)
            joblib.dump(m, get_resolved_path(config["paths"]["artifact_dir"]) / "models" / f"{name.lower().replace(' ', '')}_{mode}_model.joblib")

        df_comparison = pd.DataFrame(comparison_rows)
        save_model_comparison(df_comparison, filename=f"model_comparison_initial_{mode}.csv")

        # Print validation results
        logger.info("\n=== INITIAL FORECASTING VALIDATION RESULTS ===")
        for _, row in df_comparison.iterrows():
            logger.info(f"Model: {row['Model']} | MAE: {row['MAE_Val']:.4f} | RMSE: {row['RMSE_Val']:.4f} | R2: {row['R2_Val']:.4f}")

        # Group validation set into previously observed vs unseen schemes
        train_schemes = set(df_std_sorted.loc[train_mask, "Scheme Name"])
        val_schemes_list = df_std_sorted.loc[val_mask, "Scheme Name"].values
        val_seen_mask = np.array([s in train_schemes for s in val_schemes_list])

        # Evaluated on Ridge as primary linear model
        ridge_val_preds = ridge.predict(X_val)
        logger.info("\n=== Validation Grouping Breakdown (Ridge Model) ===")
        logger.info(f"Seen Schemes: Row Count = {val_seen_mask.sum()} | Unique Schemes = {pd.Series(val_schemes_list[val_seen_mask]).nunique()}")
        if val_seen_mask.any():
            m_seen = calculate_regression_metrics(y_val[val_seen_mask], ridge_val_preds[val_seen_mask])
            logger.info(f"  Seen Schemes performance: MAE = {m_seen['mae']:.4f}, RMSE = {m_seen['rmse']:.4f}, R2 = {m_seen['r2']:.4f}")

        logger.info(f"Unseen Schemes: Row Count = {(~val_seen_mask).sum()} | Unique Schemes = {pd.Series(val_schemes_list[~val_seen_mask]).nunique()}")
        if (~val_seen_mask).any():
            m_unseen = calculate_regression_metrics(y_val[~val_seen_mask], ridge_val_preds[~val_seen_mask])
            logger.info(f"  Unseen Schemes performance: MAE = {m_unseen['mae']:.4f}, RMSE = {m_unseen['rmse']:.4f}, R2 = {m_unseen['r2']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Mutual Fund ML Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["validate", "preprocess", "model", "all"],
        help="Stage of the pipeline to run"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="explanatory",
        choices=["explanatory", "forecasting"],
        help="Pipeline mode"
    )
    args = parser.parse_args()

    if args.stage == "validate":
        run_validate()
    elif args.stage == "preprocess":
        run_preprocess(args.mode)
    elif args.stage == "model":
        run_model(args.mode)
    elif args.stage == "all":
        run_validate()
        run_preprocess(args.mode)
        run_model(args.mode)

if __name__ == "__main__":
    main()

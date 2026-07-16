import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Add src folder to Python path for importing mutual_fund_ml
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mutual_fund_ml.config import load_config, get_resolved_path
from mutual_fund_ml.utils import setup_logger, set_seed, ensure_dir
from mutual_fund_ml.evaluation import calculate_regression_metrics

logger = setup_logger("ensemble_validation", log_file="outputs/ensemble_validation.log")

def main():
    logger.info("Starting Bounded Ensemble Validation stage...")
    config = load_config()
    set_seed(config["parameters"]["random_seed"])

    # Load processed modeling data
    processed_dir = get_resolved_path(config["paths"]["processed_data_dir"])
    processed_path = processed_dir / "modelling_dataset_forecasting.csv"

    if not processed_path.exists():
        logger.error(f"Processed dataset not found: {processed_path}. Run preprocessing first!")
        return

    df_proc = pd.read_csv(processed_path).rename(columns={"Unnamed: 0": "row_id"}).set_index("row_id")

    train_mask = df_proc["split"] == "train"
    val_mask = df_proc["split"] == "val"
    test_mask = df_proc["split"] == "test"

    feature_cols = [c for c in df_proc.columns if c not in ["target", "split", "TS_Month End", "Scheme Name", "target_month"]]

    # Sort training subset chronologically by target_month
    df_train_subset = df_proc.loc[train_mask].sort_values(by="target_month").reset_index(drop=True)
    X_train = df_train_subset[feature_cols].values
    y_train = df_train_subset["target"].values
    train_target_months = df_train_subset["target_month"].values

    X_val = df_proc.loc[val_mask, feature_cols].values
    y_val = df_proc.loc[val_mask, "target"].values
    val_target_months = df_proc.loc[val_mask, "target_month"].values

    # 1. Define custom chronological CV folds
    custom_cv_folds = []
    fold_boundaries = [
        (202112, 202201, 202206),
        (202206, 202207, 202212),
        (202212, 202301, 202306),
        (202306, 202307, 202312),
        (202312, 202401, 202412)
    ]

    for fold, (train_limit, val_start, val_limit) in enumerate(fold_boundaries):
        train_idx_cv = np.where(train_target_months <= train_limit)[0]
        val_idx_cv = np.where((train_target_months >= val_start) & (train_target_months <= val_limit))[0]
        custom_cv_folds.append((train_idx_cv, val_idx_cv))

    logger.info(f"Loaded training shape: {X_train.shape} | Validation shape: {X_val.shape}")

    # 2. Setup RandomizedSearchCV for Random Forest
    rf_param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 5, 6, 8],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 8]
    }

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base,
        param_distributions=rf_param_dist,
        n_iter=15,
        cv=custom_cv_folds,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1
    )

    logger.info("Tuning Random Forest Regressor (15 candidates)...")
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    rf_runtime = time.time() - start_time
    rf_best = rf_search.best_estimator_
    logger.info(f"Random Forest Tuning Complete (Best parameters: {rf_search.best_params_} | Runtime: {rf_runtime:.2f}s)")

    # 3. Setup RandomizedSearchCV for HistGradientBoosting
    hgb_param_dist = {
        "max_iter": [50, 100, 150],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_leaf": [5, 10, 20]
    }

    hgb_base = HistGradientBoostingRegressor(random_state=42)
    hgb_search = RandomizedSearchCV(
        hgb_base,
        param_distributions=hgb_param_dist,
        n_iter=15,
        cv=custom_cv_folds,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1
    )

    logger.info("Tuning HistGradientBoosting Regressor (15 candidates)...")
    start_time = time.time()
    hgb_search.fit(X_train, y_train)
    hgb_runtime = time.time() - start_time
    hgb_best = hgb_search.best_estimator_
    logger.info(f"HistGradientBoosting Tuning Complete (Best parameters: {hgb_search.best_params_} | Runtime: {hgb_runtime:.2f}s)")

    # Helper to calculate metrics on CV folds
    def get_fold_level_metrics(model, X, y, cv):
        fold_maes = []
        fold_rmses = []
        fold_r2s = []

        for train_idx, val_idx in cv:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_v, y_v = X[val_idx], y[val_idx]

            # Refit on fold train partition
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_tr, y_tr)
            preds = model_fold.predict(X_v)

            mae = mean_absolute_error(y_v, preds)
            rmse = np.sqrt(mean_squared_error(y_v, preds))
            r2 = r2_score(y_v, preds)

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)

        return fold_maes, fold_rmses, fold_r2s

    # Calculate Fold-Level Metrics
    logger.info("Calculating fold-level validation metrics for RF best estimator...")
    rf_maes, rf_rmses, rf_r2s = get_fold_level_metrics(rf_best, X_train, y_train, custom_cv_folds)

    logger.info("Calculating fold-level validation metrics for HGB best estimator...")
    hgb_maes, hgb_rmses, hgb_r2s = get_fold_level_metrics(hgb_best, X_train, y_train, custom_cv_folds)

    # Load Ridge model from explanatory stage as baseline check
    models_dir = get_resolved_path(config["paths"]["artifact_dir"]) / "models"
    ridge_path = models_dir / "ridge_forecasting_model.joblib"
    if ridge_path.exists():
        ridge = joblib.load(ridge_path)
    else:
        # Fallback fit
        logger.warning("Ridge model not found in artifacts, fitting a temporary one...")
        ridge = fit_ridge_cv(X_train, y_train, cv=custom_cv_folds)

    # Save the new ensemble model checkpoints
    joblib.dump(rf_best, models_dir / "randomforest_forecasting_model.joblib")
    joblib.dump(hgb_best, models_dir / "gradientboosting_forecasting_model.joblib")

    # 4. Evaluate Validation Set (2025) Metrics
    # Predictions
    rf_val_preds = rf_best.predict(X_val)
    hgb_val_preds = hgb_best.predict(X_val)
    ridge_val_preds = ridge.predict(X_val)

    # Training performance for gap check
    rf_train_preds = rf_best.predict(X_train)
    hgb_train_preds = hgb_best.predict(X_train)

    # Baselines: Previous-month persistence
    avg_col_idx = feature_cols.index("Average")
    y_val_prev = X_val[:, avg_col_idx]

    # Validation Overall Metrics
    rf_val_metrics = calculate_regression_metrics(y_val, rf_val_preds)
    hgb_val_metrics = calculate_regression_metrics(y_val, hgb_val_preds)
    ridge_val_metrics = calculate_regression_metrics(y_val, ridge_val_preds)
    prev_val_metrics = calculate_regression_metrics(y_val, y_val_prev)

    # Training Metrics for train-val gap check
    rf_train_metrics = calculate_regression_metrics(y_train, rf_train_preds)
    hgb_train_metrics = calculate_regression_metrics(y_train, hgb_train_preds)

    # 5. Seen vs Unseen Validation Groups Analysis
    # Load raw to match category distribution and seen/unseen
    df_raw = pd.read_excel(get_resolved_path(config["paths"]["raw_data_dir"]) / config["paths"]["raw_filename"], sheet_name="Rawdata")
    df_raw = df_raw.sort_values(by=["Scheme Name", "TS_Month End"]).reset_index(drop=True)
    df_raw["target"] = df_raw.groupby("Scheme Name")["Average"].shift(-1)
    df_raw["target_month"] = df_raw.groupby("Scheme Name")["TS_Month End"].shift(-1)

    years = df_raw["TS_Month End"] // 100
    months = df_raw["TS_Month End"] % 100
    total_months = years * 12 + months
    df_raw["_total_months"] = total_months
    df_raw["_next_total_months"] = df_raw.groupby("Scheme Name")["_total_months"].shift(-1)
    df_raw["_month_diff"] = df_raw["_next_total_months"] - df_raw["_total_months"]
    df_raw.loc[df_raw["_month_diff"] != 1, "target"] = np.nan
    df_raw.loc[df_raw["_month_diff"] != 1, "target_month"] = np.nan
    df_raw = df_raw.dropna(subset=["target"]).reset_index(drop=True)

    # Set splits same as pipeline
    train_mask_raw = df_raw["target_month"] <= config["parameters"]["chronological_split"]["train_end"]
    val_mask_raw = (df_raw["target_month"] > config["parameters"]["chronological_split"]["train_end"]) & (df_raw["target_month"] <= config["parameters"]["chronological_split"]["val_end"])

    train_schemes = set(df_raw.loc[train_mask_raw, "Scheme Name"])
    val_schemes_list = df_raw.loc[val_mask_raw, "Scheme Name"].values
    val_categories = df_raw.loc[val_mask_raw, "Category"].values
    val_months = df_raw.loc[val_mask_raw, "target_month"].values

    val_seen_mask = np.array([s in train_schemes for s in val_schemes_list])

    # 6. Feature Importances & Permutation Importance
    # Impurity feature importance (Random Forest)
    rf_importances = rf_best.feature_importances_
    df_rf_imp = pd.DataFrame({"Feature": feature_cols, "Impurity_Importance": rf_importances}).sort_values(by="Impurity_Importance", ascending=False)

    # Permutation Importances on validation set
    logger.info("Calculating permutation importances on validation set (this may take up to 2 minutes)...")
    perm_rf = permutation_importance(rf_best, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
    perm_hgb = permutation_importance(hgb_best, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)

    df_perm_rf = pd.DataFrame({"Feature": feature_cols, "RF_Perm_Importance": perm_rf.importances_mean}).sort_values(by="RF_Perm_Importance", ascending=False)
    df_perm_hgb = pd.DataFrame({"Feature": feature_cols, "HGB_Perm_Importance": perm_hgb.importances_mean}).sort_values(by="HGB_Perm_Importance", ascending=False)

    # 7. Formulate Output Report
    report = []
    report.append("======================================================================")
    report.append("MUTUAL FUND ANALYSIS - BOUNDED ENSEMBLE VALIDATION REPORT")
    report.append("======================================================================\n")

    report.append("1. FOLD-LEVEL METRICS & STABILITY")
    report.append("---------------------------------")

    def add_model_fold_summary(name, maes, rmses, r2s):
        report.append(f"Model: {name}")
        report.append(f"  Fold 1: MAE={maes[0]:.4f} | RMSE={rmses[0]:.4f} | R2={r2s[0]:.4f}")
        report.append(f"  Fold 2: MAE={maes[1]:.4f} | RMSE={rmses[1]:.4f} | R2={r2s[1]:.4f}")
        report.append(f"  Fold 3: MAE={maes[2]:.4f} | RMSE={rmses[2]:.4f} | R2={r2s[2]:.4f}")
        report.append(f"  Fold 4: MAE={maes[3]:.4f} | RMSE={rmses[3]:.4f} | R2={r2s[3]:.4f}")
        report.append(f"  Fold 5: MAE={maes[4]:.4f} | RMSE={rmses[4]:.4f} | R2={r2s[4]:.4f}")
        report.append(f"  Mean:   MAE={np.mean(maes):.4f} (std={np.std(maes):.4f}) | RMSE={np.mean(rmses):.4f} (std={np.std(rmses):.4f}) | R2={np.mean(r2s):.4f} (std={np.std(r2s):.4f})\n")

    add_model_fold_summary("Random Forest", rf_maes, rf_rmses, rf_r2s)
    add_model_fold_summary("Hist Gradient Boosting", hgb_maes, hgb_rmses, hgb_r2s)

    report.append("2. 2025 VALIDATION METRICS & PERSISTENCE COMPARISON")
    report.append("--------------------------------------------------")
    report.append(f"{'Model':<30} | {'MAE_Val':<10} | {'RMSE_Val':<10} | {'R2_Val':<10}")
    report.append("-" * 72)
    report.append(f"{'Previous-Month Persistence':<30} | {prev_val_metrics['mae']:<10.4f} | {prev_val_metrics['rmse']:<10.4f} | {prev_val_metrics['r2']:<10.4f}")
    report.append(f"{'Ridge':<30} | {ridge_val_metrics['mae']:<10.4f} | {ridge_val_metrics['rmse']:<10.4f} | {ridge_val_metrics['r2']:<10.4f}")
    report.append(f"{'Random Forest':<30} | {rf_val_metrics['mae']:<10.4f} | {rf_val_metrics['rmse']:<10.4f} | {rf_val_metrics['r2']:<10.4f}")
    report.append(f"{'Hist Gradient Boosting':<30} | {hgb_val_metrics['mae']:<10.4f} | {hgb_val_metrics['rmse']:<10.4f} | {hgb_val_metrics['r2']:<10.4f}")
    report.append("")

    # Persistence baseline beats checks
    def check_beats(model_name, metrics):
        mae_beats = metrics["mae"] < prev_val_metrics["mae"]
        rmse_beats = metrics["rmse"] < prev_val_metrics["rmse"]
        r2_beats = metrics["r2"] > prev_val_metrics["r2"]
        report.append(f"Does {model_name} beat Previous-Month Persistence?")
        report.append(f"  MAE:  {'YES' if mae_beats else 'NO'} ({metrics['mae']:.4f} vs {prev_val_metrics['mae']:.4f})")
        report.append(f"  RMSE: {'YES' if rmse_beats else 'NO'} ({metrics['rmse']:.4f} vs {prev_val_metrics['rmse']:.4f})")
        report.append(f"  R2:   {'YES' if r2_beats else 'NO'} ({metrics['r2']:.4f} vs {prev_val_metrics['r2']:.4f})\n")

    check_beats("Random Forest", rf_val_metrics)
    check_beats("Hist Gradient Boosting", hgb_val_metrics)

    report.append("3. TRAINING-VERSUS-VALIDATION GAP (OVERFITTING CHECK)")
    report.append("-----------------------------------------------------")
    report.append(f"Random Forest:")
    report.append(f"  Train MAE: {rf_train_metrics['mae']:.4f} | Val MAE: {rf_val_metrics['mae']:.4f} | Gap: {rf_val_metrics['mae'] - rf_train_metrics['mae']:.4f}")
    report.append(f"  Train R2:  {rf_train_metrics['r2']:.4f} | Val R2:  {rf_val_metrics['r2']:.4f} | Gap: {rf_train_metrics['r2'] - rf_val_metrics['r2']:.4f}")
    report.append(f"Hist Gradient Boosting:")
    report.append(f"  Train MAE: {hgb_train_metrics['mae']:.4f} | Val MAE: {hgb_val_metrics['mae']:.4f} | Gap: {hgb_val_metrics['mae'] - hgb_train_metrics['mae']:.4f}")
    report.append(f"  Train R2:  {hgb_train_metrics['r2']:.4f} | Val R2:  {hgb_val_metrics['r2']:.4f} | Gap: {hgb_train_metrics['r2'] - hgb_val_metrics['r2']:.4f}\n")

    report.append("4. SEEN VERSUS UNSEEN SCHEME PERFORMANCE BREAKDOWN (VAL)")
    report.append("--------------------------------------------------------")

    def format_split_breakdown(group_name, mask, X_g, y_g, category_g, months_g, schemes_g):
        g_rows = mask.sum()
        g_schemes = len(set(schemes_g[mask]))
        g_mean = np.mean(y_g[mask])
        g_std = np.std(y_g[mask])

        # Persistence Baseline
        g_prev = X_g[mask, avg_col_idx]
        g_prev_mae = mean_absolute_error(y_g[mask], g_prev)
        g_prev_rmse = np.sqrt(mean_squared_error(y_g[mask], g_prev))
        g_prev_r2 = r2_score(y_g[mask], g_prev)

        # Model predictions
        g_rf = rf_val_preds[mask]
        g_rf_mae = mean_absolute_error(y_g[mask], g_rf)
        g_rf_rmse = np.sqrt(mean_squared_error(y_g[mask], g_rf))
        g_rf_r2 = r2_score(y_g[mask], g_rf)

        g_hgb = hgb_val_preds[mask]
        g_hgb_mae = mean_absolute_error(y_g[mask], g_hgb)
        g_hgb_rmse = np.sqrt(mean_squared_error(y_g[mask], g_hgb))
        g_hgb_r2 = r2_score(y_g[mask], g_hgb)

        # Category distribution
        cat_counts = pd.Series(category_g[mask]).value_counts()
        cat_dist_str = ", ".join([f"{k}: {v}" for k, v in cat_counts.items()])

        # Monthly counts
        month_counts = pd.Series(months_g[mask]).value_counts().sort_index()
        month_counts_str = ", ".join([f"{k}: {v}" for k, v in month_counts.items()])

        report.append(f"Group: {group_name}")
        report.append(f"  Observations:  {g_rows:,} rows | {g_schemes} unique schemes")
        report.append(f"  Target Stats:  Mean={g_mean:.4f} | StdDev={g_std:.4f}")
        report.append(f"  Persistence Baseline: MAE={g_prev_mae:.4f} | RMSE={g_prev_rmse:.4f} | R2={g_prev_r2:.4f}")
        report.append(f"  Random Forest:        MAE={g_rf_mae:.4f} | RMSE={g_rf_rmse:.4f} | R2={g_rf_r2:.4f}")
        report.append(f"  Hist Gradient Boosting:MAE={g_hgb_mae:.4f} | RMSE={g_hgb_rmse:.4f} | R2={g_hgb_r2:.4f}")
        report.append(f"  Category Distribution: {cat_dist_str}")
        report.append(f"  Monthly Observations:  {month_counts_str}\n")

    format_split_breakdown("SEEN SCHEMES (in training)", val_seen_mask, X_val, y_val, val_categories, val_months, val_schemes_list)
    format_split_breakdown("UNSEEN SCHEMES (newly appeared in validation)", ~val_seen_mask, X_val, y_val, val_categories, val_months, val_schemes_list)

    report.append("5. TOP FEATURE IMPORTANCE & PERMUTATION IMPORTANCE")
    report.append("-------------------------------------------------")
    report.append("Random Forest - Top 10 Impurity Importance:")
    for i, r in df_rf_imp.head(10).iterrows():
        report.append(f"  {r['Feature']:<35}: {r['Impurity_Importance']:.4f}")
    report.append("")

    report.append("Random Forest - Top 10 Permutation Importance:")
    for i, r in df_perm_rf.head(10).iterrows():
        report.append(f"  {r['Feature']:<35}: {r['RF_Perm_Importance']:.4f}")
    report.append("")

    report.append("Hist Gradient Boosting - Top 10 Permutation Importance:")
    for i, r in df_perm_hgb.head(10).iterrows():
        report.append(f"  {r['Feature']:<35}: {r['HGB_Perm_Importance']:.4f}")
    report.append("")

    report.append("6. BEST TUNED PARAMETERS")
    report.append("------------------------")
    report.append(f"Random Forest Best Parameters:         {rf_search.best_params_}")
    report.append(f"Hist Gradient Boosting Best Params:    {hgb_search.best_params_}")
    report.append("======================================================================\n")

    report_str = "\n".join(report)
    print(report_str)

    # Save report
    tables_dir = get_resolved_path(config["paths"]["output_dir"]) / "tables"
    ensure_dir(tables_dir)
    with open(tables_dir / "ensemble_validation_results.txt", "w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info(f"Ensemble validation results report exported to: {tables_dir / 'ensemble_validation_results.txt'}")

if __name__ == "__main__":
    main()

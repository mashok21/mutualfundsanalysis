import pandas as pd
from pathlib import Path
from .config import load_config, get_resolved_path
from .utils import setup_logger

logger = setup_logger("reporting")

def get_tables_path() -> Path:
    """Returns the absolute path to the outputs/tables folder."""
    config = load_config()
    out_dir = config["paths"]["output_dir"]
    path = get_resolved_path(out_dir) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_model_comparison(df_comparison: pd.DataFrame, filename: str = "model_comparison.csv") -> Path:
    """Saves a model comparison table to the tables folder in both CSV and Excel formats."""
    path_dir = get_tables_path()

    # Save CSV
    csv_path = path_dir / filename
    df_comparison.to_csv(csv_path, index=False)
    logger.info(f"Model comparison table saved to CSV: {csv_path}")

    # Save Excel
    xlsx_path = path_dir / filename.replace(".csv", ".xlsx")
    try:
        df_comparison.to_excel(xlsx_path, index=False)
        logger.info(f"Model comparison table saved to Excel: {xlsx_path}")
    except Exception as e:
        logger.warning(f"Could not save comparison to Excel: {e}")

    return csv_path

def save_predictions_and_residuals(df_preds: pd.DataFrame, stage_name: str = "supervised_models") -> Path:
    """Saves predictions and residuals to output directory for later evaluation."""
    config = load_config()
    out_dir = config["paths"]["output_dir"]
    dest_dir = get_resolved_path(out_dir) / stage_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    path = dest_dir / "predictions.csv"
    df_preds.to_csv(path, index=False)
    logger.info(f"Predictions and residuals exported to: {path}")
    return path

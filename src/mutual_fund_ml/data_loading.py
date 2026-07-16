import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .config import load_config, get_resolved_path
from .utils import setup_logger

logger = setup_logger("data_loading")

def get_raw_data_path() -> Path:
    """Returns the path to the raw Excel dataset."""
    config = load_config()
    raw_dir = config["paths"]["raw_data_dir"]
    filename = config["paths"]["raw_filename"]
    return get_resolved_path(raw_dir) / filename

def get_interim_data_path() -> Path:
    """Returns the path to the interim cleaned dataset."""
    config = load_config()
    interim_dir = config["paths"]["interim_data_dir"]
    filename = config["paths"]["cleaned_filename"]
    return get_resolved_path(interim_dir) / filename

def get_processed_data_path() -> Path:
    """Returns the path to the processed modelling dataset."""
    config = load_config()
    processed_dir = config["paths"]["processed_data_dir"]
    filename = config["paths"]["processed_filename"]
    return get_resolved_path(processed_dir) / filename

def load_raw_data() -> pd.DataFrame:
    """Loads the raw mutual fund Excel sheet."""
    path = get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    logger.info(f"Loading raw data from: {path}")
    # Read the sheet named 'Rawdata'
    return pd.read_excel(path, sheet_name="Rawdata")

def load_interim_data() -> pd.DataFrame:
    """Loads the cleaned interim CSV dataset."""
    path = get_interim_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Interim data file not found at: {path}")
    logger.info(f"Loading interim data from: {path}")
    return pd.read_csv(path)

def load_processed_data() -> pd.DataFrame:
    """Loads the final model-ready CSV dataset."""
    path = get_processed_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found at: {path}")
    logger.info(f"Loading processed data from: {path}")
    return pd.read_csv(path)

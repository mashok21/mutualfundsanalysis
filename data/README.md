# Data Folder Documentation

This directory contains all data inputs and outputs used in the project, separated into three distinct lifecycles.

---

## 1. Directory Breakdown

### `data/raw/`
*   **Immutability Rule:** This directory holds the raw, unedited source files (specifically `20260622_Scheme_MF_PCA_Rawdata.xlsx` containing the 27,107 observations).
*   **Safety Warning:** **NEVER edit, modify, or overwrite any file inside this folder.** All cleanups must be performed programmatically and saved in subsequent directories.

### `data/interim/`
*   **Cleaned Inputs:** Holds datasets that have undergone structural validation and general cleaning (handling column names, removing exact duplicates, correcting data types) but have **not** been transformed, scaled, or split.
*   **Key File:** `cleaned_dataset.csv`

### `data/processed/`
*   **Modeling-Ready Data:** Holds the final numerical features, normalized values, target variables, and splitting metadata.
*   **Key File:** `modelling_dataset.csv`

---

## 2. Generation Process
Data flows through the following pipeline:
1. `data/raw/` $\rightarrow$ Checked and clean duplicates/types $\rightarrow$ Saved to `data/interim/cleaned_dataset.csv` via `00_data_audit.ipynb`.
2. `data/interim/cleaned_dataset.csv` $\rightarrow$ Preprocessed and scaled $\rightarrow$ Saved to `data/processed/modelling_dataset.csv` via `02_preprocessing_and_feature_engineering.ipynb`.

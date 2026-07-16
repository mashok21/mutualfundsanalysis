# Mutual Fund Analysis

A modular Python project for analyzing mutual fund scheme characteristics.
The project covers descriptive analysis, structural PCA, unsupervised
clustering, contemporaneous explanatory analysis, and a governed next-month
forecasting exercise on scheme-level panel data.

---

## Data

The underlying dataset is proprietary and is **not included** in this
repository. `data/raw/`, `data/interim/`, and `data/processed/` are present
only as empty, git-ignored directories; see `data/README.md` for a
description of their intended contents and the data-processing lifecycle.

To use this project you must supply your own compatible dataset and update
`config/config.yaml` accordingly.

---

## Repository contents

This initial publication is intentionally conservative. It contains:

```
.
├── config/
│   └── config.yaml            # Centralized project parameters
├── data/
│   └── README.md               # Description of the data lifecycle (no data)
├── docs/
│   └── linux_migration_report.md
├── src/
│   └── mutual_fund_ml/          # Reusable pipeline package
├── tests/                       # Automated unit and integration tests
├── run_pipeline.py              # Command-line pipeline orchestrator
├── run_ensemble_validation.py   # Ensemble-model validation utility
├── run_final_evaluation.py      # Final-holdout evaluation entry point
├── run_tests.py                 # Standalone test runner
├── pyproject.toml
└── requirements.txt
```

Detailed analysis notebooks and extended methodology/report documents are
retained locally pending a separate confidentiality and methodology review,
and are not part of this initial commit.

---

## Installation (Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running tests

```bash
python -m pytest -v
```

## Safe validation

The pipeline supports a validation-only stage that checks configuration,
schema, and data-loading wiring without requiring the final holdout to be
opened:

```bash
python run_pipeline.py --stage validate --mode forecasting
```

`run_final_evaluation.py` executes the project's final forecasting holdout
evaluation. **It must not be re-run for model selection or tuning purposes**
— see [Forecasting governance](#forecasting-governance) below.

---

## Forecasting governance

This project enforces a strict, frozen forecasting protocol:

- The final forecasting holdout period has already been evaluated once and
  is now **closed**. It must not be reopened for model development, tuning,
  or search.
- **Frozen conclusion:** no fitted model demonstrated incremental
  next-month forecasting value over the strongest simple baseline, the
  zero-return forecast, on the final holdout.
- Ridge was the best-performing fitted model tested, but it is **not**
  approved as a production forecasting model.
- Splits are chronological, not shuffled, to avoid look-ahead leakage.

The primary demonstrated value of this project lies in its descriptive
analysis, fund characterisation, structural PCA, clustering, and
contemporaneous (non-forecasting) explanatory analysis — not in next-month
return prediction.

---

## Project origin

This Linux-native repository was migrated from a prior local working copy.
The former working copy is retained only as a read-only backup and is not
part of this repository's history.

# Prospective Data Ingestion Protocol

This document governs how future raw dataset drops (e.g. a May 2026 or June
2026 extract) are received, validated, and admitted into the pipeline. It
exists to protect two things: the integrity of historical raw files, and the
frozen January-April 2026 forecasting holdout (`target_month` 202601-202604,
per `config/config.yaml` `parameters.chronological_split`).

No new raw drop may be processed by `run_pipeline.py` or any modeling script
until it has passed every step below.

---

## 1. Purity Rules

- Historical raw files under `data/raw/` are **never overwritten, edited, or
  deleted**. Each new drop is landed as a new, separate file.
- New files are versioned with a `YYYYMMDD` date prefix, following the
  existing convention already used for the current raw extract (e.g.
  `20260622_Scheme_MF_PCA_Rawdata.xlsx`). The date is the extract date, not
  the ingestion date.
- `data/raw/` is strictly append-only. If a correction to a prior drop is
  required, it arrives as a new dated file with a note in the ingestion log
  (Step 10) explaining the correction — the original file is untouched.

## 2. Diagnostic Checks (recorded before anything else runs)

For every new file, record and store locally (never overwrite a prior
record):

- SHA-256 hash of the file
- File size in bytes
- Sheet names and sheet count (for `.xlsx` sources)
- Row count and column count per sheet
- Count of unique `Scheme Name` values and unique `TS_Month End` values

These diagnostics are the baseline evidence that the file landed intact and
establish what changed relative to the prior drop.

## 3. Schema Drift Detection

Compare the new file's schema against the currently active raw file and flag
any of the following before proceeding:

- Added, removed, or renamed columns
- Changed dtypes on existing columns (e.g. a numeric column arriving as text)
- Material shifts in missingness rate per column (a large jump suggests an
  upstream extraction problem, not a real data pattern)
- Any change to the definition or units of return-derived variables
  (`Average`, or any column feeding `target_col`) — these require explicit
  sign-off before use, since a silent redefinition would corrupt both the
  explanatory and forecasting tracks without raising an error downstream.

## 4. Analytical Allocation Rules

Before any model is fit or any exploratory notebook is pointed at the new
data, its role must be explicitly declared as exactly one of:

1. **New descriptive context** — extends panel diagnostics, EDA, and
   clustering only. No forecasting or explanatory regression use.
2. **Explanatory update** — the contemporaneous (`Average`) explanatory OLS
   track may be refit including the new months.
3. **Prospective forward holdout** — the new month(s) are reserved,
   untouched by any fitting or tuning, for a future locked evaluation event
   analogous to the January-April 2026 holdout.

This declaration is made and recorded **before** any code touches the new
data. Retroactively reclassifying a month after inspecting model performance
on it is prohibited.

## 5. Frozen Holdout Isolation (hard constraint)

- The January-April 2026 holdout (`target_month` 202601-202604) is
  permanently frozen. It has already been scored once by
  `run_final_evaluation.py`. It must never be added back into a training,
  validation, or hyperparameter-tuning set, regardless of how future data
  arrives.
- New months (202605 onward) are new prospective data under Step 4's
  allocation rules — they are a separate concern from the already-consumed
  holdout and do not reopen it.
- Any script that fits or tunes a model must derive its training data using
  a mask strictly bounded by `parameters.chronological_split.val_end`
  (currently `202512`) or earlier. Data at or beyond
  `parameters.chronological_split.test_end` may only be used with
  `.predict()`, never `.fit()`.

## 6. Panel Key Integrity

The composite key `Scheme Name` + `TS_Month End` must be unique across the
full raw panel after a new drop is merged in for analysis. A duplicate key
indicates either a re-extraction of an existing period (which should have
arrived as a correction under Step 1) or a data-vendor error, and must be
resolved before ingestion proceeds.

## 7. Ingestion Workflow

1. Land the new file in `data/raw/` under its dated filename (Step 1).
2. Run diagnostics (Step 2) and store the record locally.
3. Run schema drift detection (Step 3) against the previously active raw
   file.
4. Declare the analytical allocation (Step 4) in writing before touching
   pipeline code.
5. Run the automated ingestion guard test suite
   (`tests/test_data_ingestion_guard.py`) to confirm raw immutability, panel
   key uniqueness, and holdout isolation hold before any model touches the
   merged data.
6. Only after all of the above pass does the new data become eligible for
   the analytical use declared in Step 4.

## 8. Automated Enforcement

`tests/test_data_ingestion_guard.py` provides machine-checked guarantees for
the invariants in Steps 1, 5, and 6 (raw file immutability, holdout
never-fit, panel key uniqueness). These tests run as part of the standard
`pytest` suite and must pass before any pipeline change involving new data
is considered complete.

## 9. Local-Only Scope

Raw data files, ingestion diagnostic records, and schema drift logs remain
local artifacts under `data/raw/` and are not committed to the public
repository, consistent with the existing `.gitignore` quarantine of
proprietary data. Only this protocol document and the automated guard test
are part of the public core source tree.

## 10. Audit Trail

Every ingestion event (a new file landing, its diagnostics, its schema drift
findings, and its analytical allocation declaration) is recorded locally in
an ingestion log kept alongside `data/raw/`. The log is append-only, mirrors
the purity rule in Step 1, and is the reference record if a later dispute
arises about when a given month was first admitted and for what purpose.

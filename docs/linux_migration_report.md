# Linux Migration Report

## Project

- Project: `mutualfundsanalysis`
- Active working directory: `~/projects/mutualfundsanalysis`
- Remote repository: `mashok21/mutualfundsanalysis`
- Linux environment: Ubuntu 24.04 LTS on WSL 2
- Python environment: Linux-native virtual environment under `.venv`

## Migration scope

The validated Windows workspace was copied into a Linux-native WSL directory.
The Linux directory is now the sole active editable working copy.

The former Windows workspace remains intact as a temporary read-only backup.

## Files migrated

The migration retained:

- Source code
- Configuration
- Clean source notebooks
- Tests
- Documentation
- Local proprietary data
- Local generated outputs
- Local model and preprocessing artifacts
- Historical reconciliation material

## Initial publication scope

The initial public commit is intentionally conservative. It contains only
core source code, configuration, tests, packaging files, entry-point
scripts, a data-directory placeholder README, and this migration report.

Detailed analysis notebooks and extended documentation (analysis plan, data
dictionary, feature provenance, methodology, model card, validation report,
and the final forecasting assessment) are retained locally pending a
separate confidentiality and methodology-alignment review. They are
quarantined from the initial commit via `.git/info/exclude` and are not
part of this repository's public history.

## Files excluded from GitHub

The following categories remain locally available but are excluded from Git:

- Raw, interim and processed proprietary datasets
- Executed notebooks
- Source notebooks (quarantined pending methodology review)
- Extended documentation and analytical reports (quarantined pending review)
- Scheme-level PCA scores and cluster assignments
- Row-level predictions and detailed error reports
- Serialized models and preprocessors
- Generated analytical outputs
- Private migration logs and hashes
- Superseded baseline-run evidence
- Original legacy notebooks pending separate sanitisation

## Linux validation

- Project package imported successfully from the repository root.
- Python dependency consistency check passed.
- Thirteen pytest tests passed on Linux.
- Tests include leakage, chronological splitting, exact adjacent-month
  forecasting and deterministic PCA-input selection checks.
- The forecasting validation-stage smoke test passed.
- Generated validation files were written only to ignored locations.
- Raw-data hashes matched before and after copying and validation.
- Proprietary files remained physically present but untracked.
- Source notebooks contained no stored execution outputs.

## Methodological governance

The January-April 2026 forecasting holdout remains consumed and frozen.

The migration did not:

- Retune any forecasting model
- Reopen the final holdout for model selection
- Add XGBoost
- Change the forecasting feature set
- Change the target or chronological split
- Reinstate the superseded 99.79% result as forecasting evidence

## GitHub status

The public repository was confirmed empty before the initial staging operation.

The initial commit was pushed to the public repository. Local HEAD and
origin/main were verified identical immediately after the push. The
published tree was independently re-checked and confirmed to contain none
of the excluded categories listed above.

## Final verdict

The initial public GitHub migration completed successfully. The repository
was populated from the Linux-native WSL workspace. The initial publication
was deliberately conservative. Local HEAD and origin/main matched after the
first push.

Proprietary data, generated outputs, models and private audit evidence
remained local and ignored. Notebooks and detailed analytical reports
remain locally preserved and quarantined pending methodology review. The
Windows backup remained untouched.

No forecasting model was retuned. The January-April 2026 final holdout was
not reused.

PASS

# Audit Finding: PCA & Forecasting Architecture Reconciliation

**Status:** Final
**Date:** 2026-07-16
**Scope:** Reconciliation of the legacy spreadsheet-based PCA/clustering analysis (Stonelink, prepared by Nandhini Harikrishnan, received by email 2026-07-15) against the current `mutual_fund_ml` Python codebase.

---

## 1. Executive Finding Summary

The project has transitioned from an unconstrained, single-block spreadsheet correlation analysis to a governed, leakage-safe Python pipeline (`mutual_fund_ml`). The defining architectural difference is not model sophistication — both eras use standard, well-understood techniques (PCA, K-Means, linear/tree regression) — it is **governance**: the current codebase encodes variable-role decisions (structural vs. performance-derived, explanatory vs. forecasting) as inspectable configuration and code assertions, rather than leaving them to spreadsheet convention.

This reconciliation surfaces two distinct predictive illusions, from two different sources, that this audit treats separately rather than conflating:

1. **The legacy PCA's circularity flaw** — performance ratios mathematically derived from the target return were loaded into the same unsupervised PCA as structural portfolio variables, contaminating several components with the target's own information content.
2. **This codebase's own early over-fit illusion** — an initial, superseded validation pass (`docs/validation_report.md`, Section 6) reported a Gradient Boosting R² of 99.79% using a random 80/20 train/test split on panel data. This figure did **not** originate from the legacy spreadsheet — Nandhini's workbook contains no supervised regression or R² claim at all, only PCA and K-Means outputs. It was this project's own early methodological error, and it was self-corrected by the introduction of a proper chronological walk-forward holdout, documented in `docs/final_forecasting_assessment.md` and `docs/model_card.md`.

Both illusions share a root cause worth naming plainly: **evaluating a time-ordered panel as if it were i.i.d. cross-sectional data inflates apparent predictive power.** The legacy spreadsheet did this via variable contamination inside PCA; this project's own early pass did it via a non-chronological split. The current, final architecture closes both paths.

---

## 2. The Predictive Illusion & Target Leakage Breakdown

### 2a. Legacy circularity (Nandhini's PCA)

Several of the legacy workbook's ratio-type variables are mathematically non-separable from the target return `Average`:

| Variable | Formula | Contains `Average`? |
|---|---|---|
| Sharpe | `(R_p − R_f) / σ(R_p)` | Yes — numerator |
| Sortino | `(R_p − R_f) / downside_σ(R_p)` | Yes — numerator |
| Jensen's Alpha | `R_p − [R_f + β(R_m − R_f)]` | Yes — direct term |
| Fama | Return decomposition | Yes — by construction |
| Information Ratio | `(R_p − R_m) / TrackingError` | Yes — numerator |

Loading these into the same PCA as structural variables (HHI, Top-N concentration, cap allocation) means the resulting components are not purely structural — they partially re-express the target itself. Any downstream regression built on those component scores would be regressing the target on transformations of the target: inflated fit statistics, not genuine signal. This is a structural flaw in the component construction, independent of which supervised model is later run on top of it.

### 2b. This project's own early over-fit illusion (superseded)

Per `docs/validation_report.md` §6, an early validation pass reported:

| Model | R² (Test) |
|---|---|
| Gradient Boosting | **0.9979** |
| Decision Tree | 0.9912 |
| Random Forest | 0.9879 |
| Ridge | 0.9758 |

This evaluation used a **random 80/20 split** (5,422 test rows) on panel data where the same scheme appears across many adjacent months with strongly autocorrelated returns, and where the 1-month lagged `Average` was already a feature. A random split lets near-duplicate, highly correlated rows from the same scheme land on both sides of the train/test boundary — the model does not need to forecast anything; it needs only to recognize the scheme and interpolate a recent value it has already partially seen. High R² under this design does not indicate forecasting skill.

### 2c. The corrected, authoritative finding

`docs/final_forecasting_assessment.md` and `docs/model_card.md` replace the above with a proper 5-fold chronological walk-forward validation and a frozen, never-reused Jan–Apr 2026 holdout (1,978 rows):

| Model | RMSE (Holdout) | R² (Holdout) |
|---|---|---|
| Zero-Return Baseline | 4.4422 | −0.0373 |
| Category Historical-Mean | 5.0069 | −0.3177 |
| Development Unconditional-Mean | 5.0263 | −0.3280 |
| Ridge | 5.2620 | −0.4555 |
| Random Forest | 5.8139 | −0.7768 |
| Previous-Month Persistence | 7.1799 | −1.7098 |

**Finding:** No fitted model — Ridge, Random Forest, or otherwise — outperforms a constant Zero-Return baseline on the frozen holdout. Short-horizon mutual fund returns in this dataset are dominated by market-wide noise that these structural/portfolio-attribute features do not resolve. This is the authoritative, governance-compliant result; the 99.79%/97.6% figures in §2b are documented as superseded, not as current model performance.

---

## 3. Variance Decomposition & The Dominant Time Effect

Panel diagnostics (`outputs/tables/panel_diagnostics.txt`; 27,107 observations, 516 unique schemes, 76 monthly periods, unbalanced panel):

| Component | Value | Share of total variance |
|---|---|---|
| Overall variance (`Average`) | 19.069 | 100% |
| Variance of month means | 18.655 | **97.8%** |
| Between-scheme variance | 1.434 | 7.5% |
| Category-mean variance | 0.071 | 0.37% |

Month-to-month variation explains essentially all of the spread in returns — "a rising tide lifts all boats." Which fund you hold explains a small fraction (~7.5%), and fund *category* explains almost nothing (<0.4%). This is independent corroboration of §2c: if cross-sectional fund attributes carry so little of the variance to begin with, no amount of structural feature engineering on those attributes should be expected to produce strong return forecasts — the negative-R² holdout result is the expected outcome given this decomposition, not an anomaly.

---

## 4. Structural Taxonomy vs. Legacy Mixed Components

### PCA design

| | Legacy (Nandhini) | Current (`mutual_fund_ml`) |
|---|---|---|
| Component count | 13 (reported) | 11 (`pca_n_components: 11`, config-driven) |
| Input block | Single, mixed structural + performance-ratio block | Isolated structural/valuation/concentration block only |
| Feature ordering | Positional (spreadsheet column order) | Alphabetically sorted, assertion-enforced (`pca.py`: `assert features_list == sorted(features_list)`) |
| Feature contract | Implicit (workbook layout) | Explicit, versioned JSON (`artifacts/metadata/pca_input_features_explanatory.json`) |
| Feature count (explanatory) | N/A | **25** (verified: `AALC_Equity Allocation (%)` … `log_AUM`; target `Average` absent) |
| PC1 | Reported 27.5% var, concentration factor | **48.65%** var (verified, current fit), same concentration/diversification character — HHI, Top-N Stocks/Sectors load positively; `TS_Total Equity Stocks Count`, `TS_Total Sector Count`, `log_AUM` load negatively |
| Cumulative variance (11 PCs) | N/A | **96.02%** (verified) |

### Clustering — the platform's verified operational utility

K-Means (k=3) on the PC score space, verified from `outputs/clustering/cluster_profiles.csv` and `cluster_metrics.csv`:

| Cluster | Count | PC1 centroid | Character |
|---|---|---|---|
| 2 | 36 | +8.71 | Ultra-Concentrated / High-Conviction — extreme positive concentration score |
| 1 | 282 | +0.83 | Core Concentrated, near-average — largest group |
| 0 | 198 | −3.07 | Diversified, broad-market / mid-small-cap tilt — extreme negative concentration score |

(Silhouette = 0.290, Davies-Bouldin = 1.158, Calinski-Harabasz = 261.6, Cophenetic correlation vs. hierarchical = 0.657, counts sum to 516 — matches the panel's unique scheme count exactly.)

This taxonomy — not return forecasting — is the platform's demonstrated, defensible use case: an active-share / concentration-style audit tool for classifying funds by structural positioning, independent of the (negative) forecasting result in §2c.

---

## 5. Namespacing & Environment Safety Controls

A structural defect was identified and closed in this audit cycle: `save_pca_artifacts()` previously wrote every PCA artifact (model, loadings, explained variance, scores, feature-list metadata) to a single shared, mode-agnostic filename. Because `explanatory` mode (contemporaneous, target excluded) and `forecasting` mode (next-month target, `Average` intentionally included as a documented AR(1) lag feature) both called this function, whichever mode ran most recently silently overwrote the other's artifact — producing exactly the kind of ambiguity that caused this audit to initially misdiagnose a clean explanatory-mode design as leaking the target.

**Fix applied** (`src/mutual_fund_ml/pca.py`, `run_pipeline.py`, commit `483de87`):
- `get_pca_model_path(mode)` and `save_pca_artifacts(..., mode)` now namespace every output: `pca_model_{mode}.joblib`, `pca_input_features_{mode}.json`, `pca_loadings_{mode}.csv`, `explained_variance_{mode}.csv`, `pca_scores_{mode}.csv`.
- Stale, ambiguous unsuffixed artifacts were removed.
- Explanatory-mode artifacts were regenerated and verified clean (25 features, `Average` absent, deterministic-sort assertion passing).
- Forecasting mode was deliberately **not** refit in this cycle — its AR-lag design is intentional, not a defect, and refitting it would constitute retuning a forecasting model outside this audit's scope.

This closes the artifact-identity gap without altering either mode's modeling logic.

---

## Summary Verdict

| Finding | Status |
|---|---|
| Legacy PCA circularity (Sharpe/Sortino/Jensen's Alpha/Fama in components) | Confirmed, not present in current design |
| Project's own early 99.79% R² illusion (random split) | Confirmed superseded by chronological holdout |
| Current explanatory PCA free of target leakage | Confirmed (25 features, verified) |
| Forecasting-mode AR lag feature | Confirmed intentional, unchanged, not leakage |
| Variance dominated by time, not fund attributes | Confirmed (97.8% vs. 7.5% vs. 0.37%) |
| Artifact namespace collision between modes | Confirmed and fixed this cycle |
| Frozen Jan–Apr 2026 holdout reused or retuned | Not reused; not retuned |

**Overall:** The current architecture does not produce a better-forecasting model than the legacy work — on the frozen holdout, no model beats a naive zero-return guess, and that is documented as the honest finding rather than concealed. What the current architecture provides that the legacy workbook did not is **governance**: leakage exclusion enforced by configuration and tests, deterministic and auditable feature identity, and an explicit, documented distinction between exploratory/diagnostic structure (PCA, clustering) and forecasting claims — so that an illusion like the 99.79% figure in §2b, when it occurred, was caught, superseded, and left on the record rather than shipped as a result.

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def run_panel_diagnostics(df: pd.DataFrame, target_col: str = "Average") -> Dict[str, Any]:
    """Calculates diagnostics for panel data and returns a dictionary of results."""
    # 1. Basic Counts
    num_schemes = df["Scheme Name"].nunique()
    num_periods = df["TS_Month End"].nunique()
    total_obs = len(df)

    # 2. Balance analysis
    obs_counts = df["Scheme Name"].value_counts()
    is_balanced = (obs_counts.nunique() == 1) and (obs_counts.iloc[0] == num_periods)
    panel_status = "Balanced" if is_balanced else "Unbalanced"

    # 3. Observations per scheme statistics
    stats_obs = obs_counts.describe().to_dict()

    # 4. Variation decomposition (Within vs Between)
    overall_mean = df[target_col].mean()
    scheme_means = df.groupby("Scheme Name")[target_col].mean()

    # Between Variance (across scheme averages)
    between_var = scheme_means.var(ddof=1)
    between_std = np.sqrt(between_var)

    # Within Variance (de-meaned observations)
    demeaned = df[target_col] - df["Scheme Name"].map(scheme_means)
    within_var = demeaned.var(ddof=1)
    within_std = np.sqrt(within_var)

    # Overall Variance
    overall_var = df[target_col].var(ddof=1)
    overall_std = np.sqrt(overall_var)

    # 5. Month and Category effects
    month_means = df.groupby("TS_Month End")[target_col].mean()
    month_var = month_means.var(ddof=1)

    cat_means = df.groupby("Category")[target_col].mean()
    cat_var = cat_means.var(ddof=1)

    results = {
        "num_schemes": num_schemes,
        "num_periods": num_periods,
        "total_obs": total_obs,
        "panel_status": panel_status,
        "obs_per_scheme_mean": stats_obs["mean"],
        "obs_per_scheme_std": stats_obs["std"],
        "obs_per_scheme_min": stats_obs["min"],
        "obs_per_scheme_25pct": stats_obs["25%"],
        "obs_per_scheme_50pct": stats_obs["50%"],
        "obs_per_scheme_75pct": stats_obs["75%"],
        "obs_per_scheme_max": stats_obs["max"],
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "between_std": between_std,
        "within_std": within_std,
        "month_variance": month_var,
        "category_variance": cat_var
    }

    return results

def format_panel_diagnostics(results: Dict[str, Any]) -> str:
    """Formats panel diagnostics dictionary into a readable report string."""
    report = f"""PANEL DATA DIAGNOSTICS REPORT
=============================
Total Observations:         {results['total_obs']:,}
Number of unique schemes (n): {results['num_schemes']}
Number of unique periods (T): {results['num_periods']}
Panel Status:               {results['panel_status']}

Observations per Scheme Statistics:
----------------------------------
  Mean:   {results['obs_per_scheme_mean']:.2f}
  StdDev: {results['obs_per_scheme_std']:.2f}
  Min:    {results['obs_per_scheme_min']:.0f}
  25%:    {results['obs_per_scheme_25pct']:.0f}
  50%:    {results['obs_per_scheme_50pct']:.0f}
  75%:    {results['obs_per_scheme_75pct']:.0f}
  Max:    {results['obs_per_scheme_max']:.0f}

Target Variation Decomposition (col: Average):
----------------------------------------------
  Overall Mean:             {results['overall_mean']:.6f}
  Overall Standard Dev:      {results['overall_std']:.6f}
  Between-Scheme Std Dev:    {results['between_std']:.6f}
  Within-Scheme Std Dev:     {results['within_std']:.6f}

  Variance of Month Means:   {results['month_variance']:.6f}
  Variance of Category Means:{results['category_variance']:.6f}
"""
    return report

def save_panel_diagnostics(results: Dict[str, Any], output_path: Path) -> None:
    """Saves panel diagnostics report to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_str = format_panel_diagnostics(results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_str)

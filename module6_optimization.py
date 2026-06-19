"""
module6_optimization.py
─────────────────────────────────────────────────────────────────────────────
MODULE 6 : MULTI-OBJECTIVE PRIORITY OPTIMIZATION
─────────────────────────────────────────────────────────────────────────────

Goal
────
Given a limited area budget (N grid cells to intervene in), identify the
most impactful set of cells to target and recommend the single best
mitigation strategy for each selected cell.

Optimization Formulation
────────────────────────
Priority Score for cell i:

  P(i) = α × S(i) + β × C(i)

where:
  S(i) = Temperature Severity Score
         = (LST_i − LST_min) / (LST_max − LST_min)
         Ranks cells by how dangerously hot they are relative to the study area.

  C(i) = Cooling Sensitivity Score
         = max(ΔT_i across all scenarios) / max(ΔT across all cells)
         Ranks cells by how much benefit they get from the best intervention.

  α = CFG.alpha (default 0.6) — weight on temperature severity
  β = CFG.beta  (default 0.4) — weight on cooling sensitivity
  α + β = 1 (complementary weights)

Outputs
───────
  • optimization_results.csv — top-N cells with priority scores and
    recommended scenario.
  • Console report with the optimization summary.

Run standalone:
    python module6_optimization.py

Dependencies: pandas, numpy
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd

from config import CFG, get_synthetic_dataframe
from module3_predictive_modeling import run_modeling
from module5_scenario_simulation import run_simulation

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Module6")


# ─── SCORE COMPUTATION ────────────────────────────────────────────────────────

def compute_severity_score(lst_values: np.ndarray) -> np.ndarray:
    """
    Min-max normalise LST values to produce a severity score in [0, 1].

    S(i) = (LST_i − LST_min) / (LST_max − LST_min)

    A score of 1.0 = the hottest cell in the study area.
    A score of 0.0 = the coolest cell.

    Parameters
    ----------
    lst_values : np.ndarray — LST_Celsius values for all test cells

    Returns
    -------
    np.ndarray — severity scores in [0, 1]
    """
    lst_min = lst_values.min()
    lst_max = lst_values.max()

    if lst_max == lst_min:
        log.warning("All LST values identical — severity scores set to 0.5.")
        return np.full_like(lst_values, 0.5, dtype=float)

    return (lst_values - lst_min) / (lst_max - lst_min)


def compute_cooling_sensitivity(results_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the cooling sensitivity score for each cell.

    C(i) = max(ΔT across scenarios for cell i) / max(ΔT across ALL cells)

    Uses the per-cell maximum cooling achievable across all scenarios.
    This ensures we reward cells that can benefit greatly from *some*
    intervention, regardless of which one.

    Parameters
    ----------
    results_df : pd.DataFrame — output of module5.simulate_scenarios()
                               must contain columns starting with 'DeltaT_'

    Returns
    -------
    np.ndarray — sensitivity scores in [0, 1]
    """
    delta_cols = [c for c in results_df.columns if c.startswith("DeltaT_")]
    if not delta_cols:
        raise ValueError("No DeltaT columns found in results_df.")

    # Per-cell maximum cooling across all scenarios
    max_dt_per_cell = results_df[delta_cols].max(axis=1).values

    global_max = max_dt_per_cell.max()
    if global_max <= 0:
        log.warning("No positive cooling found — sensitivity scores set to 0.")
        return np.zeros(len(results_df))

    return np.clip(max_dt_per_cell / global_max, 0.0, 1.0)


def compute_priority_score(severity: np.ndarray,
                             sensitivity: np.ndarray,
                             alpha: float = CFG.alpha,
                             beta: float  = CFG.beta) -> np.ndarray:
    """
    Combine severity and sensitivity into a single priority score.

    P(i) = α × S(i) + β × C(i)

    Parameters
    ----------
    severity    : np.ndarray — temperature severity scores [0, 1]
    sensitivity : np.ndarray — cooling sensitivity scores  [0, 1]
    alpha       : float      — weight on severity  (default 0.6)
    beta        : float      — weight on sensitivity (default 0.4)

    Returns
    -------
    np.ndarray — priority scores in [0, 1]
    """
    if not np.isclose(alpha + beta, 1.0):
        log.warning("α + β = %.2f ≠ 1.0. Normalising.", alpha + beta)
        total = alpha + beta
        alpha /= total
        beta  /= total

    priority = alpha * severity + beta * sensitivity
    return priority


def identify_best_scenario(results_df: pd.DataFrame) -> pd.Series:
    """
    For each cell, identify the scenario name that achieves the maximum ΔT.

    Parameters
    ----------
    results_df : pd.DataFrame — must contain DeltaT_* columns

    Returns
    -------
    pd.Series — best scenario name per cell
    """
    delta_cols = [c for c in results_df.columns if c.startswith("DeltaT_")]
    # idxmax returns the column name of the max ΔT for each row
    best_col = results_df[delta_cols].idxmax(axis=1)
    # Clean up the column name to human-readable scenario name
    best_scenario = (
        best_col
        .str.replace("DeltaT_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )
    return best_scenario


# ─── MAIN OPTIMIZATION ENGINE ─────────────────────────────────────────────────

def run_optimization(df_test: pd.DataFrame,
                      results_df: pd.DataFrame,
                      budget_n: int = CFG.budget_n,
                      alpha: float = CFG.alpha,
                      beta: float  = CFG.beta) -> pd.DataFrame:
    """
    Full multi-objective priority optimization.

    Parameters
    ----------
    df_test    : pd.DataFrame — test set with geographic coordinates + LST
    results_df : pd.DataFrame — scenario ΔT results from Module 5
    budget_n   : int          — number of target cells (area budget)
    alpha      : float        — severity weight
    beta       : float        — sensitivity weight

    Returns
    -------
    top_n_df : pd.DataFrame — top-N cells with:
        longitude, latitude, LST_Celsius, Severity_Score,
        Sensitivity_Score, Priority_Score, Best_Scenario,
        Best_DeltaT_C, hotspot_category (if available)
    """
    log.info("Running priority optimization: N=%d, α=%.2f, β=%.2f",
             budget_n, alpha, beta)

    df_test     = df_test.reset_index(drop=True)
    results_df  = results_df.reset_index(drop=True)

    # ── Get LST values ─────────────────────────────────────────────────────────
    if "LST_Celsius" not in df_test.columns:
        raise ValueError("df_test must contain 'LST_Celsius' column.")
    lst_values = df_test["LST_Celsius"].values

    # ── Compute component scores ───────────────────────────────────────────────
    severity    = compute_severity_score(lst_values)
    sensitivity = compute_cooling_sensitivity(results_df)
    priority    = compute_priority_score(severity, sensitivity, alpha, beta)

    # ── Identify best scenario per cell ───────────────────────────────────────
    best_scenario = identify_best_scenario(results_df)

    # ── Compute max achievable ΔT per cell ────────────────────────────────────
    delta_cols    = [c for c in results_df.columns if c.startswith("DeltaT_")]
    best_delta_t  = results_df[delta_cols].max(axis=1).values

    # ── Assemble full optimization DataFrame ──────────────────────────────────
    opt_cols = {
        "Severity_Score"    : severity,
        "Sensitivity_Score" : sensitivity,
        "Priority_Score"    : priority,
        "Best_Scenario"     : best_scenario.values,
        "Best_DeltaT_C"     : best_delta_t,
    }

    # Include geo + LST columns from df_test
    geo_keep = ["longitude", "latitude", "LST_Celsius"]
    if "hotspot_category" in df_test.columns:
        geo_keep.append("hotspot_category")

    opt_df = df_test[geo_keep].copy()
    for col, vals in opt_cols.items():
        opt_df[col] = vals

    # Also attach all DeltaT columns for transparency
    for col in delta_cols:
        opt_df[col] = results_df[col].values

    # ── Select top-N cells by priority score ──────────────────────────────────
    opt_df = opt_df.sort_values("Priority_Score", ascending=False)
    top_n  = min(budget_n, len(opt_df))
    top_n_df = opt_df.head(top_n).reset_index(drop=True)
    top_n_df["Rank"] = np.arange(1, top_n + 1)

    log.info("Selected top %d cells from %d candidates.", top_n, len(opt_df))
    return top_n_df, opt_df


# ─── STRATEGY RECOMMENDATION REPORT ──────────────────────────────────────────

def build_strategy_report(top_n_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the recommended strategies across the selected target cells.

    Returns a DataFrame showing how many cells each scenario is recommended
    for, along with average expected cooling.
    """
    report = (
        top_n_df.groupby("Best_Scenario")
        .agg(
            Num_Cells    = ("Best_Scenario", "count"),
            Avg_DeltaT_C = ("Best_DeltaT_C", "mean"),
            Avg_Priority = ("Priority_Score", "mean"),
            Avg_LST_C    = ("LST_Celsius", "mean"),
        )
        .round(3)
        .sort_values("Num_Cells", ascending=False)
        .reset_index()
    )
    return report


def print_optimization_report(top_n_df: pd.DataFrame,
                                strategy_report: pd.DataFrame,
                                budget_n: int) -> None:
    """Print a formatted optimization report to stdout."""
    print("\n" + "=" * 70)
    print(f"  MULTI-OBJECTIVE OPTIMIZATION REPORT  (Budget N = {budget_n} cells)")
    print("=" * 70)

    print(f"\n  Total cells selected : {len(top_n_df)}")
    print(f"  Avg Priority Score   : {top_n_df['Priority_Score'].mean():.3f}")
    print(f"  Avg LST (°C)         : {top_n_df['LST_Celsius'].mean():.2f}")
    print(f"  Expected avg cooling : {top_n_df['Best_DeltaT_C'].mean():+.3f} °C")

    print("\n  ── Strategy Allocation ──────────────────────────────────────────")
    for _, row in strategy_report.iterrows():
        pct = 100 * row["Num_Cells"] / len(top_n_df)
        print(f"  {row['Best_Scenario']:<22s}  "
              f"{row['Num_Cells']:>4d} cells ({pct:.1f}%)  "
              f"avg ΔT={row['Avg_DeltaT_C']:+.3f}°C")

    print("\n  ── Top 10 Target Cells ──────────────────────────────────────────")
    print(top_n_df[["Rank", "longitude", "latitude", "LST_Celsius",
                    "Priority_Score", "Best_Scenario", "Best_DeltaT_C"]]
          .head(10).to_string(index=False))
    print("=" * 70 + "\n")


# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def run_full_optimization(simulation_results: dict = None,
                           modeling_results: dict = None,
                           budget_n: int = CFG.budget_n,
                           alpha: float = CFG.alpha,
                           beta: float  = CFG.beta) -> dict:
    """
    Full optimization pipeline.

    Parameters
    ----------
    simulation_results : dict | None — output of module5.run_simulation()
    modeling_results   : dict | None — output of module3.run_modeling()
    budget_n           : int         — number of target cells
    alpha, beta        : float       — optimization weights

    Returns
    -------
    dict with keys:
        top_n_df        : pd.DataFrame — selected target cells
        full_opt_df     : pd.DataFrame — all cells with scores
        strategy_report : pd.DataFrame — strategy allocation summary
    """
    if simulation_results is None:
        if modeling_results is None:
            log.info("Running Module 3 (modeling) …")
            modeling_results = run_modeling()
        log.info("Running Module 5 (simulation) …")
        simulation_results = run_simulation(modeling_results)

    results_df  = simulation_results["results_df"]
    df_test_geo = simulation_results["df_test_geo"]

    # df_test_geo may not have LST_Celsius — fall back to df_test from modeling
    if "LST_Celsius" not in df_test_geo.columns and modeling_results:
        df_test = modeling_results["df_test"].reset_index(drop=True)
    else:
        df_test = df_test_geo.reset_index(drop=True)

    top_n_df, full_opt_df = run_optimization(
        df_test, results_df, budget_n, alpha, beta
    )
    strategy_report = build_strategy_report(top_n_df)
    print_optimization_report(top_n_df, strategy_report, budget_n)

    # Persist results
    top_n_df.to_csv(CFG.optim_csv, index=False)
    log.info("Optimization results saved → %s", CFG.optim_csv)

    return {
        "top_n_df"        : top_n_df,
        "full_opt_df"     : full_opt_df,
        "strategy_report" : strategy_report,
    }


# ─── STANDALONE EXECUTION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 6 : MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 70)

    budget = int(sys.argv[1]) if len(sys.argv) > 1 else CFG.budget_n

    opt_results = run_full_optimization(budget_n=budget)
    top_n       = opt_results["top_n_df"]

    print(f"\n✓ Optimization results saved → {CFG.optim_csv}")
    print(f"✓ Top {len(top_n)} target cells identified.")
    print(f"✓ Dominant strategy: '{opt_results['strategy_report'].iloc[0]['Best_Scenario']}'")

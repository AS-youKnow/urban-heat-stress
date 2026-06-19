"""
module5_scenario_simulation.py
─────────────────────────────────────────────────────────────────────────────
MODULE 5 : ADVANCED SCENARIO SIMULATION
─────────────────────────────────────────────────────────────────────────────

Models three urban heat mitigation policy interventions:

  1. Urban Greening   — NDVI +0.20, NDBI -0.05
     (Planting trees, parks, green roofs, urban forests)

  2. Cool Roofs/Albedo — NDBI -0.25
     (High-albedo coatings, reflective surfaces, cool pavements)

  3. Mixed Strategy   — NDVI +0.15, NDBI -0.15
     (Combined moderate greening + reflective surface upgrade)

For each scenario:
  • Perturb the relevant features in the test set.
  • Clip perturbed values to physically valid ranges [−1, 1].
  • Generate new LST predictions using the trained XGBoost model.
  • Compute ΔT = LST_baseline − LST_scenario (positive = cooling).
  • Aggregate mean, median, and 95th-percentile cooling per scenario.

Output: scenario_results.csv with per-cell ΔT for all three scenarios.

Run standalone:
    python module5_scenario_simulation.py

Dependencies: pandas, numpy, xgboost
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd

from config import CFG, get_synthetic_dataframe
from module3_predictive_modeling import run_modeling, load_model

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Module5")


# ─── SCENARIO APPLICATION ─────────────────────────────────────────────────────

def apply_scenario(X_test: pd.DataFrame, deltas: dict) -> pd.DataFrame:
    """
    Apply feature perturbations to a copy of the test feature matrix.

    Parameters
    ----------
    X_test  : pd.DataFrame — baseline test features
    deltas  : dict         — {column_name: delta_value} to add to each feature
                             e.g. {'NDVI': +0.20, 'NDBI': -0.05}

    Returns
    -------
    X_perturbed : pd.DataFrame — feature matrix after intervention

    Notes
    -----
    • Only NDVI and NDBI are perturbed; other features remain unchanged.
    • NDVI / NDBI are clipped to [-1, 1] post-perturbation.
    • Built_Fraction and Pop_Density are left unchanged as they represent
      structural/demographic properties that change on longer timescales.
    """
    X_pert = X_test.copy()

    for feature, delta in deltas.items():
        if feature not in X_pert.columns:
            log.warning("Feature '%s' not in DataFrame — skipping.", feature)
            continue

        X_pert[feature] = X_pert[feature] + delta

        # Clip spectral indices to their physical valid range
        if feature in ("NDVI", "NDBI"):
            X_pert[feature] = X_pert[feature].clip(-1.0, 1.0)

    return X_pert


def compute_delta_t(baseline_lst: np.ndarray,
                    scenario_lst: np.ndarray) -> np.ndarray:
    """
    Compute per-cell cooling effect.

    ΔT = LST_baseline − LST_scenario
    Positive ΔT → intervention causes cooling (desired effect).
    Negative ΔT → intervention causes warming (should not occur for
                  sensible scenarios but handled defensively).

    Returns
    -------
    np.ndarray — per-cell ΔT values (°C)
    """
    delta_t = baseline_lst - scenario_lst
    return delta_t


# ─── SIMULATION ENGINE ────────────────────────────────────────────────────────

def simulate_scenarios(model,
                        X_test: pd.DataFrame,
                        baseline_lst: np.ndarray,
                        scenarios: dict = None) -> pd.DataFrame:
    """
    Run all intervention scenarios and compile per-cell ΔT results.

    Parameters
    ----------
    model        : fitted XGBRegressor
    X_test       : pd.DataFrame — test set features (baseline)
    baseline_lst : np.ndarray  — baseline LST predictions (°C)
    scenarios    : dict | None — {name: {feature: delta}} mapping.
                                  Defaults to CFG.scenarios.

    Returns
    -------
    pd.DataFrame with one column per scenario (delta_T values) plus
    a 'Best_Scenario' column indicating which intervention gives most cooling
    for each cell.
    """
    if scenarios is None:
        scenarios = CFG.scenarios

    results = {}

    for name, deltas in scenarios.items():
        log.info("Simulating scenario: '%s'  (deltas: %s)", name, deltas)

        X_pert      = apply_scenario(X_test, deltas)
        scenario_lst = model.predict(X_pert)
        delta_t     = compute_delta_t(baseline_lst, scenario_lst)

        col_name           = f"DeltaT_{name.replace(' ', '_')}"
        results[col_name]  = delta_t

        mean_cooling   = delta_t.mean()
        median_cooling = np.median(delta_t)
        p95_cooling    = np.percentile(delta_t, 95)
        n_benefited    = (delta_t > 0).sum()

        log.info(
            "  Mean ΔT=%.3f°C | Median=%.3f°C | P95=%.3f°C | "
            "Cells cooled: %d / %d",
            mean_cooling, median_cooling, p95_cooling, n_benefited, len(delta_t)
        )

    results_df = pd.DataFrame(results, index=X_test.index)

    # Identify the best scenario (max cooling) for each cell
    results_df["Best_Scenario"] = results_df.idxmax(axis=1).str.replace(
        "DeltaT_", "", regex=False
    ).str.replace("_", " ", regex=False)

    return results_df


# ─── SUMMARY TABLE ────────────────────────────────────────────────────────────

def build_summary_table(results_df: pd.DataFrame,
                          baseline_lst: np.ndarray) -> pd.DataFrame:
    """
    Create a concise summary DataFrame comparing all scenarios.

    Columns: Scenario | Mean ΔT (°C) | Median ΔT | 95th Pct ΔT |
             Max ΔT | % Cells Cooled | Avg Baseline LST

    Parameters
    ----------
    results_df   : pd.DataFrame — output of simulate_scenarios()
    baseline_lst : np.ndarray  — baseline LST values

    Returns
    -------
    pd.DataFrame — one row per scenario
    """
    delta_cols = [c for c in results_df.columns if c.startswith("DeltaT_")]
    rows = []

    for col in delta_cols:
        dt  = results_df[col].values
        name = col.replace("DeltaT_", "").replace("_", " ")
        rows.append({
            "Scenario"          : name,
            "Mean_DeltaT_C"     : round(float(dt.mean()), 3),
            "Median_DeltaT_C"   : round(float(np.median(dt)), 3),
            "P95_DeltaT_C"      : round(float(np.percentile(dt, 95)), 3),
            "Max_DeltaT_C"      : round(float(dt.max()), 3),
            "Pct_Cells_Cooled"  : round(float((dt > 0).mean() * 100), 1),
            "Avg_Baseline_LST"  : round(float(baseline_lst.mean()), 2),
        })

    summary = pd.DataFrame(rows).sort_values("Mean_DeltaT_C", ascending=False)
    return summary


# ─── PRINT REPORT ─────────────────────────────────────────────────────────────

def print_scenario_report(summary: pd.DataFrame) -> None:
    """Print a formatted scenario comparison table to stdout."""
    print("\n" + "=" * 70)
    print("  SCENARIO SIMULATION RESULTS")
    print("=" * 70)
    print(f"  Baseline mean LST: {summary['Avg_Baseline_LST'].mean():.2f} °C\n")

    for _, row in summary.iterrows():
        print(f"  ▶ {row['Scenario']}")
        print(f"      Mean cooling (ΔT)  : {row['Mean_DeltaT_C']:+.3f} °C")
        print(f"      Median ΔT          : {row['Median_DeltaT_C']:+.3f} °C")
        print(f"      95th-pct ΔT        : {row['P95_DeltaT_C']:+.3f} °C")
        print(f"      Cells cooled       : {row['Pct_Cells_Cooled']:.1f} %")
        print()

    best = summary.iloc[0]["Scenario"]
    print(f"  ★ Most effective strategy: '{best}'")
    print("=" * 70 + "\n")


# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def run_simulation(modeling_results: dict = None) -> dict:
    """
    End-to-end scenario simulation pipeline.

    Parameters
    ----------
    modeling_results : dict | None
        Output of module3_predictive_modeling.run_modeling().
        If None, re-runs the modeling pipeline.

    Returns
    -------
    dict with keys:
        results_df  : pd.DataFrame — per-cell ΔT for all scenarios
        summary     : pd.DataFrame — aggregated summary table
        df_test_geo : pd.DataFrame — test set with coordinates + results merged
    """
    if modeling_results is None:
        log.info("No modeling results supplied — running Module 3 …")
        modeling_results = run_modeling()

    model        = modeling_results["model"]
    X_test       = modeling_results["X_test"]
    df_test      = modeling_results["df_test"]
    baseline_lst = modeling_results["metrics"]["y_pred"]

    log.info("Running %d scenarios on %d test cells …",
             len(CFG.scenarios), len(X_test))

    results_df = simulate_scenarios(model, X_test, baseline_lst)
    summary    = build_summary_table(results_df, baseline_lst)
    print_scenario_report(summary)

    # Merge geo-coordinates with scenario results for mapping
    geo_cols = [c for c in ["longitude", "latitude", "LST_Celsius", "hotspot_category"]
                if c in df_test.columns]
    df_test_geo = pd.concat(
        [df_test[geo_cols].reset_index(drop=True), results_df.reset_index(drop=True)],
        axis=1
    )

    # Save to disk
    full_results = pd.concat(
        [df_test.reset_index(drop=True), results_df.reset_index(drop=True)],
        axis=1
    )
    full_results.to_csv(CFG.scenario_csv, index=False)
    log.info("Scenario results saved → %s", CFG.scenario_csv)

    summary.to_csv(CFG.scenario_csv.replace(".csv", "_summary.csv"), index=False)

    return {
        "results_df"  : results_df,
        "summary"     : summary,
        "df_test_geo" : df_test_geo,
    }


# ─── STANDALONE EXECUTION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 5 : SCENARIO SIMULATION")
    print("=" * 70)

    sim_results = run_simulation()
    summary     = sim_results["summary"]

    print(f"\n✓ Scenario results saved → {CFG.scenario_csv}")
    print(f"✓ Best strategy: '{summary.iloc[0]['Scenario']}'  "
          f"(Mean ΔT = {summary.iloc[0]['Mean_DeltaT_C']:+.3f} °C)")

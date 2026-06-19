"""
pipeline_runner.py
─────────────────────────────────────────────────────────────────────────────
END-TO-END ORCHESTRATOR
─────────────────────────────────────────────────────────────────────────────

Runs all six processing modules in sequence, validates inter-module outputs,
and produces a final summary report.

Usage:
    python pipeline_runner.py                   # synthetic data, default budget
    python pipeline_runner.py --gee             # live GEE data
    python pipeline_runner.py --budget 100      # custom budget (N cells)
    python pipeline_runner.py --alpha 0.7       # custom severity weight

Output files (all in the project directory):
    heat_grid.csv
    heat_grid_clustered.csv
    xgb_lst_model.json
    shap_summary_beeswarm.png
    shap_feature_importance.png
    scenario_results.csv
    scenario_results_summary.csv
    optimization_results.csv
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import argparse
import logging
import traceback
import io

# ── Force UTF-8 output on Windows (prevents UnicodeEncodeError in PowerShell)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import CFG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Pipeline")


# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Urban Heat Stress AI/ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_runner.py                  # Quick demo with synthetic data
  python pipeline_runner.py --gee            # Live GEE data (requires auth)
  python pipeline_runner.py --budget 100 --alpha 0.7
        """,
    )
    parser.add_argument("--gee",    action="store_true",
                        help="Use live Google Earth Engine data (requires auth)")
    parser.add_argument("--budget", type=int,   default=CFG.budget_n,
                        help=f"Number of target cells (default: {CFG.budget_n})")
    parser.add_argument("--alpha",  type=float, default=CFG.alpha,
                        help=f"Severity weight α ∈ [0,1] (default: {CFG.alpha})")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis (faster, no plots generated)")
    return parser.parse_args()


# ─── STAGE RUNNER ─────────────────────────────────────────────────────────────

def run_stage(name: str, func, *args, **kwargs):
    """
    Execute a pipeline stage with timing, error handling, and logging.

    Parameters
    ----------
    name : str  — human-readable stage name
    func        — callable to execute
    *args, **kwargs — passed to func

    Returns
    -------
    result of func(*args, **kwargs)

    Raises
    ------
    SystemExit on failure (prints traceback and exits)
    """
    print(f"\n{'─'*60}")
    print(f"  STAGE: {name}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  ✓ Completed in {elapsed:.2f}s")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ FAILED after {elapsed:.2f}s: {exc}")
        traceback.print_exc()
        sys.exit(1)


# ─── VALIDATION HELPERS ───────────────────────────────────────────────────────

def validate_dataframe(df, name: str, required_cols: list = None) -> None:
    """Assert that a DataFrame is non-empty and contains required columns."""
    assert df is not None,     f"{name} is None"
    assert len(df) > 0,        f"{name} is empty"
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"{name} missing columns: {missing}"
    print(f"    DataFrame '{name}': {len(df):,} rows × {df.shape[1]} cols — OK")


def validate_model(model) -> None:
    """Assert that a trained XGBoost model is ready for inference."""
    import numpy as np
    dummy = np.zeros((1, len(CFG.feature_cols)))
    pred  = model.predict(dummy)
    assert len(pred) == 1, "Model predict() returned wrong shape"
    print(f"    Model inference check: {pred[0]:.4f}°C — OK")


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  URBAN HEAT STRESS AI/ML PIPELINE")
    print("=" * 60)
    print(f"  Data source : {'GEE (live)' if args.gee else 'Synthetic (demo)'}")
    print(f"  Budget N    : {args.budget} cells")
    print(f"  alpha (severity): {args.alpha}")
    print(f"  beta  (sensitiv): {1.0 - args.alpha:.2f}")
    print(f"  Skip SHAP   : {args.skip_shap}")
    print("=" * 60)

    pipeline_start = time.perf_counter()

    # ────────────────────────────────────────────────────────────────────────
    # MODULE 1 — Data Ingestion
    # ────────────────────────────────────────────────────────────────────────
    from module1_data_ingestion import run_ingestion

    df = run_stage(
        "MODULE 1 — Data Ingestion & Grid Extraction",
        run_ingestion,
        use_gee=args.gee,
    )
    validate_dataframe(df, "heat_grid",
                       ["longitude", "latitude", "LST_Celsius",
                        "NDVI", "NDBI", "Pop_Density", "Built_Fraction", "LULC"])

    # ────────────────────────────────────────────────────────────────────────
    # MODULE 2 — Hotspot Clustering
    # ────────────────────────────────────────────────────────────────────────
    from module2_hotspot_clustering import run_hotspot_analysis

    df = run_stage(
        "MODULE 2 — Getis-Ord Gi* Hotspot Clustering",
        run_hotspot_analysis,
        df,
    )
    validate_dataframe(df, "clustered_grid", ["gi_zscore", "hotspot_category"])

    # Save clustered grid
    df.to_csv(CFG.grid_csv.replace(".csv", "_clustered.csv"), index=False)
    print(f"    Clustered grid saved → heat_grid_clustered.csv")

    # ────────────────────────────────────────────────────────────────────────
    # MODULE 3 — Predictive Modeling
    # ────────────────────────────────────────────────────────────────────────
    from module3_predictive_modeling import run_modeling

    mod_results = run_stage(
        "MODULE 3 — XGBoost LST Regressor",
        run_modeling,
        df,
    )
    validate_model(mod_results["model"])
    metrics = mod_results["metrics"]
    print(f"    MAE={metrics['MAE']:.4f}°C | RMSE={metrics['RMSE']:.4f}°C | "
          f"R²={metrics['R2']:.4f}")

    # ────────────────────────────────────────────────────────────────────────
    # MODULE 4 — SHAP Interpretation
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_shap:
        from module4_shap_interpretation import run_shap_analysis

        shap_results = run_stage(
            "MODULE 4 — SHAP Driver Analysis",
            run_shap_analysis,
            mod_results,
        )
        top_feat = shap_results["importance_df"].iloc[0]["Feature"]
        print(f"    Top SHAP driver: '{top_feat}'")
        print(f"    Beeswarm → {shap_results['beeswarm_path']}")
        print(f"    Bar plot → {shap_results['bar_path']}")
    else:
        log.info("SHAP stage skipped (--skip-shap flag).")
        shap_results = None

    # ────────────────────────────────────────────────────────────────────────
    # MODULE 5 — Scenario Simulation
    # ────────────────────────────────────────────────────────────────────────
    from module5_scenario_simulation import run_simulation

    sim_results = run_stage(
        "MODULE 5 — Intervention Scenario Simulation",
        run_simulation,
        mod_results,
    )
    summary = sim_results["summary"]
    best_scenario = summary.iloc[0]["Scenario"]
    print(f"    Best scenario: '{best_scenario}'  "
          f"(Mean ΔT={summary.iloc[0]['Mean_DeltaT_C']:+.3f}°C)")

    # ────────────────────────────────────────────────────────────────────────
    # MODULE 6 — Multi-Objective Optimization
    # ────────────────────────────────────────────────────────────────────────
    from module6_optimization import run_full_optimization

    opt_results = run_stage(
        f"MODULE 6 — Priority Optimization (N={args.budget})",
        run_full_optimization,
        simulation_results=sim_results,
        modeling_results=mod_results,
        budget_n=args.budget,
        alpha=args.alpha,
        beta=1.0 - args.alpha,
    )
    top_n = opt_results["top_n_df"]
    print(f"    {len(top_n)} target cells selected.")
    print(f"    Avg priority score: {top_n['Priority_Score'].mean():.3f}")
    print(f"    Avg expected cooling: {top_n['Best_DeltaT_C'].mean():+.3f}°C")

    # ────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total runtime      : {total_time:.2f}s")
    print(f"  Grid cells         : {len(df):,}")
    print(f"  Data source        : {df.attrs.get('source', 'unknown')}")
    print(f"  Model R2           : {metrics['R2']:.4f}")
    print(f"  Best scenario      : {best_scenario}")
    print(f"  Target cells       : {len(top_n)}")
    print()
    print("  Output files:")
    import os
    output_files = [
        CFG.grid_csv,
        CFG.model_path,
        CFG.scenario_csv,
        CFG.optim_csv,
        CFG.shap_beeswarm,
        CFG.shap_bar,
    ]
    for f in output_files:
        exists = "[OK]" if os.path.exists(f) else "[--]"
        print(f"    {exists}  {os.path.basename(f)}")

    print()
    print("  Launch dashboard:")
    print("    streamlit run module7_dashboard.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

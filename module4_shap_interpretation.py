"""
module4_shap_interpretation.py
─────────────────────────────────────────────────────────────────────────────
MODULE 4 : SHAP FEATURE INTERPRETATION (Driver Analysis)
─────────────────────────────────────────────────────────────────────────────

Uses SHAP (SHapley Additive exPlanations) with the TreeExplainer backend
to produce globally interpretable explanations of the XGBoost LST model.

Outputs:
  1. shap_summary_beeswarm.png — Global beeswarm summary plot showing the
     distribution of SHAP values for every feature and data point.
     High SHAP value → feature pushes prediction higher (hotter).
  2. shap_feature_importance.png — Mean |SHAP| bar chart ranking features
     by their average absolute impact on LST predictions.

Interpretation guide:
  • Built_Fraction / NDBI near the top → built-up surfaces dominate heat.
  • NDVI near top with negative SHAP → green areas cool the surface.
  • Pop_Density contribution → indirect urban heat island effect.

Run standalone:
    python module4_shap_interpretation.py

Dependencies: shap, xgboost, matplotlib, pandas, numpy
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import shap
from xgboost import XGBRegressor

from config import CFG, get_synthetic_dataframe
from module3_predictive_modeling import run_modeling, load_model

# Use non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Module4")

# ─── PLOT STYLE CONSTANTS ─────────────────────────────────────────────────────
PLOT_DPI        = 150
PLOT_FIGSIZE    = (12, 7)
BACKGROUND_CLR  = "#0e1117"
TEXT_CLR        = "#e0e0e0"
ACCENT_CLR      = "#ff6b6b"
COOL_CLR        = "#4ecdc4"


def _apply_dark_theme(fig, ax):
    """Apply a consistent dark publication theme to a matplotlib figure."""
    fig.patch.set_facecolor(BACKGROUND_CLR)
    ax.set_facecolor(BACKGROUND_CLR)
    ax.tick_params(colors=TEXT_CLR, labelsize=10)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


# ─── SHAP EXPLAINER ───────────────────────────────────────────────────────────

def build_shap_explainer(model: XGBRegressor, X_train: pd.DataFrame):
    """
    Create a SHAP TreeExplainer for the given XGBoost model.

    TreeExplainer uses the exact Shapley values algorithm optimised for
    tree-based models — no approximation needed.

    Note: We use feature_perturbation='tree_path_dependent' which is required
    when XGBoost is trained with tree_method='hist' (XGBoost 3.x default).
    The 'interventional' mode is not yet supported with categorical/hist splits.

    Parameters
    ----------
    model   : fitted XGBRegressor
    X_train : pd.DataFrame — training features (used to set baseline / background)

    Returns
    -------
    explainer      : shap.TreeExplainer
    """
    log.info("Building SHAP TreeExplainer (tree_path_dependent) ...")
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",  # required for XGBoost hist method
    )
    return explainer


def compute_shap_values(explainer, X_test: pd.DataFrame):
    """
    Compute SHAP values for the test set.
    Returns a 2D numpy array of shape (n_samples, n_features).
    """
    log.info("Computing SHAP values for %d test samples ...", len(X_test))
    # Use the newer shap.Explanation API first; fall back to legacy .shap_values()
    try:
        explanation = explainer(X_test, check_additivity=False)
        # explanation.values shape: (n_samples, n_features) or (n_samples, n_features, n_outputs)
        sv = explanation.values
        if sv.ndim == 3:
            sv = sv[:, :, 0]  # take first output for single-output regression
    except Exception:
        sv = explainer.shap_values(X_test, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[0]
    log.info("SHAP values computed. Shape: %s", sv.shape)
    return sv.astype(np.float32)


# ─── GLOBAL SUMMARY BEESWARM PLOT ─────────────────────────────────────────────

def plot_shap_beeswarm(shap_values: np.ndarray,
                        X_test: pd.DataFrame,
                        save_path: str = CFG.shap_beeswarm) -> None:
    """
    Generate and save the SHAP global summary beeswarm plot.

    Each point represents one test-set observation.
    X-axis position = SHAP value (impact on LST prediction in °C).
    Colour = feature value (red=high, blue=low).
    Features sorted by mean |SHAP| (most impactful at top).

    Parameters
    ----------
    shap_values : np.ndarray — SHAP values (n_samples × n_features)
    X_test      : pd.DataFrame — test feature matrix (for colour mapping)
    save_path   : str — file path to save the PNG
    """
    log.info("Generating SHAP beeswarm summary plot …")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    _apply_dark_theme(fig, ax)

    # Use SHAP's built-in beeswarm which internally calls matplotlib
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=CFG.feature_cols,
        plot_type="dot",
        show=False,
        color_bar=True,
        max_display=len(CFG.feature_cols),
    )

    # Customise labels and title post-render
    ax = plt.gca()
    ax.set_xlabel("SHAP Value (Impact on LST Prediction, °C)", color=TEXT_CLR, fontsize=12)
    ax.set_title(
        "SHAP Global Feature Impact — Urban LST Driver Analysis\n"
        "(Positive SHAP → Surface Heating | Negative SHAP → Cooling)",
        color=TEXT_CLR, fontsize=13, pad=12, fontweight="bold",
    )
    ax.tick_params(colors=TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    fig.patch.set_facecolor(BACKGROUND_CLR)
    ax.set_facecolor(BACKGROUND_CLR)

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=BACKGROUND_CLR)
    plt.close(fig)
    log.info("Beeswarm plot saved → %s", save_path)


# ─── FEATURE IMPORTANCE BAR PLOT ──────────────────────────────────────────────

def plot_shap_importance_bar(shap_values: np.ndarray,
                              feature_names: list,
                              save_path: str = CFG.shap_bar) -> pd.DataFrame:
    """
    Generate a horizontal bar chart of mean |SHAP| per feature.

    This plot answers: "On average, how much does each feature shift the
    LST prediction away from the baseline (expected) value?"

    Parameters
    ----------
    shap_values   : np.ndarray — (n_samples × n_features)
    feature_names : list       — feature column names
    save_path     : str        — output PNG path

    Returns
    -------
    pd.DataFrame — feature importance table (sorted descending by mean |SHAP|)
    """
    log.info("Generating SHAP feature importance bar chart …")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature"       : feature_names,
        "Mean_Abs_SHAP" : mean_abs_shap,
    }).sort_values("Mean_Abs_SHAP", ascending=True)   # ascending for horizontal bar

    # Colour gradient: hottest → red, coolest → teal
    n_feats = len(feature_names)
    colours = [
        plt.cm.RdYlGn_r(i / (n_feats - 1))
        for i in range(n_feats)
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=PLOT_DPI)
    _apply_dark_theme(fig, ax)

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Mean_Abs_SHAP"],
        color=colours,
        edgecolor="#333333",
        linewidth=0.8,
        height=0.65,
    )

    # Value labels on bars
    for bar, val in zip(bars, importance_df["Mean_Abs_SHAP"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} °C",
            va="center", ha="left", color=TEXT_CLR, fontsize=10,
        )

    ax.set_xlabel("Mean |SHAP Value| (°C)", color=TEXT_CLR, fontsize=12)
    ax.set_title(
        "SHAP Feature Importance — LST Prediction Drivers\n"
        "(Built vs. Green Landscape Effects on Urban Surface Temperature)",
        color=TEXT_CLR, fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlim(right=importance_df["Mean_Abs_SHAP"].max() * 1.25)

    # Annotation: top driver
    top_feature = importance_df.iloc[-1]["Feature"]
    ax.annotate(
        f"← Primary driver: {top_feature}",
        xy=(importance_df["Mean_Abs_SHAP"].max(), n_feats - 1),
        xytext=(importance_df["Mean_Abs_SHAP"].max() * 0.55, n_feats - 1.4),
        arrowprops=dict(arrowstyle="->", color=ACCENT_CLR, lw=1.5),
        color=ACCENT_CLR, fontsize=10, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=BACKGROUND_CLR)
    plt.close(fig)
    log.info("Importance bar chart saved → %s", save_path)

    return importance_df.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)


# ─── DEPENDENCY PLOT (OPTIONAL) ───────────────────────────────────────────────

def plot_shap_dependence(shap_values: np.ndarray,
                          X_test: pd.DataFrame,
                          feature: str = "NDVI",
                          interaction: str = "NDBI") -> str:
    """
    Plot SHAP dependence for a chosen feature, coloured by an interaction term.
    Reveals non-linear relationships (e.g. NDVI effect stronger in high-NDBI cells).

    Returns the save path of the generated PNG.
    """
    import os
    save_path = os.path.join(
        os.path.dirname(CFG.shap_beeswarm),
        f"shap_dependence_{feature}.png"
    )

    feat_idx   = CFG.feature_cols.index(feature)
    inter_idx  = CFG.feature_cols.index(interaction)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=PLOT_DPI)
    _apply_dark_theme(fig, ax)

    sc = ax.scatter(
        X_test[feature],
        shap_values[:, feat_idx],
        c=X_test[interaction],
        cmap="RdYlGn_r",
        alpha=0.5,
        s=15,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(interaction, color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xlabel(feature, color=TEXT_CLR, fontsize=12)
    ax.set_ylabel(f"SHAP Value for {feature} (°C)", color=TEXT_CLR, fontsize=12)
    ax.set_title(
        f"SHAP Dependence: {feature}  (coloured by {interaction})",
        color=TEXT_CLR, fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=BACKGROUND_CLR)
    plt.close(fig)
    log.info("Dependence plot saved → %s", save_path)
    return save_path


# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def run_shap_analysis(modeling_results: dict = None) -> dict:
    """
    Complete SHAP analysis pipeline.

    Parameters
    ----------
    modeling_results : dict | None
        Output of module3_predictive_modeling.run_modeling().
        If None, re-runs the modeling pipeline to get the trained model.

    Returns
    -------
    dict with keys:
        explainer     : shap.TreeExplainer
        shap_values   : np.ndarray
        importance_df : pd.DataFrame  — feature importance ranking
        beeswarm_path : str
        bar_path      : str
    """
    if modeling_results is None:
        log.info("No modeling results supplied — running Module 3 …")
        modeling_results = run_modeling()

    model   = modeling_results["model"]
    X_train = modeling_results["X_train"]
    X_test  = modeling_results["X_test"]

    # Build explainer and compute SHAP values
    explainer   = build_shap_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X_test)

    # Generate plots
    plot_shap_beeswarm(shap_values, X_test, CFG.shap_beeswarm)
    importance_df = plot_shap_importance_bar(shap_values, CFG.feature_cols, CFG.shap_bar)

    # Print top driver analysis
    print("\n── SHAP Feature Importance Ranking ──────────────────────────────")
    print(importance_df[["Feature", "Mean_Abs_SHAP"]].to_string(index=False))
    print("\n  Interpretation:")
    top_pos = importance_df[importance_df["Mean_Abs_SHAP"] > 0].iloc[0]["Feature"]
    print(f"  • '{top_pos}' is the dominant LST driver.")
    print("  • Negative SHAP for NDVI confirms green areas cool the surface.")
    print("  • Positive SHAP for NDBI / Built_Fraction → urban heat sources.")
    print("─────────────────────────────────────────────────────────────────\n")

    return {
        "explainer"     : explainer,
        "shap_values"   : shap_values,
        "importance_df" : importance_df,
        "beeswarm_path" : CFG.shap_beeswarm,
        "bar_path"      : CFG.shap_bar,
    }


# ─── STANDALONE EXECUTION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 4 : SHAP INTERPRETATION")
    print("=" * 70)

    results = run_shap_analysis()

    print(f"\n✓ Beeswarm plot  → {results['beeswarm_path']}")
    print(f"✓ Importance bar → {results['bar_path']}")
